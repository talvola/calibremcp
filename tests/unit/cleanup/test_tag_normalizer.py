"""Unit + integration tests for the tag normalizer."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from calibre_mcp.cleanup import proposals, tag_normalizer
from calibre_mcp.cleanup.proposals import Proposal
from calibre_mcp.cleanup.tag_normalizer import classify, load_canonical

# ---------------------------------------------------------------------------
# classify() — pure rule logic
# ---------------------------------------------------------------------------


@pytest.fixture
def canonical_vocab() -> dict[str, str]:
    """Mimic Erik's vocabulary at a small scale."""
    names = [
        "Fiction",
        "Nonfiction",
        "Science Fiction",
        "Fantasy",
        "Mystery",
        "Cookbooks",
        "Cooking",
        "Horror",
        "Adventure",
        "Space Opera",
        "Cthulhu",
        "Cocktails",
        "Cozy Mystery",
        "Star Wars",
    ]
    return {n.casefold(): n for n in names}


@pytest.mark.parametrize(
    "value, expected_verdict, reason_substr",
    [
        # 1. Exact case-fold match → approve.
        ("Cthulhu", "approve", "matches canonical"),
        ("cthulhu", "approve", "matches canonical"),
        ("SCIENCE FICTION", "approve", "matches canonical"),
        ("Mystery", "approve", "matches canonical"),
        # 2. Noise patterns → reject.
        ("Horror tales, American", "reject", "LCSH"),
        ("Detective and mystery stories, English", "reject", "LCSH"),
        ("France -- History -- Louis XIV, 1643-1715 -- Fiction", "reject", "LoC"),
        ("FICTION / Christian / Futuristic", "reject", "BISAC"),
        ("Nonfiction / Cooking / General", "reject", "BISAC"),
        ("_isfdb", "reject", "underscore"),
        ("_calibre_uuid", "reject", "underscore"),
        ("prose_contemporary", "reject", "snake"),
        ("media_tie_in", "reject", "snake"),
        ("epub3", "reject", "format/source"),
        ("Amazon", "reject", "format/source"),
        ("kindle", "reject", "format/source"),
        ("2014", "reject", "date-shaped"),
        ("2014-11-10", "reject", "date-shaped"),
        ("", "reject", "empty"),
        ("   ", "reject", "empty"),
        # 3. Defer: legitimate-looking new tags + ambiguous variants.
        ("Mythos", "defer", "novel"),
        ("Dystopian", "defer", "novel"),
        ("sf", "defer", "novel"),  # stylistic variant — needs human/LLM
        ("cookbook", "defer", "novel"),  # singular-vs-plural — same
        ("Wraeththu", "defer", "novel"),  # one-off, looks plausible
    ],
)
def test_classify(value: str, expected_verdict: str, reason_substr: str, canonical_vocab) -> None:
    verdict, reason = classify(value, canonical_vocab)
    assert verdict == expected_verdict, f"{value!r}: got {verdict!r}, reason={reason!r}"
    assert reason_substr.lower() in reason.lower(), f"{value!r}: reason {reason!r} doesn't contain {reason_substr!r}"


def test_classify_short_acronyms_not_misread_as_snake_case(canonical_vocab) -> None:
    """A two-word lowercase tag isn't snake_case (no underscore). Should
    defer, not reject — could be a legit lowercase variant the user has."""
    verdict, _ = classify("space opera", canonical_vocab)
    assert verdict == "approve"  # canonical match (case-insensitive)
    verdict, _ = classify("foobar", canonical_vocab)
    assert verdict == "defer"  # no underscore, so not snake_case rule


@pytest.mark.parametrize(
    "value, expected_verdict, reason_substr",
    [
        # Version markers from EPUB tooling.
        ("(v5.0)",     "reject", "version"),
        ("(v 1.2)",    "reject", "version"),
        # Bare punctuation.
        ("-",          "reject", "punctuation"),
        ("--",         "reject", "punctuation"),
        ("...",        "reject", "punctuation"),
        # Typo prefix: 'cThriller' is a corruption of 'Thriller'.
        ("cThriller",  "reject", "typo"),
        ("aFiction",   "reject", "typo"),
    ],
)
def test_classify_extra_noise_patterns(
    value: str, expected_verdict: str, reason_substr: str, canonical_vocab
) -> None:
    verdict, reason = classify(value, canonical_vocab)
    assert verdict == expected_verdict
    assert reason_substr.lower() in reason.lower()


def test_classify_noise_pattern_wins_over_canonical(canonical_vocab) -> None:
    """Critical: pre-existing junk tags in the vocabulary must not approve
    fresh proposals that match. ``Epub3`` may already be in Erik's tags
    (from prior auto-imports) but new proposals to add it should still be
    rejected as format-marker noise so we don't propagate the pollution."""
    polluted_vocab = {**canonical_vocab, "epub3": "Epub3", "amazon": "amazon"}
    verdict, reason = classify("Epub3", polluted_vocab)
    assert verdict == "reject"
    assert "format/source" in reason.lower()
    verdict, reason = classify("amazon", polluted_vocab)
    assert verdict == "reject"


def test_classify_rejects_author_and_series_names(canonical_vocab) -> None:
    """Author and series names that leak into OPF subjects belong in the
    dedicated authors/series fields, not as tags."""
    names = {"donna leon", "wraeththu", "vorkosigan", "fletch"}
    for v in ("Donna Leon", "wraeththu", "Vorkosigan", "Fletch"):
        verdict, reason = classify(v, canonical_vocab, author_or_series_names=names)
        assert verdict == "reject", f"{v!r} should reject"
        assert "author or series" in reason.lower()


def test_classify_canonical_match_still_works_when_not_noise(canonical_vocab) -> None:
    """Reordering noise-before-canonical doesn't break the legitimate
    canonical-match approval path."""
    verdict, reason = classify("Cthulhu", canonical_vocab)
    assert verdict == "approve"
    assert "canonical" in reason.lower()


# ---------------------------------------------------------------------------
# load_canonical()
# ---------------------------------------------------------------------------


def test_load_canonical_returns_casefolded_keys(tmp_path: Path) -> None:
    db_path = tmp_path / "calibre.db"
    conn = sqlite3.connect(db_path)
    try:
        # Real Calibre schema enforces NOT NULL on tags.name; the loader's
        # NULL/whitespace guards exist to handle stripped-to-empty corner
        # cases, not actual NULL rows.
        conn.executescript("""
            CREATE TABLE tags (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT NOT NULL
            );
            INSERT INTO tags (name) VALUES
              ('Science Fiction'), ('Cthulhu'), ('Cookbooks'), ('  ');
        """)
        conn.commit()
    finally:
        conn.close()

    cal = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cal.row_factory = sqlite3.Row
    try:
        canonical = load_canonical(cal)
    finally:
        cal.close()

    assert canonical == {
        "science fiction": "Science Fiction",
        "cthulhu": "Cthulhu",
        "cookbooks": "Cookbooks",
    }


# ---------------------------------------------------------------------------
# run() — integration with proposals DB
# ---------------------------------------------------------------------------


def _make_calibre_with_tags(path: Path, tag_names: list[str]) -> None:
    """Minimal Calibre schema for tag-normalizer tests. The normalizer also
    queries authors and series for its name-rejection rules, so create those
    empty tables too — real Calibre always has them."""
    conn = sqlite3.connect(path)
    try:
        conn.executescript("""
            CREATE TABLE tags (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT NOT NULL
            );
            CREATE TABLE authors (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT NOT NULL
            );
            CREATE TABLE series (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT NOT NULL
            );
        """)
        for name in tag_names:
            conn.execute("INSERT INTO tags (name) VALUES (?)", (name,))
        conn.commit()
    finally:
        conn.close()


def _seed_tag_proposal(pc: sqlite3.Connection, book_id: int, value: str) -> int:
    run_id = proposals.start_run(
        pc,
        source="opf",
        library_root=Path("/tmp/lib"),  # noqa: S108
        metadata_db_path=Path("/tmp/lib/metadata.db"),  # noqa: S108
        dry_run=False,
    )
    proposals.insert_proposal(
        pc,
        run_id,
        Proposal(
            book_id=book_id,
            field="tags.add",
            proposed_value=value,
            source="opf",
            confidence=0.6,
        ),
    )
    return pc.execute(
        "SELECT id FROM proposals WHERE book_id=? AND proposed_value=?",
        (book_id, value),
    ).fetchone()[0]


def test_run_classifies_existing_proposals(tmp_path: Path) -> None:
    calibre_db = tmp_path / "metadata.db"
    _make_calibre_with_tags(calibre_db, ["Science Fiction", "Cthulhu", "Mystery"])
    proposals_db = tmp_path / "proposals.db"
    pc = proposals.connect(proposals_db)
    seeded = [
        _seed_tag_proposal(pc, 1, "Cthulhu"),  # canonical match → approve
        _seed_tag_proposal(pc, 2, "Mythos"),  # novel → defer
        _seed_tag_proposal(pc, 3, "Horror tales, American"),  # LoC → reject
        _seed_tag_proposal(pc, 4, "FICTION / Christian / Future"),  # BISAC → reject
        _seed_tag_proposal(pc, 5, "_isfdb"),  # noise → reject
    ]
    pc.close()

    summary = tag_normalizer.run(calibre_db=calibre_db, proposals_db=proposals_db)
    assert summary.examined == 5
    assert summary.approved == 1
    assert summary.rejected == 3
    assert summary.deferred == 1

    # Verify the per-row outcomes.
    pc = sqlite3.connect(proposals_db)
    pc.row_factory = sqlite3.Row
    try:
        statuses = {r["id"]: r["status"] for r in pc.execute("SELECT id, status FROM proposals WHERE field='tags.add'")}
    finally:
        pc.close()
    assert statuses[seeded[0]] == "approved"  # Cthulhu
    assert statuses[seeded[1]] == "proposed"  # Mythos (deferred = left as-is)
    assert statuses[seeded[2]] == "rejected"  # Horror tales, American
    assert statuses[seeded[3]] == "rejected"  # BISAC path
    assert statuses[seeded[4]] == "rejected"  # _isfdb


def test_run_dry_run_does_not_update(tmp_path: Path) -> None:
    calibre_db = tmp_path / "metadata.db"
    _make_calibre_with_tags(calibre_db, ["Cthulhu"])
    proposals_db = tmp_path / "proposals.db"
    pc = proposals.connect(proposals_db)
    pid = _seed_tag_proposal(pc, 1, "Cthulhu")
    pc.close()

    summary = tag_normalizer.run(
        calibre_db=calibre_db,
        proposals_db=proposals_db,
        dry_run=True,
    )
    assert summary.approved == 1
    pc = sqlite3.connect(proposals_db)
    try:
        status = pc.execute("SELECT status FROM proposals WHERE id=?", (pid,)).fetchone()[0]
    finally:
        pc.close()
    # Status unchanged in dry-run.
    assert status == "proposed"
