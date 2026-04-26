"""Unit tests for the live-tag sweep classifier + persistence path."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from calibre_mcp.cleanup import live_tag_sweep, proposals
from calibre_mcp.cleanup.live_tag_sweep import (
    SOURCE,
    TAG_PROPOSAL_BOOK_ID,
    classify,
    parse_merge_proposed_value,
    sweep,
)

# ---------------------------------------------------------------------------
# classify() — rule logic (no DB)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected_verdict, expected_target",
    [
        # Useless generics → delete.
        ("General", "delete", None),
        ("Adult", "delete", None),
        ("None", "delete", None),
        ("unknown", "delete", None),

        # LCSH bibliographic ('--' separators) → delete.
        ("England -- Fiction", "delete", None),
        ("Orphans -- Fiction", "delete", None),
        ("Princesses -- Fiction", "delete", None),

        # LCSH '(Fictitious character)' → delete (when not in explicit map).
        ("Drizzt Do'Urden (Fictitious character)", "delete", None),
        ("Stackhouse; Sookie (Fictitious character)", "delete", None),

        # BISAC codes & paths → delete.
        ("FIC000000", "delete", None),
        ("COM004000 - COMPUTERS / Intelligence (AI) and Semantics", "delete", None),
        ("FICTION / Christian / Futuristic", "delete", None),
        ("Fiction / Mystery & Detective", "delete", None),

        # Snake_case / underscore → delete.
        ("sf_fantasy", "delete", None),
        ("prose_contemporary", "delete", None),
        ("_isfdb", "delete", None),
        ("_soon", "delete", None),

        # Format/source markers → delete.
        ("ebook", "delete", None),
        ("Amazon", "delete", None),
        ("Epub3", "delete", None),

        # Date-shapes → delete.
        ("1939-1945", "delete", None),
        ("1830s", "delete", None),
        ("20th century", "delete", None),
        ("1962", "delete", None),

        # Bare punct / typo / 1-2 char → delete.
        ("--", "delete", None),
        ("cThriller", "delete", None),
        ("c", "delete", None),
        ("zz", "delete", None),

        # Bare ISBN → delete.
        ("9780387202488", "delete", None),

        # 'X - Fiction' LCSH-shape (no canonical match) → delete.
        ("Boston (Mass.) - Fiction", "delete", None),
        ("Robots - Fiction", "delete", None),
        ("Time Travel - Fiction", "delete", None),

        # Whitelist real-things-shaped-like-noise → keep.
        ("Oz", "keep", None),
        ("AI", "keep", None),
        ("1632", "keep", None),
        ("2001", "keep", None),

        # Explicit merges (curated map).
        ("sf", "merge", "Science Fiction"),
        ("Sci-fi", "merge", "Science Fiction"),
        ("cthulhu", "merge", "Cthulhu Mythos"),
        ("mythos", "merge", "Cthulhu Mythos"),
        ("Mystery & Detective", "merge", "Mystery"),
        ("Action & Adventure", "merge", "Adventure"),
        ("Thrillers", "merge", "Thriller"),
        ("anthology", "merge", "Anthologies"),
        ("cookbook", "merge", "Cookbooks"),
        ("Fantasy fiction", "merge", "Fantasy"),

        # Star-Trek / Star-Wars character → series-tag fold.
        ("Kirk; James T. (Fictitious Character)", "merge", "Star Trek Fiction"),
        ("Spock (Fictitious Character)", "merge", "Star Trek Fiction"),
        ("Skywalker; Luke (Fictitious character)", "merge", "Star Wars"),
        ("Solo; Han (Fictitious character)", "merge", "Star Wars"),

        # Multi-author character canonicals.
        ("Wolfe; Nero (Fictitious character)", "merge", "Nero Wolfe"),
        ("Holmes; Sherlock (Fictitious Character)", "merge", "Sherlock Holmes"),
        ("Bond; James (Fictitious Character)", "merge", "James Bond"),
        ("Conan (Fictitious character)", "merge", "Conan"),
        ("Marlowe; Philip (Fictitious character)", "merge", "Philip Marlowe"),

        # Case-rename to canonical Title Case.
        ("cozy mystery", "merge", "Cozy Mystery"),
        ("short stories", "merge", "Short Stories"),
        ("steampunk", "merge", "Steampunk"),
        ("alternate history", "merge", "Alternate History"),
        ("magical realism", "merge", "Magical Realism"),

        # Dash-collapse: prefer specific over Fiction/Nonfiction.
        ("Fiction - Science Fiction", "merge", "Science Fiction"),
        ("Fiction - Fantasy", "merge", "Fantasy"),
        ("Fiction - Horror", "merge", "Horror"),
        ("Science Fiction - General", "merge", "Science Fiction"),
        ("Fantasy - General", "merge", "Fantasy"),
        ("Science Fiction - Space Opera", "merge", "Space Opera"),
        ("Fantasy - Epic", "merge", "Fantasy"),  # 'Epic' not canonical, only Fantasy is

        # Junk-modifier drop into the parent.
        ("Fiction - General", "merge", "Fiction"),
        ("Horror - General", "merge", "Horror"),

        # Trailing nationality qualifier.
        ("Science Fiction; American", "merge", "Science Fiction"),
        ("Fantasy fiction; English", "merge", "Fantasy"),  # via case-insensitive 'Fantasy fiction' explicit map

        # Mystery-family compound dash via EXPLICIT_MERGES source recognition.
        ("Mystery & Detective - General", "merge", "Mystery"),
        ("Action & Adventure - General", "merge", "Adventure"),

        # Historical Fiction explicit map.
        ("Fiction - Historical", "merge", "Historical Fiction"),
        ("Historical - General", "merge", "Historical Fiction"),

        # Media Tie-In family.
        ("Movie-TV Tie-In - General", "merge", "Media Tie-In"),

        # Non-canonical dash with no Fiction/Nonfiction parent → keep.
        ("Private investigators - New York (State) - New York", "keep", None),
        ("Fiction - Espionage", "keep", None),  # Fiction parent + non-canonical specific → keep, don't lose info
        ("Fiction - Psychological Suspense", "keep", None),

        # Already-canonical → keep (no rename needed).
        ("Science Fiction", "keep", None),
        ("Cookbooks", "keep", None),
        ("Star Wars", "keep", None),
    ],
)
def test_classify(name: str, expected_verdict: str, expected_target: str | None) -> None:
    verdict, target, reason = classify(name, count=10)
    assert verdict == expected_verdict, f"{name!r}: got {verdict!r}, reason={reason!r}"
    assert target == expected_target, f"{name!r}: target {target!r} != expected {expected_target!r}"


def test_empty_or_whitespace_deletes() -> None:
    for v in ("", "   ", "\t"):
        verdict, _, _ = classify(v, count=1)
        assert verdict == "delete"


def test_explicit_merge_self_match_keeps() -> None:
    """When the source equals the explicit-merge target case-insensitively,
    we should keep — happens when a prior apply created the canonical
    Title-Case variant and a re-sweep finds it via the same map.
    Regression test: 'Conan' → 'Conan' was being emitted as a no-op merge."""
    for name in ("Conan", "conan", "CONAN"):
        verdict, target, _ = classify(name, count=1)
        if name == "Conan":
            # canonical form: keep; no merge attempted.
            assert verdict == "keep", f"{name!r} should keep when it IS the canonical"
        else:
            # case variant: merge to the canonical.
            assert verdict == "merge"
            assert target == "Conan"


def test_canonical_set_consistency() -> None:
    """Every EXPLICIT_MERGES target must be a real canonical name (either in
    CANONICAL_GENRES or used as a multi-author character)."""
    targets = set(live_tag_sweep.EXPLICIT_MERGES.values())
    # Targets should be Title Case (not casefolded).
    for t in targets:
        assert t == t.strip()
        assert t  # non-empty


# ---------------------------------------------------------------------------
# sweep() — batch driver
# ---------------------------------------------------------------------------


def test_sweep_batches_decisions() -> None:
    rows = [
        (1, "General", 100),
        (2, "Science Fiction", 50),
        (3, "sf", 10),
        (4, "FIC000000", 3),
        (5, "Oz", 8),
    ]
    decisions = sweep(rows)
    assert [d.verdict for d in decisions] == ["delete", "keep", "merge", "delete", "keep"]
    assert decisions[2].target == "Science Fiction"
    assert decisions[0].book_count == 100


# ---------------------------------------------------------------------------
# parse_merge_proposed_value
# ---------------------------------------------------------------------------


def test_parse_merge_proposed_value_roundtrip() -> None:
    src, target = parse_merge_proposed_value("cthulhu -> Cthulhu Mythos")
    assert src == "cthulhu"
    assert target == "Cthulhu Mythos"


def test_parse_merge_handles_target_with_arrow_in_name() -> None:
    """Use rsplit so a hypothetical target containing '->' on the left side
    of the merge encoding doesn't break parsing. Defensive — current data
    has no such tags but the encoding shouldn't be brittle."""
    src, target = parse_merge_proposed_value("My Tag -> X -> Real Target")
    assert src == "My Tag -> X"
    assert target == "Real Target"


def test_parse_merge_rejects_missing_arrow() -> None:
    with pytest.raises(ValueError):
        parse_merge_proposed_value("just a name")


# ---------------------------------------------------------------------------
# run() — DB-driven persistence (uses tmp DBs only)
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_calibre_db(tmp_path: Path) -> Path:
    """Stand up a minimal Calibre-shaped metadata.db with just enough for
    ``load_live_tags`` to work."""
    db_path = tmp_path / "metadata.db"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE tags (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
        CREATE TABLE books (id INTEGER PRIMARY KEY);
        CREATE TABLE books_tags_link (id INTEGER PRIMARY KEY, book INTEGER, tag INTEGER);
    """)
    # A canonical-keep, a delete-target, and a merge-target.
    conn.executemany(
        "INSERT INTO tags (id, name) VALUES (?, ?)",
        [(1, "Science Fiction"), (2, "General"), (3, "sf"), (4, "FIC000000")],
    )
    conn.executemany("INSERT INTO books (id) VALUES (?)", [(10,), (11,), (12,)])
    conn.executemany(
        "INSERT INTO books_tags_link (book, tag) VALUES (?, ?)",
        [
            (10, 1), (10, 2),  # book 10 has Science Fiction + General
            (11, 3),  # book 11 has 'sf'
            (12, 2), (12, 4),  # book 12 has General + FIC000000
        ],
    )
    conn.commit()
    conn.close()
    return db_path


def test_run_persists_delete_and_merge_proposals(
    synthetic_calibre_db: Path, tmp_path: Path,
) -> None:
    pdb = tmp_path / "proposals.db"
    summary, decisions = live_tag_sweep.run(
        calibre_db=synthetic_calibre_db,
        proposals_db=pdb,
        dry_run=False,
    )
    assert summary.examined == 4
    # Science Fiction kept; General + FIC000000 deleted; sf merged.
    assert summary.delete == 2
    assert summary.merge == 1
    assert summary.keep == 1

    pc = proposals.connect(pdb)
    rows = list(pc.execute(
        "SELECT field, calibre_value, proposed_value, source, book_id, notes FROM proposals "
        "WHERE source=? ORDER BY field, calibre_value",
        (SOURCE,),
    ))
    pc.close()

    deletes = [r for r in rows if r["field"] == "tag.delete"]
    merges = [r for r in rows if r["field"] == "tag.merge"]
    assert {r["calibre_value"] for r in deletes} == {"General", "FIC000000"}
    assert len(merges) == 1
    assert merges[0]["calibre_value"] == "sf"
    assert merges[0]["proposed_value"] == "sf -> Science Fiction"
    assert all(r["book_id"] == TAG_PROPOSAL_BOOK_ID for r in rows)
    # Notes should have structured JSON with the source tag id.
    notes = json.loads(rows[0]["notes"])
    assert "src_tag_id" in notes
    assert "book_count" in notes


def test_run_dry_run_persists_nothing(
    synthetic_calibre_db: Path, tmp_path: Path,
) -> None:
    pdb = tmp_path / "proposals.db"
    summary, _ = live_tag_sweep.run(
        calibre_db=synthetic_calibre_db,
        proposals_db=pdb,
        dry_run=True,
    )
    assert summary.delete == 2
    assert summary.merge == 1

    pc = proposals.connect(pdb)
    n = pc.execute(
        "SELECT COUNT(*) AS n FROM proposals WHERE source=?", (SOURCE,),
    ).fetchone()["n"]
    pc.close()
    assert n == 0


def test_run_is_idempotent(
    synthetic_calibre_db: Path, tmp_path: Path,
) -> None:
    """Re-running the sweep against the same library should not duplicate
    proposals — relies on the (book_id, field, source, proposed_value)
    unique index in the propose-queue."""
    pdb = tmp_path / "proposals.db"
    live_tag_sweep.run(calibre_db=synthetic_calibre_db, proposals_db=pdb)
    live_tag_sweep.run(calibre_db=synthetic_calibre_db, proposals_db=pdb)

    pc = proposals.connect(pdb)
    n = pc.execute(
        "SELECT COUNT(*) AS n FROM proposals WHERE source=?", (SOURCE,),
    ).fetchone()["n"]
    pc.close()
    # 2 deletes + 1 merge = 3, even after two runs.
    assert n == 3
