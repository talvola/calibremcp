"""Unit tests for author_normalizer — key detection, canonical selection,
sweep logic, and apply-layer integration (author.merge in plan/execute)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from calibre_mcp.cleanup import author_normalizer as an
from calibre_mcp.cleanup import apply as apply_mod
from calibre_mcp.cleanup import proposals
from calibre_mcp.cleanup.proposals import Proposal


# ---------------------------------------------------------------------------
# _normalize_key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Arthur C. Clarke", "arthur c clarke"),
        ("Arthur C Clarke", "arthur c clarke"),
        ("Robert A. Heinlein", "robert a heinlein"),
        ("Robert A Heinlein", "robert a heinlein"),
        ("Brian M. Sammons.", "brian m sammons"),   # trailing period stripped
        ("Aristotle.", "aristotle"),
        # Consecutive single-letter initials collapse to one token
        ("J.R.R. Tolkien", "jrr tolkien"),
        ("JRR Tolkien", "jrr tolkien"),
        ("George R. R. Martin", "george rr martin"),  # R. R. → rr
        ("George R R Martin", "george rr martin"),
        ("George R .R. Martin", "george rr martin"),  # space before period
        ("George R.R. Martin", "george rr martin"),
        ("H. P. Lovecraft", "hp lovecraft"),          # H. P. → hp
        ("H P Lovecraft", "hp lovecraft"),
        ("H.P. Lovecraft", "hp lovecraft"),
        ("B. V. Larson", "bv larson"),
        ("B.V. Larson", "bv larson"),
        ("M.C. Beaton", "mc beaton"),
        ("Mc Beaton", "mc beaton"),
        # Underscore normalisation
        ("MICHAEL_P. KUBE-MCDOWELL", "michael p kube-mcdowell"),
        ("Michael P. Kube-McDowell", "michael p kube-mcdowell"),
        # Single initial not adjacent to another single → not collapsed
        ("L. Neil Smith", "l neil smith"),
        ("L Neil Smith", "l neil smith"),
    ],
)
def test_normalize_key(name: str, expected: str) -> None:
    assert an._normalize_key(name) == expected


# ---------------------------------------------------------------------------
# _pick_canonical
# ---------------------------------------------------------------------------


def test_pick_canonical_prefers_periods() -> None:
    variants = [("Arthur C Clarke", 5), ("Arthur C. Clarke", 3)]
    assert an._pick_canonical(variants) == "Arthur C. Clarke"


def test_pick_canonical_tiebreaks_by_book_count() -> None:
    variants = [("Arthur C. Clarke", 5), ("Arthur C. Clarke ", 3)]
    # Both have 1 period; most books wins (stripped names differ only in trailing space)
    result = an._pick_canonical(variants)
    assert result in ("Arthur C. Clarke", "Arthur C. Clarke")


def test_pick_canonical_strips_trailing_period() -> None:
    variants = [("Aristotle.", 20), ("Aristotle", 5)]
    # "Aristotle." has 1 period but it's a trailing period — after strip it has 0.
    # "Aristotle" has 0 periods.
    # Tie → most books ("Aristotle." with 20). Trailing period stripped → "Aristotle".
    assert an._pick_canonical(variants) == "Aristotle"


def test_pick_canonical_strips_trailing_period_from_winner() -> None:
    # Only one variant has a trailing period; it also has more books.
    variants = [("Brian M. Sammons.", 10), ("Brian M. Sammons", 8), ("Brian M Sammons", 3)]
    # "Brian M. Sammons." → stripped = "Brian M. Sammons" → 1 period → score (1, 10)
    # "Brian M. Sammons" → 1 period → score (1, 8)
    # tiebreak: 10 > 8 → "Brian M. Sammons." wins → stripped → "Brian M. Sammons"
    assert an._pick_canonical(variants) == "Brian M. Sammons"


def test_pick_canonical_jrr_tolkien() -> None:
    variants = [("J.R.R. Tolkien", 50), ("JRR Tolkien", 2)]
    assert an._pick_canonical(variants) == "J.R.R. Tolkien"


def test_pick_canonical_george_martin() -> None:
    variants = [("George R.R. Martin", 40), ("George Rr Martin", 1)]
    assert an._pick_canonical(variants) == "George R.R. Martin"


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------


def test_sweep_produces_merge_per_non_canonical() -> None:
    clusters = [
        [("Arthur C. Clarke", 10), ("Arthur C Clarke", 2)],
    ]
    decisions = an.sweep(clusters)
    assert len(decisions) == 1
    d = decisions[0]
    assert d.src_name == "Arthur C Clarke"
    assert d.canonical == "Arthur C. Clarke"
    assert d.book_count == 2


def test_sweep_skips_exact_canonical() -> None:
    clusters = [
        [("H. P. Lovecraft", 100), ("H P Lovecraft", 5)],
    ]
    decisions = an.sweep(clusters)
    # Only the variant (H P Lovecraft) should be in the merge list.
    assert len(decisions) == 1
    assert decisions[0].src_name == "H P Lovecraft"
    assert decisions[0].canonical == "H. P. Lovecraft"


def test_sweep_three_way_cluster() -> None:
    clusters = [
        [("Brian M Sammons", 1), ("Brian M. Sammons", 5), ("Brian M. Sammons.", 3)],
    ]
    decisions = an.sweep(clusters)
    # canonical = "Brian M. Sammons" (max periods after strip, tiebreak by count)
    # "Brian M. Sammons" is exact canonical → skip
    # "Brian M Sammons" → merge to "Brian M. Sammons"
    # "Brian M. Sammons." → name != canonical ("Brian M. Sammons." != "Brian M. Sammons") → merge
    names = {d.src_name for d in decisions}
    assert "Brian M Sammons" in names
    assert "Brian M. Sammons." in names
    assert "Brian M. Sammons" not in names


def test_sweep_trailing_period_variant_gets_merge() -> None:
    """Trailing-period variants must be merged even though stripped they equal canonical."""
    clusters = [
        [("Aristotle", 50), ("Aristotle.", 2)],
    ]
    decisions = an.sweep(clusters)
    assert len(decisions) == 1
    assert decisions[0].src_name == "Aristotle."
    assert decisions[0].canonical == "Aristotle"


def test_sweep_empty_clusters() -> None:
    assert an.sweep([]) == []


def test_sweep_single_member_clusters_produce_nothing() -> None:
    clusters = [[("Only Author", 10)]]
    assert an.sweep(clusters) == []


# ---------------------------------------------------------------------------
# EXPLICIT_OVERRIDES — typo periods and cross-cluster unification
# ---------------------------------------------------------------------------


def test_sweep_override_reverses_typo_period() -> None:
    """'Henry. Petroski' wins by period count, but override corrects direction."""
    clusters = [[("Henry Petroski", 7), ("Henry. Petroski", 1)]]
    decisions = an.sweep(clusters)
    # Only the typo should be merged; the correct form should not appear as src.
    assert len(decisions) == 1
    assert decisions[0].src_name == "Henry. Petroski"
    assert decisions[0].canonical == "Henry Petroski"


def test_sweep_override_unifies_cross_cluster() -> None:
    """G. R. R. Martin and G. R.R. Martin are separate clusters but both
    should converge on 'George R.R. Martin'."""
    cluster_rr = [("George R.R. Martin", 40), ("George Rr Martin", 1)]
    cluster_r_r = [("George R. R. Martin", 5), ("George R R Martin", 2), ("George R .R. Martin", 1)]
    decisions = an.sweep([cluster_rr, cluster_r_r])
    targets = {d.canonical for d in decisions}
    sources = {d.src_name for d in decisions}
    # All non-canonical variants should merge to the compact form.
    assert targets == {"George R.R. Martin"}
    assert "George R.R. Martin" not in sources  # never a source
    assert "George Rr Martin" in sources
    assert "George R. R. Martin" in sources
    assert "George R R Martin" in sources
    assert "George R .R. Martin" in sources


def test_sweep_override_preserves_capitalisation() -> None:
    """'Keith R. A. Decandido' loses the capital D; override restores it."""
    clusters = [[("Keith R A DeCandido", 2), ("Keith R. A. Decandido", 1)]]
    decisions = an.sweep(clusters)
    for d in decisions:
        assert d.canonical == "Keith R.A. DeCandido"
    sources = {d.src_name for d in decisions}
    assert "Keith R A DeCandido" in sources
    assert "Keith R. A. Decandido" in sources


def test_sweep_explicit_overrides_cover_expected_cases() -> None:
    """Smoke-check that EXPLICIT_OVERRIDES contains entries for known problem cases."""
    overrides = an.EXPLICIT_OVERRIDES
    assert overrides.get("Henry. Petroski") == "Henry Petroski"
    assert overrides.get("Tyler. Cowen") == "Tyler Cowen"
    assert overrides.get("George R. R. Martin") == "George R.R. Martin"
    assert overrides.get("P. C. Cast") == "P.C. Cast"
    assert overrides.get("Keith R. A. Decandido") == "Keith R.A. DeCandido"
    assert overrides.get("Ursula K. le Guin") == "Ursula K. Le Guin"


# ---------------------------------------------------------------------------
# parse_merge_proposed_value
# ---------------------------------------------------------------------------


def test_parse_merge_proposed_value_roundtrip() -> None:
    src, target = an.parse_merge_proposed_value("Arthur C Clarke -> Arthur C. Clarke")
    assert src == "Arthur C Clarke"
    assert target == "Arthur C. Clarke"


def test_parse_merge_proposed_value_rejects_missing_arrow() -> None:
    with pytest.raises(ValueError, match="missing ' -> '"):
        an.parse_merge_proposed_value("Arthur C Clarke")


# ---------------------------------------------------------------------------
# apply integration: _apply_author_ops and _format_authors
# ---------------------------------------------------------------------------


def test_apply_author_ops_renames_variant() -> None:
    current = ["Arthur C Clarke", "Isaac Asimov"]
    changes = {"arthur c clarke": "Arthur C. Clarke"}
    result = apply_mod._apply_author_ops(current, changes)
    assert result == ["Arthur C. Clarke", "Isaac Asimov"]


def test_apply_author_ops_dedupes_when_canonical_present() -> None:
    current = ["Arthur C. Clarke", "Arthur C Clarke", "Isaac Asimov"]
    changes = {"arthur c clarke": "Arthur C. Clarke"}
    result = apply_mod._apply_author_ops(current, changes)
    assert result.count("Arthur C. Clarke") == 1
    assert "Isaac Asimov" in result
    assert "Arthur C Clarke" not in result


def test_apply_author_ops_preserves_unchanged_authors() -> None:
    current = ["Isaac Asimov", "Robert Heinlein"]
    changes = {"arthur c clarke": "Arthur C. Clarke"}  # not in current
    result = apply_mod._apply_author_ops(current, changes)
    assert result == ["Isaac Asimov", "Robert Heinlein"]


def test_format_authors_single() -> None:
    assert apply_mod._format_authors(["Robert A. Heinlein"]) == "Robert A. Heinlein"


def test_format_authors_multiple() -> None:
    result = apply_mod._format_authors(["Author A", "Author B", "Author C"])
    assert result == "Author A & Author B & Author C"


def test_format_authors_empty() -> None:
    assert apply_mod._format_authors([]) == ""


# ---------------------------------------------------------------------------
# apply integration: plan() expands author.merge into per-book commands
# ---------------------------------------------------------------------------


def _make_calibre_db_with_authors(
    library_path: Path, books: dict[int, list[str]],
) -> None:
    """Seed a minimal Calibre DB with books and authors."""
    library_path.mkdir(parents=True, exist_ok=True)
    db_path = library_path / "metadata.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript("""
            CREATE TABLE identifiers (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              book INTEGER NOT NULL, type TEXT NOT NULL, val TEXT NOT NULL
            );
            CREATE TABLE authors (
              id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL
            );
            CREATE TABLE books_authors_link (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              book INTEGER NOT NULL, author INTEGER NOT NULL
            );
            CREATE TABLE tags (id INTEGER PRIMARY KEY, name TEXT);
            CREATE TABLE books_tags_link (id INTEGER PRIMARY KEY, book INTEGER, tag INTEGER);
        """)
        name_to_id: dict[str, int] = {}
        for book_id, author_names in books.items():
            for name in author_names:
                if name not in name_to_id:
                    cur = conn.execute("INSERT INTO authors (name) VALUES (?)", (name,))
                    name_to_id[name] = cur.lastrowid
                conn.execute(
                    "INSERT INTO books_authors_link (book, author) VALUES (?, ?)",
                    (book_id, name_to_id[name]),
                )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture()
def db(tmp_path: Path) -> sqlite3.Connection:
    db_path = tmp_path / "proposals.db"
    conn = proposals.connect(db_path)
    yield conn
    conn.close()


def _seed_author_merge(
    db: sqlite3.Connection,
    *,
    src: str,
    canonical: str,
    status: str = "approved",
) -> int:
    run_id = proposals.start_run(
        db,
        source="author_normalizer",
        library_root=Path("/tmp/lib"),  # noqa: S108
        metadata_db_path=Path("/tmp/lib/metadata.db"),  # noqa: S108
        dry_run=False,
    )
    encoded = f"{src} -> {canonical}"
    proposals.insert_proposal(
        db,
        run_id,
        Proposal(
            book_id=0,
            field="author.merge",
            calibre_value=src,
            proposed_value=encoded,
            source="author_normalizer",
            confidence=1.0,
        ),
    )
    pid = db.execute(
        "SELECT id FROM proposals WHERE field='author.merge' AND calibre_value=?",
        (src,),
    ).fetchone()["id"]
    if status != "proposed":
        proposals.set_status(db, pid, status)
    return pid


def test_plan_author_merge_expands_to_each_affected_book(
    tmp_path: Path, db: sqlite3.Connection,
) -> None:
    library = tmp_path / "lib"
    _make_calibre_db_with_authors(library, {
        10: ["Arthur C Clarke", "Isaac Asimov"],
        20: ["Arthur C Clarke"],
        30: ["Isaac Asimov"],  # not affected
    })
    pid = _seed_author_merge(db, src="Arthur C Clarke", canonical="Arthur C. Clarke")

    cmds = list(apply_mod.plan(db, library_path=library))
    by_book = {c.book_id: c for c in cmds}
    assert set(by_book) == {10, 20}
    for cmd in by_book.values():
        assert cmd.proposal_ids == ()
        assert cmd.author_op_proposal_ids == (pid,)
        authors_arg = next(a for a in cmd.argv if a.startswith("--field=authors:"))
        assert "Arthur C Clarke" not in authors_arg
        assert "Arthur C. Clarke" in authors_arg


def test_plan_author_merge_dedupes_when_canonical_already_present(
    tmp_path: Path, db: sqlite3.Connection,
) -> None:
    library = tmp_path / "lib"
    _make_calibre_db_with_authors(library, {
        10: ["Arthur C. Clarke", "Arthur C Clarke"],
    })
    _seed_author_merge(db, src="Arthur C Clarke", canonical="Arthur C. Clarke")

    cmds = list(apply_mod.plan(db, library_path=library))
    assert len(cmds) == 1
    authors_arg = next(a for a in cmds[0].argv if a.startswith("--field=authors:"))
    authors_value = authors_arg.removeprefix("--field=authors:")
    parts = [p.strip() for p in authors_value.split(" & ")]
    assert parts.count("Arthur C. Clarke") == 1
    assert "Arthur C Clarke" not in parts


def test_plan_author_merge_no_affected_books_yields_nothing(
    tmp_path: Path, db: sqlite3.Connection,
) -> None:
    library = tmp_path / "lib"
    _make_calibre_db_with_authors(library, {10: ["Isaac Asimov"]})
    _seed_author_merge(db, src="Arthur C Clarke", canonical="Arthur C. Clarke")
    cmds = list(apply_mod.plan(db, library_path=library))
    assert cmds == []
