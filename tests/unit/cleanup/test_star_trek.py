"""Unit tests for star_trek — title parsing and proposal generation."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from calibre_mcp.cleanup import star_trek, proposals
from calibre_mcp.cleanup.star_trek import ParsedBook, parse_title, _YEAR_PREFIX_RE


# ---------------------------------------------------------------------------
# parse_title
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected_series, expected_index, expected_title",
    [
        # Standard TOS numbered
        (
            "Star Trek: The Original Series - 001 - Mission to Horatius",
            "Star Trek: The Original Series", 1, "Mission to Horatius",
        ),
        # TNG numbered — three digits
        (
            "Star Trek: The Next Generation - 018 - Vendetta",
            "Star Trek: The Next Generation", 18, "Vendetta",
        ),
        # Corp of Engineers — long series
        (
            "Star Trek: Corp of Engineers - 074 - Remembrance of Things Past - Book 2",
            "Star Trek: Corp of Engineers", 74, "Remembrance of Things Past - Book 2",
        ),
        # Single-digit index
        (
            "Star Trek: Gateways - 1 - One Small Step",
            "Star Trek: Gateways", 1, "One Small Step",
        ),
        # Bantam Episodes sub-series embedded in TOS
        (
            "Star Trek: The Original Series - Bantam Episodes - 001 - Charlie's Law",
            "Star Trek: The Original Series - Bantam Episodes", 1, "Charlie's Law",
        ),
        # Bantam Novels
        (
            "Star Trek: The Original Series - Bantam Novels - 007 - Vulcan",
            "Star Trek: The Original Series - Bantam Novels", 7, "Vulcan",
        ),
        # Nested crossover — series_name from FIRST sep, real_title from LAST sep
        (
            "Star Trek: 24th Century Crossover - 001 - Destiny - 01 - Gods of Night",
            "Star Trek: 24th Century Crossover", 1, "Gods of Night",
        ),
        (
            "Star Trek: 24th Century Crossover - 005 - Typhon Pact - 01 - Zero Sum Game",
            "Star Trek: 24th Century Crossover", 5, "Zero Sum Game",
        ),
        # Sub-arc title containing a dash (not a separator — no digit before the dash)
        (
            "Star Trek: Deep Space Nine - 030 - The Rebels 1 - The Conquered",
            "Star Trek: Deep Space Nine", 30, "The Rebels 1 - The Conquered",
        ),
        (
            "Star Trek: The Original Series - 101 - My Brother's Keeper 1 - Republic",
            "Star Trek: The Original Series", 101, "My Brother's Keeper 1 - Republic",
        ),
        # Shatnerverse (no series set in Calibre — tests the no-series path)
        (
            "Star Trek: Shatnerverse - 003 - Avenger",
            "Star Trek: Shatnerverse", 3, "Avenger",
        ),
        # Starfleet Academy sub-series
        (
            "Star Trek: Starfleet Academy - The Next Generation - 01 - Worf's First Adventure",
            "Star Trek: Starfleet Academy - The Next Generation", 1, "Worf's First Adventure",
        ),
        # Mirror Universe
        (
            "Star Trek: Mirror Universe - 001 - Glass Empires",
            "Star Trek: Mirror Universe", 1, "Glass Empires",
        ),
        # Two-digit index, no leading zero
        (
            "Star Trek: Section 31 - 1 - Cloak",
            "Star Trek: Section 31", 1, "Cloak",
        ),
    ],
)
def test_parse_title_parametrized(
    raw: str,
    expected_series: str,
    expected_index: int,
    expected_title: str,
) -> None:
    result = parse_title(raw)
    assert result is not None
    series_name, series_index, real_title = result
    assert series_name == expected_series
    assert series_index == expected_index
    assert real_title == expected_title


@pytest.mark.parametrize(
    "raw",
    [
        # No 'Star Trek:' prefix — cookbooks etc.
        "Star Trek - Food Recipes from the Replicator: The Most Delicious Dishes",
        "Star Trek Cocktails: A Stellar Compendium",
        # No separator at all
        "Star Trek: Klingon",
        "Star Trek: The Captain's Table",
        # Logs — number at end, no trailing ' - Title'
        "Star Trek: The Original Series - Log 01",
        "Star Trek: The Original Series - Log 10",
        # Unnumbered suffix (XXX is not digits)
        "Star Trek: The Original Series - XXX - The Aliens Are Coming!",
        # Already-stripped clean title (no separator)
        "Gods of Night",
        "Mission to Horatius",
        # 'Star Trek The Next Generation' without colon
        "Star Trek The Next Generation: Available Light",
        # Prometheus without colon
        "Star Trek Prometheus: The Root of All Rage",
    ],
)
def test_parse_title_no_match(raw: str) -> None:
    assert parse_title(raw) is None


def test_parse_title_nested_uses_last_separator_for_title() -> None:
    """The real title comes from the LAST separator, not the first."""
    result = parse_title(
        "Star Trek: 24th Century Crossover - 013 - The Fall - 01 - Revelation and Dust"
    )
    assert result is not None
    series_name, series_index, real_title = result
    assert series_name == "Star Trek: 24th Century Crossover"
    assert series_index == 13
    assert real_title == "Revelation and Dust"


def test_parse_title_sub_arc_dash_not_split() -> None:
    """A dash in the real title that is NOT preceded by a digit is not a separator."""
    result = parse_title(
        "Star Trek: The Next Generation - 053 - Invasion! 2 - The Soldiers of Fear"
    )
    assert result is not None
    _, _, real_title = result
    assert real_title == "Invasion! 2 - The Soldiers of Fear"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _make_calibre_db(
    path: Path,
    books: list[tuple[int, str, float, str]],  # (id, title, idx, series_name or "")
) -> None:
    """Minimal Calibre DB with optional series assignments."""
    conn = sqlite3.connect(path)
    try:
        conn.executescript("""
            CREATE TABLE books (
              id INTEGER PRIMARY KEY,
              title TEXT NOT NULL,
              series_index REAL NOT NULL DEFAULT 1.0
            );
            CREATE TABLE series (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
            CREATE TABLE books_series_link (
              id INTEGER PRIMARY KEY, book INTEGER, series INTEGER
            );
            CREATE TABLE authors (id INTEGER PRIMARY KEY, name TEXT);
            CREATE TABLE books_authors_link (id INTEGER PRIMARY KEY, book INTEGER, author INTEGER);
            CREATE TABLE identifiers (id INTEGER PRIMARY KEY, book INTEGER, type TEXT, val TEXT);
            CREATE TABLE tags (id INTEGER PRIMARY KEY, name TEXT);
            CREATE TABLE books_tags_link (id INTEGER PRIMARY KEY, book INTEGER, tag INTEGER);
        """)
        series_map: dict[str, int] = {}
        for book_id, title, idx, sname in books:
            conn.execute(
                "INSERT INTO books (id, title, series_index) VALUES (?, ?, ?)",
                (book_id, title, idx),
            )
            if sname:
                if sname not in series_map:
                    sid = len(series_map) + 1
                    series_map[sname] = sid
                    conn.execute("INSERT INTO series (id, name) VALUES (?, ?)", (sid, sname))
                conn.execute(
                    "INSERT INTO books_series_link (book, series) VALUES (?, ?)",
                    (book_id, series_map[sname]),
                )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture()
def prop_db(tmp_path: Path) -> sqlite3.Connection:
    conn = proposals.connect(tmp_path / "proposals.db")
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# run() integration
# ---------------------------------------------------------------------------


def test_run_dry_run_parses_books(tmp_path: Path) -> None:
    _make_calibre_db(tmp_path / "metadata.db", [
        (1, "Star Trek: The Original Series - 001 - Mission to Horatius", 1.0,
         "Star Trek: The Original Series"),
        (2, "Star Trek: Corp of Engineers - 001 - Belly of the Beast", 1.0,
         "Star Trek: Corp of Engineers"),
    ])
    summary, books = star_trek.run(
        calibre_db=tmp_path / "metadata.db",
        proposals_db=tmp_path / "proposals.db",
        dry_run=True,
    )
    assert summary.examined == 2
    assert summary.proposals_emitted == 0
    assert len(books) == 2
    titles = {b.proposed_title for b in books}
    assert titles == {"Mission to Horatius", "Belly of the Beast"}


def test_run_title_only_for_books_with_series(tmp_path: Path) -> None:
    """Books that already have a Calibre series get only a title proposal."""
    _make_calibre_db(tmp_path / "metadata.db", [
        (1, "Star Trek: The Original Series - 001 - Mission to Horatius", 1.0,
         "Star Trek: The Original Series"),
    ])
    proposals_db = tmp_path / "proposals.db"
    summary, books = star_trek.run(
        calibre_db=tmp_path / "metadata.db",
        proposals_db=proposals_db,
        dry_run=False,
    )
    assert summary.proposals_emitted == 1

    conn = sqlite3.connect(proposals_db)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT field FROM proposals WHERE source='star_trek' ORDER BY field"
        ).fetchall()
    finally:
        conn.close()

    fields = {r["field"] for r in rows}
    assert fields == {"title"}


def test_run_title_value_correct(tmp_path: Path) -> None:
    _make_calibre_db(tmp_path / "metadata.db", [
        (1, "Star Trek: Voyager - 003 - Ragnarok", 3.0, "Star Trek: Voyager"),
    ])
    proposals_db = tmp_path / "proposals.db"
    star_trek.run(
        calibre_db=tmp_path / "metadata.db",
        proposals_db=proposals_db,
        dry_run=False,
    )
    conn = sqlite3.connect(proposals_db)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT proposed_value, calibre_value FROM proposals WHERE field='title'"
        ).fetchone()
    finally:
        conn.close()
    assert row["proposed_value"] == "Ragnarok"
    assert row["calibre_value"] == "Star Trek: Voyager - 003 - Ragnarok"


def test_run_series_proposals_for_no_series_books(tmp_path: Path) -> None:
    """Books with no Calibre series get title + series + series_index proposals."""
    _make_calibre_db(tmp_path / "metadata.db", [
        # No series ("") — Shatnerverse with wrong series_index
        (1, "Star Trek: Shatnerverse - 003 - Avenger", 1.0, ""),
    ])
    proposals_db = tmp_path / "proposals.db"
    star_trek.run(
        calibre_db=tmp_path / "metadata.db",
        proposals_db=proposals_db,
        dry_run=False,
    )
    conn = sqlite3.connect(proposals_db)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT field, proposed_value FROM proposals ORDER BY field"
        ).fetchall()
    finally:
        conn.close()

    fields = {r["field"]: r["proposed_value"] for r in rows}
    assert "title" in fields
    assert fields["title"] == "Avenger"
    assert "series" in fields
    assert fields["series"] == "Star Trek: Shatnerverse"
    assert "series_index" in fields
    assert float(fields["series_index"]) == pytest.approx(3.0)


def test_run_no_series_index_when_already_correct(tmp_path: Path) -> None:
    """No series_index proposal when the current index already equals the parsed one."""
    _make_calibre_db(tmp_path / "metadata.db", [
        (1, "Star Trek: Section 31 - 1 - Cloak", 1.0, ""),  # index=1, parsed=1 → no change
    ])
    proposals_db = tmp_path / "proposals.db"
    star_trek.run(
        calibre_db=tmp_path / "metadata.db",
        proposals_db=proposals_db,
        dry_run=False,
    )
    conn = sqlite3.connect(proposals_db)
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM proposals WHERE field='series_index'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert count == 0


def test_run_nested_crossover_title(tmp_path: Path) -> None:
    """24th Century Crossover nested titles: real_title from last separator."""
    _make_calibre_db(tmp_path / "metadata.db", [
        (
            1,
            "Star Trek: 24th Century Crossover - 001 - Destiny - 01 - Gods of Night",
            1.0,
            "Star Trek: Destiny",  # series already set correctly
        ),
    ])
    proposals_db = tmp_path / "proposals.db"
    star_trek.run(
        calibre_db=tmp_path / "metadata.db",
        proposals_db=proposals_db,
        dry_run=False,
    )
    conn = sqlite3.connect(proposals_db)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT field, proposed_value FROM proposals ORDER BY field"
        ).fetchall()
    finally:
        conn.close()
    fields = {r["field"]: r["proposed_value"] for r in rows}
    # Only title proposal — series already set
    assert set(fields.keys()) == {"title"}
    assert fields["title"] == "Gods of Night"


def test_run_skips_no_match(tmp_path: Path) -> None:
    """Titles without a ' - NNN - ' separator are skipped."""
    _make_calibre_db(tmp_path / "metadata.db", [
        (1, "Star Trek: The Original Series - Log 01", 1.0,
         "Star Trek: The Original Series"),
        (2, "Star Trek: Klingon", 1.0, ""),
        (3, "Star Trek: The Original Series - 001 - Mission to Horatius", 1.0,
         "Star Trek: The Original Series"),
    ])
    summary, books = star_trek.run(
        calibre_db=tmp_path / "metadata.db",
        proposals_db=tmp_path / "proposals.db",
        dry_run=True,
    )
    assert summary.skipped_no_match == 2
    assert len(books) == 1


def test_run_skips_no_series_index_when_already_one(tmp_path: Path) -> None:
    """No series_index proposal when parsed index equals current index of 1.0."""
    _make_calibre_db(tmp_path / "metadata.db", [
        (1, "Star Trek: Shatnerverse - 001 - The Ashes of Eden", 1.0, ""),
    ])
    star_trek.run(
        calibre_db=tmp_path / "metadata.db",
        proposals_db=tmp_path / "proposals.db",
        dry_run=False,
    )
    conn = sqlite3.connect(tmp_path / "proposals.db")
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM proposals WHERE field='series_index'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert count == 0


def test_run_is_idempotent(tmp_path: Path) -> None:
    """Running twice produces the same proposals, not duplicates."""
    _make_calibre_db(tmp_path / "metadata.db", [
        (1, "Star Trek: The Original Series - 001 - Mission to Horatius", 1.0,
         "Star Trek: The Original Series"),
    ])
    proposals_db = tmp_path / "proposals.db"
    star_trek.run(calibre_db=tmp_path / "metadata.db", proposals_db=proposals_db)
    star_trek.run(calibre_db=tmp_path / "metadata.db", proposals_db=proposals_db)

    conn = sqlite3.connect(proposals_db)
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM proposals WHERE field='title'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert count == 1


def test_run_bantam_episodes(tmp_path: Path) -> None:
    """Bantam Episodes sub-series embedded within TOS."""
    _make_calibre_db(tmp_path / "metadata.db", [
        (1, "Star Trek: The Original Series - Bantam Episodes - 001 - Charlie's Law",
         1.0, "Star Trek: The Original Series - Bantam Episodes"),
    ])
    star_trek.run(
        calibre_db=tmp_path / "metadata.db",
        proposals_db=tmp_path / "proposals.db",
        dry_run=False,
    )
    conn = sqlite3.connect(tmp_path / "proposals.db")
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT field, proposed_value FROM proposals WHERE field='title'"
        ).fetchone()
    finally:
        conn.close()
    assert row["proposed_value"] == "Charlie's Law"


# ---------------------------------------------------------------------------
# _YEAR_PREFIX_RE and run_year_fixup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2298 - The Sundered", "The Sundered"),
        ("2311 - Serpents Among the Ruins", "Serpents Among the Ruins"),
        ("2328-2346 - The Art of the Impossible", "The Art of the Impossible"),
        ("2319 - One Constant Star", "One Constant Star"),
        ("2355-2357 - Deny Thy Father", "Deny Thy Father"),
    ],
)
def test_year_prefix_re_strips_year(raw: str, expected: str) -> None:
    m = _YEAR_PREFIX_RE.match(raw)
    assert m is not None
    assert raw[m.end():].strip() == expected


@pytest.mark.parametrize(
    "raw",
    [
        "The Sundered",             # already clean
        "234 - Too short",          # only 3 digits, not a year
        "Star Trek: Something",     # doesn't start with digits
    ],
)
def test_year_prefix_re_no_match(raw: str) -> None:
    assert _YEAR_PREFIX_RE.match(raw) is None


def test_run_year_fixup_emits_title_proposals(tmp_path: Path) -> None:
    _make_calibre_db(tmp_path / "metadata.db", [
        (1, "2298 - The Sundered", 1.0, "Star Trek: The Lost Era"),
        (2, "2311 - Serpents Among the Ruins", 2.0, "Star Trek: The Lost Era"),
        (3, "2328-2346 - The Art of the Impossible", 3.0, "Star Trek: The Lost Era"),
    ])
    summary, books = star_trek.run_year_fixup(
        calibre_db=tmp_path / "metadata.db",
        proposals_db=tmp_path / "proposals.db",
        series_name="Star Trek: The Lost Era",
        dry_run=False,
    )
    assert summary.examined == 3
    assert summary.proposals_emitted == 3
    assert {b.proposed_title for b in books} == {
        "The Sundered", "Serpents Among the Ruins", "The Art of the Impossible"
    }


def test_run_year_fixup_dry_run(tmp_path: Path) -> None:
    _make_calibre_db(tmp_path / "metadata.db", [
        (1, "2298 - The Sundered", 1.0, "Star Trek: The Lost Era"),
    ])
    summary, books = star_trek.run_year_fixup(
        calibre_db=tmp_path / "metadata.db",
        proposals_db=tmp_path / "proposals.db",
        series_name="Star Trek: The Lost Era",
        dry_run=True,
    )
    assert summary.proposals_emitted == 0
    assert len(books) == 1
    assert books[0].proposed_title == "The Sundered"


def test_run_year_fixup_skips_clean_titles(tmp_path: Path) -> None:
    _make_calibre_db(tmp_path / "metadata.db", [
        (1, "The Sundered", 1.0, "Star Trek: The Lost Era"),  # already clean
    ])
    summary, _ = star_trek.run_year_fixup(
        calibre_db=tmp_path / "metadata.db",
        proposals_db=tmp_path / "proposals.db",
        series_name="Star Trek: The Lost Era",
        dry_run=False,
    )
    assert summary.skipped_no_match == 1
    assert summary.proposals_emitted == 0


def test_run_mixed_series_and_no_series(tmp_path: Path) -> None:
    """Mix of books with and without series — correct proposal counts."""
    _make_calibre_db(tmp_path / "metadata.db", [
        (1, "Star Trek: The Original Series - 001 - Mission to Horatius", 1.0,
         "Star Trek: The Original Series"),
        (2, "Star Trek: Shatnerverse - 002 - The Return", 1.0, ""),  # no series
    ])
    proposals_db = tmp_path / "proposals.db"
    summary, books = star_trek.run(
        calibre_db=tmp_path / "metadata.db",
        proposals_db=proposals_db,
        dry_run=False,
    )
    assert summary.proposals_emitted == 2
    assert len(books) == 2

    conn = sqlite3.connect(proposals_db)
    try:
        by_field = dict(conn.execute(
            "SELECT field, COUNT(*) FROM proposals GROUP BY field"
        ).fetchall())
    finally:
        conn.close()
    # Book 1 (has series): title only
    # Book 2 (no series): title + series + series_index
    assert by_field["title"] == 2
    assert by_field["series"] == 1
    assert by_field["series_index"] == 1
