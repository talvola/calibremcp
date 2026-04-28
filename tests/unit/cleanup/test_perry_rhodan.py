"""Unit tests for perry_rhodan — title parsing and proposal generation."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from calibre_mcp.cleanup import perry_rhodan, proposals
from calibre_mcp.cleanup.perry_rhodan import ParsedBook, parse_atlan_title, parse_title


# ---------------------------------------------------------------------------
# parse_title
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected_title, expected_num, expected_letter, expected_index",
    [
        (
            "Perry Rhodan – 01 – Perry Rhodan (001) Enterprise Stardust",
            "Enterprise Stardust", 1, "", 1.0,
        ),
        (
            "Perry Rhodan – 01 – Perry Rhodan (002a) The Radiant Dome",
            "The Radiant Dome", 2, "a", 2.1,
        ),
        (
            "Perry Rhodan – 01 – Perry Rhodan (002b) Twilight of The Gods",
            "Twilight of The Gods", 2, "b", 2.2,
        ),
        (
            "Perry Rhodan – 01 – Perry Rhodan (010) The Ghosts of Gol",
            "The Ghosts of Gol", 10, "", 10.0,
        ),
        (
            "Perry Rhodan – 01 – Perry Rhodan (100) something",
            "something", 100, "", 100.0,
        ),
        # em-dash variant (already tested above) vs plain hyphen
        (
            "Perry Rhodan - 01 - Perry Rhodan (005a) Space Battle in the Vega Sector",
            "Space Battle in the Vega Sector", 5, "a", 5.1,
        ),
    ],
)
def test_parse_title_parametrized(
    raw: str,
    expected_title: str,
    expected_num: int,
    expected_letter: str,
    expected_index: float,
) -> None:
    result = parse_title(raw)
    assert result is not None
    real_title, book_number, letter, index = result
    assert real_title == expected_title
    assert book_number == expected_num
    assert letter == expected_letter
    assert abs(index - expected_index) < 1e-9


@pytest.mark.parametrize(
    "raw",
    [
        "Foundation",
        "Perry Rhodan",                          # no pattern
        "Star Trek: Mirror Universe - 001 - Glass Empires",
        "Perry Rhodan – The Unknown",            # missing the number block
    ],
)
def test_parse_title_no_match(raw: str) -> None:
    assert parse_title(raw) is None


def test_parse_title_strips_leading_dash_from_real_title() -> None:
    """Some raw titles have '(030) - To Arkon' — the leading '- ' must be stripped."""
    result = parse_title("Perry Rhodan – 01 – Perry Rhodan (030) - To Arkon")
    assert result is not None
    real_title, book_number, letter, index = result
    assert real_title == "To Arkon"
    assert book_number == 30
    assert abs(index - 30.0) < 1e-9


def test_letter_decimal_c() -> None:
    """Third variant letter → .3."""
    result = parse_title("Perry Rhodan – 01 – Perry Rhodan (003c) Third Part")
    assert result is not None
    _, _, letter, index = result
    assert letter == "c"
    assert abs(index - 3.3) < 1e-9


# ---------------------------------------------------------------------------
# run() integration
# ---------------------------------------------------------------------------


def _make_calibre_db(path: Path, books: list[tuple[int, str, float]]) -> None:
    """Minimal Calibre DB: [(book_id, title, series_index)]."""
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
        conn.execute("INSERT INTO series (id, name) VALUES (1, 'Perry Rhodan')")
        for book_id, title, idx in books:
            conn.execute(
                "INSERT INTO books (id, title, series_index) VALUES (?, ?, ?)",
                (book_id, title, idx),
            )
            conn.execute(
                "INSERT INTO books_series_link (book, series) VALUES (?, 1)", (book_id,)
            )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture()
def prop_db(tmp_path: Path) -> sqlite3.Connection:
    conn = proposals.connect(tmp_path / "proposals.db")
    yield conn
    conn.close()


def test_run_dry_run_parses_books(tmp_path: Path, prop_db: sqlite3.Connection) -> None:
    _make_calibre_db(tmp_path / "metadata.db", [
        (1, "Perry Rhodan – 01 – Perry Rhodan (001) Enterprise Stardust", 1.0),
        (2, "Perry Rhodan – 01 – Perry Rhodan (002a) The Radiant Dome", 1.0),
    ])
    summary, books = perry_rhodan.run(
        calibre_db=tmp_path / "metadata.db",
        proposals_db=tmp_path / "proposals.db",
        dry_run=True,
    )
    assert summary.examined == 2
    assert summary.proposals_emitted == 0
    assert len(books) == 2
    titles = {b.real_title for b in books}
    assert titles == {"Enterprise Stardust", "The Radiant Dome"}


def test_run_emits_title_and_index_proposals(tmp_path: Path) -> None:
    _make_calibre_db(tmp_path / "metadata.db", [
        (1, "Perry Rhodan – 01 – Perry Rhodan (007) Fortress of the Six Moons", 1.0),
    ])
    proposals_db = tmp_path / "proposals.db"

    summary, books = perry_rhodan.run(
        calibre_db=tmp_path / "metadata.db",
        proposals_db=proposals_db,
        dry_run=False,
    )
    assert summary.proposals_emitted == 1

    conn = sqlite3.connect(proposals_db)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT field, calibre_value, proposed_value, status FROM proposals ORDER BY field"
        ).fetchall()
    finally:
        conn.close()

    fields = {r["field"]: r for r in rows}
    assert "title" in fields
    assert fields["title"]["proposed_value"] == "Fortress of the Six Moons"
    assert fields["title"]["status"] == "proposed"

    assert "series_index" in fields
    assert float(fields["series_index"]["proposed_value"]) == pytest.approx(7.0)


def test_run_letter_variant_emits_fractional_index(tmp_path: Path) -> None:
    _make_calibre_db(tmp_path / "metadata.db", [
        (1, "Perry Rhodan – 01 – Perry Rhodan (002b) Twilight of The Gods", 1.0),
    ])
    summary, books = perry_rhodan.run(
        calibre_db=tmp_path / "metadata.db",
        proposals_db=tmp_path / "proposals.db",
        dry_run=False,
    )
    conn = sqlite3.connect(tmp_path / "proposals.db")
    conn.row_factory = sqlite3.Row
    try:
        idx_row = conn.execute(
            "SELECT proposed_value FROM proposals WHERE field='series_index'"
        ).fetchone()
    finally:
        conn.close()
    assert idx_row is not None
    assert float(idx_row["proposed_value"]) == pytest.approx(2.2)


def test_run_skips_already_correct_index(tmp_path: Path) -> None:
    """If series_index already equals the parsed value, no series_index proposal."""
    _make_calibre_db(tmp_path / "metadata.db", [
        (1, "Perry Rhodan – 01 – Perry Rhodan (005) Space Foo", 5.0),
    ])
    perry_rhodan.run(
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


def test_run_skips_unmatched_titles(tmp_path: Path) -> None:
    _make_calibre_db(tmp_path / "metadata.db", [
        (1, "Perry Rhodan – 01 – Perry Rhodan (001) Enterprise Stardust", 1.0),
        (2, "Perry Rhodan – Some Other Format", 1.0),
    ])
    summary, books = perry_rhodan.run(
        calibre_db=tmp_path / "metadata.db",
        proposals_db=tmp_path / "proposals.db",
        dry_run=True,
    )
    assert summary.skipped_no_match == 1
    assert len(books) == 1


def test_run_is_idempotent(tmp_path: Path) -> None:
    """Running twice produces the same proposals, not duplicates."""
    _make_calibre_db(tmp_path / "metadata.db", [
        (1, "Perry Rhodan – 01 – Perry Rhodan (001) Enterprise Stardust", 1.0),
    ])
    proposals_db = tmp_path / "proposals.db"
    perry_rhodan.run(calibre_db=tmp_path / "metadata.db", proposals_db=proposals_db)
    perry_rhodan.run(calibre_db=tmp_path / "metadata.db", proposals_db=proposals_db)

    conn = sqlite3.connect(proposals_db)
    try:
        count = conn.execute("SELECT COUNT(*) FROM proposals WHERE field='title'").fetchone()[0]
    finally:
        conn.close()
    assert count == 1


# ---------------------------------------------------------------------------
# parse_atlan_title
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected_title, expected_num, expected_letter, expected_index",
    [
        (
            "Perry Rhodan – 01 – Perry Rhodan Atlan (01) Spider Desert",
            "Spider Desert", 1, "", 1.0,
        ),
        (
            "Perry Rhodan – 01 – Perry Rhodan Atlan (02) Flight from Tarkihl",
            "Flight from Tarkihl", 2, "", 2.0,
        ),
        (
            "Perry Rhodan – 01 – Perry Rhodan Atlan (2b) Menace of Atomigeddon",
            "Menace of Atomigeddon", 2, "b", 2.2,
        ),
        (
            "Perry Rhodan – 01 – Perry Rhodan Atlan (05) War of the Ghosts",
            "War of the Ghosts", 5, "", 5.0,
        ),
        # plain-hyphen separator
        (
            "Perry Rhodan - 01 - Perry Rhodan Atlan (03) Pale Country Pursuit",
            "Pale Country Pursuit", 3, "", 3.0,
        ),
    ],
)
def test_parse_atlan_title_parametrized(
    raw: str,
    expected_title: str,
    expected_num: int,
    expected_letter: str,
    expected_index: float,
) -> None:
    result = parse_atlan_title(raw)
    assert result is not None
    real_title, book_number, letter, index = result
    assert real_title == expected_title
    assert book_number == expected_num
    assert letter == expected_letter
    assert abs(index - expected_index) < 1e-9


def test_parse_atlan_title_no_match_on_main_series() -> None:
    """Main-series titles must NOT be matched by parse_atlan_title."""
    assert parse_atlan_title(
        "Perry Rhodan – 01 – Perry Rhodan (001) Enterprise Stardust"
    ) is None


def test_parse_title_no_match_on_atlan() -> None:
    """Atlan titles must NOT be matched by parse_title."""
    assert parse_title(
        "Perry Rhodan – 01 – Perry Rhodan Atlan (01) Spider Desert"
    ) is None


# ---------------------------------------------------------------------------
# run() — Atlan proposals
# ---------------------------------------------------------------------------


def _make_calibre_db_with_series(
    path: "Path",
    books: "list[tuple[int, str, float, str]]",
) -> None:
    """Calibre DB supporting multiple series: [(book_id, title, series_index, series_name)]."""
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
        # collect unique series names
        series_map: dict[str, int] = {}
        for _, _, _, sname in books:
            if sname not in series_map:
                sid = len(series_map) + 1
                series_map[sname] = sid
                conn.execute("INSERT INTO series (id, name) VALUES (?, ?)", (sid, sname))
        for book_id, title, idx, sname in books:
            conn.execute(
                "INSERT INTO books (id, title, series_index) VALUES (?, ?, ?)",
                (book_id, title, idx),
            )
            conn.execute(
                "INSERT INTO books_series_link (book, series) VALUES (?, ?)",
                (book_id, series_map[sname]),
            )
        conn.commit()
    finally:
        conn.close()


def test_run_atlan_emits_series_proposal(tmp_path: "Path") -> None:
    _make_calibre_db(tmp_path / "metadata.db", [
        (1, "Perry Rhodan – 01 – Perry Rhodan Atlan (01) Spider Desert", 1.0),
    ])
    proposals_db = tmp_path / "proposals.db"

    summary, books = perry_rhodan.run(
        calibre_db=tmp_path / "metadata.db",
        proposals_db=proposals_db,
        dry_run=False,
    )
    assert summary.proposals_emitted == 1
    assert len(books) == 1
    assert books[0].series == "Perry Rhodan: Atlan"

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
    assert fields["title"] == "Spider Desert"
    assert "series" in fields
    assert fields["series"] == "Perry Rhodan: Atlan"
    # series_index=1.0 already matches parsed index 1.0, so no series_index proposal
    assert "series_index" not in fields


def test_run_atlan_letter_variant(tmp_path: "Path") -> None:
    _make_calibre_db(tmp_path / "metadata.db", [
        (1, "Perry Rhodan – 01 – Perry Rhodan Atlan (2b) Menace of Atomigeddon", 1.0),
    ])
    perry_rhodan.run(
        calibre_db=tmp_path / "metadata.db",
        proposals_db=tmp_path / "proposals.db",
        dry_run=False,
    )
    conn = sqlite3.connect(tmp_path / "proposals.db")
    conn.row_factory = sqlite3.Row
    try:
        idx = conn.execute(
            "SELECT proposed_value FROM proposals WHERE field='series_index'"
        ).fetchone()
        series = conn.execute(
            "SELECT proposed_value FROM proposals WHERE field='series'"
        ).fetchone()
    finally:
        conn.close()
    assert float(idx["proposed_value"]) == pytest.approx(2.2)
    assert series["proposed_value"] == "Perry Rhodan: Atlan"


def test_run_mixed_main_and_atlan(tmp_path: "Path") -> None:
    """Main-series and Atlan books in the same DB are both parsed correctly."""
    _make_calibre_db(tmp_path / "metadata.db", [
        (1, "Perry Rhodan – 01 – Perry Rhodan (001) Enterprise Stardust", 1.0),
        (2, "Perry Rhodan – 01 – Perry Rhodan Atlan (01) Spider Desert", 1.0),
    ])
    summary, books = perry_rhodan.run(
        calibre_db=tmp_path / "metadata.db",
        proposals_db=tmp_path / "proposals.db",
        dry_run=True,
    )
    assert summary.examined == 2
    assert len(books) == 2
    series_names = {b.series for b in books}
    assert series_names == {"Perry Rhodan", "Perry Rhodan: Atlan"}
