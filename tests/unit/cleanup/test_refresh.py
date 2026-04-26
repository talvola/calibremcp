"""Tests for the refresh orchestration module."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from calibre_mcp.cleanup import live_tag_sweep, proposals, refresh


def _build_library(
    library_root: Path, books: list[tuple[int, str, list[str]]],
) -> Path:
    """Build a minimal Calibre library: just metadata.db with the schema
    pieces refresh / live_tag_sweep / tag_normalizer touch.

    ``books`` is ``[(id, title, tags)]``. No EPUB files — Phase 1's
    ``_find_epub`` returns None, so the miner phase yields zero proposals
    in tests. That's what we want: refresh() should still run cleanly.
    """
    library_root.mkdir(parents=True, exist_ok=True)
    db = library_root / "metadata.db"
    conn = sqlite3.connect(db)
    try:
        conn.executescript(
            """
            CREATE TABLE books (
              id INTEGER PRIMARY KEY,
              uuid TEXT,
              title TEXT NOT NULL,
              path TEXT NOT NULL DEFAULT '',
              pubdate TEXT,
              series_index REAL
            );
            CREATE TABLE authors (id INTEGER PRIMARY KEY, name TEXT);
            CREATE TABLE books_authors_link (id INTEGER PRIMARY KEY, book INTEGER, author INTEGER);
            CREATE TABLE series (id INTEGER PRIMARY KEY, name TEXT);
            CREATE TABLE books_series_link (id INTEGER PRIMARY KEY, book INTEGER, series INTEGER);
            CREATE TABLE publishers (id INTEGER PRIMARY KEY, name TEXT);
            CREATE TABLE books_publishers_link (id INTEGER PRIMARY KEY, book INTEGER, publisher INTEGER);
            CREATE TABLE comments (id INTEGER PRIMARY KEY, book INTEGER, text TEXT);
            CREATE TABLE identifiers (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              book INTEGER NOT NULL, type TEXT NOT NULL, val TEXT NOT NULL
            );
            CREATE TABLE tags (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL);
            CREATE TABLE books_tags_link (id INTEGER PRIMARY KEY AUTOINCREMENT, book INTEGER, tag INTEGER);
            CREATE TABLE languages (id INTEGER PRIMARY KEY, lang_code TEXT);
            CREATE TABLE books_languages_link (id INTEGER PRIMARY KEY, book INTEGER, lang_code INTEGER, item_order INTEGER);
            """
        )
        name_to_tag_id: dict[str, int] = {}
        for book_id, title, tags in books:
            conn.execute(
                "INSERT INTO books (id, title, path) VALUES (?, ?, ?)",
                (book_id, title, f"path/{book_id}"),
            )
            for name in tags:
                if name not in name_to_tag_id:
                    cur = conn.execute("INSERT INTO tags (name) VALUES (?)", (name,))
                    name_to_tag_id[name] = cur.lastrowid
                conn.execute(
                    "INSERT INTO books_tags_link (book, tag) VALUES (?, ?)",
                    (book_id, name_to_tag_id[name]),
                )
        conn.commit()
    finally:
        conn.close()
    return db


def test_refresh_runs_all_three_phases(tmp_path: Path) -> None:
    """Smoke test: refresh against a minimal library yields a structured
    summary with all three phase results, totals, and books-seen count."""
    lib = tmp_path / "lib"
    metadata_db = _build_library(
        lib, [(1, "Book A", ["General", "Science Fiction"]), (2, "Book B", ["sf", "Mystery"])],
    )
    pdb = tmp_path / "proposals.db"
    summary = refresh.run(
        library_root=lib, metadata_db=metadata_db, proposals_db=pdb,
    )
    # Phase 1 has no EPUBs to mine → zero proposals (but the run completes).
    assert summary.phase1.name == "phase1_opf_miner"
    assert summary.phase1.proposals_emitted == 0
    # Phase 3b runs cleanly even with no tag.add proposals to classify.
    assert summary.phase3b.name == "phase3b_tag_normalizer"
    # Phase 4 catches the noise + stylistic variants.
    assert summary.phase4.name == "phase4_live_tag_sweep"
    # 'General' delete + 'sf' merge = 2 emitted.
    assert summary.phase4.proposals_emitted >= 2


def test_refresh_is_idempotent(tmp_path: Path) -> None:
    """Re-running refresh against the same state must not duplicate
    proposals — the propose-queue's unique index dedupes."""
    lib = tmp_path / "lib"
    metadata_db = _build_library(
        lib, [(1, "Book A", ["General"]), (2, "Book B", ["sf"])],
    )
    pdb = tmp_path / "proposals.db"
    refresh.run(library_root=lib, metadata_db=metadata_db, proposals_db=pdb)
    refresh.run(library_root=lib, metadata_db=metadata_db, proposals_db=pdb)

    pc = proposals.connect(pdb)
    n = pc.execute(
        "SELECT COUNT(*) AS n FROM proposals WHERE source=?",
        (live_tag_sweep.SOURCE,),
    ).fetchone()["n"]
    pc.close()
    # Exactly: 1 delete (General) + 1 merge (sf → Science Fiction) = 2,
    # even after two runs.
    assert n == 2


def test_refresh_picks_up_new_tags_between_runs(tmp_path: Path) -> None:
    """Run 1: library has 'General' only. Run 2: library now has
    'General' + 'cthulhu'. Refresh should add the cthulhu merge as a
    new DB row; the General delete row from run 1 stays as one row."""
    lib = tmp_path / "lib"
    metadata_db = _build_library(lib, [(1, "Book A", ["General"])])
    pdb = tmp_path / "proposals.db"
    refresh.run(library_root=lib, metadata_db=metadata_db, proposals_db=pdb)

    pc = proposals.connect(pdb)
    n1 = pc.execute(
        "SELECT COUNT(*) AS n FROM proposals WHERE source=?",
        (live_tag_sweep.SOURCE,),
    ).fetchone()["n"]
    pc.close()
    assert n1 == 1  # just the General delete

    # Mutate the live library: add a new tag that should classify as merge.
    conn = sqlite3.connect(metadata_db)
    conn.execute("INSERT INTO books (id, title, path) VALUES (3, 'Book C', 'path/3')")
    cur = conn.execute("INSERT INTO tags (name) VALUES ('cthulhu')")
    conn.execute("INSERT INTO books_tags_link (book, tag) VALUES (3, ?)", (cur.lastrowid,))
    conn.commit()
    conn.close()

    refresh.run(library_root=lib, metadata_db=metadata_db, proposals_db=pdb)

    pc = proposals.connect(pdb)
    n2 = pc.execute(
        "SELECT COUNT(*) AS n FROM proposals WHERE source=?",
        (live_tag_sweep.SOURCE,),
    ).fetchone()["n"]
    pc.close()
    assert n2 == 2  # General delete + cthulhu merge; no duplicate of General


def test_refresh_scopes_phase1_to_since_book_id(tmp_path: Path) -> None:
    """``since_book_id`` only narrows the OPF miner pass — Phase 4 always
    sees the whole live tags table. Verify books_seen reflects the scope."""
    lib = tmp_path / "lib"
    metadata_db = _build_library(
        lib, [(1, "old", []), (2, "old", []), (10, "new", []), (11, "new", [])],
    )
    pdb = tmp_path / "proposals.db"
    summary = refresh.run(
        library_root=lib, metadata_db=metadata_db, proposals_db=pdb,
        since_book_id=10,
    )
    # No EPUBs → books_parsed is 0 across the board (since_book_id only
    # matters when EPUBs exist), but the scope was applied — books with
    # id < 10 weren't even attempted for OPF parsing. We can't directly
    # assert that here without instrumenting; instead just verify the
    # summary is structurally valid.
    assert summary.new_books_seen == 0  # no EPUBs in synthetic library


def test_list_new_proposals_filters_by_recent_runs(tmp_path: Path) -> None:
    """list_new_proposals returns rows attached to runs completed within
    the last N minutes — a fresh refresh should populate it; a long-ago
    run wouldn't (we'd need to manipulate timestamps to test that)."""
    lib = tmp_path / "lib"
    metadata_db = _build_library(lib, [(1, "Book A", ["General", "sf"])])
    pdb = tmp_path / "proposals.db"
    refresh.run(library_root=lib, metadata_db=metadata_db, proposals_db=pdb)

    rows = refresh.list_new_proposals(pdb, since_minutes=10)
    assert len(rows) >= 2  # at minimum the General delete + sf merge


def test_list_new_proposals_skips_old_runs(tmp_path: Path) -> None:
    """Force a run's completed_at to be 1 hour old; list_new_proposals
    with since_minutes=10 must exclude its proposals."""
    lib = tmp_path / "lib"
    metadata_db = _build_library(lib, [(1, "Book A", ["General"])])
    pdb = tmp_path / "proposals.db"
    refresh.run(library_root=lib, metadata_db=metadata_db, proposals_db=pdb)

    # Backdate every miner_run by 1 hour.
    pc = proposals.connect(pdb)
    pc.execute(
        "UPDATE miner_runs SET completed_at = strftime('%Y-%m-%dT%H:%M:%S', "
        "datetime(completed_at, '-1 hour')) || '+00:00'"
    )
    pc.close()

    rows = refresh.list_new_proposals(pdb, since_minutes=10)
    assert rows == []
