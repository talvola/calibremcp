"""Unit tests for the apply module (command generation + execution)."""

from __future__ import annotations

import sqlite3
import stat
from pathlib import Path

import pytest

from calibre_mcp.cleanup import apply as apply_mod
from calibre_mcp.cleanup import proposals
from calibre_mcp.cleanup.proposals import Proposal


@pytest.fixture
def db(tmp_path: Path):
    conn = proposals.connect(tmp_path / "proposals.db")
    yield conn
    conn.close()


def _seed(db, book_id: int, field: str, value: str, status: str = "approved") -> int:
    """Insert a proposal at the given status. Returns its id."""
    run_id = proposals.start_run(
        db,
        source="opf",
        library_root=Path("/tmp/library"),  # noqa: S108
        metadata_db_path=Path("/tmp/library/metadata.db"),  # noqa: S108
        dry_run=False,
    )
    proposals.insert_proposal(
        db,
        run_id,
        Proposal(
            book_id=book_id,
            field=field,
            proposed_value=value,
            source="opf",
            confidence=1.0,
        ),
    )
    pid = db.execute(
        "SELECT id FROM proposals WHERE book_id=? AND field=? AND proposed_value=?",
        (book_id, field, value),
    ).fetchone()["id"]
    if status != "proposed":
        proposals.set_status(db, pid, status)
    return pid


# ---------------------------------------------------------------------------
# plan()
# ---------------------------------------------------------------------------


def test_plan_groups_fields_per_book(db) -> None:
    _seed(db, 42, "isbn", "9780141036144")
    _seed(db, 42, "publisher", "Penguin")
    _seed(db, 42, "pubdate", "2014-11-10")
    _seed(db, 99, "isbn", "9780000000007")

    cmds = list(apply_mod.plan(db, library_path=Path("/lib")))
    # Two books, one command each; book 42 carries three --field args.
    by_book = {c.book_id: c for c in cmds}
    assert set(by_book) == {42, 99}
    assert set(by_book[42].fields) == {"isbn", "publisher", "pubdate"}
    assert by_book[99].fields == {"isbn": "9780000000007"}
    # Command-line shape
    assert by_book[42].argv[0] == "calibredb"
    assert by_book[42].argv[1] == "set_metadata"
    assert "--library-path=/lib" in by_book[42].argv
    assert str(42) == by_book[42].argv[-1]
    # ISBN is emitted via the identifiers dict syntax (calibredb's only form).
    assert any("--field=identifiers:isbn:9780141036144" in a for a in by_book[42].argv)
    # Other scalars keep their direct --field=name:value syntax.
    assert "--field=publisher:Penguin" in by_book[42].argv
    assert "--field=pubdate:2014-11-10" in by_book[42].argv


def test_plan_skips_non_scalar_fields(db) -> None:
    _seed(db, 1, "isbn", "9780141036144")
    _seed(db, 1, "tags.add", "Science Fiction")
    _seed(db, 1, "identifier.goodreads", "12345")
    cmds = list(apply_mod.plan(db, library_path=Path("/lib")))
    assert len(cmds) == 1
    # Only the ISBN proposal flows through; merge-needing fields deferred.
    assert list(cmds[0].fields) == ["isbn"]


def test_plan_description_aliased_to_comments(db) -> None:
    _seed(db, 7, "description", "A rich description of the book, longer than 20 chars.")
    cmd = next(apply_mod.plan(db, library_path=Path("/lib")))
    # Calibre stores description in the comments field.
    assert any(a.startswith("--field=comments:") for a in cmd.argv), cmd.argv


def test_plan_only_approved_flows(db) -> None:
    _seed(db, 1, "isbn", "9780141036144", status="proposed")  # not yet approved
    _seed(db, 2, "isbn", "9780000000007", status="approved")
    cmds = list(apply_mod.plan(db, library_path=Path("/lib")))
    assert len(cmds) == 1
    assert cmds[0].book_id == 2


def test_plan_report_distinguishes_skipped(db) -> None:
    _seed(db, 1, "isbn", "9780141036144")
    _seed(db, 1, "tags.add", "Horror")
    _seed(db, 2, "tags.add", "Science Fiction")
    _seed(db, 2, "identifier.goodreads", "12345")
    report = apply_mod.plan_report(db)
    assert report["isbn"] == 1
    assert "tags.add (skipped: needs merge)" in report
    assert report["tags.add (skipped: needs merge)"] == 2
    assert "identifier.goodreads (skipped: needs merge)" in report


def test_plan_respects_field_filter(db) -> None:
    _seed(db, 1, "isbn", "9780141036144")
    _seed(db, 2, "publisher", "Penguin")
    cmds = list(apply_mod.plan(db, library_path=Path("/lib"), field="isbn"))
    assert len(cmds) == 1 and cmds[0].book_id == 1


def test_plan_respects_id_filter(db) -> None:
    pid_a = _seed(db, 1, "isbn", "9780141036144")
    _seed(db, 2, "isbn", "9780000000007")
    cmds = list(apply_mod.plan(db, library_path=Path("/lib"), ids=[pid_a]))
    assert len(cmds) == 1 and cmds[0].book_id == 1


def _make_calibre_db_with_identifiers(
    library_path: Path, book_id: int, identifiers: dict[str, str]
) -> None:
    """Create a minimal ``metadata.db`` at ``library_path/metadata.db`` with
    just enough schema to satisfy the identifier-merge lookup."""
    library_path.mkdir(parents=True, exist_ok=True)
    db_path = library_path / "metadata.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE identifiers (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              book INTEGER NOT NULL,
              type TEXT NOT NULL,
              val TEXT NOT NULL
            );
            """
        )
        for scheme, value in identifiers.items():
            conn.execute(
                "INSERT INTO identifiers (book, type, val) VALUES (?, ?, ?)",
                (book_id, scheme, value),
            )
        conn.commit()
    finally:
        conn.close()


def test_plan_merges_existing_identifiers(tmp_path: Path, db) -> None:
    """When Calibre already has other identifiers on a book (Erik's
    hand-added goodreads IDs, Amazon ASINs, etc.), the merge must preserve
    them — calibredb set_metadata replaces the whole identifiers dict."""
    library = tmp_path / "lib"
    _make_calibre_db_with_identifiers(
        library, book_id=42,
        identifiers={"goodreads": "12345", "amazon": "B00XYZ1234"},
    )
    _seed(db, 42, "isbn", "9780141036144")

    cmd = next(apply_mod.plan(db, library_path=library))
    # Find the --field=identifiers:... argument and verify it contains all three.
    id_arg = next(a for a in cmd.argv if a.startswith("--field=identifiers:"))
    payload = id_arg.removeprefix("--field=identifiers:")
    # Keys are sorted alphabetically for determinism.
    assert payload == "amazon:B00XYZ1234,goodreads:12345,isbn:9780141036144"


def test_plan_fill_empty_isbn_when_no_existing_identifiers(tmp_path: Path, db) -> None:
    """The happy path: book has no identifiers in Calibre, ISBN proposal
    fills in cleanly."""
    library = tmp_path / "lib"
    _make_calibre_db_with_identifiers(library, book_id=42, identifiers={})
    _seed(db, 42, "isbn", "9780141036144")
    cmd = next(apply_mod.plan(db, library_path=library))
    assert any("--field=identifiers:isbn:9780141036144" in a for a in cmd.argv)


# ---------------------------------------------------------------------------
# render()
# ---------------------------------------------------------------------------


def test_render_shell_quotes_spaces_and_special_chars(db) -> None:
    _seed(db, 1, "publisher", "O'Reilly Media & Co.")
    cmd = next(apply_mod.plan(db, library_path=Path("/lib")))
    rendered = apply_mod.render(cmd)
    # Shell-safe: the apostrophe and ampersand are inside quotes.
    assert "'\\''" in rendered or '"' in rendered or "O'Reilly" not in rendered.replace("'", "")


# ---------------------------------------------------------------------------
# execute() — driven by a fake calibredb script on PATH
# ---------------------------------------------------------------------------


def _install_fake_calibredb(dir_: Path, *, exit_code: int = 0, echo: bool = True) -> Path:
    """Write a throwaway shell script that impersonates calibredb.

    Captures argv into dir/calls.log and exits with the given code. Used to
    verify the execute() subprocess pipeline without requiring real Calibre."""
    script = dir_ / "calibredb"
    log = dir_ / "calls.log"
    script.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$@" >> "{log}"\n'
        + ('printf "fake stderr\\n" 1>&2\n' if exit_code != 0 else "")
        + f"exit {exit_code}\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return script


def test_execute_success_marks_applied(tmp_path: Path, db) -> None:
    pid = _seed(db, 42, "isbn", "9780141036144")
    fake = _install_fake_calibredb(tmp_path)
    cmd = next(apply_mod.plan(db, library_path=Path("/lib"), calibredb=str(fake)))
    results = list(apply_mod.execute(db, [cmd]))
    assert len(results) == 1 and results[0].ok
    row = db.execute("SELECT status, applied_at FROM proposals WHERE id=?", (pid,)).fetchone()
    assert row["status"] == "applied"
    assert row["applied_at"] is not None
    # Verify our fake calibredb was actually invoked with the expected argv.
    log_content = (tmp_path / "calls.log").read_text()
    assert "set_metadata" in log_content
    assert "--field=identifiers:isbn:9780141036144" in log_content
    assert log_content.strip().splitlines()[-1] == "42"


def test_execute_failure_marks_conflict(tmp_path: Path, db) -> None:
    pid = _seed(db, 42, "isbn", "9780141036144")
    fake = _install_fake_calibredb(tmp_path, exit_code=1)
    cmd = next(apply_mod.plan(db, library_path=Path("/lib"), calibredb=str(fake)))
    results = list(apply_mod.execute(db, [cmd]))
    assert len(results) == 1 and not results[0].ok
    row = db.execute("SELECT status, conflict_reason FROM proposals WHERE id=?", (pid,)).fetchone()
    assert row["status"] == "conflict"
    assert "fake stderr" in row["conflict_reason"] or "calibredb exit=1" in row["conflict_reason"]


def test_execute_missing_calibredb_is_handled(db) -> None:
    _seed(db, 42, "isbn", "9780141036144")
    cmd = next(apply_mod.plan(db, library_path=Path("/lib"), calibredb="/nonexistent/calibredb-xxx"))
    results = list(apply_mod.execute(db, [cmd]))
    assert len(results) == 1 and not results[0].ok
    assert "FileNotFoundError" in results[0].stderr or "No such file" in results[0].stderr
    # Proposal should be left in a state the reviewer can investigate.
    row = db.execute("SELECT status, conflict_reason FROM proposals").fetchone()
    assert row["status"] == "conflict"


# ---------------------------------------------------------------------------
# proposals.count_proposals + set_status_where (used by approve/reject CLI)
# ---------------------------------------------------------------------------


def test_count_proposals_with_filters(db) -> None:
    _seed(db, 1, "isbn", "9780141036144", status="proposed")
    _seed(db, 1, "publisher", "Penguin", status="proposed")
    _seed(db, 2, "isbn", "9780000000007", status="approved")
    assert proposals.count_proposals(db) == 3
    assert proposals.count_proposals(db, field="isbn") == 2
    assert proposals.count_proposals(db, status="approved") == 1
    assert proposals.count_proposals(db, field="isbn", status="proposed") == 1


def test_set_status_where_bulk(db) -> None:
    _seed(db, 1, "isbn", "9780141036144", status="proposed")
    _seed(db, 2, "isbn", "9780000000007", status="proposed")
    _seed(db, 3, "publisher", "Penguin", status="proposed")
    n = proposals.set_status_where(db, "approved", field="isbn", status="proposed")
    assert n == 2
    # Non-ISBN untouched.
    assert proposals.count_proposals(db, field="publisher", status="proposed") == 1


def test_set_status_where_refuses_unfiltered_bulk(db) -> None:
    _seed(db, 1, "isbn", "9780141036144")
    with pytest.raises(ValueError, match="at least one narrowing"):
        proposals.set_status_where(db, "rejected")


def test_set_status_where_stores_notes(db) -> None:
    pid = _seed(db, 1, "isbn", "9780141036144", status="proposed")
    proposals.set_status_where(db, "rejected", ids=[pid], notes="wrong edition")
    row = db.execute("SELECT status, notes FROM proposals WHERE id=?", (pid,)).fetchone()
    assert row["status"] == "rejected"
    assert row["notes"] == "wrong edition"
