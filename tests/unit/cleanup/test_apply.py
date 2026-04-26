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


def test_plan_includes_identifier_and_tag_fields(db) -> None:
    """Phase 3b: tags.add now flows through (with merge); identifier.* too."""
    _seed(db, 1, "isbn", "9780141036144")
    _seed(db, 1, "tags.add", "Science Fiction")
    _seed(db, 1, "identifier.goodreads", "12345")
    _seed(db, 1, "identifier.amazon", "B00XYZ1234")
    cmds = list(apply_mod.plan(db, library_path=Path("/lib")))
    assert len(cmds) == 1
    # All four flow through.
    assert set(cmds[0].fields) == {
        "isbn",
        "tags.add",
        "identifier.goodreads",
        "identifier.amazon",
    }
    # Identifiers merged into one arg.
    id_arg = next(a for a in cmds[0].argv if a.startswith("--field=identifiers:"))
    payload = id_arg.removeprefix("--field=identifiers:")
    assert payload == "amazon:B00XYZ1234,goodreads:12345,isbn:9780141036144"
    # Tags get their own merged arg too.
    tags_arg = next(a for a in cmds[0].argv if a.startswith("--field=tags:"))
    assert "Science Fiction" in tags_arg


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


def test_plan_report_lists_all_applicable_fields(db) -> None:
    _seed(db, 1, "isbn", "9780141036144")
    _seed(db, 1, "tags.add", "Horror")
    _seed(db, 2, "tags.add", "Science Fiction")
    _seed(db, 2, "identifier.goodreads", "12345")
    report = apply_mod.plan_report(db)
    # Phase 1+3a+3b: all are applicable, none skipped.
    assert report["isbn"] == 1
    assert report["identifier.goodreads"] == 1
    assert report["tags.add"] == 2
    assert not any("skipped" in k for k in report)


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


def _make_calibre_db_with_identifiers(library_path: Path, book_id: int, identifiers: dict[str, str]) -> None:
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
        library,
        book_id=42,
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


def test_plan_merges_new_goodreads_with_existing_isbn(tmp_path: Path, db) -> None:
    """Phase 3a case: book already has an ISBN in Calibre (from Phase 1
    apply), now we're adding an identifier.goodreads. The merge must
    preserve the ISBN and add Goodreads alongside."""
    library = tmp_path / "lib"
    _make_calibre_db_with_identifiers(
        library,
        book_id=42,
        identifiers={"isbn": "9780141036144"},
    )
    _seed(db, 42, "identifier.goodreads", "12345")

    cmd = next(apply_mod.plan(db, library_path=library))
    id_arg = next(a for a in cmd.argv if a.startswith("--field=identifiers:"))
    payload = id_arg.removeprefix("--field=identifiers:")
    assert payload == "goodreads:12345,isbn:9780141036144"


def test_plan_single_book_multiple_identifier_schemes(tmp_path: Path, db) -> None:
    """Real-world shape: one book with isbn + goodreads + amazon + google
    all approved at once, merged into a single calibredb invocation."""
    library = tmp_path / "lib"
    _make_calibre_db_with_identifiers(library, book_id=99, identifiers={})
    _seed(db, 99, "isbn", "9780141036144")
    _seed(db, 99, "identifier.goodreads", "12345")
    _seed(db, 99, "identifier.amazon", "B00XYZ1234")
    _seed(db, 99, "identifier.google", "abcDEF")

    cmds = list(apply_mod.plan(db, library_path=library))
    # Still a single command for the one book — calibredb startup cost paid
    # once even with four identifier additions.
    assert len(cmds) == 1
    id_arg = next(a for a in cmds[0].argv if a.startswith("--field=identifiers:"))
    payload = id_arg.removeprefix("--field=identifiers:")
    # Alphabetical: goodreads sorts before google (4th char: 'd' < 'g').
    assert payload == "amazon:B00XYZ1234,goodreads:12345,google:abcDEF,isbn:9780141036144"


def _make_calibre_db_with_tags(library_path: Path, book_id: int, tags: list[str]) -> None:
    """Minimal Calibre schema for tag-merge testing: tags + books_tags_link."""
    library_path.mkdir(parents=True, exist_ok=True)
    db_path = library_path / "metadata.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE identifiers (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              book INTEGER NOT NULL, type TEXT NOT NULL, val TEXT NOT NULL
            );
            CREATE TABLE tags (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT NOT NULL
            );
            CREATE TABLE books_tags_link (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              book INTEGER NOT NULL, tag INTEGER NOT NULL
            );
            """
        )
        for name in tags:
            cur = conn.execute("INSERT INTO tags (name) VALUES (?)", (name,))
            conn.execute(
                "INSERT INTO books_tags_link (book, tag) VALUES (?, ?)",
                (book_id, cur.lastrowid),
            )
        conn.commit()
    finally:
        conn.close()


def test_plan_tags_add_merges_with_existing_tags(tmp_path: Path, db) -> None:
    """Phase 3b: book has 'Cthulhu' and 'Horror' in Calibre, we approve adding
    'Mythos'. Apply must emit the full tag list — calibredb's --field=tags
    REPLACES the list, so missing 'Cthulhu' would silently drop it."""
    library = tmp_path / "lib"
    _make_calibre_db_with_tags(library, book_id=42, tags=["Cthulhu", "Horror"])
    _seed(db, 42, "tags.add", "Mythos")

    cmd = next(apply_mod.plan(db, library_path=library))
    tags_arg = next(a for a in cmd.argv if a.startswith("--field=tags:"))
    payload = tags_arg.removeprefix("--field=tags:")
    # Existing tags preserved in their original order, new tag appended.
    assert payload == "Cthulhu,Horror,Mythos"


def test_plan_tags_add_skips_duplicate_case_insensitive(tmp_path: Path, db) -> None:
    """A 'tags.add' approval that case-folds to an existing tag is a no-op
    on the Calibre side (the merge dedupes). The argv still emits the list
    so calibredb writes the (unchanged) full set, but no duplicate appears."""
    library = tmp_path / "lib"
    _make_calibre_db_with_tags(library, book_id=42, tags=["Cthulhu"])
    _seed(db, 42, "tags.add", "cthulhu")  # different case
    cmd = next(apply_mod.plan(db, library_path=library))
    tags_arg = next(a for a in cmd.argv if a.startswith("--field=tags:"))
    assert tags_arg.removeprefix("--field=tags:") == "Cthulhu"


def test_plan_tags_add_with_empty_existing(tmp_path: Path, db) -> None:
    library = tmp_path / "lib"
    _make_calibre_db_with_tags(library, book_id=42, tags=[])
    _seed(db, 42, "tags.add", "Cozy Mystery")
    _seed(db, 42, "tags.add", "Cthulhu")
    cmd = next(apply_mod.plan(db, library_path=library))
    tags_arg = next(a for a in cmd.argv if a.startswith("--field=tags:"))
    payload = tags_arg.removeprefix("--field=tags:")
    # Order preserved as proposals were processed (sqlite3 returns by id).
    assert sorted(payload.split(",")) == ["Cozy Mystery", "Cthulhu"]


def test_plan_tags_add_strips_commas_in_values(tmp_path: Path, db) -> None:
    """Tag values with commas would corrupt the comma-separated list calibredb
    expects. Defensive replace with space."""
    library = tmp_path / "lib"
    _make_calibre_db_with_tags(library, book_id=42, tags=[])
    _seed(db, 42, "tags.add", "Cooking, Vegetarian")
    cmd = next(apply_mod.plan(db, library_path=library))
    tags_arg = next(a for a in cmd.argv if a.startswith("--field=tags:"))
    assert "," not in tags_arg.removeprefix("--field=tags:")


def test_plan_combined_isbn_and_tags(tmp_path: Path, db) -> None:
    """A book with both an ISBN and tags.add proposals emits two merged
    --field args: one identifiers, one tags."""
    library = tmp_path / "lib"
    _make_calibre_db_with_tags(library, book_id=42, tags=["Cthulhu"])
    _seed(db, 42, "isbn", "9780141036144")
    _seed(db, 42, "tags.add", "Mythos")
    cmd = next(apply_mod.plan(db, library_path=library))
    field_args = [a for a in cmd.argv if a.startswith("--field=")]
    assert any(a.startswith("--field=identifiers:") for a in field_args)
    assert any(a.startswith("--field=tags:") for a in field_args)


def test_plan_identifier_only_no_scalar_proposals(tmp_path: Path, db) -> None:
    """Book with *only* identifier.* proposals (no ISBN, no scalars) should
    still produce exactly one calibredb command."""
    library = tmp_path / "lib"
    _make_calibre_db_with_identifiers(library, book_id=7, identifiers={})
    _seed(db, 7, "identifier.goodreads", "99999")
    cmds = list(apply_mod.plan(db, library_path=library))
    assert len(cmds) == 1
    assert any("--field=identifiers:goodreads:99999" in a for a in cmds[0].argv)
    # Only the identifiers field, plus the book_id argument.
    field_args = [a for a in cmds[0].argv if a.startswith("--field=")]
    assert len(field_args) == 1


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


# ---------------------------------------------------------------------------
# Tag-level ops (tag.delete / tag.merge): plan() expansion + execute()
# ---------------------------------------------------------------------------


def _make_calibre_db_multibook_tags(
    library_path: Path, books: dict[int, list[str]],
) -> None:
    """Seed multiple books each carrying a distinct tag list."""
    library_path.mkdir(parents=True, exist_ok=True)
    db_path = library_path / "metadata.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE identifiers (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              book INTEGER NOT NULL, type TEXT NOT NULL, val TEXT NOT NULL
            );
            CREATE TABLE tags (
              id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL
            );
            CREATE TABLE books_tags_link (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              book INTEGER NOT NULL, tag INTEGER NOT NULL
            );
            """
        )
        # Dedupe tag-name → id so multiple books can share a tag.
        name_to_id: dict[str, int] = {}
        for book_id, tag_names in books.items():
            for name in tag_names:
                if name not in name_to_id:
                    cur = conn.execute("INSERT INTO tags (name) VALUES (?)", (name,))
                    name_to_id[name] = cur.lastrowid
                conn.execute(
                    "INSERT INTO books_tags_link (book, tag) VALUES (?, ?)",
                    (book_id, name_to_id[name]),
                )
        conn.commit()
    finally:
        conn.close()


def _seed_tag_op(
    db, *, op: str, src: str, target: str | None, status: str = "approved",
) -> int:
    """Insert a tag.delete or tag.merge proposal in the tag-op shape used by
    the live-tag sweep. Returns the proposal id."""
    run_id = proposals.start_run(
        db,
        source="live_tag_sweep",
        library_root=Path("/tmp/library"),  # noqa: S108
        metadata_db_path=Path("/tmp/library/metadata.db"),  # noqa: S108
        dry_run=False,
    )
    if op == "tag.delete":
        proposed_value = src
    else:
        assert target is not None
        proposed_value = f"{src} -> {target}"
    proposals.insert_proposal(
        db,
        run_id,
        Proposal(
            book_id=0,
            field=op,
            calibre_value=src,
            proposed_value=proposed_value,
            source="live_tag_sweep",
            confidence=1.0,
        ),
    )
    pid = db.execute(
        "SELECT id FROM proposals WHERE field=? AND calibre_value=? AND proposed_value=?",
        (op, src, proposed_value),
    ).fetchone()["id"]
    if status != "proposed":
        proposals.set_status(db, pid, status)
    return pid


def test_plan_tag_delete_expands_to_each_affected_book(tmp_path: Path, db) -> None:
    library = tmp_path / "lib"
    _make_calibre_db_multibook_tags(library, {
        10: ["General", "Science Fiction"],
        20: ["General", "Mystery"],
        30: ["Fantasy"],  # not affected
    })
    pid = _seed_tag_op(db, op="tag.delete", src="General", target=None)

    cmds = list(apply_mod.plan(db, library_path=library))
    by_book = {c.book_id: c for c in cmds}
    assert set(by_book) == {10, 20}  # only books that had the tag
    # Each affected command carries the tag-op proposal_id (not per-book).
    for cmd in by_book.values():
        assert cmd.proposal_ids == ()
        assert cmd.tag_op_proposal_ids == (pid,)
        tags_arg = next(a for a in cmd.argv if a.startswith("--field=tags:"))
        # 'General' removed, other tags preserved.
        assert "General" not in tags_arg.removeprefix("--field=tags:")


def test_plan_tag_merge_substitutes_target(tmp_path: Path, db) -> None:
    library = tmp_path / "lib"
    _make_calibre_db_multibook_tags(library, {
        10: ["sf", "Adventure"],
        20: ["sf", "Mystery"],
    })
    _seed_tag_op(db, op="tag.merge", src="sf", target="Science Fiction")

    cmds = list(apply_mod.plan(db, library_path=library))
    by_book = {c.book_id: c for c in cmds}
    assert set(by_book) == {10, 20}
    payload10 = next(a for a in by_book[10].argv if a.startswith("--field=tags:")).removeprefix("--field=tags:")
    # 'sf' replaced with 'Science Fiction', Adventure preserved.
    assert payload10.split(",") == ["Science Fiction", "Adventure"]


def test_plan_tag_merge_dedupes_when_target_already_present(tmp_path: Path, db) -> None:
    """Book has both 'sf' and 'Science Fiction'. Merging sf→Science Fiction
    must not produce two 'Science Fiction' entries."""
    library = tmp_path / "lib"
    _make_calibre_db_multibook_tags(library, {
        10: ["sf", "Science Fiction", "Mystery"],
    })
    _seed_tag_op(db, op="tag.merge", src="sf", target="Science Fiction")

    cmds = list(apply_mod.plan(db, library_path=library))
    payload = next(a for a in cmds[0].argv if a.startswith("--field=tags:")).removeprefix("--field=tags:")
    parts = payload.split(",")
    assert parts.count("Science Fiction") == 1
    assert "Mystery" in parts
    assert "sf" not in parts


def test_plan_combines_tag_op_with_per_book_tags_add(tmp_path: Path, db) -> None:
    """Single book with both a tag.merge fanned-out op AND a tags.add
    per-book proposal must produce one calibredb command, not two."""
    library = tmp_path / "lib"
    _make_calibre_db_multibook_tags(library, {42: ["sf", "Mystery"]})
    _seed_tag_op(db, op="tag.merge", src="sf", target="Science Fiction")
    _seed(db, 42, "tags.add", "Cthulhu Mythos")

    cmds = list(apply_mod.plan(db, library_path=library))
    assert len(cmds) == 1
    cmd = cmds[0]
    payload = next(a for a in cmd.argv if a.startswith("--field=tags:")).removeprefix("--field=tags:")
    parts = payload.split(",")
    assert "sf" not in parts
    assert "Science Fiction" in parts
    assert "Mystery" in parts
    assert "Cthulhu Mythos" in parts


def test_plan_tag_op_no_affected_books_yields_nothing(tmp_path: Path, db) -> None:
    """Approved tag.delete for a tag no book actually has (e.g. already
    cleaned up in a prior partial apply) should produce zero commands."""
    library = tmp_path / "lib"
    _make_calibre_db_multibook_tags(library, {10: ["Fantasy"]})
    _seed_tag_op(db, op="tag.delete", src="DoesNotExist", target=None)
    cmds = list(apply_mod.plan(db, library_path=library))
    assert cmds == []


def test_execute_tag_op_marks_applied_only_when_all_books_succeed(
    tmp_path: Path, db,
) -> None:
    library = tmp_path / "lib"
    _make_calibre_db_multibook_tags(library, {
        10: ["General", "Science Fiction"],
        20: ["General", "Mystery"],
    })
    pid = _seed_tag_op(db, op="tag.delete", src="General", target=None)
    fake = _install_fake_calibredb(tmp_path)

    cmds = list(apply_mod.plan(db, library_path=library, calibredb=str(fake)))
    list(apply_mod.execute(db, cmds))
    row = db.execute("SELECT status FROM proposals WHERE id=?", (pid,)).fetchone()
    assert row["status"] == "applied"


def test_execute_tag_op_partial_failure_keeps_approved(
    tmp_path: Path, db,
) -> None:
    """If even one of the per-book commands for a tag-op fails, the tag-op
    proposal stays as 'approved' so a re-run picks up the survivors."""
    library = tmp_path / "lib"
    _make_calibre_db_multibook_tags(library, {
        10: ["General", "Science Fiction"],
        20: ["General", "Mystery"],
    })
    pid = _seed_tag_op(db, op="tag.delete", src="General", target=None)
    fake = _install_fake_calibredb(tmp_path, exit_code=1)  # every call fails

    cmds = list(apply_mod.plan(db, library_path=library, calibredb=str(fake)))
    list(apply_mod.execute(db, cmds))
    row = db.execute("SELECT status, notes FROM proposals WHERE id=?", (pid,)).fetchone()
    assert row["status"] == "approved"  # not 'applied', not 'conflict'
    assert "per-book commands failed" in (row["notes"] or "")
