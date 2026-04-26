"""Translate approved proposals into ``calibredb set_metadata`` invocations.

The v1 apply step is intentionally conservative: it only handles scalar
fields whose backfill semantics are pure replacement (``isbn``, ``publisher``,
``pubdate``, ``description``, ``series``, ``series_index``). List-valued
fields (``tags.add``) and non-ISBN identifiers have merge semantics —
``calibredb`` replaces the whole ``identifiers`` dict and the whole ``tags``
list, so writing a single new value without reading the current set first
would clobber existing data. Phase 3 will add those with a proper merge.

The caller decides whether to actually execute the emitted commands or just
print them (dry-run). Execution happens via ``subprocess.run``; when
``calibredb`` isn't available on ``$PATH`` (e.g., you're driving the
pipeline from a WSL workstation that talks to Calibre via CIFS), dry-run
output is the safest path — copy the commands to run on the NAS side.
"""

from __future__ import annotations

import logging
import shlex
import sqlite3
import subprocess
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from calibre_mcp.cleanup import calibre_reader, live_tag_sweep, proposals

log = logging.getLogger(__name__)

# Scalar-replacement fields: safe to overwrite without reading current state.
# (ISBN is *not* in this set — although conceptually scalar, calibredb routes
# it through the identifiers dict, which needs read-merge.)
_APPLICABLE_SCALAR_FIELDS: frozenset[str] = frozenset({"publisher", "pubdate", "description", "series", "series_index"})

# Identifier proposal fields route through the merged identifiers-dict path.
# Phase 1 originally supported just 'isbn'; Phase 3a extends to any scheme
# via the 'identifier.<scheme>' prefix convention.
_IDENTIFIER_FIELDS: frozenset[str] = frozenset({"isbn"})


def _is_applicable(field: str) -> bool:
    """True for every proposal field the current apply pipeline can handle.

    Includes:
      * scalar fields (publisher, pubdate, description, series, series_index)
      * ``isbn`` and ``identifier.<scheme>`` (merged into the identifiers dict)
      * ``tags.add`` (merged into Calibre's tags list)
      * ``tag.delete`` / ``tag.merge`` (tag-level ops; expanded per-book)
    """
    return (
        field in _APPLICABLE_SCALAR_FIELDS
        or field in _IDENTIFIER_FIELDS
        or field.startswith("identifier.")
        or field == "tags.add"
        or field in _TAG_OP_FIELDS
    )


_TAG_OP_FIELDS: frozenset[str] = frozenset({"tag.delete", "tag.merge"})


# Proposal field -> calibredb --field name (they mostly match; the main
# exception is 'description' which Calibre stores as 'comments').
_FIELD_ALIASES: dict[str, str] = {
    "description": "comments",
}


@dataclass(frozen=True, slots=True)
class ApplyCommand:
    """One ``calibredb set_metadata`` invocation for a single book.

    A book with multiple approved proposals yields one command with multiple
    ``--field`` arguments so calibredb only pays its startup cost once per
    book.

    ``proposal_ids`` are per-book proposals — they get marked applied as
    soon as the command succeeds. ``tag_op_proposal_ids`` are tag-level
    proposals (``tag.delete`` / ``tag.merge``) that fan out across many
    books; the executor tracks them separately and only marks them applied
    once *every* affected book's command succeeds.
    """

    book_id: int
    proposal_ids: tuple[int, ...]
    tag_op_proposal_ids: tuple[int, ...]
    fields: dict[str, str]  # proposal-field -> proposed_value
    argv: tuple[str, ...]  # fully-constructed calibredb argv


@dataclass(frozen=True, slots=True)
class ApplyResult:
    command: ApplyCommand
    ok: bool
    returncode: int
    stdout: str
    stderr: str


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def plan(
    conn: sqlite3.Connection,
    *,
    library_path: Path,
    ids: Sequence[int] | None = None,
    field: str | None = None,
    source: str | None = None,
    limit: int | None = None,
    calibredb: str = "calibredb",
) -> Iterator[ApplyCommand]:
    """Yield an ``ApplyCommand`` for every book with approved proposals in
    applicable fields.

    Per-book proposals (scalar fields, identifiers, ``tags.add``) drive the
    main grouping. Tag-level proposals (``tag.delete``, ``tag.merge``) are
    expanded by looking up which books currently carry the source tag and
    folding the resulting per-book ops into the same grouping, so a book
    affected by both a tags.add and a tag.merge gets a single command.
    """
    rows = proposals.list_proposals(
        conn,
        status="approved",
        field=field,
        source=source,
        limit=limit if limit is not None else 1_000_000,
    )
    if ids is not None:
        allowed = set(ids)
        rows = [r for r in rows if r["id"] in allowed]

    per_book_rows: list[sqlite3.Row] = []
    tag_op_rows: list[sqlite3.Row] = []
    for row in rows:
        if not _is_applicable(row["field"]):
            continue
        if row["field"] in _TAG_OP_FIELDS:
            tag_op_rows.append(row)
        else:
            per_book_rows.append(row)

    by_book: dict[int, list[sqlite3.Row]] = defaultdict(list)
    for row in per_book_rows:
        by_book[row["book_id"]].append(row)

    # Open Calibre's metadata.db read-only for identifier-merge / tags lookups.
    # If it isn't present (tests using synthetic library paths), fall back
    # to "no current state" — safe because the caller wouldn't actually be
    # executing calibredb against that path anyway.
    calibre_metadata_db = library_path / "metadata.db"
    cal_conn: sqlite3.Connection | None = None
    if calibre_metadata_db.exists():
        cal_conn = calibre_reader.open_readonly(calibre_metadata_db)

    # Expand each tag-op proposal to {book_id: {src_cf: target_or_None}}
    # plus a {book_id: [proposal_id]} fan-out so the executor can track
    # tag-op completion across multiple per-book commands.
    book_tag_changes: dict[int, dict[str, str | None]] = defaultdict(dict)
    book_tag_op_ids: dict[int, list[int]] = defaultdict(list)
    if cal_conn is not None:
        for op_row in tag_op_rows:
            src = op_row["calibre_value"]
            if not src:
                continue
            target: str | None
            if op_row["field"] == "tag.merge":
                _, target = live_tag_sweep.parse_merge_proposed_value(op_row["proposed_value"])
            else:  # tag.delete
                target = None
            for book_id in _books_with_tag(cal_conn, src):
                # Key by casefold so the per-book apply matches case-insensitively.
                book_tag_changes[book_id][src.casefold()] = target
                book_tag_op_ids[book_id].append(op_row["id"])

    try:
        all_book_ids = sorted(set(by_book) | set(book_tag_changes))
        for book_id in all_book_ids:
            book_rows = by_book.get(book_id, [])
            argv: list[str] = [calibredb, "set_metadata", f"--library-path={library_path}"]
            fields_map: dict[str, str] = {}
            identifier_updates: dict[str, str] = {}
            tag_additions: list[str] = []

            for row in book_rows:
                field = row["field"]
                value = row["proposed_value"]
                fields_map[field] = value

                if field == "isbn":
                    identifier_updates["isbn"] = value
                elif field.startswith("identifier."):
                    scheme = field.split(".", 1)[1]
                    identifier_updates[scheme] = value
                elif field == "tags.add":
                    tag_additions.append(value)
                else:
                    calibre_field = _FIELD_ALIASES.get(field, field)
                    argv.append(f"--field={calibre_field}:{value}")

            if identifier_updates:
                current = _current_identifiers(cal_conn, book_id) if cal_conn else {}
                current.update(identifier_updates)
                argv.append(f"--field=identifiers:{_format_identifiers(current)}")

            tag_changes = book_tag_changes.get(book_id, {})
            if tag_additions or tag_changes:
                # calibredb's --field=tags:... REPLACES the whole list, so
                # read current tags and apply removals/renames/additions
                # in one shot.
                current_tags = _current_tags(cal_conn, book_id) if cal_conn else []
                new_tags = _apply_tag_ops(
                    current_tags, additions=tag_additions, changes=tag_changes,
                )
                argv.append(f"--field=tags:{_format_tags(new_tags)}")

            argv.append(str(book_id))
            yield ApplyCommand(
                book_id=book_id,
                proposal_ids=tuple(r["id"] for r in book_rows),
                tag_op_proposal_ids=tuple(book_tag_op_ids.get(book_id, [])),
                fields=fields_map,
                argv=tuple(argv),
            )
    finally:
        if cal_conn is not None:
            cal_conn.close()


def _books_with_tag(cal_conn: sqlite3.Connection, tag_name: str) -> list[int]:
    """Return book ids currently carrying ``tag_name`` (case-insensitive)."""
    rows = cal_conn.execute(
        """
        SELECT btl.book FROM books_tags_link btl
        JOIN tags t ON t.id = btl.tag
        WHERE LOWER(t.name) = LOWER(?)
        """,
        (tag_name,),
    ).fetchall()
    return [int(r["book"]) for r in rows]


def _apply_tag_ops(
    current: list[str],
    *,
    additions: list[str],
    changes: dict[str, str | None],
) -> list[str]:
    """Compute a book's new tag list by applying tag-level ops.

    ``changes`` maps casefolded source-tag to either a target name (rename)
    or ``None`` (delete). ``additions`` are tag-level adds from the
    ``tags.add`` queue. Order: keep originals where unchanged, substitute
    renamed ones in-place, drop deletes, then append novel additions.
    Case-insensitive dedupe across the merged result."""
    seen_ci: set[str] = set()
    out: list[str] = []
    for tag in current:
        cf = tag.casefold()
        if cf in changes:
            target = changes[cf]
            if target is None:
                continue  # deleted
            target_cf = target.casefold()
            if target_cf in seen_ci:
                continue
            out.append(target)
            seen_ci.add(target_cf)
        else:
            if cf in seen_ci:
                continue
            out.append(tag)
            seen_ci.add(cf)
    for tag in additions:
        cf = tag.casefold()
        if cf in seen_ci:
            continue
        out.append(tag)
        seen_ci.add(cf)
    return out


def _current_identifiers(cal_conn: sqlite3.Connection, book_id: int) -> dict[str, str]:
    """Read the current identifiers dict for a book from Calibre's DB."""
    rows = cal_conn.execute("SELECT type, val FROM identifiers WHERE book = ?", (book_id,)).fetchall()
    out: dict[str, str] = {}
    for r in rows:
        scheme = (r["type"] or "").strip().lower()
        value = (r["val"] or "").strip()
        if scheme and value:
            out[scheme] = value
    return out


def _format_identifiers(ids: dict[str, str]) -> str:
    """Render an identifiers dict in calibredb's ``scheme1:val1,scheme2:val2``
    form. Sorted for deterministic output."""
    return ",".join(f"{k}:{v}" for k, v in sorted(ids.items()))


def _current_tags(cal_conn: sqlite3.Connection, book_id: int) -> list[str]:
    """Read the current tag list for a book from Calibre's DB."""
    rows = cal_conn.execute(
        """
        SELECT t.name FROM tags t
        JOIN books_tags_link btl ON btl.tag = t.id
        WHERE btl.book = ?
        """,
        (book_id,),
    ).fetchall()
    return [(r["name"] or "").strip() for r in rows if (r["name"] or "").strip()]


def _merge_tags(current: list[str], additions: list[str]) -> list[str]:
    """Union ``current`` with ``additions``, preserving original order and
    appending only tags not already present (case-insensitive)."""
    seen_ci: set[str] = set()
    out: list[str] = []
    for tag in current:
        cf = tag.casefold()
        if cf not in seen_ci:
            seen_ci.add(cf)
            out.append(tag)
    for tag in additions:
        cf = tag.casefold()
        if cf not in seen_ci:
            seen_ci.add(cf)
            out.append(tag)
    return out


def _format_tags(tags: list[str]) -> str:
    """Render the tag list as calibredb expects: comma-separated.

    Calibre splits on commas, so tag values containing commas would corrupt
    the list. Real tags don't usually contain commas, but we strip just in
    case to fail safe rather than silently mis-split."""
    return ",".join(t.replace(",", " ") for t in tags)


def plan_report(conn: sqlite3.Connection) -> dict[str, int]:
    """Summary of what ``plan`` would emit vs skip, grouped by field.

    Useful so the reviewer can see ``'tags.add: 1674 skipped (merge needed)'``
    instead of silently dropping them."""
    out: dict[str, int] = {}
    for row in proposals.list_proposals(conn, status="approved", limit=1_000_000):
        key = row["field"]
        if not _is_applicable(key):
            key = f"{key} (skipped: needs merge)"
        out[key] = out.get(key, 0) + 1
    return out


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def execute(
    conn: sqlite3.Connection,
    commands: Iterator[ApplyCommand] | Sequence[ApplyCommand],
    *,
    timeout: float = 30.0,
    on_result: Callable[[ApplyResult], None] | None = None,
) -> Iterator[ApplyResult]:
    """Run each ``ApplyCommand`` via subprocess and update proposal statuses.

    Per-book proposals → ``applied`` on success, ``conflict`` on failure
    (immediate, per-command). Tag-op proposals fan out across many books
    so they're tracked separately and only marked applied once *every*
    affected book's command succeeded; partial failures leave them as
    ``approved`` (so a re-run will retry the survivors)."""
    tag_op_outcomes: dict[int, list[bool]] = defaultdict(list)
    try:
        for cmd in commands:
            try:
                result = subprocess.run(  # noqa: S603 — argv list, not shell
                    list(cmd.argv),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
                _mark_failed(conn, cmd.proposal_ids, f"{type(exc).__name__}: {exc}")
                for tag_op_id in cmd.tag_op_proposal_ids:
                    tag_op_outcomes[tag_op_id].append(False)
                yield ApplyResult(
                    command=cmd,
                    ok=False,
                    returncode=-1,
                    stdout="",
                    stderr=f"{type(exc).__name__}: {exc}",
                )
                continue

            ok = result.returncode == 0
            if ok:
                if cmd.proposal_ids:
                    _mark_applied(conn, cmd.proposal_ids)
            else:
                if cmd.proposal_ids:
                    _mark_failed(
                        conn,
                        cmd.proposal_ids,
                        f"calibredb exit={result.returncode}: {result.stderr.strip() or result.stdout.strip()}",
                    )
            for tag_op_id in cmd.tag_op_proposal_ids:
                tag_op_outcomes[tag_op_id].append(ok)
            ar = ApplyResult(
                command=cmd,
                ok=ok,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
            if on_result is not None:
                on_result(ar)
            yield ar
    finally:
        # Tag-op proposals: mark applied only when every affected book's
        # command succeeded. Otherwise leave as approved so a re-run picks
        # up the survivors. Notes carry the failure breakdown either way.
        for tag_op_id, results in tag_op_outcomes.items():
            failed = sum(1 for r in results if not r)
            if failed == 0:
                _mark_applied(conn, [tag_op_id])
            else:
                conn.execute(
                    "UPDATE proposals SET notes = COALESCE(notes || '; ', '') || ? "
                    "WHERE id = ? AND status='approved'",
                    (f"apply: {failed}/{len(results)} per-book commands failed", tag_op_id),
                )


def _mark_applied(conn: sqlite3.Connection, proposal_ids: Sequence[int]) -> None:
    now = datetime.now(UTC).isoformat()
    placeholders = ",".join(["?"] * len(proposal_ids))
    conn.execute(
        f"""
        UPDATE proposals
        SET status = 'applied', applied_at = ?
        WHERE id IN ({placeholders})
        """,  # noqa: S608 — placeholders is '?,?,?' string; ids bind via params
        [now, *proposal_ids],
    )


def _mark_failed(conn: sqlite3.Connection, proposal_ids: Sequence[int], reason: str) -> None:
    placeholders = ",".join(["?"] * len(proposal_ids))
    conn.execute(
        f"""
        UPDATE proposals
        SET status          = 'conflict',
            conflict_reason = ?,
            reviewed_at     = ?
        WHERE id IN ({placeholders})
        """,  # noqa: S608 — placeholders is '?,?,?' string; ids bind via params
        [reason, datetime.now(UTC).isoformat(), *proposal_ids],
    )


# ---------------------------------------------------------------------------
# Rendering (for dry-run / logs)
# ---------------------------------------------------------------------------


def render(cmd: ApplyCommand) -> str:
    """Return a shell-quoted, copy-pasteable version of the calibredb argv."""
    return " ".join(shlex.quote(a) for a in cmd.argv)
