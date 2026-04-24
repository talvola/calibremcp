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

from calibre_mcp.cleanup import calibre_reader, proposals

log = logging.getLogger(__name__)

# Proposal fields whose backfill is a pure replacement — safe to apply
# without reading Calibre's current state first. Deferred for Phase 3:
#   - 'tags.add' (list merge)
#   - 'identifier.*' (dict merge, except isbn which Calibre treats as scalar)
_APPLICABLE_SCALAR_FIELDS: frozenset[str] = frozenset(
    {"isbn", "publisher", "pubdate", "description", "series", "series_index"}
)

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
    book."""

    book_id: int
    proposal_ids: tuple[int, ...]
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
    applicable scalar fields.

    ISBN is special: ``calibredb set_metadata`` doesn't expose a top-level
    ``isbn`` field — you set it via ``--field identifiers:isbn:XXXX``, which
    REPLACES the entire identifiers dict. So for any book with an approved
    ISBN proposal we first read the book's current identifiers from
    Calibre's metadata.db, merge in the new ISBN, and emit the full
    dict-shaped argument. This preserves any hand-added identifiers
    (goodreads, amazon, etc.) the user already has on the book.

    Proposals for fields that still need merge logic but aren't yet
    supported (tags.add, non-ISBN identifiers) are skipped — see
    ``_APPLICABLE_SCALAR_FIELDS`` and ``plan_report``.
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

    by_book: dict[int, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        if row["field"] not in _APPLICABLE_SCALAR_FIELDS:
            continue
        by_book[row["book_id"]].append(row)

    # Open Calibre's metadata.db read-only for the identifier merge lookup.
    # If it isn't present (tests using synthetic library paths), the merge
    # falls back to "no existing identifiers" — safe because the caller
    # wouldn't actually be executing calibredb against that path anyway.
    calibre_metadata_db = library_path / "metadata.db"
    cal_conn: sqlite3.Connection | None = None
    if calibre_metadata_db.exists():
        cal_conn = calibre_reader.open_readonly(calibre_metadata_db)

    try:
        for book_id, book_rows in sorted(by_book.items()):
            argv: list[str] = [calibredb, "set_metadata", f"--library-path={library_path}"]
            fields_map: dict[str, str] = {}
            new_isbn: str | None = None

            for row in book_rows:
                if row["field"] == "isbn":
                    new_isbn = row["proposed_value"]
                    fields_map["isbn"] = new_isbn
                    continue
                calibre_field = _FIELD_ALIASES.get(row["field"], row["field"])
                argv.append(f"--field={calibre_field}:{row['proposed_value']}")
                fields_map[row["field"]] = row["proposed_value"]

            if new_isbn is not None:
                current = _current_identifiers(cal_conn, book_id) if cal_conn else {}
                current["isbn"] = new_isbn
                argv.append(f"--field=identifiers:{_format_identifiers(current)}")

            argv.append(str(book_id))
            yield ApplyCommand(
                book_id=book_id,
                proposal_ids=tuple(r["id"] for r in book_rows),
                fields=fields_map,
                argv=tuple(argv),
            )
    finally:
        if cal_conn is not None:
            cal_conn.close()


def _current_identifiers(cal_conn: sqlite3.Connection, book_id: int) -> dict[str, str]:
    """Read the current identifiers dict for a book from Calibre's DB."""
    rows = cal_conn.execute(
        "SELECT type, val FROM identifiers WHERE book = ?", (book_id,)
    ).fetchall()
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


def plan_report(conn: sqlite3.Connection) -> dict[str, int]:
    """Summary of what ``plan`` would emit vs skip, grouped by field.

    Useful so the reviewer can see ``'tags.add: 1715 skipped (merge needed)'``
    instead of silently dropping them."""
    out: dict[str, int] = {}
    for row in proposals.list_proposals(conn, status="approved", limit=1_000_000):
        key = row["field"]
        if key not in _APPLICABLE_SCALAR_FIELDS:
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

    On success: proposal rows → ``status='applied'``, ``applied_at`` set.
    On failure: proposal rows → ``status='conflict'`` with ``conflict_reason``
    carrying stderr, so the reviewer can see why and retry.
    """
    for cmd in commands:
        try:
            result = subprocess.run(  # noqa: S603 — argv list, not shell
                list(cmd.argv),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            # calibredb not on PATH, or hung. Mark proposals so the reviewer
            # sees what happened; they can re-approve and re-run later.
            _mark_failed(conn, cmd.proposal_ids, f"{type(exc).__name__}: {exc}")
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
            _mark_applied(conn, cmd.proposal_ids)
        else:
            _mark_failed(
                conn,
                cmd.proposal_ids,
                f"calibredb exit={result.returncode}: {result.stderr.strip() or result.stdout.strip()}",
            )
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
