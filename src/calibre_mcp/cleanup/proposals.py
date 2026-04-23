"""Propose-queue SQLite layer for the cleanup pipeline.

Thin wrapper around ``cleanup_proposals.db``. Uses stdlib sqlite3 (no
SQLAlchemy) so this module stays self-contained and easy to extract.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).parent / "schema.sql"

_VALID_STATUSES = frozenset({"proposed", "approved", "rejected", "applied", "superseded", "conflict"})


@dataclass(frozen=True, slots=True)
class Proposal:
    """A single proposed metadata change for one book field."""

    book_id: int
    field: str
    proposed_value: str
    source: str
    confidence: float
    book_uuid: str | None = None
    calibre_value: str | None = None
    conflict_reason: str | None = None
    notes: str | None = None


def connect(db_path: Path | str) -> sqlite3.Connection:
    """Open (and initialize on first use) the proposals database."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None)  # autocommit; we manage BEGIN/COMMIT
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))


def start_run(
    conn: sqlite3.Connection,
    *,
    source: str,
    library_root: Path,
    metadata_db_path: Path,
    dry_run: bool,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO miner_runs (source, library_root, metadata_db_path, dry_run)
        VALUES (?, ?, ?, ?)
        """,
        (source, str(library_root), str(metadata_db_path), int(dry_run)),
    )
    run_id = cur.lastrowid
    if run_id is None:
        raise RuntimeError("sqlite3 did not return a lastrowid for miner_runs insert")
    return run_id


def finish_run(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    books_scanned: int,
    books_with_epub: int,
    books_parsed: int,
    proposals_emitted: int,
    errors: int,
    notes: str | None = None,
) -> None:
    conn.execute(
        """
        UPDATE miner_runs
        SET completed_at      = ?,
            books_scanned     = ?,
            books_with_epub   = ?,
            books_parsed      = ?,
            proposals_emitted = ?,
            errors            = ?,
            notes             = ?
        WHERE id = ?
        """,
        (
            datetime.now(UTC).isoformat(),
            books_scanned,
            books_with_epub,
            books_parsed,
            proposals_emitted,
            errors,
            notes,
            run_id,
        ),
    )


def record_error(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    book_id: int | None,
    book_path: str | None,
    error: str,
) -> None:
    conn.execute(
        "INSERT INTO miner_errors (run_id, book_id, book_path, error) VALUES (?, ?, ?, ?)",
        (run_id, book_id, book_path, error),
    )


def insert_proposal(conn: sqlite3.Connection, run_id: int, p: Proposal) -> bool:
    """Insert a proposal. Idempotent on ``(book_id, field, source, proposed_value)``.

    Returns True if a new row was inserted, False if that exact proposal was
    already recorded (so re-running the miner is a no-op).
    """
    status = "conflict" if p.conflict_reason else "proposed"
    try:
        conn.execute(
            """
            INSERT INTO proposals
                (book_id, book_uuid, field, calibre_value, proposed_value,
                 source, confidence, status, conflict_reason, notes, run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                p.book_id,
                p.book_uuid,
                p.field,
                p.calibre_value,
                p.proposed_value,
                p.source,
                p.confidence,
                status,
                p.conflict_reason,
                p.notes,
                run_id,
            ),
        )
        return True
    except sqlite3.IntegrityError:
        return False


def summary_counts(conn: sqlite3.Connection) -> dict[str, dict[str, int]]:
    """Return ``{field: {status: count}}`` across all proposals."""
    cur = conn.execute("SELECT field, status, COUNT(*) AS n FROM proposals GROUP BY field, status")
    result: dict[str, dict[str, int]] = {}
    for row in cur:
        result.setdefault(row["field"], {})[row["status"]] = row["n"]
    return result


def list_proposals(
    conn: sqlite3.Connection,
    *,
    field: str | None = None,
    status: str | None = None,
    source: str | None = None,
    limit: int = 50,
) -> list[sqlite3.Row]:
    query = "SELECT * FROM proposals WHERE 1=1"
    params: list[Any] = []
    if field is not None:
        query += " AND field = ?"
        params.append(field)
    if status is not None:
        query += " AND status = ?"
        params.append(status)
    if source is not None:
        query += " AND source = ?"
        params.append(source)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    return list(conn.execute(query, params))


def _build_filter_clauses(
    *,
    field: str | None = None,
    status: str | None = None,
    source: str | None = None,
    ids: Sequence[int] | None = None,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
) -> tuple[str, list[Any]]:
    """Assemble a WHERE-clause fragment (without the ``WHERE`` keyword) and
    bound parameters from the given filters. Returns ``('1=1', [])`` when
    no filters are supplied so the caller can concatenate safely."""
    clauses: list[str] = []
    params: list[Any] = []
    if field is not None:
        clauses.append("field = ?")
        params.append(field)
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    if source is not None:
        clauses.append("source = ?")
        params.append(source)
    if ids:
        placeholders = ",".join(["?"] * len(ids))
        clauses.append(f"id IN ({placeholders})")
        params.extend(ids)
    if min_confidence is not None:
        clauses.append("confidence >= ?")
        params.append(min_confidence)
    if max_confidence is not None:
        clauses.append("confidence <= ?")
        params.append(max_confidence)
    return (" AND ".join(clauses) if clauses else "1=1"), params


def count_proposals(
    conn: sqlite3.Connection,
    *,
    field: str | None = None,
    status: str | None = None,
    source: str | None = None,
    ids: Sequence[int] | None = None,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
) -> int:
    """Count proposals matching the filters. Used to preview bulk operations."""
    where, params = _build_filter_clauses(
        field=field,
        status=status,
        source=source,
        ids=ids,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
    )
    row = conn.execute(f"SELECT COUNT(*) AS n FROM proposals WHERE {where}", params).fetchone()  # noqa: S608 — {where} is a keyword-only fragment from _build_filter_clauses; values flow via bound params
    return int(row["n"])


def set_status_where(
    conn: sqlite3.Connection,
    new_status: str,
    *,
    notes: str | None = None,
    field: str | None = None,
    status: str | None = None,
    source: str | None = None,
    ids: Sequence[int] | None = None,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
) -> int:
    """Bulk-update status for proposals matching filters. Returns rowcount.

    ``notes`` is merged via COALESCE so callers can leave earlier notes
    intact by passing None. Without any filter this refuses to run (would
    update the entire table) — callers must pass at least one narrowing arg."""
    if new_status not in _VALID_STATUSES:
        raise ValueError(f"invalid status: {new_status!r}")
    if all(v is None or v == [] for v in (field, status, source, ids, min_confidence, max_confidence)):
        raise ValueError("refusing to update without any filter — pass at least one narrowing argument")
    where, params = _build_filter_clauses(
        field=field,
        status=status,
        source=source,
        ids=ids,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
    )
    cur = conn.execute(
        f"""
        UPDATE proposals
        SET status      = ?,
            reviewed_at = ?,
            notes       = COALESCE(?, notes)
        WHERE {where}
        """,  # noqa: S608 — {where} is keyword-only from _build_filter_clauses; values bind as params
        [new_status, datetime.now(UTC).isoformat(), notes, *params],
    )
    return cur.rowcount


def set_status(
    conn: sqlite3.Connection,
    proposal_id: int,
    status: str,
    *,
    notes: str | None = None,
) -> None:
    if status not in _VALID_STATUSES:
        raise ValueError(f"invalid status: {status!r}")
    conn.execute(
        """
        UPDATE proposals
        SET status      = ?,
            reviewed_at = ?,
            notes       = COALESCE(?, notes)
        WHERE id = ?
        """,
        (status, datetime.now(UTC).isoformat(), notes, proposal_id),
    )
