"""Tag-level cleanup: delete junk tags, merge near-duplicates.

Unlike the per-book proposals system, these operations work at the library
level — a tag rename is one SQL UPDATE on the tags table and instantly affects
every book that carries it.  Calibre-Web reads SQLite on each request so
changes are visible immediately; Calibre desktop needs a library refresh.

Three operation types
---------------------
delete  — strip tag from all books and drop from tags table
merge   — redirect all books from old_tag to existing canonical, drop old_tag
rename  — change tag name in place (canonical doesn't exist yet)

Passes
------
Pass 1 — mechanical deletes: URL tags, garbage keyword strings, LCSH catalog
         tags (geographic qualifiers, war/conflict entries, TV program refs).
         No human judgment needed; pattern-matched with high confidence.

Pass 2 — canonical merges: known near-duplicate pairs (Non-Fiction→Nonfiction,
         SciFi→Science Fiction, etc.).  Small hardcoded list; human-reviewed.

Pass 3 — compound tag triage: semicolon / ampersand compound tags.
         Handled separately once the list is clean from passes 1+2.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

OpKind = Literal["delete", "merge", "rename"]


@dataclass
class TagOp:
    op: OpKind
    source_tag: str
    target_tag: str | None  # None for delete
    reason: str
    books_affected: int
    pass_num: int = 1


@dataclass
class ScanResult:
    total_tags: int
    singleton_count: int
    ops: list[TagOp] = field(default_factory=list)

    @property
    def pass1_ops(self) -> list[TagOp]:
        return [o for o in self.ops if o.pass_num == 1]

    @property
    def pass2_ops(self) -> list[TagOp]:
        return [o for o in self.ops if o.pass_num == 2]

    @property
    def deletes(self) -> list[TagOp]:
        return [o for o in self.ops if o.op == "delete"]

    @property
    def merges(self) -> list[TagOp]:
        return [o for o in self.ops if o.op in ("merge", "rename")]


# ---------------------------------------------------------------------------
# Pattern matchers for Pass 1
# ---------------------------------------------------------------------------

_URL_RE        = re.compile(r'^https?://')
_GEO_ABBR_RE   = re.compile(r'\([A-Z][a-z]+\.\)')          # (N.Y.), (Mass.)
_TV_PROG_RE    = re.compile(r'\(Television Program\)', re.I)
_YEAR_RANGE_RE = re.compile(r'\b\d{4}-\d{4}\b')
_WAR_KEYWORDS  = re.compile(
    r'\b(War|Conflict|Revolution|Operation|Uprising|Occupation|Rebellion)\b'
)

# Multi-semicolon = definitely LCSH compound subject heading
_MULTI_SEMI_RE = re.compile(r'.+;.+;')

_MAX_TAG_LEN = 60  # tags longer than this are keyword-stuffed garbage


# ---------------------------------------------------------------------------
# Pass 2: hardcoded canonical merge pairs
# ---------------------------------------------------------------------------

# (source_tag, canonical_tag) — source will be merged/renamed into canonical.
# Only pairs where the merge is unambiguous and the canonical clearly dominates.
CANONICAL_MERGES: list[tuple[str, str]] = [
    ("Non-Fiction",            "Nonfiction"),
    ("Non Fiction",            "Nonfiction"),
    ("SciFi",                  "Science Fiction"),
    ("Horror & Ghost Stories", "Horror"),
    ("Thrillers & Suspense",   "Thriller"),
    # Leave 'history' (441) and 'Historical' (299) alone — distinct meanings.
    # Leave 'Star Trek' (64) — may carry non-fiction books, decide later.
    # Leave compound & tags for Pass 3.
]


# ---------------------------------------------------------------------------
# Core scanner
# ---------------------------------------------------------------------------

def scan(cal: sqlite3.Connection) -> ScanResult:
    """Analyse Calibre's tag table and classify tags into operations.

    Returns a ScanResult with Pass-1 delete ops and Pass-2 merge ops.
    Does NOT write anything — call save_ops() to persist the plan.
    """
    rows = cal.execute("""
        SELECT t.name, COUNT(btl.book) AS cnt
        FROM tags t
        LEFT JOIN books_tags_link btl ON btl.tag = t.id
        GROUP BY t.id
        ORDER BY t.name COLLATE NOCASE
    """).fetchall()

    tag_counts: dict[str, int] = {r["name"]: r["cnt"] for r in rows}
    ops: list[TagOp] = []

    # ── Pass 1: mechanical deletes ──────────────────────────────────────────
    for name, cnt in tag_counts.items():

        if _URL_RE.match(name):
            ops.append(TagOp("delete", name, None, "URL tag", cnt, pass_num=1))
            continue

        if len(name) > _MAX_TAG_LEN:
            ops.append(TagOp("delete", name, None,
                             f"keyword-stuffed string ({len(name)} chars)", cnt, pass_num=1))
            continue

        if _TV_PROG_RE.search(name):
            ops.append(TagOp("delete", name, None, "LCSH TV-program tag", cnt, pass_num=1))
            continue

        if _GEO_ABBR_RE.search(name):
            ops.append(TagOp("delete", name, None, "LCSH geographic qualifier", cnt, pass_num=1))
            continue

        # LCSH war/event tags: require semicolon PLUS war keyword or year range
        if ";" in name and (_WAR_KEYWORDS.search(name) or _YEAR_RANGE_RE.search(name)):
            ops.append(TagOp("delete", name, None, "LCSH historical-event tag", cnt, pass_num=1))
            continue

        # Multi-semicolon compound subjects (3+ parts) — always LCSH noise
        if _MULTI_SEMI_RE.match(name):
            ops.append(TagOp("delete", name, None, "LCSH multi-part subject", cnt, pass_num=1))
            continue

    # ── Pass 2: canonical merges ────────────────────────────────────────────
    source_tags_in_pass1 = {o.source_tag for o in ops}  # don't double-queue

    for src, dst in CANONICAL_MERGES:
        if src not in tag_counts or src in source_tags_in_pass1:
            continue
        cnt = tag_counts[src]
        if dst in tag_counts:
            # Destination already exists → merge (redirect books, drop source)
            ops.append(TagOp("merge", src, dst,
                             f"canonical duplicate of {dst!r}", cnt, pass_num=2))
        else:
            # Destination doesn't exist yet → rename in place
            ops.append(TagOp("rename", src, dst,
                             f"rename to canonical form {dst!r}", cnt, pass_num=2))

    return ScanResult(
        total_tags=len(tag_counts),
        singleton_count=sum(1 for c in tag_counts.values() if c == 1),
        ops=ops,
    )


# ---------------------------------------------------------------------------
# Persistence: save / load ops in proposals.db
# ---------------------------------------------------------------------------

def save_ops(ops_conn: sqlite3.Connection, ops: list[TagOp]) -> int:
    """Upsert tag ops into proposals.db.  Returns count of newly inserted rows.

    Uses INSERT OR IGNORE so running scan twice is idempotent — existing
    pending ops are preserved unchanged.
    """
    inserted = 0
    for op in ops:
        cur = ops_conn.execute("""
            INSERT OR IGNORE INTO tag_ops
                (op, source_tag, target_tag, reason, books_affected, pass)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (op.op, op.source_tag, op.target_tag, op.reason,
              op.books_affected, op.pass_num))
        inserted += cur.rowcount
    ops_conn.execute("COMMIT")
    ops_conn.execute("BEGIN")
    return inserted


def load_ops(
    ops_conn: sqlite3.Connection,
    *,
    status: str = "pending",
    pass_num: int | None = None,
) -> list[TagOp]:
    """Load ops from proposals.db, optionally filtered by pass."""
    query = "SELECT * FROM tag_ops WHERE status = ?"
    params: list = [status]
    if pass_num is not None:
        query += " AND pass = ?"
        params.append(pass_num)
    query += " ORDER BY pass, op, source_tag COLLATE NOCASE"
    rows = ops_conn.execute(query, params).fetchall()
    return [
        TagOp(
            op=r["op"],
            source_tag=r["source_tag"],
            target_tag=r["target_tag"],
            reason=r["reason"],
            books_affected=r["books_affected"],
            pass_num=r["pass"],
        )
        for r in rows
    ]


def skip_op(ops_conn: sqlite3.Connection, source_tag: str, notes: str = "") -> None:
    """Mark a pending op as skipped (won't be applied)."""
    ops_conn.execute("""
        UPDATE tag_ops SET status = 'skipped', notes = ?
        WHERE source_tag = ? AND status = 'pending'
    """, (notes, source_tag))
    ops_conn.execute("COMMIT")
    ops_conn.execute("BEGIN")


# ---------------------------------------------------------------------------
# Apply: execute ops against Calibre's DB
# ---------------------------------------------------------------------------

def open_calibre_rw(metadata_db: Path) -> sqlite3.Connection:
    """Open Calibre's metadata.db in read-write mode for tag operations."""
    conn = sqlite3.connect(metadata_db)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@dataclass
class ApplyResult:
    applied: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)


def apply_ops(
    cal: sqlite3.Connection,
    ops_conn: sqlite3.Connection,
    ops: list[TagOp],
    *,
    dry_run: bool = True,
) -> ApplyResult:
    """Execute a list of TagOps against Calibre's DB.

    When ``dry_run=True`` (default), validates the ops and reports what
    would happen without modifying anything.
    """
    result = ApplyResult()

    for op in ops:
        try:
            src_id = _tag_id(cal, op.source_tag)
            if src_id is None:
                # Tag no longer exists — already cleaned up or never existed
                result.skipped += 1
                if not dry_run:
                    _mark_applied(ops_conn, op.source_tag, notes="tag not found — skipped")
                continue

            if op.op == "delete":
                if not dry_run:
                    cal.execute("DELETE FROM books_tags_link WHERE tag = ?", (src_id,))
                    cal.execute("DELETE FROM tags WHERE id = ?", (src_id,))
                    _mark_applied(ops_conn, op.source_tag)

            elif op.op == "merge":
                dst_id = _tag_id(cal, op.target_tag)
                if dst_id is None:
                    result.errors.append(
                        f"merge {op.source_tag!r}→{op.target_tag!r}: destination not found"
                    )
                    continue
                if not dry_run:
                    # Move books that only have the source tag over to the destination.
                    # OR IGNORE handles books that already carry both.
                    cal.execute("""
                        UPDATE OR IGNORE books_tags_link
                        SET tag = ?
                        WHERE tag = ?
                    """, (dst_id, src_id))
                    # Remove any remaining links (books that had both — the IGNORE above
                    # left the duplicate source link in place).
                    cal.execute("DELETE FROM books_tags_link WHERE tag = ?", (src_id,))
                    cal.execute("DELETE FROM tags WHERE id = ?", (src_id,))
                    _mark_applied(ops_conn, op.source_tag)

            elif op.op == "rename":
                if _tag_id(cal, op.target_tag) is not None:
                    result.errors.append(
                        f"rename {op.source_tag!r}→{op.target_tag!r}: "
                        f"target already exists — use merge instead"
                    )
                    continue
                if not dry_run:
                    cal.execute("UPDATE tags SET name = ? WHERE id = ?",
                                (op.target_tag, src_id))
                    _mark_applied(ops_conn, op.source_tag)

            result.applied += 1

        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"{op.source_tag!r}: {exc}")

    if not dry_run:
        cal.execute("COMMIT")
        cal.execute("BEGIN")

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tag_id(cal: sqlite3.Connection, name: str | None) -> int | None:
    if not name:
        return None
    row = cal.execute("SELECT id FROM tags WHERE name = ?", (name,)).fetchone()
    return int(row["id"]) if row else None


def _mark_applied(ops_conn: sqlite3.Connection, source_tag: str, notes: str = "") -> None:
    ops_conn.execute("""
        UPDATE tag_ops
        SET status = 'applied', applied_at = ?, notes = ?
        WHERE source_tag = ?
    """, (datetime.now(UTC).isoformat(), notes, source_tag))
