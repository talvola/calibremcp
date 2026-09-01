"""Read-only view of a Calibre metadata.db.

Exposes one ``BookRecord`` per book with everything the miner needs to diff
against OPF-derived metadata. Uses stdlib sqlite3 opened read-only so it
cannot write to the live library.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BookRecord:
    """One row from Calibre's library, shaped for diffing."""

    id: int
    uuid: str | None
    title: str
    path: str  # relative to library_root
    authors: tuple[str, ...]
    publisher: str | None
    pubdate: str | None  # YYYY-MM-DD or None (Calibre sentinels mapped to None)
    description: str | None  # from comments.text
    series: str | None
    series_index: str | None  # kept as string to preserve fractionals
    tags_ci: frozenset[str]  # casefolded for easy membership checks
    isbn: str | None  # from identifiers where type='isbn'
    identifiers: dict[str, str]  # full map: scheme -> value
    language: str | None  # first lang_code
    # Added 2026-09-01 for the MCP read path. Defaults keep older call sites working.
    author_sort: str | None = None  # books.author_sort — governs Calibre-Web's author display order
    tags: tuple[str, ...] = ()  # stored-case tag names, ordered by name (what Calibre-Web shows)


def open_readonly(metadata_db: Path) -> sqlite3.Connection:
    """Open Calibre's metadata.db in read-only mode via URI.

    Why: ``mode=ro`` guarantees no accidental writes even if a caller forgets
    to wrap statements in BEGIN/COMMIT. Why: ``immutable=1`` skips SQLite's
    WAL/SHM probe and shared-lock acquisition — both of which are unreliable
    over CIFS and produce intermittent "unable to open database file" errors.
    Safe here because the canonical workflow keeps the library mounted ro and
    stops Calibre-Web before any apply pass."""
    uri = f"file:{Path(metadata_db).resolve()}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def book_count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS n FROM books").fetchone()
    return int(row["n"])


def iter_books(conn: sqlite3.Connection, *, limit: int | None = None) -> Iterator[BookRecord]:
    """Yield BookRecords in stable id order.

    We batch lookups for authors / tags / identifiers / languages per book
    rather than one giant join (keeps memory predictable on a 21k-book library)."""
    query = """
        SELECT b.id, b.uuid, b.title, b.path, b.pubdate,
               b.series_index,
               s.name  AS series_name,
               p.name  AS publisher_name,
               c.text  AS comments_text
        FROM books b
        LEFT JOIN books_series_link    bsl ON bsl.book = b.id
        LEFT JOIN series               s   ON s.id = bsl.series
        LEFT JOIN books_publishers_link bpl ON bpl.book = b.id
        LEFT JOIN publishers           p   ON p.id = bpl.publisher
        LEFT JOIN comments             c   ON c.book = b.id
        ORDER BY b.id
    """
    if limit is not None:
        query += f" LIMIT {int(limit)}"

    for row in conn.execute(query):
        book_id = int(row["id"])
        authors = _authors_for(conn, book_id)
        identifiers = _identifiers_for(conn, book_id)
        tag_names = _tag_names_for(conn, book_id)
        language = _language_for(conn, book_id)
        yield BookRecord(
            id=book_id,
            uuid=row["uuid"],
            title=row["title"],
            path=row["path"],
            authors=authors,
            publisher=_strip_or_none(row["publisher_name"]),
            pubdate=_normalize_calibre_date(row["pubdate"]),
            description=_strip_or_none(row["comments_text"]),
            series=_strip_or_none(row["series_name"]),
            series_index=_format_series_index(row["series_index"]),
            tags_ci=frozenset(t.casefold() for t in tag_names),
            isbn=identifiers.get("isbn"),
            identifiers=identifiers,
            language=language,
            tags=tag_names,  # author_sort left None: only the MCP read path needs it
        )


def _authors_for(conn: sqlite3.Connection, book_id: int) -> tuple[str, ...]:
    rows = conn.execute(
        """
        SELECT a.name FROM authors a
        JOIN books_authors_link bal ON bal.author = a.id
        WHERE bal.book = ?
        ORDER BY bal.id
        """,
        (book_id,),
    ).fetchall()
    return tuple(r["name"] for r in rows if r["name"])


def _identifiers_raw_for(conn: sqlite3.Connection, book_id: int) -> dict[str, str]:
    """Identifiers exactly as stored (type case preserved), in row order.

    Calibre desktop lowercases identifier types, but Calibre-Web stores
    whatever case the import/edit supplied, so this library carries mixed
    and all-caps junk types (``ISBN``, ``FICTIONDB``, ``MOBI-ASIN``). Anything
    shown to a human or agent must use this form so it can be compared
    against the Calibre-Web edit form and against what was written.
    """
    rows = conn.execute(
        "SELECT type, val FROM identifiers WHERE book = ? ORDER BY id",
        (book_id,),
    ).fetchall()
    out: dict[str, str] = {}
    for r in rows:
        t = (r["type"] or "").strip()
        v = (r["val"] or "").strip()
        if t and v:
            out[t] = v
    return out


def _identifiers_for(conn: sqlite3.Connection, book_id: int) -> dict[str, str]:
    """Lowercase-keyed identifiers — for the cleanup pipeline's lookups only."""
    return {t.lower(): v for t, v in _identifiers_raw_for(conn, book_id).items()}


def _tag_names_for(conn: sqlite3.Connection, book_id: int) -> tuple[str, ...]:
    """Stored-case tag names for one book, ordered by name (case-insensitive).

    This is the form Calibre-Web displays (its ``Books.tags`` relationship is
    ``order_by="Tags.name"``). Use this — not ``_tags_for`` — whenever the
    result is shown to a human or an agent that will compare against what it
    wrote: the library's canonical tags are mostly Title Case with a handful
    of lowercase exceptions (``cookbooks``, ``nonfiction``, ``thriller``).
    """
    rows = conn.execute(
        """
        SELECT t.name FROM tags t
        JOIN books_tags_link btl ON btl.tag = t.id
        WHERE btl.book = ?
        ORDER BY t.name COLLATE NOCASE
        """,
        (book_id,),
    ).fetchall()
    return tuple(r["name"] for r in rows if r["name"])


def _tags_for(conn: sqlite3.Connection, book_id: int) -> frozenset[str]:
    """Casefolded tag set — for membership checks in the cleanup pipeline only."""
    return frozenset(t.casefold() for t in _tag_names_for(conn, book_id))


def _language_for(conn: sqlite3.Connection, book_id: int) -> str | None:
    row = conn.execute(
        """
        SELECT l.lang_code FROM languages l
        JOIN books_languages_link bll ON bll.lang_code = l.id
        WHERE bll.book = ?
        ORDER BY bll.item_order
        LIMIT 1
        """,
        (book_id,),
    ).fetchone()
    return _strip_or_none(row["lang_code"]) if row else None


def _strip_or_none(v: str | None) -> str | None:
    if v is None:
        return None
    s = v.strip()
    return s or None


def _normalize_calibre_date(raw: str | None) -> str | None:
    """Calibre stores missing pubdate as the sentinel year 0101-01-01. Map to None."""
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("0101-") or s.startswith("0001-"):
        return None
    # Calibre pubdates are typically ISO 8601; we only keep the date portion.
    return s.split("T", 1)[0].split(" ", 1)[0] or None


def _format_series_index(value: float | int | str | None) -> str | None:
    """Calibre stores series_index as REAL. Render it without trailing zeros so
    integers render as '1' and fractionals like 1.5 survive as '1.5'.

    NB: Calibre can't natively distinguish Erik's '1.3' omnibus notation from
    '1.3' meaning a side-story — whatever Calibre has, we pass through.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    # Integer-valued floats should render cleanly: 1.0 -> '1'.
    if isinstance(value, (int, float)) and float(value).is_integer():
        return str(int(value))
    # Keep a reasonable precision; strip trailing zeros.
    formatted = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return formatted or None


@contextmanager
def open_library(metadata_db: Path) -> Iterator[sqlite3.Connection]:
    """Context manager for a read-only Calibre DB connection."""
    conn = open_readonly(metadata_db)
    try:
        yield conn
    finally:
        conn.close()
