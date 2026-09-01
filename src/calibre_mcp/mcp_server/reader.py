"""Read-only queries against Calibre's metadata.db for the MCP tool surface.

Builds on ``calibre_mcp.cleanup.calibre_reader.open_readonly`` (immutable=1 +
mode=ro, CIFS-friendly). Reads don't conflict with Calibre-Web's writer lock,
so the MCP server can answer queries at full speed even while Calibre-Web is
editing.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from calibre_mcp.cleanup.calibre_reader import (
    BookRecord,
    _format_series_index,
    _identifiers_raw_for,
    _language_for,
    _normalize_calibre_date,
    _strip_or_none,
    _tag_names_for,
)


@contextmanager
def open_library(metadata_db: Path | str) -> Iterator[sqlite3.Connection]:
    """Open Calibre's metadata.db read-only, *with* WAL visibility.

    Distinct from ``cleanup.calibre_reader.open_readonly`` which uses
    ``immutable=1``. That flag is needed on the CIFS WSL path (the cleanup
    pipeline) to skip the SHM/WAL probe — but it also tells SQLite the
    database will never change, so SQLite caches everything and ignores the
    WAL file. Calibre-Web writes through WAL, so on the MCP path (NAS bind
    mount, no CIFS) we must NOT use immutable: reads need to follow the WAL
    to see fresh writes from Calibre-Web.

    Requires that ``metadata.db-wal`` and ``metadata.db-shm`` are accessible
    in the same directory as ``metadata.db`` (mount the books directory, not
    just the .db file).
    """
    uri = f"file:{Path(metadata_db).resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _display_authors_for(conn: sqlite3.Connection, book_id: int) -> tuple[tuple[str, ...], str | None]:
    """Authors in **Calibre-Web display order**, plus the raw ``author_sort``.

    Calibre-Web does not order authors by ``books_authors_link.id`` (Calibre
    desktop's order). Its ``order_authors()`` walks ``books.author_sort``,
    splits on ``&``, and emits each author whose ``authors.sort`` matches, in
    that sequence; anything left over is appended in link order. Editing
    authors through the Calibre-Web form never rewrites existing link rows,
    so link order goes stale the moment a book's author order is changed via
    Calibre-Web — reading link order would then contradict what Erik sees.

    We mirror ``order_authors`` exactly so ``calibre_get_book`` agrees with
    the Calibre-Web edit form and detail page.
    """
    row = conn.execute("SELECT author_sort FROM books WHERE id = ?", (book_id,)).fetchone()
    author_sort = _strip_or_none(row["author_sort"]) if row else None
    links = conn.execute(
        """
        SELECT a.id, a.name, a.sort FROM authors a
        JOIN books_authors_link bal ON bal.author = a.id
        WHERE bal.book = ?
        ORDER BY bal.id
        """,
        (book_id,),
    ).fetchall()
    remaining: dict[int, str] = {int(r["id"]): r["name"] for r in links if r["name"]}
    by_sort: dict[str, int] = {}
    for r in links:
        if r["sort"] and r["sort"].strip() not in by_sort:
            by_sort[r["sort"].strip()] = int(r["id"])

    ordered: list[str] = []
    for piece in (author_sort or "").split("&"):
        aid = by_sort.get(piece.strip())
        if aid is not None and aid in remaining:
            ordered.append(remaining.pop(aid))
    ordered.extend(remaining.values())  # dict preserves link order for leftovers
    return tuple(ordered), author_sort


@dataclass(frozen=True, slots=True)
class BookSummary:
    """Lightweight book row for list/search responses."""

    id: int
    title: str
    authors: tuple[str, ...]
    series: str | None
    series_index: str | None


@dataclass(frozen=True, slots=True)
class FacetCount:
    """A tag/series/author/publisher with its book count."""

    name: str
    book_count: int


class CalibreReader:
    """Thin query layer over Calibre's metadata.db.

    Opens a new connection per call so multiple FastMCP worker threads don't
    fight over a single connection. Reads are cheap (CIFS or local).
    """

    def __init__(self, metadata_db: Path | str) -> None:
        self.metadata_db = Path(metadata_db)

    def book_count(self) -> int:
        with open_library(self.metadata_db) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM books").fetchone()[0])

    def search_books(self, query: str, limit: int = 20) -> list[BookSummary]:
        """Search title / author / series substring (case-insensitive)."""
        q = f"%{query.strip()}%"
        sql = """
            SELECT DISTINCT b.id, b.title, b.series_index, s.name AS series_name
            FROM books b
            LEFT JOIN books_series_link bsl ON bsl.book = b.id
            LEFT JOIN series s              ON s.id = bsl.series
            LEFT JOIN books_authors_link bal ON bal.book = b.id
            LEFT JOIN authors a              ON a.id = bal.author
            WHERE b.title LIKE ? COLLATE NOCASE
               OR a.name  LIKE ? COLLATE NOCASE
               OR s.name  LIKE ? COLLATE NOCASE
            ORDER BY b.id DESC
            LIMIT ?
        """
        with open_library(self.metadata_db) as conn:
            rows = conn.execute(sql, (q, q, q, int(limit))).fetchall()
            return [
                BookSummary(
                    id=int(r["id"]),
                    title=r["title"] or "",
                    authors=_display_authors_for(conn, int(r["id"]))[0],
                    series=_strip_or_none(r["series_name"]),
                    series_index=_format_series_index(r["series_index"]),
                )
                for r in rows
            ]

    def get_book(self, book_id: int) -> BookRecord | None:
        sql = """
            SELECT b.id, b.uuid, b.title, b.path, b.pubdate,
                   b.series_index,
                   s.name AS series_name,
                   p.name AS publisher_name,
                   c.text AS comments_text
            FROM books b
            LEFT JOIN books_series_link    bsl ON bsl.book = b.id
            LEFT JOIN series               s   ON s.id = bsl.series
            LEFT JOIN books_publishers_link bpl ON bpl.book = b.id
            LEFT JOIN publishers           p   ON p.id = bpl.publisher
            LEFT JOIN comments             c   ON c.book = b.id
            WHERE b.id = ?
        """
        with open_library(self.metadata_db) as conn:
            row = conn.execute(sql, (int(book_id),)).fetchone()
            if row is None:
                return None
            bid = int(row["id"])
            authors, author_sort = _display_authors_for(conn, bid)
            tag_names = _tag_names_for(conn, bid)
            identifiers = _identifiers_raw_for(conn, bid)  # stored case, never normalized
            isbn = next((v for k, v in identifiers.items() if k.lower() == "isbn"), None)
            return BookRecord(
                id=bid,
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
                isbn=isbn,
                identifiers=identifiers,
                language=_language_for(conn, bid),
                author_sort=author_sort,
                tags=tag_names,
            )

    def get_author_sort(self, book_id: int) -> str | None:
        """Raw ``books.author_sort`` — the field Calibre-Web keys author display order off."""
        with open_library(self.metadata_db) as conn:
            row = conn.execute("SELECT author_sort FROM books WHERE id = ?", (int(book_id),)).fetchone()
            return _strip_or_none(row["author_sort"]) if row else None

    def list_series(self, query: str = "", limit: int = 50) -> list[FacetCount]:
        return self._facet("series", "books_series_link", "series", query, limit)

    def list_tags(self, query: str = "", limit: int = 100) -> list[FacetCount]:
        return self._facet("tags", "books_tags_link", "tag", query, limit)

    def list_authors(self, query: str = "", limit: int = 50) -> list[FacetCount]:
        return self._facet("authors", "books_authors_link", "author", query, limit)

    def list_publishers(self, query: str = "", limit: int = 50) -> list[FacetCount]:
        return self._facet("publishers", "books_publishers_link", "publisher", query, limit)

    def search_advanced(
        self,
        *,
        title_contains: str | None = None,
        author_contains: str | None = None,
        series_contains: str | None = None,
        publisher_contains: str | None = None,
        language: str | None = None,
        tags_all: list[str] | None = None,
        tags_any: list[str] | None = None,
        tags_none: list[str] | None = None,
        pubdate_after: str | None = None,
        pubdate_before: str | None = None,
        has_series: bool | None = None,
        has_isbn: bool | None = None,
        has_cover: bool | None = None,
        identifier_type: str | None = None,
        identifier_value: str | None = None,
        limit: int = 50,
    ) -> list[BookSummary]:
        """Multi-field AND-combined search across the library.

        All filters are optional; only the non-None ones constrain the result.
        Tag matching is case-insensitive and exact. ``tags_all`` requires every
        listed tag to be present, ``tags_any`` requires at least one, and
        ``tags_none`` excludes any book carrying any of them.
        """
        clauses: list[str] = []
        params: list[object] = []

        if title_contains:
            clauses.append("b.title LIKE ? COLLATE NOCASE")
            params.append(f"%{title_contains}%")
        if author_contains:
            clauses.append(
                "EXISTS (SELECT 1 FROM books_authors_link bal "
                "JOIN authors a ON a.id = bal.author "
                "WHERE bal.book = b.id AND a.name LIKE ? COLLATE NOCASE)"
            )
            params.append(f"%{author_contains}%")
        if series_contains:
            clauses.append(
                "EXISTS (SELECT 1 FROM books_series_link bsl "
                "JOIN series s2 ON s2.id = bsl.series "
                "WHERE bsl.book = b.id AND s2.name LIKE ? COLLATE NOCASE)"
            )
            params.append(f"%{series_contains}%")
        if publisher_contains:
            clauses.append(
                "EXISTS (SELECT 1 FROM books_publishers_link bpl "
                "JOIN publishers p ON p.id = bpl.publisher "
                "WHERE bpl.book = b.id AND p.name LIKE ? COLLATE NOCASE)"
            )
            params.append(f"%{publisher_contains}%")
        if language:
            clauses.append(
                "EXISTS (SELECT 1 FROM books_languages_link bll "
                "JOIN languages lg ON lg.id = bll.lang_code "
                "WHERE bll.book = b.id AND lg.lang_code = ? COLLATE NOCASE)"
            )
            params.append(language)
        if tags_all:
            placeholders = ",".join("?" * len(tags_all))
            clauses.append(
                f"(SELECT COUNT(DISTINCT t.id) FROM books_tags_link btl "
                f"JOIN tags t ON t.id = btl.tag "
                f"WHERE btl.book = b.id AND t.name COLLATE NOCASE IN ({placeholders})) = ?"
            )
            params.extend(tags_all)
            params.append(len(tags_all))
        if tags_any:
            placeholders = ",".join("?" * len(tags_any))
            clauses.append(
                f"EXISTS (SELECT 1 FROM books_tags_link btl "
                f"JOIN tags t ON t.id = btl.tag "
                f"WHERE btl.book = b.id AND t.name COLLATE NOCASE IN ({placeholders}))"
            )
            params.extend(tags_any)
        if tags_none:
            placeholders = ",".join("?" * len(tags_none))
            clauses.append(
                f"NOT EXISTS (SELECT 1 FROM books_tags_link btl "
                f"JOIN tags t ON t.id = btl.tag "
                f"WHERE btl.book = b.id AND t.name COLLATE NOCASE IN ({placeholders}))"
            )
            params.extend(tags_none)
        if pubdate_after:
            clauses.append("b.pubdate >= ?")
            params.append(pubdate_after)
        if pubdate_before:
            clauses.append("b.pubdate <= ?")
            params.append(pubdate_before)
        if has_series is not None:
            op = "EXISTS" if has_series else "NOT EXISTS"
            clauses.append(f"{op} (SELECT 1 FROM books_series_link bsl WHERE bsl.book = b.id)")
        if has_isbn is not None:
            op = "EXISTS" if has_isbn else "NOT EXISTS"
            clauses.append(f"{op} (SELECT 1 FROM identifiers i WHERE i.book = b.id AND i.type = 'isbn')")
        if has_cover is not None:
            clauses.append("b.has_cover = ?")
            params.append(1 if has_cover else 0)
        if identifier_type and identifier_value:
            clauses.append(
                "EXISTS (SELECT 1 FROM identifiers i "
                "WHERE i.book = b.id AND i.type = ? COLLATE NOCASE "
                "AND i.val = ? COLLATE NOCASE)"
            )
            params.extend([identifier_type, identifier_value])

        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        sql = f"""
            SELECT b.id, b.title, b.series_index,
                   (SELECT s.name FROM books_series_link bsl
                    JOIN series s ON s.id = bsl.series
                    WHERE bsl.book = b.id LIMIT 1) AS series_name
            FROM books b
            {where}
            ORDER BY b.id DESC
            LIMIT ?
        """  # noqa: S608 — clauses are constructed with placeholders, no user-injected SQL
        params.append(int(limit))

        with open_library(self.metadata_db) as conn:
            rows = conn.execute(sql, params).fetchall()
            return [
                BookSummary(
                    id=int(r["id"]),
                    title=r["title"] or "",
                    authors=_display_authors_for(conn, int(r["id"]))[0],
                    series=_strip_or_none(r["series_name"]),
                    series_index=_format_series_index(r["series_index"]),
                )
                for r in rows
            ]

    def _facet(
        self,
        table: str,
        link_table: str,
        link_col: str,
        query: str,
        limit: int,
    ) -> list[FacetCount]:
        q = query.strip()
        where = ""
        params: list[object] = []
        if q:
            where = "WHERE t.name LIKE ? COLLATE NOCASE"
            params.append(f"%{q}%")
        params.append(int(limit))
        sql = f"""
            SELECT t.name, COUNT(l.book) AS n
            FROM {table} t
            LEFT JOIN {link_table} l ON l.{link_col} = t.id
            {where}
            GROUP BY t.id
            ORDER BY n DESC, t.name
            LIMIT ?
        """  # noqa: S608 — fixed identifiers, no user input
        with open_library(self.metadata_db) as conn:
            return [
                FacetCount(name=r["name"] or "", book_count=int(r["n"])) for r in conn.execute(sql, params).fetchall()
            ]
