"""Library-consult-first series resolution.

Series name is the join key Calibre-Web uses to list books in that series:
two slightly-different spellings ("On Track" vs "On Track...") split into
two unrelated views. Before proposing a series name we always check whether
the library already has one we should reuse verbatim.

Resolution order:
  1. Series among books by the same author — prefer verbatim.
  2. Series among books by the same publisher — prefer verbatim.
  3. Fuzzy match (normalized) against every series in the library.
  4. None found — caller may propose a new canonical name.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SeriesCandidate:
    name: str
    book_count: int
    reason: str  # 'same_author' | 'same_publisher' | 'fuzzy_match'


def normalize(name: str) -> str:
    """Casefold + strip non-alphanumerics so "On Track..." == "on track"."""
    return re.sub(r"[^a-z0-9]+", "", name.casefold())


def find_library_candidates(
    conn: sqlite3.Connection,
    *,
    author_id: int | None,
    publisher_id: int | None,
    proposed_name: str | None = None,
) -> list[SeriesCandidate]:
    """Find existing series names worth consulting before proposing.

    When ``proposed_name`` is given, also surfaces fuzzy matches against
    every series in the library — that's the variant-detection guard.
    """
    out: list[SeriesCandidate] = []
    seen: set[str] = set()

    if author_id is not None:
        for row in conn.execute(
            """
            SELECT s.name AS name, COUNT(DISTINCT bsl.book) AS n
            FROM series s
            JOIN books_series_link bsl ON bsl.series = s.id
            JOIN books_authors_link bal ON bal.book = bsl.book
            WHERE bal.author = ?
            GROUP BY s.name
            ORDER BY n DESC
            """,
            (author_id,),
        ):
            if row["name"] and row["name"] not in seen:
                out.append(
                    SeriesCandidate(name=row["name"], book_count=int(row["n"]), reason="same_author")
                )
                seen.add(row["name"])

    if publisher_id is not None:
        for row in conn.execute(
            """
            SELECT s.name AS name, COUNT(DISTINCT bsl.book) AS n
            FROM series s
            JOIN books_series_link bsl ON bsl.series = s.id
            JOIN books_publishers_link bpl ON bpl.book = bsl.book
            WHERE bpl.publisher = ?
            GROUP BY s.name
            ORDER BY n DESC
            """,
            (publisher_id,),
        ):
            if row["name"] and row["name"] not in seen:
                out.append(
                    SeriesCandidate(name=row["name"], book_count=int(row["n"]), reason="same_publisher")
                )
                seen.add(row["name"])

    if proposed_name:
        target = normalize(proposed_name)
        if target:
            for row in conn.execute(
                """
                SELECT s.name AS name, COUNT(DISTINCT bsl.book) AS n
                FROM series s LEFT JOIN books_series_link bsl ON bsl.series = s.id
                GROUP BY s.name
                """
            ):
                name = row["name"]
                if not name or name in seen:
                    continue
                if normalize(name) == target:
                    out.append(
                        SeriesCandidate(name=name, book_count=int(row["n"]), reason="fuzzy_match")
                    )
                    seen.add(name)

    return out


def author_id_for_book(conn: sqlite3.Connection, book_id: int) -> int | None:
    """Primary author id (first by link insertion order)."""
    row = conn.execute(
        "SELECT author FROM books_authors_link WHERE book = ? ORDER BY id LIMIT 1",
        (book_id,),
    ).fetchone()
    return int(row["author"]) if row else None


def publisher_id_for_book(conn: sqlite3.Connection, book_id: int) -> int | None:
    row = conn.execute(
        "SELECT publisher FROM books_publishers_link WHERE book = ? LIMIT 1",
        (book_id,),
    ).fetchone()
    return int(row["publisher"]) if row else None
