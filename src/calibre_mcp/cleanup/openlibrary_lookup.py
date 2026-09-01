"""OpenLibrary search for ISBN, cover URLs, and edition metadata.

Polite primary source for the single-book audit. Uses the ``/search.json``
endpoint — no API key required. Caller decides how to score candidates;
this module just shapes the JSON into a typed hit.

Not bulk-friendly; intended for one-book-at-a-time gold-standard audits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

SOURCE = "openlibrary"

_SEARCH_URL = "https://openlibrary.org/search.json"
_COVER_URL_FMT = "https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"

_DEFAULT_UA = "calibremcp/0.1 (https://github.com/talvola/calibremcp; library cleanup)"


@dataclass(frozen=True, slots=True)
class OpenLibraryHit:
    """One candidate from an OpenLibrary search result."""

    work_key: str  # e.g. "/works/OL12345W"
    title: str
    authors: tuple[str, ...]
    first_publish_year: int | None
    isbn_list: tuple[str, ...]
    cover_id: int | None
    publisher_list: tuple[str, ...]

    @property
    def cover_url(self) -> str | None:
        return _COVER_URL_FMT.format(cover_id=self.cover_id) if self.cover_id else None

    @property
    def best_isbn(self) -> str | None:
        """Prefer ISBN-13 (longer); fall back to first ISBN-10."""
        isbn13 = next((i for i in self.isbn_list if len(i) == 13), None)
        return isbn13 or (self.isbn_list[0] if self.isbn_list else None)


def search(
    *,
    title: str,
    author: str,
    client: httpx.Client | None = None,
    limit: int = 5,
) -> list[OpenLibraryHit]:
    """Search OpenLibrary for works matching ``title`` + ``author``.

    Returns up to ``limit`` hits in OpenLibrary's relevance order. Network
    or parse failures return an empty list and log a warning.
    """
    params = {
        "title": title,
        "author": author,
        "limit": str(limit),
        "fields": "key,title,author_name,first_publish_year,isbn,cover_i,publisher",
    }
    own_client = client is None
    if own_client:
        client = httpx.Client(
            headers={"User-Agent": _DEFAULT_UA, "Accept": "application/json"},
            timeout=20.0,
        )
    try:
        try:
            r = client.get(_SEARCH_URL, params=params)
            r.raise_for_status()
            payload = r.json()
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("openlibrary search failed for %r / %r: %s", title, author, exc)
            return []
    finally:
        if own_client:
            client.close()

    docs = payload.get("docs", []) or []
    out: list[OpenLibraryHit] = []
    for d in docs:
        isbns = tuple(_clean_isbn(i) for i in (d.get("isbn") or []) if i)
        out.append(
            OpenLibraryHit(
                work_key=d.get("key") or "",
                title=(d.get("title") or "").strip(),
                authors=tuple(d.get("author_name") or []),
                first_publish_year=d.get("first_publish_year"),
                isbn_list=tuple(i for i in isbns if i),
                cover_id=d.get("cover_i"),
                publisher_list=tuple(d.get("publisher") or []),
            )
        )
    return out


def _clean_isbn(raw: str) -> str:
    """Strip hyphens / spaces from an ISBN string; keep digits and trailing X."""
    return "".join(c for c in raw if c.isdigit() or c.upper() == "X")
