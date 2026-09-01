"""Google Books search for cross-checking ISBN and surfacing Amazon ASIN.

Used as the secondary source after OpenLibrary. The ``volumes`` endpoint
returns ``industryIdentifiers`` which carries ISBN-10/ISBN-13 explicitly
and occasionally an ASIN under ``type="OTHER"``.

No API key needed for low-volume queries; rate-limited per IP.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

SOURCE = "google_books"

_VOLUMES_URL = "https://www.googleapis.com/books/v1/volumes"

_DEFAULT_UA = "calibremcp/0.1 (https://github.com/talvola/calibremcp; library cleanup)"


@dataclass(frozen=True, slots=True)
class GoogleBooksHit:
    """One candidate volume from a Google Books search."""

    volume_id: str
    title: str
    subtitle: str | None
    authors: tuple[str, ...]
    publisher: str | None
    published_date: str | None
    isbn_10: str | None
    isbn_13: str | None
    other_identifiers: dict[str, str]
    thumbnail_url: str | None
    description: str | None


def search(
    *,
    title: str,
    author: str,
    client: httpx.Client | None = None,
    limit: int = 5,
) -> list[GoogleBooksHit]:
    """Search Google Books for volumes matching ``title`` + ``author``."""
    q = f'intitle:"{title}" inauthor:"{author}"'
    params = {"q": q, "maxResults": str(limit)}
    own_client = client is None
    if own_client:
        client = httpx.Client(
            headers={"User-Agent": _DEFAULT_UA, "Accept": "application/json"},
            timeout=20.0,
        )
    try:
        try:
            r = client.get(_VOLUMES_URL, params=params)
            r.raise_for_status()
            payload = r.json()
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("google_books search failed for %r / %r: %s", title, author, exc)
            return []
    finally:
        if own_client:
            client.close()

    items = payload.get("items") or []
    out: list[GoogleBooksHit] = []
    for item in items:
        vi = item.get("volumeInfo") or {}
        idents = vi.get("industryIdentifiers") or []
        isbn_10 = next(
            (i["identifier"] for i in idents if i.get("type") == "ISBN_10" and i.get("identifier")),
            None,
        )
        isbn_13 = next(
            (i["identifier"] for i in idents if i.get("type") == "ISBN_13" and i.get("identifier")),
            None,
        )
        other = {
            i.get("type", ""): i.get("identifier", "")
            for i in idents
            if i.get("type") not in ("ISBN_10", "ISBN_13")
            and i.get("type")
            and i.get("identifier")
        }
        images = vi.get("imageLinks") or {}
        out.append(
            GoogleBooksHit(
                volume_id=item.get("id") or "",
                title=(vi.get("title") or "").strip(),
                subtitle=vi.get("subtitle") or None,
                authors=tuple(vi.get("authors") or []),
                publisher=vi.get("publisher") or None,
                published_date=vi.get("publishedDate") or None,
                isbn_10=isbn_10,
                isbn_13=isbn_13,
                other_identifiers=other,
                thumbnail_url=images.get("thumbnail") or images.get("smallThumbnail") or None,
                description=vi.get("description") or None,
            )
        )
    return out
