"""Goodreads book-page scraper for cover URL, Amazon ASIN, and series info.

Called by ``book_audit`` once we have a Goodreads numeric ID (from
``goodreads_lookup``). Extracts:

  * Cover URL — ``<meta property="og:image">`` (Amazon-CDN-hosted)
  * Amazon ASIN — ``amazon.com/dp/`` & ``/gp/product/`` links, filtered to
    B-prefixed (Kindle/Audible). Non-B 10-char codes are ISBN-10 mirrors,
    redundant with the ISBN identifier.
  * Series — Goodreads stores ``bookSeries`` as Apollo cache refs
    (``Series:kca://series/...``) which we resolve to the flat
    ``"Series:...":{title,webUrl}`` entries elsewhere in the page.
  * Title + ISBN + author — from the schema.org JSON-LD ``Book`` block,
    used as a sanity check against the ID we looked up.
  * Parenthetical heuristic — when ``bookSeries`` is empty, a title like
    ``"X (On Track)"`` often encodes the publisher's series in parens.
    Surfaced separately so callers can treat it as low-confidence.

Polite: one request per book, realistic Firefox UA. Goodreads has so far
served 200s for this endpoint without bot challenges.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

SOURCE = "goodreads_page"

_DEFAULT_UA = "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"


@dataclass(frozen=True, slots=True)
class SeriesRef:
    title: str
    user_position: str | None
    web_url: str | None = None


@dataclass(frozen=True, slots=True)
class GoodreadsPageData:
    goodreads_id: str
    title: str | None
    primary_author: str | None
    isbn: str | None
    cover_url: str | None
    asin: str | None
    asins_all: tuple[str, ...]
    series: tuple[SeriesRef, ...]
    series_heuristic: SeriesRef | None  # from "(Name, #N)" parenthetical
    genres: tuple[str, ...]


def fetch(client: httpx.Client, goodreads_id: str) -> GoodreadsPageData | None:
    """Fetch the page and parse it. Returns None on HTTP error.

    Caller passes a pre-configured client (UA + follow_redirects=True);
    if you want a default one, use ``new_client()``.
    """
    url = f"https://www.goodreads.com/book/show/{goodreads_id}"
    try:
        r = client.get(url)
    except httpx.HTTPError as exc:
        log.warning("goodreads page %s: %s", goodreads_id, exc)
        return None
    if r.status_code != 200:
        log.warning("goodreads page %s: HTTP %d", goodreads_id, r.status_code)
        return None
    return parse(goodreads_id, r.text)


def new_client() -> httpx.Client:
    """Default httpx client configured for Goodreads page scraping."""
    return httpx.Client(
        headers={"User-Agent": _DEFAULT_UA, "Accept": "text/html,application/xhtml+xml"},
        follow_redirects=True,
        timeout=30.0,
    )


# ---------------------------------------------------------------------------
# Pure parser
# ---------------------------------------------------------------------------


def parse(goodreads_id: str, html: str) -> GoodreadsPageData:
    """Parse raw HTML — pure, no I/O — exposed for testing against fixtures."""
    title, isbn, author = _from_json_ld(html)
    series = _resolve_series(html)
    # Even if structured series exist, Goodreads titles often append the
    # canonical series + index as " (Name, #N)". When that parses cleanly,
    # surface it — useful for picking a canonical name when the page lists
    # multiple sub-series.
    heuristic = _parenthetical_heuristic(title)
    asins_all = _all_asins(html)
    return GoodreadsPageData(
        goodreads_id=goodreads_id,
        title=title,
        primary_author=author,
        isbn=isbn,
        cover_url=_og_image(html),
        asin=asins_all[0] if asins_all else None,
        asins_all=asins_all,
        series=series,
        series_heuristic=heuristic,
        genres=_genres(html),
    )


_OG_IMAGE_RE = re.compile(r'<meta\s+property="og:image"\s+content="([^"]+)"')


def _og_image(html: str) -> str | None:
    m = _OG_IMAGE_RE.search(html)
    return m.group(1) if m else None


_LD_JSON_RE = re.compile(
    r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.DOTALL,
)


def _from_json_ld(html: str) -> tuple[str | None, str | None, str | None]:
    """Pull title / ISBN / primary-author from the first Book JSON-LD block."""
    for m in _LD_JSON_RE.finditer(html):
        try:
            data = json.loads(m.group(1))
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(data, list):
            data = next(
                (d for d in data if isinstance(d, dict) and d.get("@type") == "Book"),
                None,
            )
        if not isinstance(data, dict) or data.get("@type") != "Book":
            continue
        a = data.get("author")
        if isinstance(a, list) and a:
            a = a[0]
        author = a.get("name") if isinstance(a, dict) else None
        return data.get("name"), data.get("isbn"), author
    return None, None, None


_ASIN_LINK_RE = re.compile(r"amazon\.com/(?:dp|gp/product|gp/aw/d)/([A-Z0-9]{10})")


def _all_asins(html: str) -> tuple[str, ...]:
    """All B-prefixed ASINs in document order. Non-B 10-char codes are ISBN-10."""
    seen: list[str] = []
    for m in _ASIN_LINK_RE.finditer(html):
        asin = m.group(1)
        if asin.startswith("B") and asin not in seen:
            seen.append(asin)
    return tuple(seen)


# bookSeries is an array of refs+positions; the Series records themselves
# live as flat cache entries elsewhere in the page. We use Python rather
# than regex to walk the bracket-balanced array so nested objects don't
# confuse a naive `.*?]` pattern.
_BOOKSERIES_START = '"bookSeries":['


def _resolve_series(html: str) -> tuple[SeriesRef, ...]:
    """Parse bookSeries refs, then resolve each to a Series cache entry.

    Goodreads stores entries as ``[{"__typename":"BookSeries",
    "userPosition":"N","series":{"__ref":"Series:kca://..."}}]`` —
    json-load the balanced array slice and walk it.
    """
    idx = html.find(_BOOKSERIES_START)
    if idx < 0:
        return ()
    body = _balanced_slice(html, idx + len(_BOOKSERIES_START) - 1)  # include '['
    if not body or body == "[]":
        return ()
    try:
        entries = json.loads(body)
    except (ValueError, json.JSONDecodeError) as exc:
        log.warning("bookSeries parse failed: %s", exc)
        return ()
    if not isinstance(entries, list):
        return ()
    out: list[SeriesRef] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        ref = (e.get("series") or {}).get("__ref")
        if not ref:
            continue
        resolved = _lookup_series_ref(html, ref)
        if resolved is None:
            continue
        title, url = resolved
        out.append(SeriesRef(title=title, user_position=e.get("userPosition"), web_url=url))
    return tuple(out)


def _balanced_slice(html: str, start: int) -> str | None:
    """Return ``html[start:end]`` covering one balanced bracket group ([] or {}).

    Naive — ignores quoted brackets — but adequate here because Goodreads'
    bookSeries payload doesn't have square brackets inside string values."""
    if start >= len(html):
        return None
    open_ch = html[start]
    close_ch = {"[": "]", "{": "}"}.get(open_ch)
    if close_ch is None:
        return None
    depth = 0
    for i in range(start, len(html)):
        c = html[i]
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return html[start : i + 1]
    return None


def _lookup_series_ref(html: str, ref: str) -> tuple[str, str | None] | None:
    """Find ``"<ref>":{... "title":"X", "webUrl":"Y"}`` in the Apollo cache."""
    pattern = re.escape(f'"{ref}"') + r':\{[^}]*"title":"([^"]+)"(?:[^}]*"webUrl":"([^"]+)")?'
    m = re.search(pattern, html)
    if not m:
        return None
    return m.group(1), m.group(2)


_PARENTHETICAL_RE = re.compile(r"\(([^()]+)\)\s*$")
_NAME_WITH_INDEX_RE = re.compile(r"^(?P<name>.+?)[,\s]+#\s*(?P<idx>[\d.]+)\s*$")
_FORMAT_KEYWORDS = ("edition", "paperback", "hardcover", "annotated", "illustrated", "reprint", "ebook", "audiobook")


def _parenthetical_heuristic(title: str | None) -> SeriesRef | None:
    """Parse trailing ``(Name, #N)`` or ``(Name)`` from a Goodreads-style title.

    Returns a SeriesRef with ``user_position`` populated when ``#N`` is present.
    Skips parentheticals that look like edition/format labels.
    """
    if not title:
        return None
    m = _PARENTHETICAL_RE.search(title.strip())
    if not m:
        return None
    body = m.group(1).strip()
    if not body or any(k in body.lower() for k in _FORMAT_KEYWORDS):
        return None
    n = _NAME_WITH_INDEX_RE.match(body)
    if n:
        return SeriesRef(title=n.group("name").strip(), user_position=n.group("idx"))
    return SeriesRef(title=body, user_position=None)


_GENRE_RE = re.compile(r'"BookGenre","genre":\{"__typename":"Genre","name":"([^"]+)"')


def _genres(html: str) -> tuple[str, ...]:
    seen: list[str] = []
    for m in _GENRE_RE.finditer(html):
        g = m.group(1)
        if g and g not in seen:
            seen.append(g)
    return tuple(seen)
