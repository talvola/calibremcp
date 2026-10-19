"""Calibre MCP server — FastMCP tool surface.

Two transport modes:

- **stdio** (default): single-user, credentials come from env / credentials
  file. Used by local Claude Desktop / Claude Code.
- **streamable-http**: multi-client via OAuth 2.1. Each MCP access token
  carries the Calibre-Web session cookie of the user who logged in. Used
  by Claude.ai (incl. iOS).

In stdio mode ``get_calibre_web()`` returns the module-level client logged
in at startup. In HTTP mode it builds a fresh client per request, seeded
with the cookie stored on the requesting OAuth access token.
"""

from __future__ import annotations

import html as _html
import ipaddress
import os
import socket
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from mcp.server.fastmcp import FastMCP

from calibre_mcp.cleanup.calibre_reader import BookRecord

from .api import CalibreWebClient
from .reader import AuthorSortCheck, BookSummary, CalibreReader

mcp = FastMCP(
    "calibre",
    host="0.0.0.0",
    port=8101,
    streamable_http_path="/mcp",
)

# Module-level handles. Populated in main() so tools can reach them without
# threading them through every signature.
_reader: CalibreReader | None = None
_calibre_web: CalibreWebClient | None = None  # stdio-mode singleton
_oauth_provider: Any = None  # set in HTTP mode


def get_reader() -> CalibreReader:
    if _reader is None:
        raise RuntimeError("Calibre metadata.db not configured. Set CALIBRE_METADATA_DB.")
    return _reader


def get_calibre_web() -> CalibreWebClient:
    """Return a Calibre-Web client appropriate to the current transport.

    - stdio: the module-level singleton logged in at startup.
    - HTTP/OAuth: a fresh client seeded with the current access token's
      stored session cookie. The caller is responsible for not leaking the
      client across requests (we make a new one per call).
    """
    if _calibre_web is not None:
        return _calibre_web

    if _oauth_provider is not None:
        from mcp.server.auth.middleware.auth_context import get_access_token

        from .auth import StoredAccessToken

        access_token = get_access_token()
        if access_token is not None and isinstance(access_token, StoredAccessToken):
            return CalibreWebClient(
                access_token.cw_url,
                session_cookie=access_token.cw_session_cookie,
            )

    raise RuntimeError(
        "Calibre-Web not configured. Either set CALIBRE_WEB_URL/USERNAME/PASSWORD (stdio) or sign in via OAuth (HTTP)."
    )


# ===== Formatting helpers =====


def _format_book_summary(b: BookSummary) -> str:
    authors = ", ".join(b.authors) if b.authors else "—"
    series = ""
    if b.series:
        series = f" [{b.series}"
        if b.series_index:
            series += f" #{b.series_index}"
        series += "]"
    return f"[{b.id}] {b.title} — {authors}{series}"


def _format_book_detail(b: BookRecord, author_sort_check: AuthorSortCheck | None = None) -> str:
    lines = [f"**{b.title}** (id={b.id})"]
    if b.authors:
        lines.append("Authors: " + " & ".join(b.authors))
    if b.author_sort:
        lines.append("Author sort: " + b.author_sort)
    if author_sort_check is not None and (warning := _author_sort_warning(author_sort_check)):
        lines.append(warning)
    if b.series:
        s = b.series
        if b.series_index:
            s += f" #{b.series_index}"
        lines.append("Series: " + s)
    if b.publisher:
        lines.append("Publisher: " + b.publisher)
    if b.pubdate:
        lines.append("Pubdate: " + b.pubdate)
    if b.language:
        lines.append("Language: " + b.language)
    if b.tags:
        # Stored case, as Calibre-Web shows it — never the casefolded tags_ci.
        lines.append("Tags: " + ", ".join(b.tags))
    if b.identifiers:
        # Types exactly as stored — Calibre-Web keeps import case, and the
        # agent needs to see e.g. ``ISBN``/``FICTIONDB`` to know to clean them.
        idents = ", ".join(f"{k}={v}" for k, v in sorted(b.identifiers.items(), key=lambda kv: kv[0].lower()))
        lines.append("Identifiers: " + idents)
    if b.description:
        desc = b.description.strip()
        if len(desc) > 600:
            desc = desc[:600].rstrip() + "…"
        lines.append("\n" + desc)
    return "\n".join(lines)


# ===== Read tools =====


@mcp.tool()
def calibre_search_books(query: str, limit: int = 20) -> str:
    """Search the Calibre library by title, author, or series substring (case-insensitive).

    Returns up to `limit` books as one-line summaries, newest first.
    """
    books = get_reader().search_books(query, limit=limit)
    if not books:
        return f"No books found matching {query!r}."
    return "\n".join(_format_book_summary(b) for b in books)


@mcp.tool()
def calibre_get_book(book_id: int) -> str:
    """Fetch full metadata for one book by its Calibre id.

    Authors are listed in Calibre-Web display order (driven by ``author_sort``,
    shown on its own line), joined with ``&`` like the ``authors`` write field.
    If an ``author_sort`` piece matches no author's stored sort name, a ⚠ line
    says so: Calibre-Web then shows that author last, so the listed order is
    a fallback, not necessarily the intended credit.
    Tags are returned in their stored case, so the output can be compared
    directly against what ``calibre_update_book`` wrote.
    """
    reader = get_reader()
    book = reader.get_book(book_id)
    if book is None:
        return f"No book with id {book_id}."
    return _format_book_detail(book, reader.check_author_sort(book_id))


@mcp.tool()
def calibre_list_series(query: str = "", limit: int = 50) -> str:
    """List series in the library, optionally filtered by substring match.

    Sorted by book count (most-populous first).
    """
    rows = get_reader().list_series(query, limit=limit)
    if not rows:
        return "No matching series."
    return "\n".join(f"{r.name} ({r.book_count})" for r in rows)


@mcp.tool()
def calibre_list_tags(query: str = "", limit: int = 100) -> str:
    """List tags in the library, optionally filtered by substring match.

    Sorted by book count.
    """
    rows = get_reader().list_tags(query, limit=limit)
    if not rows:
        return "No matching tags."
    return "\n".join(f"{r.name} ({r.book_count})" for r in rows)


@mcp.tool()
def calibre_list_authors(query: str = "", limit: int = 50) -> str:
    """List authors in the library, optionally filtered by substring match.

    Sorted by book count.
    """
    rows = get_reader().list_authors(query, limit=limit)
    if not rows:
        return "No matching authors."
    return "\n".join(f"{r.name} ({r.book_count})" for r in rows)


@mcp.tool()
def calibre_search_advanced(
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
) -> str:
    """Multi-field search of the Calibre library. All filters are optional and AND-combined.

    Tag semantics:
    - ``tags_all``: book must have EVERY listed tag (intersection).
    - ``tags_any``: book must have at least one of the listed tags (union).
    - ``tags_none``: exclude books carrying any of the listed tags.
    Tag matching is exact and case-insensitive — use ``calibre_list_tags`` to
    discover the canonical names.

    Boolean-presence filters (``has_series``, ``has_isbn``, ``has_cover``):
    pass True/False to require presence/absence; leave None to ignore.

    Date filters (``pubdate_after`` / ``pubdate_before``) take ISO ``YYYY-MM-DD``
    and are inclusive at both ends.

    ``identifier_type``/``identifier_value`` together find books matching a
    specific identifier (e.g. ``identifier_type="goodreads"``,
    ``identifier_value="7456085"``). Common types: ``isbn``, ``goodreads``,
    ``amazon``, ``google``, ``uri``.

    ``language`` is the Calibre lang_code (``eng``, ``ger``, ``fra``, ...).
    Unknown or absent codes return an empty result with no error — if you
    expected matches and got none, cross-check against ``calibre_list_*`` or
    the library's actual lang_code values.

    Returns up to ``limit`` books as one-line summaries, newest first.
    """
    books = get_reader().search_advanced(
        title_contains=title_contains,
        author_contains=author_contains,
        series_contains=series_contains,
        publisher_contains=publisher_contains,
        language=language,
        tags_all=tags_all,
        tags_any=tags_any,
        tags_none=tags_none,
        pubdate_after=pubdate_after,
        pubdate_before=pubdate_before,
        has_series=has_series,
        has_isbn=has_isbn,
        has_cover=has_cover,
        identifier_type=identifier_type,
        identifier_value=identifier_value,
        limit=limit,
    )
    if not books:
        return "No books match those filters."
    return "\n".join(_format_book_summary(b) for b in books)


@mcp.tool()
def calibre_list_publishers(query: str = "", limit: int = 50) -> str:
    """List publishers in the library, optionally filtered by substring match.

    Sorted by book count.
    """
    rows = get_reader().list_publishers(query, limit=limit)
    if not rows:
        return "No matching publishers."
    return "\n".join(f"{r.name} ({r.book_count})" for r in rows)


# ===== Write tools =====


@mcp.tool()
def calibre_update_book(
    book_id: int,
    title: str | None = None,
    authors: str | None = None,
    author_sort: str | None = None,
    tags: str | None = None,
    series: str | None = None,
    series_index: float | None = None,
    pubdate: str | None = None,
    publisher: str | None = None,
    languages: str | None = None,
    comments: str | None = None,
    rating: int | None = None,
    identifiers: dict[str, str] | None = None,
) -> str:
    """Update one or more metadata fields on a Calibre book.

    Any argument left as ``None`` is unchanged. The update goes through
    Calibre-Web's edit form so Calibre-Web remains the only writer.

    Field semantics:
    - ``authors``: **ampersand-separated** display names per Calibre's convention,
      e.g. ``"Joyce Lavene & Jim Lavene"``. Commas inside a single name are
      reserved for sort form (``"Lavene, Joyce"``). As a convenience, if the
      input has commas but no ``&`` AND each comma-separated piece looks like
      a multi-word display name (each has a space), the tool auto-converts to
      ``&``. Single-word pieces (sort form) are left alone.
    - ``author_sort``: controls **author display order**. Calibre-Web keys
      display off this field, not off the author link rows. On every form
      save — *including saves that only change comments or tags* — it
      re-derives ``author_sort`` from the authors in their currently displayed
      order, so normally passing ``authors`` in the desired order is enough.
      Supply ``author_sort`` explicitly (``"Last, First & Last, First"``,
      surnames in display order) only when you need to override that
      derivation. Each piece must be one of the book's authors' **stored**
      sort names (Calibre-Web matches exactly; a piece like ``"Cast, Pc"``
      for stored ``"Cast, PC"`` silently pushes that author to the end).
      The tool fixes case-only differences and refuses anything else, listing
      the valid names. Setting both makes two writes (form POST then AJAX);
      the AJAX call runs after the form POST so it wins. ``calibre_get_book``
      shows the stored value on its ``Author sort:`` line, and this tool's
      diff reports the real before/after value.
    - ``tags``: comma-separated string; full replacement of existing tags.
    - ``series``: bare series name; pair with ``series_index`` to set the position.
      Setting ``series`` alone on a book with no prior index stores ``1.0``
      (Calibre's default) — the response flags this; there is no "unset" index.
    - ``pubdate``: ``YYYY-MM-DD``.
    - ``languages``: language name like ``English`` (not the ``eng`` code).
    - ``identifiers``: ``{type: value}`` dict. Full replacement — pass ``{}`` to
      clear all. ``None`` (the default) preserves existing identifiers. Common
      types Calibre-Web renders as links: ``isbn``, ``goodreads``, ``amazon``.

    Returns a summary of the changes applied.
    """
    cw = get_calibre_web()
    before = cw.get_book_form_state(book_id)
    author_sort_before = _read_author_sort(book_id)
    check_before = _check_author_sort(book_id)

    updates: dict[str, Any] = {}
    if title is not None:
        updates["title"] = title
    if authors is not None:
        updates["authors"] = _normalize_authors(authors)
    if tags is not None:
        updates["tags"] = tags
    if series is not None:
        updates["series"] = series
    if series_index is not None:
        updates["series_index"] = f"{series_index:g}"
    if pubdate is not None:
        updates["pubdate"] = pubdate
    if publisher is not None:
        updates["publisher"] = publisher
    if languages is not None:
        updates["languages"] = languages
    if comments is not None:
        updates["comments"] = comments
    if rating is not None:
        updates["rating"] = str(rating)

    cw.edit_book(book_id, updates, identifiers=identifiers)

    notes: list[str] = []

    # author_sort goes through a separate AJAX endpoint, applied AFTER the
    # form post so it overrides Calibre-Web's auto-derivation from authors.
    # Validated against the post-form authors, since a mismatched piece would
    # silently reorder the displayed credit.
    author_sort_after: str | None = None
    if author_sort is not None:
        resolved, problem = _resolve_author_sort(author_sort, _check_author_sort(book_id))
        if problem is not None:
            notes.append(f"  author_sort NOT applied: {problem}")
        else:
            author_sort_after = cw.set_author_sort(book_id, resolved)

    # Read-back to confirm + show the diff.
    after = cw.get_book_form_state(book_id)
    diffs = _diff_states(before, after)

    # author_sort isn't in the form, so diff it from metadata.db. Fall back to
    # the AJAX endpoint's echoed value when the DB read is unavailable/stale.
    # Calibre-Web re-derives author_sort from ``authors`` (in the order given)
    # on every form save, so this line also surfaces pure reorders, which the
    # form's ``authors`` string (rendered in author_sort order) can hide.
    if author_sort_after is None:
        author_sort_after = _read_author_sort(book_id)
    author_sort_changed = author_sort_after is not None and author_sort_after != (author_sort_before or "")
    if author_sort_changed:
        diffs.append(f"  author_sort: {author_sort_before or ''!r} → {author_sort_after!r}")
        if authors is None and author_sort is None and check_before is not None and not check_before.consistent:
            unmatched = ", ".join(repr(p) for p in check_before.unmatched)
            notes.append(
                f"  note: author_sort was already out of sync ({unmatched} matched no author), so Calibre-Web "
                "was showing those authors last. Every form save re-derives author_sort from the displayed "
                "order, so this edit made that order permanent; the visible credit did not change. If it is "
                "wrong, call again with authors in the intended order."
            )

    check_after = _check_author_sort(book_id)
    if check_after is not None and (warning := _author_sort_warning(check_after)):
        notes.append(warning)

    if (
        series is not None
        and series_index is None
        and before.series != after.series
        and after.series_index.strip() in {"1", "1.0"}
    ):
        notes.append(
            "  note: series_index defaulted to 1 (Calibre's default) — pass series_index to set the real position."
        )
    if not diffs:
        msg = f"Submitted update for book {book_id}, but no fields changed."
        return msg + ("\n" + "\n".join(notes) if notes else "")
    return f"Updated book {book_id}:\n" + "\n".join(diffs + notes)


def _read_author_sort(book_id: int) -> str | None:
    """Best-effort ``books.author_sort`` read; None if metadata.db isn't configured."""
    try:
        return get_reader().get_author_sort(book_id)
    except Exception:  # noqa: BLE001 — diagnostics only, never fail a write over it
        return None


def _check_author_sort(book_id: int) -> AuthorSortCheck | None:
    """Best-effort :meth:`CalibreReader.check_author_sort`; None if metadata.db is unavailable."""
    try:
        return get_reader().check_author_sort(book_id)
    except Exception:  # noqa: BLE001 — diagnostics only, never fail a write over it
        return None


def _author_sort_warning(check: AuthorSortCheck) -> str | None:
    """A ⚠ line when ``author_sort`` has pieces Calibre-Web can't match to an author."""
    if check.consistent:
        return None
    unmatched = ", ".join(repr(p) for p in check.unmatched)
    stored = " & ".join(check.author_sorts)
    return (
        f"  ⚠ author_sort out of sync: {unmatched} matches no author's stored sort ({stored!r}). "
        "Calibre-Web shows unmatched authors last, so the Authors order may not be the intended credit. "
        "Fix with calibre_update_book(authors=...) in the intended order."
    )


def _resolve_author_sort(value: str, check: AuthorSortCheck | None) -> tuple[str, str | None]:
    """Map a caller's ``author_sort`` onto the book's stored author sort names.

    Calibre-Web orders the credit by exact matches between ``author_sort``
    pieces and ``authors.sort``, so the override must name each linked author
    exactly once, spelled as stored. Case-only differences are corrected;
    anything else returns ``(value, problem)`` and the caller must not write.
    Without a readable metadata.db there's nothing to check against, so the
    value passes through unchanged.
    """
    if check is None:
        return value, None
    stored = check.author_sorts
    by_fold: dict[str, str] = {}
    for s in stored:
        by_fold.setdefault(s.strip().casefold(), s)
    resolved: list[str] = []
    unknown: list[str] = []
    for piece in (p.strip() for p in value.split("&")):
        if not piece:
            continue
        if piece in stored:
            resolved.append(piece)
        elif (hit := by_fold.get(piece.casefold())) is not None:
            resolved.append(hit)
        else:
            unknown.append(piece)
    valid = " & ".join(repr(s) for s in stored)
    if unknown:
        names = ", ".join(repr(p) for p in unknown)
        return value, f"{names} matches no author on this book; use the stored sort names: {valid}."
    if sorted(resolved) != sorted(stored):
        return value, f"author_sort must list each of the book's authors exactly once: {valid}."
    return " & ".join(resolved), None


@mcp.tool()
def calibre_set_cover_from_url(book_id: int, cover_url: str) -> str:
    """Replace the cover by giving Calibre-Web a URL to fetch.

    We pre-resolve cross-domain redirects (e.g. covers.openlibrary.org ->
    archive.org's view_archive.php) because Calibre-Web's server-side
    downloader doesn't follow them — passing the original redirecting URL
    results in no cover being stored, silently. We do a HEAD/GET with
    follow_redirects, hand Calibre-Web the final resolved URL, then probe
    the stored cover and report the actual byte count so the caller can
    verify the change took.
    """
    import httpx as _httpx

    if (why := _reject_non_public_url(cover_url)) is not None:
        return f"Refusing to fetch {cover_url}: {why}"

    resolved = cover_url
    try:
        with _httpx.Client(follow_redirects=True, timeout=10.0) as c:
            r = c.get(cover_url)
            if (why := _reject_non_public_url(str(r.url))) is not None:
                return f"Refusing cover URL that redirected to {r.url}: {why}"
            if r.status_code >= 400:
                return f"Could not fetch {cover_url}: HTTP {r.status_code}"
            content_type = r.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                return f"URL did not return an image (content-type: {content_type!r})."
            resolved = str(r.url)
    except _httpx.HTTPError as e:
        return f"Could not pre-resolve cover URL: {e}"

    cw = get_calibre_web()
    cw.edit_book(book_id, {"cover_url": resolved})

    # Verify the cover changed by re-fetching it. Calibre-Web serves a small
    # placeholder when no real cover exists, so we report size for sanity-
    # checking. Any sub-2KB response is almost certainly the placeholder.
    try:
        blob, _ = cw.get_cover_bytes(book_id)
        size = len(blob)
    except Exception:  # noqa: BLE001
        size = -1

    note = f"Set cover for book {book_id} (resolved URL: {resolved})."
    if size >= 0:
        note += f" Calibre-Web now serves a {size:,}-byte cover."
        if size < 2048:
            note += " (Suspiciously small — likely a placeholder, not the real cover.)"
    return note


@mcp.tool()
def calibre_set_cover_from_file(book_id: int, file_path: str) -> str:
    """Replace the cover by uploading a local image file.

    File must be one of: .jpg, .jpeg, .png, .webp.
    """
    if _calibre_web is None:
        # In HTTP mode "local" means the server container's filesystem, which
        # no client should be able to read from. Use calibre_set_cover_from_url.
        return (
            "calibre_set_cover_from_file is only available on the local stdio server; use calibre_set_cover_from_url."
        )
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        return f"File not found: {file_path}"
    ext = path.suffix.lower().lstrip(".")
    mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
    if ext not in mime_map:
        return f"Unsupported cover extension {ext!r}; need jpg/jpeg/png/webp."
    cw = get_calibre_web()
    cw.edit_book(book_id, {}, cover_file=(path.name, path.read_bytes(), mime_map[ext]))
    return f"Uploaded cover for book {book_id} from {file_path}"


@mcp.tool()
def calibre_view_cover(book_id: int) -> str:
    """Return summary info about the current cover (size + mime type).

    Calibre-Web serves a generic placeholder image when a book has no cover
    set — so a non-zero byte count from ``/cover/<id>`` is NOT proof that a
    real cover exists. Any response under 2KB is flagged as likely-
    placeholder; everything else is assumed to be a real cover.
    """
    cw = get_calibre_web()
    try:
        blob, mime = cw.get_cover_bytes(book_id)
    except Exception as e:  # noqa: BLE001
        return f"Could not fetch cover for book {book_id}: {e}"
    size = len(blob)
    if size < 2048:
        return (
            f"Book {book_id}: Calibre-Web returned a {size:,}-byte {mime} — "
            f"this is almost certainly a placeholder; no real cover set."
        )
    return f"Book {book_id} cover: {size:,} bytes, {mime}"


def _reject_non_public_url(url: str) -> str | None:
    """Return a reason string if ``url`` must not be fetched server-side, else None.

    The MCP server runs with host networking on the NAS, so an arbitrary URL
    fetch could reach Portainer, Calibre-Web's admin, or anything else on the
    LAN. Cover sources are public CDNs (OpenLibrary, archive.org, Amazon), so
    we only allow http(s) to hosts that resolve exclusively to public
    addresses. Checked on the caller's URL and again on the redirect target.
    """
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"}:
        return f"scheme {parts.scheme!r} not allowed"
    host = parts.hostname
    if not host:
        return "no host in URL"
    try:
        infos = socket.getaddrinfo(
            host, parts.port or (443 if parts.scheme == "https" else 80), proto=socket.IPPROTO_TCP
        )
    except socket.gaierror as e:
        return f"cannot resolve host: {e}"
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            return f"host resolves to non-public address {ip}"
    return None


def _normalize_authors(authors: str) -> str:
    """Convert comma-separated display names to Calibre's ``&`` convention.

    Calibre's authors field uses ``&`` between authors; commas inside a name
    are reserved for sort form (e.g. ``"Lavene, Joyce"``). LLMs prompted with
    "comma-separated" frequently produce ``"Joyce Lavene, Jim Lavene"`` which
    Calibre then stores as a single author with a comma in the name.

    Heuristic: only split on commas when every comma-separated piece looks
    like a multi-word display name (each contains a space). That keeps
    legitimate sort-form names like ``"Lavene, Joyce"`` (single piece is one
    word) intact and only intervenes when the input is clearly a multi-name
    list with the wrong separator. Inputs already using ``&`` are passed
    through unchanged.
    """
    if "&" in authors or "," not in authors:
        return authors
    pieces = [p.strip() for p in authors.split(",")]
    pieces = [p for p in pieces if p]
    if pieces and all(" " in p for p in pieces):
        return " & ".join(pieces)
    return authors


def _diff_states(before, after) -> list[str]:
    """Render the human-readable diff between two BookFormState snapshots."""
    lines: list[str] = []
    for field_name in (
        "title",
        "authors",
        "tags",
        "series",
        "series_index",
        "pubdate",
        "publisher",
        "languages",
        "rating",
        "comments",
    ):
        b = getattr(before, field_name)
        a = getattr(after, field_name)
        if field_name == "comments":
            # Calibre-Web's form round-trip rewrites CRLF→LF even when comments
            # weren't edited; don't report that as a change.
            b = b.replace("\r\n", "\n").replace("\r", "\n")
            a = a.replace("\r\n", "\n").replace("\r", "\n")
        if b != a:
            b_short = b if len(b) <= 80 else b[:80] + "…"
            a_short = a if len(a) <= 80 else a[:80] + "…"
            lines.append(f"  {field_name}: {b_short!r} → {a_short!r}")
    if before.identifiers != after.identifiers:
        lines.append(f"  identifiers: {before.identifiers} → {after.identifiers}")
    return lines


@mcp.tool()
def calibre_stats() -> str:
    """Return total book count and connection status."""
    reader = get_reader()
    total = reader.book_count()
    parts = [f"Books in library: {total}", f"metadata.db: {reader.metadata_db}"]
    if _calibre_web is not None:
        parts.append(f"Calibre-Web: {_calibre_web.base_url} (stdio, authenticated)")
    elif _oauth_provider is not None:
        parts.append(f"Calibre-Web: {_oauth_provider.cw_url} (per-request OAuth)")
    else:
        parts.append("Calibre-Web: not configured (writes disabled)")
    return "\n".join(parts)


# ===== Entry point =====


def _load_credentials_file() -> tuple[str | None, str | None]:
    """Read username/password from ``calibre-credentials.env`` if present.

    Format: a single line ``username:password``. Located either via the
    ``CALIBRE_CREDENTIALS_FILE`` env var or by walking up from cwd looking
    for ``calibre-credentials.env`` (covers running from anywhere in repo).
    Env vars (``CALIBRE_WEB_USERNAME`` / ``CALIBRE_WEB_PASSWORD``) take
    precedence over the file.
    """
    override = os.environ.get("CALIBRE_CREDENTIALS_FILE")
    if override:
        path = Path(override)
    else:
        path = None
        for parent in [Path.cwd(), *Path.cwd().parents]:
            cand = parent / "calibre-credentials.env"
            if cand.is_file():
                path = cand
                break
    if path is None or not path.is_file():
        return None, None
    content = path.read_text(encoding="utf-8").strip()
    if not content or ":" not in content.splitlines()[0]:
        return None, None
    user, _, pw = content.splitlines()[0].partition(":")
    return user.strip() or None, pw.strip() or None


def main() -> None:
    global _reader, _calibre_web

    metadata_db = os.environ.get("CALIBRE_METADATA_DB")
    if not metadata_db:
        print("Error: CALIBRE_METADATA_DB env var required.", file=sys.stderr)
        sys.exit(1)
    _reader = CalibreReader(metadata_db)

    transport = "stdio"
    if "--transport" in sys.argv:
        idx = sys.argv.index("--transport")
        if idx + 1 < len(sys.argv):
            transport = sys.argv[idx + 1]

    if transport == "stdio":
        # Single-tenant: log in once with the configured credentials.
        file_user, file_pw = _load_credentials_file()
        cw_url = os.environ.get("CALIBRE_WEB_URL")
        cw_user = os.environ.get("CALIBRE_WEB_USERNAME") or file_user
        cw_pw = os.environ.get("CALIBRE_WEB_PASSWORD") or file_pw
        if cw_url and cw_user and cw_pw:
            client = CalibreWebClient(cw_url)
            try:
                client.login(cw_user, cw_pw)
                client.probe()
                _calibre_web = client
            except Exception as e:  # noqa: BLE001
                print(f"Calibre-Web login failed: {e}", file=sys.stderr)
                print(
                    "Continuing without Calibre-Web (writes will be disabled).",
                    file=sys.stderr,
                )
                client.close()
    elif transport == "streamable-http":
        _setup_http_auth()

    mcp.run(transport=transport)


def _setup_http_auth() -> None:
    """Wire OAuth 2.1 onto the FastMCP instance for streamable-http transport.

    Adapted directly from bar-assistant-mcp's ``_setup_http_auth``. The
    monkey-patches are workarounds for the same SDK quirks documented there:
    ``none`` token endpoint auth method for public clients (Claude.ai),
    trailing-slash on AnyHttpUrl issuer URL, and RFC 6750 WWW-Authenticate
    behavior when no token is presented.
    """
    global _oauth_provider

    from mcp.server.auth.provider import ProviderTokenVerifier
    from mcp.server.auth.settings import (
        AuthSettings,
        ClientRegistrationOptions,
        RevocationOptions,
    )
    from pydantic import AnyHttpUrl
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, RedirectResponse

    from .auth import LOGIN_PAGE_TEMPLATE, CalibreWebOAuthProvider

    cw_url = os.environ.get("CALIBRE_WEB_URL")
    if not cw_url:
        print("Error: CALIBRE_WEB_URL env var required for HTTP mode.", file=sys.stderr)
        sys.exit(1)

    issuer_url = os.environ.get("MCP_ISSUER_URL", "http://localhost:8101")

    _oauth_provider = CalibreWebOAuthProvider(
        cw_url=cw_url,
        issuer_url=issuer_url,
    )

    mcp.settings.auth = AuthSettings(
        issuer_url=AnyHttpUrl(issuer_url),
        client_registration_options=ClientRegistrationOptions(
            enabled=True,
            valid_scopes=["calibre.read", "calibre.write"],
            default_scopes=["calibre.read", "calibre.write"],
        ),
        revocation_options=RevocationOptions(enabled=True),
        required_scopes=[],
        resource_server_url=AnyHttpUrl(f"{issuer_url.rstrip('/')}/mcp"),
    )
    mcp._auth_server_provider = _oauth_provider
    mcp._token_verifier = ProviderTokenVerifier(_oauth_provider)

    # --- SDK monkey-patches (same as bar-assistant-mcp) ---

    import mcp.server.auth.routes as _auth_routes

    # Namespace dynamic client registration under /oauth/ so it doesn't collide
    # with any future Calibre-Web /register page if we ever share a hostname.
    _auth_routes.REGISTRATION_PATH = "/oauth/register"

    _orig_build_metadata = _auth_routes.build_metadata

    def _patched_build_metadata(*args, **kwargs):
        metadata = _orig_build_metadata(*args, **kwargs)
        methods = list(metadata.token_endpoint_auth_methods_supported or [])
        if "none" not in methods:
            methods.append("none")
        metadata.token_endpoint_auth_methods_supported = methods
        return metadata

    _auth_routes.build_metadata = _patched_build_metadata

    # AnyHttpUrl renders as "https://example.com/" — strip trailing slash on
    # issuer in the metadata JSON since some MCP clients are strict about it.
    from mcp.server.auth.json_response import PydanticJSONResponse

    _orig_render = PydanticJSONResponse.render

    def _patched_render(self, content):
        data = _orig_render(self, content)
        text = data.decode("utf-8")
        issuer_base = issuer_url.rstrip("/")
        text = text.replace(f'"{issuer_base}/"', f'"{issuer_base}"')
        return text.encode("utf-8")

    PydanticJSONResponse.render = _patched_render

    # RFC 6750 §3: when no token is presented, send plain ``Bearer`` challenge
    # without ``error="invalid_token"`` (which signals a *failed* token, not
    # a missing one). MCP SDK doesn't get this right by default.
    from mcp.server.auth.middleware.bearer_auth import RequireAuthMiddleware

    _orig_call = RequireAuthMiddleware.__call__

    async def _patched_call(self, scope, receive, send):
        from starlette.requests import HTTPConnection

        conn = HTTPConnection(scope)
        auth_header = conn.headers.get("authorization", "")
        if not auth_header.lower().startswith("bearer "):
            await self._send_auth_error(send, status_code=401, error="", description="Authentication required")
            return
        await _orig_call(self, scope, receive, send)

    RequireAuthMiddleware.__call__ = _patched_call

    _orig_send_error = RequireAuthMiddleware._send_auth_error

    async def _patched_send_error(self, send, status_code, error, description):
        import json as _json

        if not error:
            parts = []
            if description:
                parts.append(f'error_description="{description}"')
            if self.resource_metadata_url:
                parts.append(f'resource_metadata="{self.resource_metadata_url}"')
            www_authenticate = "Bearer" + (f" {', '.join(parts)}" if parts else "")
            body = {"error": "unauthorized", "error_description": description}
            body_bytes = _json.dumps(body).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": status_code,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body_bytes)).encode()),
                        (b"www-authenticate", www_authenticate.encode()),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body_bytes})
        else:
            await _orig_send_error(self, send, status_code, error, description)

    RequireAuthMiddleware._send_auth_error = _patched_send_error

    # --- Login page (GET + POST) ---

    @mcp.custom_route("/auth/login", methods=["GET"])
    async def login_page(request: Request):
        from string import Template

        code_id = request.query_params.get("code_id", "")
        html = Template(LOGIN_PAGE_TEMPLATE).safe_substitute(code_id=_html.escape(code_id, quote=True), error="")
        return HTMLResponse(html)

    @mcp.custom_route("/auth/login", methods=["POST"])
    async def login_submit(request: Request):
        from string import Template

        form = await request.form()
        code_id = str(form.get("code_id", ""))
        username = str(form.get("username", ""))
        password = str(form.get("password", ""))

        try:
            redirect_url = await _oauth_provider.complete_authorization(code_id, username, password)
            return RedirectResponse(url=redirect_url, status_code=302)
        except ValueError as e:
            # Escape both: code_id is attacker-controllable via the URL, and the
            # error text can echo Calibre-Web's flash message.
            error_html = f'<div class="error">{_html.escape(str(e))}</div>'
            html = Template(LOGIN_PAGE_TEMPLATE).safe_substitute(
                code_id=_html.escape(code_id, quote=True), error=error_html
            )
            return HTMLResponse(html, status_code=400)


if __name__ == "__main__":
    main()
