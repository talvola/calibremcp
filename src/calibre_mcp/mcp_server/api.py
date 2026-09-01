"""HTTP client for Calibre-Web's web UI.

Calibre-Web 0.6.26 has no proper REST API in vanilla form — it exposes OPDS
(read-only, Basic auth) and form-based admin endpoints used by its own UI.
This client wraps both:

- session-based login (``/login`` with csrf_token + cookie)
- form POSTs to ``/admin/book/<id>`` for edits (phase 2)
- cover uploads via the same admin form (phase 2)

Phase 1 only needs login + a simple probe so we can verify connectivity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import httpx
from bs4 import BeautifulSoup


class CalibreWebError(Exception):
    """Raised when Calibre-Web returns an unexpected response."""


_CSRF_RE = re.compile(
    r'<input[^>]+name="csrf_token"[^>]+value="([^"]+)"',
    re.IGNORECASE,
)

# Fields the /admin/book/<id> form accepts (other than identifiers + csrf_token).
# Maps caller-friendly names to the form field name Calibre-Web expects.
_EDIT_FIELDS: dict[str, str] = {
    "title": "title",
    "authors": "authors",
    "tags": "tags",
    "series": "series",
    "series_index": "series_index",
    "pubdate": "pubdate",
    "publisher": "publisher",
    "languages": "languages",
    "rating": "rating",
    "comments": "comments",
    "cover_url": "cover_url",
}
# NOTE on author_sort: Calibre-Web's /admin/book/<id> form does NOT expose an
# author_sort input — it auto-derives author_sort from the authors string on
# every save, then keys display order off it. To override, POST to the AJAX
# endpoint /ajax/editbooks/author_sort with JSON `{"pk":[id], "value":"..."}`.
# See CalibreWebClient.set_author_sort below.


@dataclass(frozen=True)
class BookFormState:
    """Snapshot of an /admin/book/<id> edit page.

    Holds the CSRF token (must be reused in the POST to satisfy Flask-WTF)
    plus the current form values, so the caller can do a clean read-modify-
    write without clobbering fields it doesn't intend to touch.
    """

    csrf_token: str
    book_id: int
    title: str = ""
    authors: str = ""
    tags: str = ""
    series: str = ""
    series_index: str = ""
    pubdate: str = ""
    publisher: str = ""
    languages: str = ""
    rating: str = ""
    comments: str = ""
    cover_url: str = ""
    identifiers: dict[str, str] = field(default_factory=dict)


class CalibreWebClient:
    """Authenticated session against a Calibre-Web instance.

    Usage:

        client = CalibreWebClient("http://192.168.1.64:38083")
        client.login("erik", "password")
        client.probe()  # raises if not logged in
    """

    def __init__(
        self,
        base_url: str,
        *,
        session_cookie: str | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            follow_redirects=False,
            headers={"User-Agent": "calibre-mcp/0.1"},
        )
        if session_cookie:
            # Pre-seed the session cookie so we don't need to log in again.
            # Calibre-Web's cookie name is "session"; path "/".
            self._client.cookies.set("session", session_cookie, path="/")

    def get_session_cookie(self) -> str | None:
        """Return the current ``session=…`` cookie value (None if not logged in)."""
        return self._client.cookies.get("session")

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> CalibreWebClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ===== Auth =====

    def login(self, username: str, password: str) -> None:
        """Establish a session.

        GET /login to grab the per-session csrf_token, then POST credentials.
        Calibre-Web answers a successful login with a 302 to the ``next`` URL;
        a failed login re-renders the form with HTTP 200.
        """
        # Step 1: fetch login page to get cookie + csrf_token.
        get_resp = self._client.get("/login")
        if get_resp.status_code != 200:
            raise CalibreWebError(
                f"unexpected status {get_resp.status_code} fetching /login"
            )
        match = _CSRF_RE.search(get_resp.text)
        if not match:
            raise CalibreWebError("could not find csrf_token in /login page")
        csrf_token = match.group(1)

        # Step 2: POST credentials. Calibre-Web's form is application/x-www-form-urlencoded.
        post_resp = self._client.post(
            "/login",
            data={
                "csrf_token": csrf_token,
                "username": username,
                "password": password,
                "next": "/",
                "remember_me": "on",
                "submit": "",
            },
        )
        # Successful login: 302 redirect. Failed login: 200 with the form re-rendered.
        if post_resp.status_code == 302:
            return
        if post_resp.status_code == 200:
            # Try to surface the error message Calibre-Web embeds.
            err = _extract_flash_error(post_resp.text) or "invalid credentials"
            raise CalibreWebError(f"login failed: {err}")
        raise CalibreWebError(
            f"unexpected status {post_resp.status_code} on POST /login"
        )

    def probe(self) -> dict[str, Any]:
        """Verify the session is authenticated. Returns minimal metadata.

        Calibre-Web's ``/me`` page (user profile) requires login and returns
        HTML; we just check the status code and that we don't get bounced
        back to ``/login``.
        """
        resp = self._client.get("/me")
        if resp.status_code == 302 and "login" in resp.headers.get("location", ""):
            raise CalibreWebError("not authenticated")
        if resp.status_code != 200:
            raise CalibreWebError(f"unexpected status {resp.status_code} on /me")
        return {"status": "ok", "url": self.base_url}

    # ===== Reads against Calibre-Web =====

    def get_book_form_state(self, book_id: int) -> BookFormState:
        """Fetch /admin/book/<id> and parse the current form values.

        Used to do read-modify-write without clobbering fields the caller
        didn't ask to change. Also returns the per-session csrf_token that
        the next POST has to echo back.
        """
        resp = self._client.get(f"/admin/book/{int(book_id)}")
        if resp.status_code == 302 and "login" in resp.headers.get("location", ""):
            raise CalibreWebError("not authenticated")
        if resp.status_code == 404:
            raise CalibreWebError(f"book {book_id} not found")
        if resp.status_code != 200:
            raise CalibreWebError(
                f"unexpected status {resp.status_code} on GET /admin/book/{book_id}"
            )
        return _parse_book_form(resp.text, book_id)

    def get_cover_bytes(self, book_id: int) -> tuple[bytes, str]:
        """Fetch the current cover image. Returns (raw_bytes, content_type).

        Calibre-Web serves covers at ``/cover/<book_id>``. Returns the raw
        bytes plus content_type so callers can render or stash it.
        """
        resp = self._client.get(f"/cover/{int(book_id)}")
        if resp.status_code == 302 and "login" in resp.headers.get("location", ""):
            raise CalibreWebError("not authenticated")
        if resp.status_code == 404:
            raise CalibreWebError(f"no cover for book {book_id}")
        if resp.status_code != 200:
            raise CalibreWebError(
                f"unexpected status {resp.status_code} on /cover/{book_id}"
            )
        return resp.content, resp.headers.get("content-type", "application/octet-stream")

    # ===== Writes =====

    def edit_book(
        self,
        book_id: int,
        updates: dict[str, Any],
        *,
        identifiers: dict[str, str] | None = None,
        cover_file: tuple[str, bytes, str] | None = None,
    ) -> None:
        """Apply a partial update to one book via the /admin/book/<id> form.

        ``updates`` maps fields in :data:`_EDIT_FIELDS` to new string values.
        Fields not in ``updates`` keep their current value (read-modify-write).

        ``identifiers`` is full-replacement semantics: if provided, the dict
        IS the new identifier set (pass ``{}`` to clear all). If ``None``,
        existing identifiers are preserved.

        ``cover_file`` is an optional ``(filename, bytes, mimetype)`` tuple
        for uploading a cover. Posted as multipart in the same form.
        """
        # Read-modify-write to preserve any fields we're not touching.
        state = self.get_book_form_state(book_id)

        # Validate field names early so callers get a useful error.
        for k in updates:
            if k not in _EDIT_FIELDS:
                raise CalibreWebError(
                    f"unknown field {k!r}; supported: {sorted(_EDIT_FIELDS)}"
                )

        form_data: dict[str, str] = {
            "csrf_token": state.csrf_token,
            "book_id": str(book_id),
            "detail_view": "on",  # matches the page's checked default
        }
        for caller_name, form_name in _EDIT_FIELDS.items():
            if caller_name in updates:
                value = updates[caller_name]
                form_data[form_name] = "" if value is None else str(value)
            else:
                form_data[form_name] = getattr(state, caller_name)

        # Identifiers: emit identifier-type-N / identifier-val-N pairs.
        idents = identifiers if identifiers is not None else state.identifiers
        for i, (itype, ival) in enumerate(idents.items()):
            if not itype or not ival:
                continue
            form_data[f"identifier-type-{i}"] = str(itype)
            form_data[f"identifier-val-{i}"] = str(ival)

        files = None
        if cover_file is not None:
            filename, blob, mime = cover_file
            files = {"btn-upload-cover": (filename, blob, mime)}

        resp = self._client.post(
            f"/admin/book/{int(book_id)}",
            data=form_data,
            files=files,
        )
        # Calibre-Web redirects to the book detail page on success.
        if resp.status_code in (200, 302):
            # 200 can also mean the form re-rendered with errors. Check for flash.
            if resp.status_code == 200:
                err = _extract_flash_error(resp.text)
                if err:
                    raise CalibreWebError(f"edit failed: {err}")
            return
        raise CalibreWebError(
            f"unexpected status {resp.status_code} on POST /admin/book/{book_id}"
        )

    def set_author_sort(self, book_id: int, author_sort: str) -> str:
        """Override the auto-derived author_sort for one book.

        Calibre-Web's edit form auto-computes author_sort from the authors
        string and uses it to key author display order, so the only way to
        flip author display order is to write author_sort directly via the
        AJAX endpoint ``POST /ajax/editbooks/author_sort``.

        Wire format (discovered by probing): JSON body with ``pk`` as a
        list-of-int and ``value`` as the sort string. Returns the new value
        Calibre-Web echoed back (which may differ from the input if it
        normalized whitespace).
        """
        # Need a fresh csrf_token from the book form; the session cookie
        # provides auth, the X-CSRFToken header satisfies Flask-WTF.
        state = self.get_book_form_state(book_id)
        resp = self._client.post(
            "/ajax/editbooks/author_sort",
            json={"pk": [int(book_id)], "value": author_sort},
            headers={"X-CSRFToken": state.csrf_token},
        )
        if resp.status_code != 200:
            raise CalibreWebError(
                f"unexpected status {resp.status_code} on POST /ajax/editbooks/author_sort"
            )
        try:
            payload = resp.json()
        except ValueError as e:
            raise CalibreWebError(f"non-JSON response from author_sort endpoint: {e}") from e
        if not payload.get("success"):
            raise CalibreWebError(
                f"author_sort update failed: {payload.get('msg', 'unknown error')}"
            )
        return str(payload.get("newValue", author_sort))


def _parse_book_form(html: str, book_id: int) -> BookFormState:
    """Parse /admin/book/<id> HTML into a :class:`BookFormState`.

    Locates the ``#book_edit_frm`` form and extracts each named input. The
    csrf_token field appears on every form in the page (Calibre-Web emits
    one per form); we grab the one inside the book-edit form to be safe.
    """
    soup = BeautifulSoup(html, "html.parser")
    form = soup.find("form", id="book_edit_frm")
    if form is None:
        raise CalibreWebError(
            f"could not locate book_edit_frm in /admin/book/{book_id}"
        )

    def _input_value(name: str) -> str:
        node = form.find(["input", "textarea", "select"], attrs={"name": name})
        if node is None:
            return ""
        if node.name == "textarea":
            return node.text or ""
        return node.get("value", "") or ""

    csrf = _input_value("csrf_token")
    if not csrf:
        raise CalibreWebError(f"csrf_token missing on /admin/book/{book_id}")

    # Identifiers: pre-populated inputs are named identifier-type-<key> /
    # identifier-val-<key>. New JS-added rows use identifier-type-<rand_id>.
    # We pair them by the suffix after the second hyphen.
    identifiers: dict[str, str] = {}
    type_inputs: dict[str, str] = {}
    val_inputs: dict[str, str] = {}
    for node in form.find_all("input"):
        n = node.get("name", "")
        if n.startswith("identifier-type-"):
            type_inputs[n[len("identifier-type-"):]] = node.get("value", "") or ""
        elif n.startswith("identifier-val-"):
            val_inputs[n[len("identifier-val-"):]] = node.get("value", "") or ""
    for key, itype in type_inputs.items():
        ival = val_inputs.get(key, "")
        if itype and ival:
            identifiers[itype] = ival

    return BookFormState(
        csrf_token=csrf,
        book_id=book_id,
        title=_input_value("title"),
        authors=_input_value("authors"),
        tags=_input_value("tags"),
        series=_input_value("series"),
        series_index=_input_value("series_index"),
        pubdate=_input_value("pubdate"),
        publisher=_input_value("publisher"),
        languages=_input_value("languages"),
        rating=_input_value("rating"),
        comments=_input_value("comments"),
        cover_url="",  # cover_url is always empty in the form (server-side reset)
        identifiers=identifiers,
    )


def _extract_flash_error(html: str) -> str | None:
    """Pull the first flashed error message out of a Calibre-Web HTML response.

    Calibre-Web wraps flash messages in ``<div class="alert alert-danger">...</div>``.
    """
    m = re.search(
        r'<div[^>]+class="[^"]*alert-danger[^"]*"[^>]*>(.*?)</div>',
        html,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None
    # strip nested tags
    return re.sub(r"<[^>]+>", "", m.group(1)).strip() or None
