"""Unit + integration tests for the Goodreads ID lookup module."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import httpx
import pytest

from calibre_mcp.cleanup import goodreads_lookup, proposals
from calibre_mcp.cleanup.goodreads_lookup import (
    _extract_goodreads_id,
    _iter_candidates,
    _record_not_found,
    lookup_one,
)

# ---------------------------------------------------------------------------
# _extract_goodreads_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_val, expected",
    [
        ("https://www.goodreads.com/book/show/60157982-foundation", "60157982"),
        ("https://www.goodreads.com/book/show/60157982", "60157982"),
        ("http://goodreads.com/book/show/1234-book-with-dashes-in-title", "1234"),
        ("/book/show/5678-slug", "5678"),
        ("/book/show/9999", "9999"),
        # Non-/book/show URLs — not a successful lookup.
        ("https://www.goodreads.com/search?q=foo", None),
        ("https://www.goodreads.com/book/isbn/9780141036144", None),
        ("", None),
        ("not a url", None),
    ],
)
def test_extract_goodreads_id(input_val: str, expected: str | None) -> None:
    assert _extract_goodreads_id(input_val) == expected


# ---------------------------------------------------------------------------
# lookup_one — using httpx.MockTransport
# ---------------------------------------------------------------------------


def _mock_client(handler) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(handler),
        follow_redirects=False,
        timeout=5.0,
    )


def test_lookup_one_found() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/book/isbn/9780553293357"
        return httpx.Response(
            301,
            headers={"location": "https://www.goodreads.com/book/show/60157982-foundation"},
        )

    with _mock_client(handler) as c:
        gid, status, note = lookup_one(c, "9780553293357")
    assert gid == "60157982"
    assert status == "found"
    assert "60157982" in (note or "")


def test_lookup_one_not_found_404() -> None:
    with _mock_client(lambda _req: httpx.Response(404)) as c:
        gid, status, note = lookup_one(c, "9999999999999")
    assert gid is None
    assert status == "not_found"
    assert note == "404"


def test_lookup_one_not_found_no_redirect_on_200() -> None:
    with _mock_client(lambda _req: httpx.Response(200, text="<html>search page</html>")) as c:
        gid, status, note = lookup_one(c, "9999999999999")
    assert gid is None
    assert status == "not_found"


def test_lookup_one_redirect_to_non_book_url() -> None:
    """Goodreads sometimes redirects to a search page for unknown ISBNs —
    should be treated as not_found, not a spurious 'found'."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(301, headers={"location": "https://www.goodreads.com/search?q=9999"})

    with _mock_client(handler) as c:
        gid, status, note = lookup_one(c, "9999999999999")
    assert gid is None
    assert status == "not_found"
    assert "search" in (note or "")


def test_lookup_one_rate_limited() -> None:
    with _mock_client(lambda _req: httpx.Response(429)) as c:
        gid, status, _note = lookup_one(c, "9780141036144")
    assert gid is None
    assert status == "rate_limited"


def test_lookup_one_http_error_returns_error_status() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("network gone")

    with _mock_client(handler) as c:
        gid, status, note = lookup_one(c, "9780141036144")
    assert gid is None
    assert status == "error"
    assert "ConnectError" in (note or "")


# ---------------------------------------------------------------------------
# _record_not_found
# ---------------------------------------------------------------------------


@pytest.fixture
def proposals_conn(tmp_path: Path):
    conn = proposals.connect(tmp_path / "proposals.db")
    yield conn
    conn.close()


def test_record_not_found_lands_as_rejected(proposals_conn) -> None:
    run_id = proposals.start_run(
        proposals_conn,
        source="goodreads_isbn",
        library_root=Path("/tmp/lib"),  # noqa: S108
        metadata_db_path=Path("/tmp/lib/metadata.db"),  # noqa: S108
        dry_run=False,
    )
    _record_not_found(proposals_conn, run_id, book_id=42, isbn="9999999999999", note="404")
    rows = list(proposals_conn.execute("SELECT book_id, field, status, conflict_reason FROM proposals"))
    assert len(rows) == 1
    assert rows[0][0] == 42
    assert rows[0][1] == "identifier.goodreads"
    assert rows[0][2] == "rejected"
    assert rows[0][3] == "not found on Goodreads"


def test_record_not_found_is_idempotent(proposals_conn) -> None:
    """A second not-found record for the same book is a no-op (unique index)."""
    run_id = proposals.start_run(
        proposals_conn,
        source="goodreads_isbn",
        library_root=Path("/tmp/lib"),  # noqa: S108
        metadata_db_path=Path("/tmp/lib/metadata.db"),  # noqa: S108
        dry_run=False,
    )
    _record_not_found(proposals_conn, run_id, 42, "9999999999999", "404")
    _record_not_found(proposals_conn, run_id, 42, "9999999999999", "404 again")
    assert proposals_conn.execute("SELECT COUNT(*) FROM proposals WHERE book_id=42").fetchone()[0] == 1


# ---------------------------------------------------------------------------
# _iter_candidates — skip logic
# ---------------------------------------------------------------------------


def _make_calibre_db(path: Path, books: list[tuple[int, dict[str, str]]]) -> None:
    """Minimal Calibre schema for candidate selection: books + identifiers."""
    conn = sqlite3.connect(path)
    try:
        conn.executescript("""
            CREATE TABLE identifiers (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              book INTEGER NOT NULL,
              type TEXT NOT NULL,
              val TEXT NOT NULL
            );
        """)
        for book_id, ids in books:
            for scheme, value in ids.items():
                conn.execute(
                    "INSERT INTO identifiers (book, type, val) VALUES (?, ?, ?)",
                    (book_id, scheme, value),
                )
        conn.commit()
    finally:
        conn.close()


def test_iter_candidates_skips_books_already_having_goodreads(tmp_path: Path, proposals_conn) -> None:
    calibre_db = tmp_path / "metadata.db"
    _make_calibre_db(
        calibre_db,
        [
            (1, {"isbn": "9780141036144"}),  # eligible
            (2, {"isbn": "9780553293357", "goodreads": "99"}),  # already has GR → skip
            (3, {"isbn": "9781250765055", "Goodreads": "88"}),  # capital G variant → skip
            (4, {"goodreads": "77"}),  # no ISBN → skip
        ],
    )
    cal = sqlite3.connect(calibre_db)
    cal.row_factory = sqlite3.Row
    try:
        candidates = list(_iter_candidates(cal, proposals_conn))
    finally:
        cal.close()
    assert candidates == [(1, "9780141036144")]


def test_iter_candidates_skips_books_already_tried(tmp_path: Path, proposals_conn) -> None:
    calibre_db = tmp_path / "metadata.db"
    _make_calibre_db(calibre_db, [(1, {"isbn": "978"}), (2, {"isbn": "979"})])

    # Mark book 1 as already tried (any goodreads proposal counts).
    run_id = proposals.start_run(
        proposals_conn,
        source="goodreads_isbn",
        library_root=Path("/tmp/l"),  # noqa: S108
        metadata_db_path=calibre_db,
        dry_run=False,
    )
    _record_not_found(proposals_conn, run_id, 1, "978", "404")

    cal = sqlite3.connect(calibre_db)
    cal.row_factory = sqlite3.Row
    try:
        candidates = list(_iter_candidates(cal, proposals_conn))
    finally:
        cal.close()
    assert candidates == [(2, "979")]


def test_iter_candidates_respects_limit(tmp_path: Path, proposals_conn) -> None:
    calibre_db = tmp_path / "metadata.db"
    _make_calibre_db(calibre_db, [(i, {"isbn": f"97800000{i:05d}"}) for i in range(1, 20)])
    cal = sqlite3.connect(calibre_db)
    cal.row_factory = sqlite3.Row
    try:
        candidates = list(_iter_candidates(cal, proposals_conn, limit=5))
    finally:
        cal.close()
    assert len(candidates) == 5


# ---------------------------------------------------------------------------
# run() — end-to-end with mocked HTTP
# ---------------------------------------------------------------------------


def test_run_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Three books: one finds a Goodreads ID, one doesn't, one errors."""
    calibre_db = tmp_path / "metadata.db"
    _make_calibre_db(
        calibre_db,
        [
            (1, {"isbn": "9780553293357"}),
            (2, {"isbn": "9999999999999"}),
            (3, {"isbn": "9780141036144"}),
        ],
    )
    proposals_db = tmp_path / "proposals.db"

    # Patch httpx.Client so run() uses the mock transport.
    real_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/book/isbn/9780553293357":
            return httpx.Response(301, headers={"location": "https://www.goodreads.com/book/show/60157982-foundation"})
        if path == "/book/isbn/9999999999999":
            return httpx.Response(404)
        return httpx.Response(500)

    def _mock_client_ctor(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(goodreads_lookup.httpx, "Client", _mock_client_ctor)

    # Zero rate to avoid the per-request sleep in tests.
    summary = goodreads_lookup.run(
        calibre_db=calibre_db,
        proposals_db=proposals_db,
        rate_sec=0.0,
    )
    assert summary.attempted == 3
    assert summary.found == 1
    assert summary.not_found == 1
    assert summary.errors == 1

    # Verify the found proposal landed at status='proposed' with the right ID.
    conn = sqlite3.connect(proposals_db)
    conn.row_factory = sqlite3.Row
    try:
        found = conn.execute(
            "SELECT book_id, proposed_value, status FROM proposals "
            "WHERE field='identifier.goodreads' AND status='proposed'"
        ).fetchall()
        assert len(found) == 1
        assert found[0]["book_id"] == 1
        assert found[0]["proposed_value"] == "60157982"

        # Not-found lands as rejected.
        rejected = conn.execute("SELECT book_id FROM proposals WHERE status='rejected'").fetchall()
        assert [r["book_id"] for r in rejected] == [2]
    finally:
        conn.close()
