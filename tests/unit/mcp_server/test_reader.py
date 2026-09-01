"""Read-path fidelity tests for the MCP server.

Regression coverage for the 2026-08/09 Claude Desktop findings: authors were
reported in link order (not Calibre-Web display order), tags were casefolded,
identifier types were lowercased, and the update diff never showed the real
prior ``author_sort``. Every read must report stored case and display order.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from calibre_mcp.mcp_server.api import BookFormState
from calibre_mcp.mcp_server.reader import CalibreReader
from calibre_mcp.mcp_server.server import _diff_states, _format_book_detail

_SCHEMA = """
CREATE TABLE books (id INTEGER PRIMARY KEY, uuid TEXT, title TEXT, path TEXT,
                    pubdate TEXT, series_index REAL, author_sort TEXT, has_cover INTEGER DEFAULT 0);
CREATE TABLE authors (id INTEGER PRIMARY KEY, name TEXT, sort TEXT);
CREATE TABLE books_authors_link (id INTEGER PRIMARY KEY, book INTEGER, author INTEGER);
CREATE TABLE tags (id INTEGER PRIMARY KEY, name TEXT);
CREATE TABLE books_tags_link (id INTEGER PRIMARY KEY, book INTEGER, tag INTEGER);
CREATE TABLE series (id INTEGER PRIMARY KEY, name TEXT);
CREATE TABLE books_series_link (id INTEGER PRIMARY KEY, book INTEGER, series INTEGER);
CREATE TABLE publishers (id INTEGER PRIMARY KEY, name TEXT);
CREATE TABLE books_publishers_link (id INTEGER PRIMARY KEY, book INTEGER, publisher INTEGER);
CREATE TABLE comments (id INTEGER PRIMARY KEY, book INTEGER, text TEXT);
CREATE TABLE identifiers (id INTEGER PRIMARY KEY, book INTEGER, type TEXT, val TEXT);
CREATE TABLE languages (id INTEGER PRIMARY KEY, lang_code TEXT);
CREATE TABLE books_languages_link (id INTEGER PRIMARY KEY, book INTEGER, lang_code INTEGER, item_order INTEGER);
"""


@pytest.fixture
def library(tmp_path: Path) -> Path:
    db = tmp_path / "metadata.db"
    conn = sqlite3.connect(db)
    conn.executescript(_SCHEMA)
    # Book 22999: The Currabinny Cookbook. Link order is Murray, Kavanagh
    # (stale — set when the book was imported), but author_sort was later
    # overridden to put Kavanagh first. Calibre-Web displays Kavanagh first.
    conn.execute(
        "INSERT INTO books VALUES (22999, 'u1', 'The Currabinny Cookbook', 'x', '2019-01-01', 1.0, "
        "'Kavanagh, James & Murray, William', 1)"
    )
    conn.execute("INSERT INTO authors VALUES (1, 'William Murray', 'Murray, William')")
    conn.execute("INSERT INTO authors VALUES (2, 'James Kavanagh', 'Kavanagh, James')")
    conn.execute("INSERT INTO books_authors_link VALUES (10, 22999, 1)")
    conn.execute("INSERT INTO books_authors_link VALUES (11, 22999, 2)")
    for tid, name in ((1, "cookbooks"), (2, "Baking"), (3, "Irish"), (4, "nonfiction")):
        conn.execute("INSERT INTO tags VALUES (?, ?)", (tid, name))
        conn.execute("INSERT INTO books_tags_link (book, tag) VALUES (22999, ?)", (tid,))
    # Mixed-case identifier types exactly as Calibre-Web stored them on import.
    conn.execute("INSERT INTO identifiers (book, type, val) VALUES (22999, 'ISBN', '9781785767081')")
    conn.execute("INSERT INTO identifiers (book, type, val) VALUES (22999, 'FICTIONDB', 'abc')")
    conn.execute("INSERT INTO identifiers (book, type, val) VALUES (22999, 'goodreads', '43159388')")

    # Book 23000: single author, no author_sort override, empty author_sort.
    conn.execute("INSERT INTO books VALUES (23000, 'u2', 'Mystery in White', 'y', '2014-01-01', 6.0, '', 0)")
    conn.execute("INSERT INTO authors VALUES (3, 'J. Jefferson Farjeon', 'Farjeon, J. Jefferson')")
    conn.execute("INSERT INTO books_authors_link VALUES (12, 23000, 3)")

    # Book 23001: author_sort names an author no longer linked + a linked one.
    conn.execute(
        "INSERT INTO books VALUES (23001, 'u3', 'Stale Sort', 'z', NULL, 1.0, 'Nobody, Some & Murray, William', 0)"
    )
    conn.execute("INSERT INTO books_authors_link VALUES (13, 23001, 2)")
    conn.execute("INSERT INTO books_authors_link VALUES (14, 23001, 1)")
    conn.commit()
    conn.close()
    return db


def test_authors_follow_author_sort_not_link_order(library: Path) -> None:
    book = CalibreReader(library).get_book(22999)
    assert book is not None
    assert book.authors == ("James Kavanagh", "William Murray")
    assert book.author_sort == "Kavanagh, James & Murray, William"


def test_authors_fall_back_to_link_order_when_author_sort_empty(library: Path) -> None:
    book = CalibreReader(library).get_book(23000)
    assert book is not None
    assert book.authors == ("J. Jefferson Farjeon",)
    assert book.author_sort is None


def test_unmatched_author_sort_entries_are_skipped_and_leftovers_keep_link_order(library: Path) -> None:
    # "Nobody, Some" isn't linked → skipped; Murray matched → first; Kavanagh
    # leftover → appended in link order. Mirrors Calibre-Web's order_authors().
    book = CalibreReader(library).get_book(23001)
    assert book is not None
    assert book.authors == ("William Murray", "James Kavanagh")


def test_search_results_use_display_order(library: Path) -> None:
    hits = CalibreReader(library).search_books("Currabinny")
    assert [h.authors for h in hits] == [("James Kavanagh", "William Murray")]
    hits = CalibreReader(library).search_advanced(title_contains="Currabinny")
    assert [h.authors for h in hits] == [("James Kavanagh", "William Murray")]


def test_tags_keep_stored_case_and_are_name_ordered(library: Path) -> None:
    book = CalibreReader(library).get_book(22999)
    assert book is not None
    assert book.tags == ("Baking", "cookbooks", "Irish", "nonfiction")
    # The pipeline's casefolded set is still there for its own matching.
    assert book.tags_ci == frozenset({"baking", "cookbooks", "irish", "nonfiction"})


def test_identifier_types_keep_stored_case(library: Path) -> None:
    book = CalibreReader(library).get_book(22999)
    assert book is not None
    assert book.identifiers == {"ISBN": "9781785767081", "FICTIONDB": "abc", "goodreads": "43159388"}
    assert book.isbn == "9781785767081"  # still found despite the uppercase type


def test_get_author_sort(library: Path) -> None:
    reader = CalibreReader(library)
    assert reader.get_author_sort(22999) == "Kavanagh, James & Murray, William"
    assert reader.get_author_sort(23000) is None
    assert reader.get_author_sort(999999) is None


def test_detail_rendering_has_no_case_or_order_transforms(library: Path) -> None:
    book = CalibreReader(library).get_book(22999)
    assert book is not None
    text = _format_book_detail(book)
    assert "Authors: James Kavanagh & William Murray" in text
    assert "Author sort: Kavanagh, James & Murray, William" in text
    assert "Tags: Baking, cookbooks, Irish, nonfiction" in text
    assert "FICTIONDB=abc, goodreads=43159388, ISBN=9781785767081" in text


def _state(**kw: object) -> BookFormState:
    return BookFormState(csrf_token="t", book_id=1, **kw)  # type: ignore[arg-type]  # noqa: S106


def test_diff_ignores_line_ending_only_comment_changes() -> None:
    before = _state(comments="<p>one</p>\r\n<p>two</p>\r\n", tags="Mystery")
    after = _state(comments="<p>one</p>\n<p>two</p>\n", tags="Mystery, Crime")
    assert _diff_states(before, after) == ["  tags: 'Mystery' → 'Mystery, Crime'"]


def test_diff_reports_real_comment_changes() -> None:
    before = _state(comments="<p>one</p>\r\n")
    after = _state(comments="<p>two</p>\n")
    assert _diff_states(before, after) == ["  comments: '<p>one</p>\\n' → '<p>two</p>\\n'"]


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:19943/api/stacks",
        "http://192.168.1.64:38083/admin",
        "http://localhost/x.jpg",
        "file:///etc/passwd",
        "ftp://example.com/x.jpg",
        "http:///no-host",
    ],
)
def test_cover_url_guard_rejects_non_public(url: str) -> None:
    from calibre_mcp.mcp_server.server import _reject_non_public_url

    assert _reject_non_public_url(url) is not None


def test_cover_url_guard_allows_public(monkeypatch: pytest.MonkeyPatch) -> None:
    import socket

    from calibre_mcp.mcp_server.server import _reject_non_public_url

    monkeypatch.setattr(socket, "getaddrinfo", lambda *_a, **_k: [(None, None, None, None, ("104.16.0.1", 443))])
    assert _reject_non_public_url("https://covers.openlibrary.org/b/isbn/123-L.jpg") is None
