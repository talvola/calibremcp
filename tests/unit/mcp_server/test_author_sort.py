"""author_sort guard rails on ``calibre_update_book``.

Regression coverage for the 2026-10-19 Claude Desktop findings: a
comments-only edit appeared to reshuffle authors, and a mismatched explicit
``author_sort`` reordered the displayed credit. Both come from Calibre-Web
matching ``author_sort`` pieces to ``authors.sort`` exactly and re-deriving
``author_sort`` from the displayed order on every form save.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from calibre_mcp.mcp_server import server
from calibre_mcp.mcp_server.api import BookFormState
from calibre_mcp.mcp_server.reader import AuthorSortCheck, CalibreReader
from calibre_mcp.mcp_server.server import _resolve_author_sort, calibre_update_book

from .test_reader import library  # noqa: F401 — pytest fixture


class FakeCalibreWeb:
    """Mimics the Calibre-Web behavior that matters here: every form save
    re-derives ``author_sort`` from the posted authors, in order."""

    def __init__(self, db: Path) -> None:
        self.db = db
        self.comments = ""
        self.author_sort_calls: list[str] = []

    def get_book_form_state(self, book_id: int) -> BookFormState:
        book = CalibreReader(self.db).get_book(book_id)
        assert book is not None
        authors = " & ".join(book.authors)
        return BookFormState(csrf_token="t", book_id=book_id, authors=authors, comments=self.comments)  # noqa: S106

    def edit_book(self, book_id: int, updates: dict[str, Any], *, identifiers: Any = None) -> None:
        names = [n.strip() for n in updates.get("authors", self.get_book_form_state(book_id).authors).split("&")]
        with sqlite3.connect(self.db) as conn:
            sorts = [conn.execute("SELECT sort FROM authors WHERE name = ?", (n,)).fetchone()[0] for n in names]
            conn.execute("UPDATE books SET author_sort = ? WHERE id = ?", (" & ".join(sorts), book_id))
        self.comments = updates.get("comments", self.comments)

    def set_author_sort(self, book_id: int, value: str) -> str:
        self.author_sort_calls.append(value)
        with sqlite3.connect(self.db) as conn:
            conn.execute("UPDATE books SET author_sort = ? WHERE id = ?", (value, book_id))
        return value


@pytest.fixture
def cw(library: Path, monkeypatch: pytest.MonkeyPatch) -> FakeCalibreWeb:  # noqa: F811
    fake = FakeCalibreWeb(library)
    monkeypatch.setattr(server, "_reader", CalibreReader(library))
    monkeypatch.setattr(server, "_calibre_web", fake)
    return fake


def test_comment_only_edit_on_out_of_sync_book_explains_the_author_sort_change(cw: FakeCalibreWeb) -> None:
    out = calibre_update_book(6526, comments="<p>new</p>")
    assert "author_sort: 'Cast, Pc & Cast, Kristin' → 'Cast, Kristin & Cast, PC'" in out
    assert "already out of sync ('Cast, Pc' matched no author)" in out
    assert "visible credit did not change" in out
    # The save healed the mismatch, so no lingering ⚠.
    assert "⚠" not in out


def test_comment_only_edit_on_consistent_book_is_quiet(cw: FakeCalibreWeb) -> None:
    out = calibre_update_book(22999, comments="<p>new</p>")
    assert "author_sort" not in out


def test_explicit_author_sort_case_is_corrected_to_stored_sort(cw: FakeCalibreWeb) -> None:
    out = calibre_update_book(6526, author_sort="Cast, pc & cast, kristin")
    assert cw.author_sort_calls == ["Cast, PC & Cast, Kristin"]
    assert "⚠" not in out


def test_explicit_author_sort_with_unknown_piece_is_refused(cw: FakeCalibreWeb) -> None:
    out = calibre_update_book(22999, author_sort="Kavanagh, Jim & Murray, William")
    assert cw.author_sort_calls == []
    assert "author_sort NOT applied: 'Kavanagh, Jim' matches no author" in out
    assert "'Murray, William' & 'Kavanagh, James'" in out


_CHECK = AuthorSortCheck(
    author_sort="Kavanagh, James & Murray, William",
    author_sorts=("Murray, William", "Kavanagh, James"),
    unmatched=(),
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Kavanagh, James & Murray, William", "Kavanagh, James & Murray, William"),
        ("  murray, william&KAVANAGH, JAMES ", "Murray, William & Kavanagh, James"),
    ],
)
def test_resolve_author_sort_accepts_permutations(value: str, expected: str) -> None:
    assert _resolve_author_sort(value, _CHECK) == (expected, None)


@pytest.mark.parametrize(
    "value",
    [
        "Kavanagh, James",  # missing an author → that author falls to the end
        "Kavanagh, James & Kavanagh, James & Murray, William",  # duplicate → shown twice
        "Kavanagh, J. & Murray, William",  # not a stored sort name
    ],
)
def test_resolve_author_sort_refuses_non_permutations(value: str) -> None:
    resolved, problem = _resolve_author_sort(value, _CHECK)
    assert problem is not None
    assert resolved == value


def test_resolve_author_sort_passes_through_without_reader() -> None:
    assert _resolve_author_sort("Anything, At All", None) == ("Anything, At All", None)
