"""Unit tests for author_lookup — OpenLibrary/VIAF resolution of missing-initial pairs."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import httpx
import pytest

from calibre_mcp.cleanup import author_lookup, proposals
from calibre_mcp.cleanup.author_lookup import (
    _isbns_for_authors,
    _ol_fetch_batch,
    _resolve_pair,
    _viaf_canonical,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_calibre_db(path: Path, data: dict[str, list[str]]) -> None:
    """Seed a minimal Calibre DB: {author_name: [isbn, ...]}.

    Each ISBN is assigned to its own book row (book_id auto-increments).
    Authors with no ISBNs are created in the authors table but have no books.
    """
    conn = sqlite3.connect(path)
    try:
        conn.executescript("""
            CREATE TABLE authors (
              id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL
            );
            CREATE TABLE books_authors_link (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              book INTEGER NOT NULL, author INTEGER NOT NULL
            );
            CREATE TABLE identifiers (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              book INTEGER NOT NULL, type TEXT NOT NULL, val TEXT NOT NULL
            );
        """)
        book_id = 1
        for author_name, isbns in data.items():
            cur = conn.execute("INSERT INTO authors (name) VALUES (?)", (author_name,))
            author_id = cur.lastrowid
            for isbn in isbns:
                conn.execute(
                    "INSERT INTO identifiers (book, type, val) VALUES (?, 'isbn', ?)",
                    (book_id, isbn),
                )
                conn.execute(
                    "INSERT INTO books_authors_link (book, author) VALUES (?, ?)",
                    (book_id, author_id),
                )
                book_id += 1
        conn.commit()
    finally:
        conn.close()


def _open_calibre(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _ol_response(isbn_to_authors: dict[str, list[str]]) -> httpx.Response:
    data = {
        f"ISBN:{isbn}": {"authors": [{"name": n} for n in authors]}
        for isbn, authors in isbn_to_authors.items()
    }
    return httpx.Response(200, json=data)


def _viaf_response(results: list[dict]) -> httpx.Response:
    return httpx.Response(200, json={"result": results})


# ---------------------------------------------------------------------------
# _ol_fetch_batch
# ---------------------------------------------------------------------------


def test_ol_fetch_batch_returns_author_names() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return _ol_response({"9780553293357": ["Robert A. Heinlein"]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = _ol_fetch_batch(["9780553293357"], client, rate_sec=0.0)

    assert result == {"9780553293357": ["Robert A. Heinlein"]}


def test_ol_fetch_batch_multiple_isbns() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return _ol_response({
            "9780553293357": ["Robert A. Heinlein"],
            "9780141036144": ["George Orwell"],
        })

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = _ol_fetch_batch(["9780553293357", "9780141036144"], client, rate_sec=0.0)

    assert set(result) == {"9780553293357", "9780141036144"}
    assert result["9780553293357"] == ["Robert A. Heinlein"]
    assert result["9780141036144"] == ["George Orwell"]


def test_ol_fetch_batch_non_200_returns_empty() -> None:
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(503))) as client:
        result = _ol_fetch_batch(["9999"], client, rate_sec=0.0)
    assert result == {}


def test_ol_fetch_batch_network_error_returns_empty() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("network gone")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = _ol_fetch_batch(["9999"], client, rate_sec=0.0)
    assert result == {}


def test_ol_fetch_batch_skips_entries_without_authors() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        data = {
            "ISBN:0001": {"title": "No Authors Here"},
            "ISBN:0002": {"authors": [{"name": "Real Author"}]},
        }
        return httpx.Response(200, json=data)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = _ol_fetch_batch(["0001", "0002"], client, rate_sec=0.0)

    assert "0001" not in result
    assert result["0002"] == ["Real Author"]


# ---------------------------------------------------------------------------
# _viaf_canonical
# ---------------------------------------------------------------------------


def test_viaf_canonical_inverts_surname_given() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return _viaf_response([{"nametype": "personal", "term": "Lansdale, Joe R., 1951-"}])

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = _viaf_canonical("Joe Lansdale", client, rate_sec=0.0)

    assert result == "Joe R. Lansdale"


def test_viaf_canonical_strips_year() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return _viaf_response([{"nametype": "personal", "term": "Dick, Philip K., 1928-1982"}])

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = _viaf_canonical("Philip Dick", client, rate_sec=0.0)

    assert result == "Philip K. Dick"


def test_viaf_canonical_takes_first_personal_skips_corporate() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return _viaf_response([
            {"nametype": "corporate", "term": "ACME Corp"},
            {"nametype": "personal", "term": "Weber, David, 1952-"},
        ])

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = _viaf_canonical("David Weber", client, rate_sec=0.0)

    assert result == "David Weber"


def test_viaf_canonical_returns_none_on_empty_results() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return _viaf_response([])

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = _viaf_canonical("Nobody Famous", client, rate_sec=0.0)

    assert result is None


def test_viaf_canonical_returns_none_on_network_error() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = _viaf_canonical("Joe Lansdale", client, rate_sec=0.0)

    assert result is None


# ---------------------------------------------------------------------------
# _isbns_for_authors
# ---------------------------------------------------------------------------


def test_isbns_for_authors_returns_isbns_by_name(tmp_path: Path) -> None:
    _make_calibre_db(tmp_path / "metadata.db", {
        "Joe Lansdale": ["9780001", "9780002"],
        "Joe R. Lansdale": ["9780003"],
    })
    conn = _open_calibre(tmp_path / "metadata.db")
    try:
        result = _isbns_for_authors(conn, ["Joe Lansdale", "Joe R. Lansdale"])
    finally:
        conn.close()

    assert set(result["Joe Lansdale"]) == {"9780001", "9780002"}
    assert result["Joe R. Lansdale"] == ["9780003"]


def test_isbns_for_authors_case_insensitive(tmp_path: Path) -> None:
    _make_calibre_db(tmp_path / "metadata.db", {"joe lansdale": ["9780001"]})
    conn = _open_calibre(tmp_path / "metadata.db")
    try:
        result = _isbns_for_authors(conn, ["Joe Lansdale"])
    finally:
        conn.close()

    assert result["Joe Lansdale"] == ["9780001"]


def test_isbns_for_authors_unknown_name_returns_empty_list(tmp_path: Path) -> None:
    _make_calibre_db(tmp_path / "metadata.db", {})
    conn = _open_calibre(tmp_path / "metadata.db")
    try:
        result = _isbns_for_authors(conn, ["Nobody Here"])
    finally:
        conn.close()

    assert result["Nobody Here"] == []


# ---------------------------------------------------------------------------
# _resolve_pair
# ---------------------------------------------------------------------------


def test_resolve_pair_ol_confirms_long_name(tmp_path: Path) -> None:
    """OL has all books credited to the long (initialed) form → merge_to_long."""
    _make_calibre_db(tmp_path / "metadata.db", {
        "Joe Lansdale": ["9780001"],
        "Joe R. Lansdale": ["9780002", "9780003"],
    })

    def handler(req: httpx.Request) -> httpx.Response:
        return _ol_response({
            "9780001": ["Joe R. Lansdale"],
            "9780002": ["Joe R. Lansdale"],
            "9780003": ["Joe R. Lansdale"],
        })

    conn = _open_calibre(tmp_path / "metadata.db")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        decision = _resolve_pair(
            "Joe Lansdale", 1, "Joe R. Lansdale", 3, conn, client, rate_sec=0.0,
        )
    conn.close()

    assert decision.verdict == "merge_to_long"
    assert decision.canonical == "Joe R. Lansdale"
    assert decision.confidence >= 0.9
    assert "9780001" in decision.evidence


def test_resolve_pair_ol_confirms_short_name(tmp_path: Path) -> None:
    """OL credits both books to the short form → merge_to_short."""
    _make_calibre_db(tmp_path / "metadata.db", {
        "Joe Lansdale": ["9780001"],
        "Joe R. Lansdale": ["9780002"],
    })

    def handler(req: httpx.Request) -> httpx.Response:
        return _ol_response({
            "9780001": ["Joe Lansdale"],
            "9780002": ["Joe Lansdale"],
        })

    conn = _open_calibre(tmp_path / "metadata.db")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        decision = _resolve_pair(
            "Joe Lansdale", 3, "Joe R. Lansdale", 1, conn, client, rate_sec=0.0,
        )
    conn.close()

    assert decision.verdict == "merge_to_short"
    assert decision.canonical == "Joe Lansdale"
    assert decision.confidence >= 0.9


def test_resolve_pair_ol_returns_new_canonical(tmp_path: Path) -> None:
    """OL has a third form not matching either DB name → new_canonical verdict."""
    _make_calibre_db(tmp_path / "metadata.db", {
        "Joe Lansdale": ["9780001"],
        "Joe Lansdale Jr": ["9780002"],
    })

    def handler(req: httpx.Request) -> httpx.Response:
        return _ol_response({
            "9780001": ["Joe R. Lansdale"],
            "9780002": ["Joe R. Lansdale"],
        })

    conn = _open_calibre(tmp_path / "metadata.db")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        decision = _resolve_pair(
            "Joe Lansdale", 1, "Joe Lansdale Jr", 1, conn, client, rate_sec=0.0,
        )
    conn.close()

    assert decision.verdict == "new_canonical"
    assert decision.canonical == "Joe R. Lansdale"


def test_resolve_pair_viaf_fallback_when_no_isbns(tmp_path: Path) -> None:
    """Both authors have no ISBNs → OL skipped; VIAF confirms the short form."""
    _make_calibre_db(tmp_path / "metadata.db", {
        "David Weber": [],
        "David M. Weber": [],
    })

    def handler(req: httpx.Request) -> httpx.Response:
        return _viaf_response([{"nametype": "personal", "term": "Weber, David, 1952-"}])

    conn = _open_calibre(tmp_path / "metadata.db")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        decision = _resolve_pair(
            "David Weber", 5, "David M. Weber", 3, conn, client, rate_sec=0.0,
        )
    conn.close()

    assert decision.verdict == "merge_to_short"
    assert decision.canonical == "David Weber"
    assert decision.confidence == 0.6
    assert decision.evidence == ["viaf"]


def test_resolve_pair_no_data_verdict(tmp_path: Path) -> None:
    """Neither OL nor VIAF have data → no_data verdict."""
    _make_calibre_db(tmp_path / "metadata.db", {
        "Alice Waters": [],
        "Alice L. Waters": [],
    })

    def handler(req: httpx.Request) -> httpx.Response:
        return _viaf_response([])

    conn = _open_calibre(tmp_path / "metadata.db")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        decision = _resolve_pair(
            "Alice Waters", 10, "Alice L. Waters", 2, conn, client, rate_sec=0.0,
        )
    conn.close()

    assert decision.verdict == "no_data"
    assert decision.canonical is None
    assert decision.confidence == 0.0


def test_resolve_pair_lower_confidence_on_split_votes(tmp_path: Path) -> None:
    """Disagreement below 75% threshold → confidence drops to 0.7."""
    _make_calibre_db(tmp_path / "metadata.db", {
        "Joe Lansdale": ["9780001", "9780002", "9780003"],
        "Joe R. Lansdale": ["9780004"],
    })

    def handler(req: httpx.Request) -> httpx.Response:
        # 3 votes for "Joe R. Lansdale", 1 vote for "Joe Lansdale" → 75% agreement
        # Exactly 0.75 → >= 0.75 threshold is True → confidence 0.9
        # Let's do 2 vs 2 (50%) → confidence 0.7
        return _ol_response({
            "9780001": ["Joe R. Lansdale"],
            "9780002": ["Joe R. Lansdale"],
            "9780003": ["Joe Lansdale"],
            "9780004": ["Joe Lansdale"],
        })

    conn = _open_calibre(tmp_path / "metadata.db")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        decision = _resolve_pair(
            "Joe Lansdale", 3, "Joe R. Lansdale", 1, conn, client, rate_sec=0.0,
        )
    conn.close()

    # 2 for short, 2 for long → top is whichever comes first in Counter.most_common
    # Either way, agreement = 0.5 < 0.75 → confidence should be 0.7
    assert decision.confidence == 0.7
    assert decision.verdict in ("merge_to_long", "merge_to_short")


# ---------------------------------------------------------------------------
# run() — end-to-end with mocked HTTP
# ---------------------------------------------------------------------------


def _make_full_calibre_db(path: Path) -> None:
    """Calibre DB with a missing-initial pair and ISBNs for resolution."""
    conn = sqlite3.connect(path)
    try:
        conn.executescript("""
            CREATE TABLE authors (
              id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL
            );
            CREATE TABLE books_authors_link (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              book INTEGER NOT NULL, author INTEGER NOT NULL
            );
            CREATE TABLE identifiers (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              book INTEGER NOT NULL, type TEXT NOT NULL, val TEXT NOT NULL
            );
            CREATE TABLE tags (id INTEGER PRIMARY KEY, name TEXT);
            CREATE TABLE books_tags_link (id INTEGER PRIMARY KEY, book INTEGER, tag INTEGER);
        """)
        cur = conn.execute("INSERT INTO authors (name) VALUES ('Joe Lansdale')")
        a_short = cur.lastrowid
        cur = conn.execute("INSERT INTO authors (name) VALUES ('Joe R. Lansdale')")
        a_long = cur.lastrowid

        for book_id in range(1, 4):
            conn.execute(
                "INSERT INTO books_authors_link (book, author) VALUES (?, ?)", (book_id, a_short)
            )
        for book_id in range(4, 44):
            conn.execute(
                "INSERT INTO books_authors_link (book, author) VALUES (?, ?)", (book_id, a_long)
            )

        conn.execute("INSERT INTO identifiers (book, type, val) VALUES (1, 'isbn', '9780001')")
        conn.execute("INSERT INTO identifiers (book, type, val) VALUES (2, 'isbn', '9780002')")
        conn.commit()
    finally:
        conn.close()


def test_run_emits_merge_proposal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calibre_db = tmp_path / "metadata.db"
    _make_full_calibre_db(calibre_db)
    proposals_db = tmp_path / "proposals.db"

    real_client = httpx.Client

    def handler(req: httpx.Request) -> httpx.Response:
        return _ol_response({
            "9780001": ["Joe R. Lansdale"],
            "9780002": ["Joe R. Lansdale"],
        })

    def _mock_client_ctor(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(author_lookup.httpx, "Client", _mock_client_ctor)

    summary, decisions = author_lookup.run(
        calibre_db=calibre_db,
        proposals_db=proposals_db,
        dry_run=False,
        rate_sec=0.0,
    )

    assert summary.examined == 1
    assert summary.merge_to_long == 1
    assert summary.proposals_emitted == 1

    conn = sqlite3.connect(proposals_db)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT field, calibre_value, proposed_value, status FROM proposals "
            "WHERE field='author.merge'"
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    assert rows[0]["calibre_value"] == "Joe Lansdale"
    assert rows[0]["proposed_value"] == "Joe Lansdale -> Joe R. Lansdale"
    assert rows[0]["status"] == "proposed"


def test_run_dry_run_emits_no_proposals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calibre_db = tmp_path / "metadata.db"
    _make_full_calibre_db(calibre_db)
    proposals_db = tmp_path / "proposals.db"

    real_client = httpx.Client

    def handler(req: httpx.Request) -> httpx.Response:
        return _ol_response({
            "9780001": ["Joe R. Lansdale"],
            "9780002": ["Joe R. Lansdale"],
        })

    def _mock_client_ctor(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(author_lookup.httpx, "Client", _mock_client_ctor)

    summary, decisions = author_lookup.run(
        calibre_db=calibre_db,
        proposals_db=proposals_db,
        dry_run=True,
        rate_sec=0.0,
    )

    assert summary.merge_to_long == 1

    conn = sqlite3.connect(proposals_db)
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM proposals WHERE field='author.merge'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert count == 0


def test_run_limit_caps_pairs_processed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--limit=0 processes no pairs and emits no proposals."""
    calibre_db = tmp_path / "metadata.db"
    _make_full_calibre_db(calibre_db)
    proposals_db = tmp_path / "proposals.db"

    real_client = httpx.Client

    def _mock_client_ctor(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(lambda _req: httpx.Response(200, json={}))
        return real_client(*args, **kwargs)

    monkeypatch.setattr(author_lookup.httpx, "Client", _mock_client_ctor)

    summary, decisions = author_lookup.run(
        calibre_db=calibre_db,
        proposals_db=proposals_db,
        dry_run=False,
        rate_sec=0.0,
        limit=0,
    )

    assert summary.examined == 0
    assert decisions == []


def test_run_no_data_not_persisted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """no_data decisions must not produce proposals regardless of min_confidence."""
    calibre_db = tmp_path / "metadata.db"
    _make_full_calibre_db(calibre_db)
    proposals_db = tmp_path / "proposals.db"

    real_client = httpx.Client

    def handler(req: httpx.Request) -> httpx.Response:
        if "openlibrary.org" in str(req.url):
            return httpx.Response(200, json={})  # OL has nothing
        return _viaf_response([])  # VIAF also has nothing

    def _mock_client_ctor(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(author_lookup.httpx, "Client", _mock_client_ctor)

    summary, decisions = author_lookup.run(
        calibre_db=calibre_db,
        proposals_db=proposals_db,
        dry_run=False,
        rate_sec=0.0,
    )

    assert summary.no_data == 1
    assert summary.proposals_emitted == 0

    conn = sqlite3.connect(proposals_db)
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM proposals WHERE field='author.merge'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert count == 0
