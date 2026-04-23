"""Integration tests for the cleanup webapp via FastAPI TestClient."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from calibre_mcp.cleanup import proposals
from calibre_mcp.cleanup.proposals import Proposal
from calibre_mcp.cleanup.web import create_app


@pytest.fixture
def seeded_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "proposals.db"
    conn = proposals.connect(db_path)
    run_id = proposals.start_run(
        conn,
        source="opf",
        library_root=Path("/tmp/library"),  # noqa: S108
        metadata_db_path=Path("/tmp/library/metadata.db"),  # noqa: S108
        dry_run=False,
    )
    seed = [
        Proposal(book_id=42, field="isbn", proposed_value="9780141036144", source="opf", confidence=1.0),
        Proposal(book_id=42, field="publisher", proposed_value="Penguin", source="opf", confidence=0.9),
        Proposal(book_id=99, field="isbn", proposed_value="9780000000007", source="opf", confidence=1.0),
        Proposal(
            book_id=7,
            field="pubdate",
            proposed_value="2014-11-10",
            calibre_value="2001-01-01",
            conflict_reason="calibre disagrees",
            source="opf",
            confidence=0.9,
        ),
    ]
    for p in seed:
        proposals.insert_proposal(conn, run_id, p)
    conn.close()
    return db_path


@pytest.fixture
def client(seeded_db: Path) -> TestClient:
    return TestClient(create_app(proposals_db=seeded_db))


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


def test_dashboard_renders_field_by_status_grid(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    body = r.text
    assert "Calibre Cleanup" in body
    # Both fields appear as row labels + both statuses as column labels.
    assert "isbn" in body
    assert "publisher" in body
    assert "pubdate" in body
    assert "proposed" in body
    assert "conflict" in body


def test_dashboard_counts_correct(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    # 3 proposed (2 isbn + 1 publisher) + 1 conflict (pubdate) = 4 total.
    assert ">4<" in r.text or ">4</strong>" in r.text or ">4</th>" in r.text


# ---------------------------------------------------------------------------
# List view + filters
# ---------------------------------------------------------------------------


def test_list_returns_all_without_filter(client: TestClient) -> None:
    r = client.get("/proposals")
    assert r.status_code == 200
    assert "9780141036144" in r.text
    assert "9780000000007" in r.text
    assert "Penguin" in r.text


def test_list_filter_by_field(client: TestClient) -> None:
    r = client.get("/proposals?field=isbn")
    assert r.status_code == 200
    # Both ISBN proposals present, publisher absent.
    assert "9780141036144" in r.text
    assert "9780000000007" in r.text
    assert "Penguin" not in r.text


def test_list_filter_by_status(client: TestClient) -> None:
    r = client.get("/proposals?status=conflict")
    assert r.status_code == 200
    # Only the pubdate conflict should appear.
    assert "2014-11-10" in r.text
    assert "9780141036144" not in r.text


def test_list_filter_by_book_id(client: TestClient) -> None:
    r = client.get("/proposals?book_id=42")
    assert r.status_code == 200
    # Both book 42 rows.
    assert "9780141036144" in r.text
    assert "Penguin" in r.text
    # Book 99's ISBN is absent.
    assert "9780000000007" not in r.text


def test_list_pagination_respects_size(client: TestClient) -> None:
    r = client.get("/proposals?size=1&page=1")
    assert r.status_code == 200
    # Only one proposed_value per page when size=1.
    # (Simplest signal: the "prev/next" navigation is present.)
    assert "next" in r.text
    assert "page 1 of" in r.text


# ---------------------------------------------------------------------------
# Status transitions
# ---------------------------------------------------------------------------


def test_approve_via_post_updates_status(client: TestClient, seeded_db: Path) -> None:
    # Grab the first ISBN proposal's id.
    with sqlite3.connect(seeded_db) as conn:
        conn.row_factory = sqlite3.Row
        pid = conn.execute("SELECT id FROM proposals WHERE proposed_value='9780141036144'").fetchone()["id"]

    r = client.post(
        f"/proposals/{pid}/status",
        data={"action": "approved", "return_to": "/proposals"},
        follow_redirects=False,
    )
    assert r.status_code == 303

    with sqlite3.connect(seeded_db) as conn:
        row = conn.execute("SELECT status FROM proposals WHERE id=?", (pid,)).fetchone()
    assert row[0] == "approved"


def test_reject_via_post(client: TestClient, seeded_db: Path) -> None:
    with sqlite3.connect(seeded_db) as conn:
        conn.row_factory = sqlite3.Row
        pid = conn.execute("SELECT id FROM proposals WHERE proposed_value='Penguin'").fetchone()["id"]
    r = client.post(
        f"/proposals/{pid}/status",
        data={"action": "rejected"},
        follow_redirects=False,
    )
    assert r.status_code == 303

    with sqlite3.connect(seeded_db) as conn:
        row = conn.execute("SELECT status FROM proposals WHERE id=?", (pid,)).fetchone()
    assert row[0] == "rejected"


def test_invalid_action_returns_400(client: TestClient) -> None:
    r = client.post("/proposals/1/status", data={"action": "banana"})
    assert r.status_code == 400


def test_missing_proposal_returns_404(client: TestClient) -> None:
    r = client.post("/proposals/99999/status", data={"action": "approved"})
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Bulk action
# ---------------------------------------------------------------------------


def test_bulk_approves_by_field(client: TestClient, seeded_db: Path) -> None:
    r = client.post(
        "/bulk",
        data={"action": "approved", "field": "isbn", "status": "proposed"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    with sqlite3.connect(seeded_db) as conn:
        approved = conn.execute("SELECT COUNT(*) FROM proposals WHERE status='approved' AND field='isbn'").fetchone()[0]
    assert approved == 2  # both ISBN proposals


def test_bulk_respects_book_id_filter(client: TestClient, seeded_db: Path) -> None:
    r = client.post(
        "/bulk",
        data={"action": "rejected", "book_id": "42"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    with sqlite3.connect(seeded_db) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT status FROM proposals WHERE book_id=42").fetchall()
        other = conn.execute("SELECT status FROM proposals WHERE book_id!=42").fetchall()
    assert all(r["status"] == "rejected" for r in rows)
    # Other books untouched.
    assert not any(r["status"] == "rejected" for r in other)
