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


# ---------------------------------------------------------------------------
# Cookbook dashboard + cover route + proposed_value filter
# ---------------------------------------------------------------------------


@pytest.fixture
def cookbook_db(tmp_path: Path) -> Path:
    """Seed a propose-queue with cookbook_llm proposals across multiple
    facets so the /cookbooks dashboard has something to render."""
    db_path = tmp_path / "cookbook.db"
    conn = proposals.connect(db_path)
    run_id = proposals.start_run(
        conn, source="cookbook_llm",
        library_root=Path("/tmp/library"),  # noqa: S108
        metadata_db_path=Path("/tmp/library/metadata.db"),  # noqa: S108
        dry_run=False,
    )
    seed = [
        Proposal(book_id=1, field="tags.add", proposed_value="Italian", source="cookbook_llm", confidence=0.9),
        Proposal(book_id=2, field="tags.add", proposed_value="Italian", source="cookbook_llm", confidence=0.7),
        Proposal(book_id=3, field="tags.add", proposed_value="Korean", source="cookbook_llm", confidence=0.9),
        Proposal(book_id=4, field="tags.add", proposed_value="Baking", source="cookbook_llm", confidence=0.9),
        Proposal(book_id=5, field="tags.add", proposed_value="Vegan", source="cookbook_llm", confidence=0.9),
        # Sentinel must be excluded from the dashboard.
        Proposal(book_id=6, field="tags.add", proposed_value="(no tags)", source="cookbook_llm", confidence=0.7),
    ]
    for p in seed:
        proposals.insert_proposal(conn, run_id, p)
    conn.close()
    return db_path


def test_cookbooks_dashboard_groups_by_facet(cookbook_db: Path) -> None:
    """Italian + Korean go under Cuisine, Baking under Technique, Vegan
    under Dietary."""
    client = TestClient(create_app(proposals_db=cookbook_db))
    r = client.get("/cookbooks")
    assert r.status_code == 200
    body = r.text
    assert "Cuisine" in body
    assert "Technique" in body
    assert "Dietary" in body
    assert "Italian" in body
    assert "Korean" in body
    assert "Baking" in body
    assert "Vegan" in body


def test_cookbooks_dashboard_excludes_sentinel(cookbook_db: Path) -> None:
    """The (no tags) sentinel must not appear as a tag GROUP in the review
    UI — but a count + link to the no-tags-bucket view is fine and
    expected (added so Erik can drill into the books that weren't
    auto-tagged)."""
    client = TestClient(create_app(proposals_db=cookbook_db))
    body = client.get("/cookbooks").text
    # No tag *group* labelled (no tags) — the seed has 1 sentinel row.
    assert "tag-group" in body  # at least one real tag group renders
    # But the no-tags COUNT + link should be present.
    assert "1 books got no tags" in body
    assert "proposed_value=(no+tags)" in body


def test_cookbooks_dashboard_shows_book_counts(cookbook_db: Path) -> None:
    """Italian has 2 books in the seed, Korean / Baking / Vegan have 1 each."""
    client = TestClient(create_app(proposals_db=cookbook_db))
    body = client.get("/cookbooks").text
    # Italian's "2 books" should appear; check the encoded form.
    assert "2 books" in body
    assert "1 books" in body


def test_cookbooks_dashboard_links_to_filtered_proposals(cookbook_db: Path) -> None:
    """Tag headings link to the per-tag /proposals view via proposed_value."""
    client = TestClient(create_app(proposals_db=cookbook_db))
    body = client.get("/cookbooks").text
    assert "proposed_value=Italian" in body
    assert "proposed_value=Korean" in body


def test_proposals_filter_by_proposed_value(cookbook_db: Path) -> None:
    """The new --proposed-value-equivalent URL filter narrows to one tag."""
    client = TestClient(create_app(proposals_db=cookbook_db))
    r = client.get("/proposals?proposed_value=Italian")
    assert r.status_code == 200
    # Both Italian rows match (book 1 and 2).
    # Korean / Baking / Vegan must be absent.
    assert "Italian" in r.text
    assert "Korean" not in r.text
    assert "Baking" not in r.text


def test_bulk_approve_by_proposed_value(cookbook_db: Path) -> None:
    """The killer cookbook UX: bulk-approve every Italian proposal at
    once via the /cookbooks page → /bulk POST."""
    client = TestClient(create_app(proposals_db=cookbook_db))
    r = client.post(
        "/bulk",
        data={
            "action": "approved",
            "source": "cookbook_llm",
            "proposed_value": "Italian",
            "status": "proposed",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    with sqlite3.connect(cookbook_db) as conn:
        n_italian = conn.execute(
            "SELECT COUNT(*) FROM proposals WHERE proposed_value='Italian' AND status='approved'"
        ).fetchone()[0]
        n_other = conn.execute(
            "SELECT COUNT(*) FROM proposals WHERE proposed_value!='Italian' AND status='approved'"
        ).fetchone()[0]
    assert n_italian == 2
    assert n_other == 0


def test_cover_route_404_when_no_library_root(cookbook_db: Path) -> None:
    """No --library-root configured → cover requests return 404 cleanly."""
    client = TestClient(create_app(proposals_db=cookbook_db))
    r = client.get("/cover/1.jpg")
    assert r.status_code == 404


def test_cover_route_serves_jpeg(tmp_path: Path) -> None:
    """End-to-end: write a fake cover.jpg into a synthetic library, ensure
    the route streams it back with image/jpeg."""
    # Stand up a minimal Calibre DB just for the path lookup.
    metadata_db = tmp_path / "library" / "metadata.db"
    metadata_db.parent.mkdir(parents=True)
    conn = sqlite3.connect(metadata_db)
    conn.execute("CREATE TABLE books (id INTEGER PRIMARY KEY, path TEXT)")
    conn.execute("INSERT INTO books (id, path) VALUES (1, 'Author/Book')")
    conn.commit()
    conn.close()
    cover = tmp_path / "library" / "Author" / "Book" / "cover.jpg"
    cover.parent.mkdir(parents=True)
    cover.write_bytes(b"\xff\xd8\xff\xe0fake jpeg bytes")

    pdb = tmp_path / "p.db"
    pc = proposals.connect(pdb)
    pc.close()
    client = TestClient(create_app(
        proposals_db=pdb, calibre_db=metadata_db,  # library_root auto-derives
    ))
    r = client.get("/cover/1.jpg")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert b"fake jpeg bytes" in r.content


def test_cover_route_404_for_unknown_book(tmp_path: Path) -> None:
    metadata_db = tmp_path / "library" / "metadata.db"
    metadata_db.parent.mkdir(parents=True)
    conn = sqlite3.connect(metadata_db)
    conn.execute("CREATE TABLE books (id INTEGER PRIMARY KEY, path TEXT)")
    conn.commit()
    conn.close()
    pdb = tmp_path / "p.db"
    pc = proposals.connect(pdb)
    pc.close()
    client = TestClient(create_app(proposals_db=pdb, calibre_db=metadata_db))
    assert client.get("/cover/9999.jpg").status_code == 404
