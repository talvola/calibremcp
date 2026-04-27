"""Unit tests for the cookbook orchestrator.

The LLM call is mocked via a fake client that returns canned
``CookbookTags`` so we test the orchestration logic — bucket iteration,
proposal emission, dedupe, sentinel insertion for empty results — without
hitting the API.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from calibre_mcp.cleanup import cookbook_tagger, cookbooks, proposals
from calibre_mcp.cleanup.cookbook_tagger import BookContext, CookbookTags

# ---------------------------------------------------------------------------
# Fake LLM client + library setup
# ---------------------------------------------------------------------------


def _build_calibre_db(
    library: Path,
    books: list[tuple[int, str, list[str], str | None]],
    cookbooks_tag_books: list[int],
) -> Path:
    """Build a synthetic Calibre metadata.db with just the schema bits the
    orchestrator touches.

    ``books`` is ``[(id, title, authors_list, description)]``.
    ``cookbooks_tag_books`` lists book ids that should carry the
    ``Cookbooks`` tag (the bucket query filters on this).
    """
    library.mkdir(parents=True, exist_ok=True)
    db = library / "metadata.db"
    conn = sqlite3.connect(db)
    try:
        conn.executescript(
            """
            CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT, path TEXT);
            CREATE TABLE authors (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT);
            CREATE TABLE books_authors_link (id INTEGER PRIMARY KEY AUTOINCREMENT, book INTEGER, author INTEGER);
            CREATE TABLE comments (id INTEGER PRIMARY KEY, book INTEGER, text TEXT);
            CREATE TABLE tags (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT);
            CREATE TABLE books_tags_link (id INTEGER PRIMARY KEY AUTOINCREMENT, book INTEGER, tag INTEGER);
            """
        )
        # Insert the Cookbooks tag once.
        cur = conn.execute("INSERT INTO tags (name) VALUES ('Cookbooks')")
        cookbooks_tag_id = cur.lastrowid

        for book_id, title, authors, description in books:
            book_dir = f"path_{book_id}"
            conn.execute(
                "INSERT INTO books (id, title, path) VALUES (?, ?, ?)",
                (book_id, title, book_dir),
            )
            for name in authors:
                cur = conn.execute("INSERT INTO authors (name) VALUES (?)", (name,))
                conn.execute(
                    "INSERT INTO books_authors_link (book, author) VALUES (?, ?)",
                    (book_id, cur.lastrowid),
                )
            if description is not None:
                conn.execute(
                    "INSERT INTO comments (book, text) VALUES (?, ?)",
                    (book_id, description),
                )
            if book_id in cookbooks_tag_books:
                conn.execute(
                    "INSERT INTO books_tags_link (book, tag) VALUES (?, ?)",
                    (book_id, cookbooks_tag_id),
                )
        conn.commit()
    finally:
        conn.close()
    return db


class _FakeMessages:
    """Stand-in for ``client.messages`` that returns canned responses
    keyed by something stable in the user content (title)."""

    def __init__(self, responses_by_title: dict[str, CookbookTags]) -> None:
        self.responses_by_title = responses_by_title
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        # Find the title in the user message text block.
        title = ""
        for block in kwargs["messages"][0]["content"]:
            if block.get("type") == "text":
                for line in block["text"].splitlines():
                    if line.startswith("Title: "):
                        title = line.removeprefix("Title: ")
                        break
        if title not in self.responses_by_title:
            raise AssertionError(f"no canned response for {title!r}")
        tags = self.responses_by_title[title]
        # Mimic the parsed_output attribute the SDK returns.
        return type("Resp", (), {"parsed_output": tags})()


class _FakeClient:
    def __init__(self, responses: dict[str, CookbookTags]) -> None:
        self.messages = _FakeMessages(responses)


@pytest.fixture
def fake_anthropic(monkeypatch: pytest.MonkeyPatch):
    """Patch anthropic.Anthropic so cookbooks.run() picks up our fake."""
    canned: dict[str, CookbookTags] = {}
    fake_client = _FakeClient(canned)

    def _factory(*_args, **_kwargs):
        return fake_client

    monkeypatch.setattr("calibre_mcp.cleanup.cookbooks.anthropic.Anthropic", _factory)
    return canned, fake_client


# ---------------------------------------------------------------------------
# _iter_bucket — bucket-tag vs book-id scoping
# ---------------------------------------------------------------------------


def test_iter_bucket_filters_by_tag(tmp_path: Path) -> None:
    """Only books carrying ``Cookbooks`` should be yielded."""
    db = _build_calibre_db(
        tmp_path / "lib",
        books=[
            (1, "A", ["x"], "d1"),
            (2, "B", ["y"], "d2"),
            (3, "C", [], None),  # not in cookbooks_tag_books → excluded
        ],
        cookbooks_tag_books=[1, 2],
    )
    from calibre_mcp.cleanup import calibre_reader
    cal = calibre_reader.open_readonly(db)
    try:
        ids = sorted(b.book_id for b in cookbooks._iter_bucket(cal, tmp_path / "lib"))
    finally:
        cal.close()
    assert ids == [1, 2]


def test_iter_bucket_book_ids_overrides_tag_filter(tmp_path: Path) -> None:
    """When ``book_ids`` is given we yield those regardless of bucket
    tag — the proof-of-concept use case (the 5 new books before they
    have any tags)."""
    db = _build_calibre_db(
        tmp_path / "lib",
        books=[(1, "A", [], None), (2, "B", [], None), (3, "C", [], None)],
        cookbooks_tag_books=[1],  # only book 1 is in the bucket
    )
    from calibre_mcp.cleanup import calibre_reader
    cal = calibre_reader.open_readonly(db)
    try:
        ids = sorted(
            b.book_id for b in cookbooks._iter_bucket(
                cal, tmp_path / "lib", book_ids=[2, 3],
            )
        )
    finally:
        cal.close()
    assert ids == [2, 3]  # respects book_ids, ignores Cookbooks tag


def test_iter_bucket_yields_authors_tuple(tmp_path: Path) -> None:
    db = _build_calibre_db(
        tmp_path / "lib",
        books=[(1, "Multi-Author", ["A", "B", "C"], "desc")],
        cookbooks_tag_books=[1],
    )
    from calibre_mcp.cleanup import calibre_reader
    cal = calibre_reader.open_readonly(db)
    try:
        book = next(cookbooks._iter_bucket(cal, tmp_path / "lib"))
    finally:
        cal.close()
    assert book.authors == ("A", "B", "C")


# ---------------------------------------------------------------------------
# _emit_proposals — happy path, sentinel for empty, dedupe
# ---------------------------------------------------------------------------


def test_emit_proposals_inserts_one_per_tag(tmp_path: Path) -> None:
    pdb = tmp_path / "p.db"
    pc = proposals.connect(pdb)
    run_id = proposals.start_run(
        pc, source=cookbook_tagger.SOURCE,
        library_root=tmp_path, metadata_db_path=tmp_path / "metadata.db",
        dry_run=False,
    )
    book = BookContext(book_id=42, title="X", authors=(), description=None, cover_path=None)
    tags = CookbookTags(cuisine=["Italian"], technique=["Baking", "Bread"], dietary=[], confidence="high")
    n = cookbooks._emit_proposals(pc, run_id, book, tags)
    assert n == 3  # 1 cuisine + 2 technique = 3 proposals

    rows = pc.execute(
        "SELECT proposed_value, status, source FROM proposals WHERE book_id=42 ORDER BY proposed_value"
    ).fetchall()
    assert [r["proposed_value"] for r in rows] == ["Baking", "Bread", "Italian"]
    assert all(r["status"] == "proposed" for r in rows)
    pc.close()


def test_emit_proposals_inserts_sentinel_for_empty(tmp_path: Path) -> None:
    """Honest-empty case: a Hawaiian-fusion book gets zero tags. We still
    record a (status='rejected') sentinel so re-runs skip the book and
    don't pay the LLM cost again."""
    pdb = tmp_path / "p.db"
    pc = proposals.connect(pdb)
    run_id = proposals.start_run(
        pc, source=cookbook_tagger.SOURCE,
        library_root=tmp_path, metadata_db_path=tmp_path / "metadata.db",
        dry_run=False,
    )
    book = BookContext(book_id=42, title="Hawaiian", authors=(), description=None, cover_path=None)
    tags = CookbookTags(cuisine=[], technique=[], dietary=[], confidence="medium")
    n = cookbooks._emit_proposals(pc, run_id, book, tags)
    assert n == 1
    row = pc.execute("SELECT proposed_value, status FROM proposals WHERE book_id=42").fetchone()
    assert row["proposed_value"] == cookbooks._NO_TAGS_SENTINEL
    assert row["status"] == "rejected"  # never picked up by apply
    pc.close()


def test_emit_proposals_is_idempotent(tmp_path: Path) -> None:
    pdb = tmp_path / "p.db"
    pc = proposals.connect(pdb)
    run_id = proposals.start_run(
        pc, source=cookbook_tagger.SOURCE,
        library_root=tmp_path, metadata_db_path=tmp_path / "metadata.db",
        dry_run=False,
    )
    book = BookContext(book_id=42, title="X", authors=(), description=None, cover_path=None)
    tags = CookbookTags(cuisine=["Italian"], technique=[], dietary=[], confidence="high")
    cookbooks._emit_proposals(pc, run_id, book, tags)
    cookbooks._emit_proposals(pc, run_id, book, tags)  # second call — dedupe
    n = pc.execute("SELECT COUNT(*) AS n FROM proposals WHERE book_id=42").fetchone()["n"]
    assert n == 1
    pc.close()


# ---------------------------------------------------------------------------
# _already_tagged — re-run skip detection
# ---------------------------------------------------------------------------


def test_already_tagged_detects_existing(tmp_path: Path) -> None:
    pdb = tmp_path / "p.db"
    pc = proposals.connect(pdb)
    run_id = proposals.start_run(
        pc, source=cookbook_tagger.SOURCE,
        library_root=tmp_path, metadata_db_path=tmp_path / "metadata.db",
        dry_run=False,
    )
    book = BookContext(book_id=42, title="X", authors=(), description=None, cover_path=None)
    cookbooks._emit_proposals(
        pc, run_id, book,
        CookbookTags(cuisine=["Italian"], technique=[], dietary=[], confidence="high"),
    )
    assert cookbooks._already_tagged(pc, 42) is True
    assert cookbooks._already_tagged(pc, 99) is False
    pc.close()


def test_already_tagged_returns_true_for_sentinel_book(tmp_path: Path) -> None:
    """The sentinel row counts as 'tagged' — that's the whole point."""
    pdb = tmp_path / "p.db"
    pc = proposals.connect(pdb)
    run_id = proposals.start_run(
        pc, source=cookbook_tagger.SOURCE,
        library_root=tmp_path, metadata_db_path=tmp_path / "metadata.db",
        dry_run=False,
    )
    book = BookContext(book_id=42, title="Hawaiian", authors=(), description=None, cover_path=None)
    cookbooks._emit_proposals(
        pc, run_id, book,
        CookbookTags(cuisine=[], technique=[], dietary=[], confidence="medium"),
    )
    assert cookbooks._already_tagged(pc, 42) is True
    pc.close()


# ---------------------------------------------------------------------------
# run() — full integration with mocked LLM
# ---------------------------------------------------------------------------


def test_run_tags_full_bucket(tmp_path: Path, fake_anthropic) -> None:
    canned, _ = fake_anthropic
    canned["Italian Cookbook"] = CookbookTags(cuisine=["Italian"], technique=[], dietary=[], confidence="high")
    canned["Korean Cookbook"] = CookbookTags(cuisine=["Korean"], technique=[], dietary=[], confidence="high")

    db = _build_calibre_db(
        tmp_path / "lib",
        books=[
            (1, "Italian Cookbook", ["A"], "Italian recipes"),
            (2, "Korean Cookbook", ["B"], "Korean recipes"),
        ],
        cookbooks_tag_books=[1, 2],
    )

    summary = cookbooks.run(
        library_root=tmp_path / "lib",
        metadata_db=db,
        proposals_db=tmp_path / "p.db",
    )
    assert summary.examined == 2
    assert summary.tagged == 2
    assert summary.skipped_already_tagged == 0
    assert summary.api_errors == 0
    assert summary.proposals_emitted == 2  # one tag each


def test_run_skips_already_tagged_books(tmp_path: Path, fake_anthropic) -> None:
    """Re-run on the same library should not re-call the LLM for any
    previously-tagged book."""
    canned, fake_client = fake_anthropic
    canned["A"] = CookbookTags(cuisine=["Italian"], technique=[], dietary=[], confidence="high")
    canned["B"] = CookbookTags(cuisine=["Korean"], technique=[], dietary=[], confidence="high")

    db = _build_calibre_db(
        tmp_path / "lib",
        books=[(1, "A", [], None), (2, "B", [], None)],
        cookbooks_tag_books=[1, 2],
    )
    cookbooks.run(
        library_root=tmp_path / "lib", metadata_db=db, proposals_db=tmp_path / "p.db",
    )
    n_calls_after_first = len(fake_client.messages.calls)
    summary2 = cookbooks.run(
        library_root=tmp_path / "lib", metadata_db=db, proposals_db=tmp_path / "p.db",
    )
    # No new calls — both books were skipped.
    assert len(fake_client.messages.calls) == n_calls_after_first
    assert summary2.skipped_already_tagged == 2
    assert summary2.tagged == 0


def test_run_retag_forces_recall(tmp_path: Path, fake_anthropic) -> None:
    """``skip_already_tagged=False`` (the --retag flag) re-calls the LLM
    even when proposals already exist — needed after taxonomy edits."""
    canned, fake_client = fake_anthropic
    canned["A"] = CookbookTags(cuisine=["Italian"], technique=[], dietary=[], confidence="high")

    db = _build_calibre_db(
        tmp_path / "lib", books=[(1, "A", [], None)], cookbooks_tag_books=[1],
    )
    cookbooks.run(
        library_root=tmp_path / "lib", metadata_db=db, proposals_db=tmp_path / "p.db",
    )
    cookbooks.run(
        library_root=tmp_path / "lib", metadata_db=db, proposals_db=tmp_path / "p.db",
        skip_already_tagged=False,
    )
    # Two API calls: original + retag.
    assert len(fake_client.messages.calls) == 2


def test_run_handles_api_errors_gracefully(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One book throws; the run continues and counts the error."""
    import anthropic

    class FailingMessages:
        def parse(self, **_kwargs):
            raise anthropic.APIError("boom", request=None, body=None)

    def _fake(*_a, **_k):
        return type("C", (), {"messages": FailingMessages()})()

    monkeypatch.setattr("calibre_mcp.cleanup.cookbooks.anthropic.Anthropic", _fake)

    db = _build_calibre_db(
        tmp_path / "lib", books=[(1, "A", [], None)], cookbooks_tag_books=[1],
    )
    summary = cookbooks.run(
        library_root=tmp_path / "lib", metadata_db=db, proposals_db=tmp_path / "p.db",
    )
    assert summary.api_errors == 1
    assert summary.tagged == 0
    assert summary.proposals_emitted == 0


# ---------------------------------------------------------------------------
# report_by_tag — per-cuisine review grouping
# ---------------------------------------------------------------------------


def test_report_by_tag_groups_and_excludes_sentinel(tmp_path: Path, fake_anthropic) -> None:
    canned, _ = fake_anthropic
    canned["Italian-1"] = CookbookTags(cuisine=["Italian"], technique=[], dietary=[], confidence="high")
    canned["Italian-2"] = CookbookTags(cuisine=["Italian"], technique=["Baking"], dietary=[], confidence="high")
    canned["Empty"] = CookbookTags(cuisine=[], technique=[], dietary=[], confidence="medium")

    db = _build_calibre_db(
        tmp_path / "lib",
        books=[
            (1, "Italian-1", [], None), (2, "Italian-2", [], None), (3, "Empty", [], None),
        ],
        cookbooks_tag_books=[1, 2, 3],
    )
    cookbooks.run(
        library_root=tmp_path / "lib", metadata_db=db, proposals_db=tmp_path / "p.db",
    )
    groups = cookbooks.report_by_tag(
        proposals_db=tmp_path / "p.db", metadata_db=db,
    )
    by_tag = {g.tag: g for g in groups}
    assert by_tag["Italian"].n_books == 2
    assert by_tag["Baking"].n_books == 1
    # Sentinel must NOT show up in the per-cuisine review — it's just an
    # internal skip marker.
    assert cookbooks._NO_TAGS_SENTINEL not in by_tag


def test_resolve_retag_book_ids_finds_books_with_tag(tmp_path: Path, fake_anthropic) -> None:
    """The --retag-tag flag scopes the retag to books currently carrying
    a specific tag (real or sentinel)."""
    canned, _ = fake_anthropic
    canned["A"] = CookbookTags(cuisine=["Italian"], technique=["Cocktails"], dietary=[], confidence="high")
    canned["B"] = CookbookTags(cuisine=["Italian"], technique=[], dietary=[], confidence="high")
    canned["C"] = CookbookTags(cuisine=[], technique=[], dietary=[], confidence="medium")  # → sentinel

    db = _build_calibre_db(
        tmp_path / "lib",
        books=[(1, "A", [], None), (2, "B", [], None), (3, "C", [], None)],
        cookbooks_tag_books=[1, 2, 3],
    )
    pdb = tmp_path / "p.db"
    cookbooks.run(library_root=tmp_path / "lib", metadata_db=db, proposals_db=pdb)

    # Now resolve which books match retag-tag=Cocktails: just book 1.
    pc = proposals.connect(pdb)
    matched = cookbooks._resolve_retag_book_ids(pc, ["Cocktails"])
    pc.close()
    assert matched == [1]

    # Multiple tags: union. Italian matches books 1 & 2; sentinel matches 3.
    pc = proposals.connect(pdb)
    matched_union = cookbooks._resolve_retag_book_ids(pc, ["Italian", "(no tags)"])
    pc.close()
    assert matched_union == [1, 2, 3]


def test_clear_book_proposals_removes_proposed_and_sentinel(tmp_path: Path, fake_anthropic) -> None:
    """_clear_book_proposals deletes status='proposed' + 'rejected' rows
    for one book — used before --retag writes fresh proposals so we
    don't end up with old + new tags mixed."""
    canned, _ = fake_anthropic
    canned["A"] = CookbookTags(cuisine=["Italian"], technique=["Cocktails"], dietary=[], confidence="high")
    db = _build_calibre_db(
        tmp_path / "lib", books=[(1, "A", [], None)], cookbooks_tag_books=[1],
    )
    pdb = tmp_path / "p.db"
    cookbooks.run(library_root=tmp_path / "lib", metadata_db=db, proposals_db=pdb)

    pc = proposals.connect(pdb)
    n_before = pc.execute("SELECT COUNT(*) FROM proposals WHERE book_id=1").fetchone()[0]
    deleted = cookbooks._clear_book_proposals(pc, 1)
    n_after = pc.execute("SELECT COUNT(*) FROM proposals WHERE book_id=1").fetchone()[0]
    pc.close()
    assert n_before == 2  # Italian + Cocktails
    assert deleted == 2
    assert n_after == 0


def test_clear_book_proposals_preserves_approved(tmp_path: Path, fake_anthropic) -> None:
    """Approved/applied rows represent decisions Erik already made —
    _clear_book_proposals must NOT delete them."""
    canned, _ = fake_anthropic
    canned["A"] = CookbookTags(cuisine=["Italian"], technique=[], dietary=[], confidence="high")
    db = _build_calibre_db(
        tmp_path / "lib", books=[(1, "A", [], None)], cookbooks_tag_books=[1],
    )
    pdb = tmp_path / "p.db"
    cookbooks.run(library_root=tmp_path / "lib", metadata_db=db, proposals_db=pdb)

    # Approve the Italian proposal.
    pc = proposals.connect(pdb)
    pc.execute("UPDATE proposals SET status='approved' WHERE proposed_value='Italian'")
    deleted = cookbooks._clear_book_proposals(pc, 1)
    remaining = pc.execute(
        "SELECT proposed_value, status FROM proposals WHERE book_id=1"
    ).fetchall()
    pc.close()
    assert deleted == 0
    assert len(remaining) == 1
    assert remaining[0]["status"] == "approved"


def test_run_with_retag_tags_clears_then_reretags(tmp_path: Path, fake_anthropic) -> None:
    """End-to-end: --retag-tag picks up old-tagged books, deletes their
    old proposals, and re-tags with whatever the LLM returns now (could
    be different — the whole point of the targeted retag)."""
    canned, fake_client = fake_anthropic
    canned["A"] = CookbookTags(cuisine=[], technique=["Cocktails"], dietary=[], confidence="high")
    db = _build_calibre_db(
        tmp_path / "lib", books=[(1, "A", [], None)], cookbooks_tag_books=[1],
    )
    pdb = tmp_path / "p.db"
    cookbooks.run(library_root=tmp_path / "lib", metadata_db=db, proposals_db=pdb)

    # Now the "LLM" changes its mind under the new prompt — Cocktails was
    # wrong; this is actually an Italian cookbook.
    canned["A"] = CookbookTags(cuisine=["Italian"], technique=[], dietary=[], confidence="high")

    summary = cookbooks.run(
        library_root=tmp_path / "lib", metadata_db=db, proposals_db=pdb,
        retag_tags=["Cocktails"],
    )
    assert summary.tagged == 1  # the retag re-called the LLM

    pc = proposals.connect(pdb)
    rows = pc.execute(
        "SELECT proposed_value, status FROM proposals WHERE book_id=1"
    ).fetchall()
    pc.close()
    # Old Cocktails row is gone; new Italian row is present.
    values = [r["proposed_value"] for r in rows]
    assert values == ["Italian"]


def test_run_with_retag_tags_skips_nonexistent_tag(tmp_path: Path, fake_anthropic) -> None:
    """--retag-tag for a tag no book carries → zero books examined."""
    canned, _ = fake_anthropic
    canned["A"] = CookbookTags(cuisine=["Italian"], technique=[], dietary=[], confidence="high")
    db = _build_calibre_db(
        tmp_path / "lib", books=[(1, "A", [], None)], cookbooks_tag_books=[1],
    )
    pdb = tmp_path / "p.db"
    cookbooks.run(library_root=tmp_path / "lib", metadata_db=db, proposals_db=pdb)

    summary = cookbooks.run(
        library_root=tmp_path / "lib", metadata_db=db, proposals_db=pdb,
        retag_tags=["Klingon"],
    )
    assert summary.examined == 0
    assert summary.tagged == 0


def test_report_by_tag_sorts_by_book_count_desc(tmp_path: Path, fake_anthropic) -> None:
    canned, _ = fake_anthropic
    for i in range(5):
        canned[f"It-{i}"] = CookbookTags(cuisine=["Italian"], technique=[], dietary=[], confidence="high")
    canned["K-0"] = CookbookTags(cuisine=["Korean"], technique=[], dietary=[], confidence="high")

    books = [(i, f"It-{i-1}" if i < 6 else f"K-{i-6}", [], None) for i in range(1, 7)]
    books[5] = (6, "K-0", [], None)  # last one is Korean
    db = _build_calibre_db(
        tmp_path / "lib", books=books, cookbooks_tag_books=[i for i, *_ in books],
    )
    cookbooks.run(
        library_root=tmp_path / "lib", metadata_db=db, proposals_db=tmp_path / "p.db",
    )
    groups = cookbooks.report_by_tag(
        proposals_db=tmp_path / "p.db", metadata_db=db,
    )
    # Italian (5) should sort before Korean (1).
    assert groups[0].tag == "Italian"
    assert groups[0].n_books == 5
    assert groups[1].tag == "Korean"
