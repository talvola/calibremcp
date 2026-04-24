"""Unit tests for the propose-queue SQLite layer and miner helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from calibre_mcp.cleanup import proposals
from calibre_mcp.cleanup.miner import (
    _BISAC_CODE_RE,
    _calibre_richer_or_equal,
    _isbn_equal,
    _pubdate_richer_or_equal,
)
from calibre_mcp.cleanup.proposals import Proposal


@pytest.fixture
def db(tmp_path: Path):
    conn = proposals.connect(tmp_path / "proposals.db")
    yield conn
    conn.close()


def _start_run(conn) -> int:
    return proposals.start_run(
        conn,
        source="opf",
        library_root=Path("/tmp/library"),  # noqa: S108 — dummy path stored as text; no file touched
        metadata_db_path=Path("/tmp/library/metadata.db"),  # noqa: S108
        dry_run=False,
    )


def test_schema_created_and_empty(db) -> None:
    assert proposals.summary_counts(db) == {}
    assert proposals.list_proposals(db) == []


def test_insert_and_dedupe(db) -> None:
    run_id = _start_run(db)
    p = Proposal(
        book_id=42,
        field="isbn",
        proposed_value="9780141036144",
        source="opf",
        confidence=1.0,
        book_uuid="uuid-42",
    )
    assert proposals.insert_proposal(db, run_id, p) is True
    # Same tuple a second time is a no-op — idempotent re-runs.
    assert proposals.insert_proposal(db, run_id, p) is False
    rows = proposals.list_proposals(db)
    assert len(rows) == 1


def test_conflict_proposals_get_conflict_status(db) -> None:
    run_id = _start_run(db)
    p = Proposal(
        book_id=1,
        field="isbn",
        proposed_value="9780141036144",
        source="opf",
        confidence=1.0,
        calibre_value="9780000000000",
        conflict_reason="calibre has different ISBN",
    )
    proposals.insert_proposal(db, run_id, p)
    rows = proposals.list_proposals(db, status="conflict")
    assert len(rows) == 1
    assert rows[0]["status"] == "conflict"
    assert rows[0]["conflict_reason"] == "calibre has different ISBN"


def test_summary_counts_groups_by_field_and_status(db) -> None:
    run_id = _start_run(db)
    proposals.insert_proposal(
        db,
        run_id,
        Proposal(
            book_id=1,
            field="isbn",
            proposed_value="9780141036144",
            source="opf",
            confidence=1.0,
        ),
    )
    proposals.insert_proposal(
        db,
        run_id,
        Proposal(
            book_id=2,
            field="isbn",
            proposed_value="9780000000007",
            source="opf",
            confidence=1.0,
        ),
    )
    proposals.insert_proposal(
        db,
        run_id,
        Proposal(
            book_id=1,
            field="publisher",
            proposed_value="Penguin",
            source="opf",
            confidence=0.9,
        ),
    )
    counts = proposals.summary_counts(db)
    assert counts == {
        "isbn": {"proposed": 2},
        "publisher": {"proposed": 1},
    }


def test_set_status_updates_reviewed_at(db) -> None:
    run_id = _start_run(db)
    proposals.insert_proposal(
        db,
        run_id,
        Proposal(
            book_id=1,
            field="isbn",
            proposed_value="9780141036144",
            source="opf",
            confidence=1.0,
        ),
    )
    pid = proposals.list_proposals(db)[0]["id"]
    proposals.set_status(db, pid, "approved", notes="looks good")
    row = proposals.list_proposals(db, status="approved")[0]
    assert row["status"] == "approved"
    assert row["reviewed_at"] is not None
    assert row["notes"] == "looks good"


def test_tags_add_multiple_rows_per_book(db) -> None:
    # List-valued fields rely on proposed_value in the uniqueness key —
    # multiple rows per (book, field, source) are allowed when values differ.
    run_id = _start_run(db)
    for tag in ("Science Fiction", "Space Opera", "Cthulhu"):
        ok = proposals.insert_proposal(
            db,
            run_id,
            Proposal(
                book_id=5,
                field="tags.add",
                proposed_value=tag,
                source="opf",
                confidence=0.6,
            ),
        )
        assert ok is True
    assert len(proposals.list_proposals(db, field="tags.add")) == 3


def test_invalid_status_rejected(db) -> None:
    run_id = _start_run(db)
    proposals.insert_proposal(
        db,
        run_id,
        Proposal(
            book_id=1,
            field="isbn",
            proposed_value="9780141036144",
            source="opf",
            confidence=1.0,
        ),
    )
    pid = proposals.list_proposals(db)[0]["id"]
    with pytest.raises(ValueError):
        proposals.set_status(db, pid, "fantastic")


@pytest.mark.parametrize(
    "calibre, opf, expected",
    [
        # Exact equality (case-insensitive) — not a conflict.
        ("Penguin", "Penguin", True),
        ("penguin", "Penguin", True),
        # Calibre already subsumes the OPF value — not a conflict.
        ("Scholastic Inc.", "Scholastic", True),
        ("Star Wars: Rebel Force", "Rebel Force", True),
        ("Tor Books", "Tor", True),
        # Real disagreement — should be flagged as conflict.
        ("Penguin", "Random House", False),
        ("Foundation", "Dune", False),
        # Too-short OPF token shouldn't trigger subsumption (guards against
        # e.g. matching 'NY' as if it were a meaningful publisher name).
        ("Scholastic Inc.", "In", False),
        # Empty values treated as not-subsumed.
        ("", "Penguin", False),
        ("Penguin", "", False),
    ],
)
def test_calibre_richer_or_equal(calibre: str, opf: str, expected: bool) -> None:
    assert _calibre_richer_or_equal(calibre, opf) is expected


@pytest.mark.parametrize(
    "calibre, opf, expected",
    [
        # Dashed vs bare — same ISBN, must NOT conflict.
        ("978-1-59017-595-8", "9781590175958", True),
        ("978 1 59017 595 8", "9781590175958", True),
        ("0-141-03614-1", "0141036141", True),
        # Exact match — equal.
        ("9781590175958", "9781590175958", True),
        # Different ISBNs (legitimate conflict — same book may have multiple
        # editions, reviewer should pick).
        ("9780141036144", "9780141036151", False),
        ("0451220749", "9781101007594", False),
        # A 'urn:uuid:...' accidentally stored in Calibre's isbn field is not
        # an ISBN at all — should not match a real one.
        ("urn:uuid:7655e3e8-a157-4775-89dd-b9f6408321bf", "9781250765055", False),
        # Empty / missing sides.
        ("", "9781590175958", False),
        ("9781590175958", "", False),
    ],
)
def test_isbn_equal(calibre: str, opf: str, expected: bool) -> None:
    assert _isbn_equal(calibre, opf) is expected


@pytest.mark.parametrize(
    "calibre, opf, expected",
    [
        # Calibre has more precision than OPF and agrees on the common prefix —
        # Calibre wins, should NOT be flagged as conflict.
        ("2025-11-11", "2025", True),
        ("2025-11", "2025", True),
        ("2025-11-11", "2025-11", True),
        # Equal.
        ("2025-11-11", "2025-11-11", True),
        ("2025", "2025", True),
        # OPF has more precision and agrees on prefix — this *should* be
        # proposed as an upgrade (not equal), so the "richer-or-equal"
        # predicate returns False (allowing the proposal to fire).
        ("2025", "2025-11-11", False),
        ("2025-11", "2025-11-11", False),
        # Real disagreements.
        ("2024", "2025", False),
        ("2025-11-11", "2025-11-12", False),
        ("2025-11", "2025-12", False),
        # Guard against "2025" being treated as a prefix of "20251111" —
        # we require the dash separator so partial-string matches don't
        # pass silently.
        ("20251111", "2025", False),
        # Empty sides.
        ("", "2025", False),
        ("2025", "", False),
    ],
)
def test_pubdate_richer_or_equal(calibre: str, opf: str, expected: bool) -> None:
    assert _pubdate_richer_or_equal(calibre, opf) is expected


@pytest.mark.parametrize(
    "value, is_bisac",
    [
        ("FIC045000", True),  # Fiction / Action & Adventure
        ("FIC000000", True),
        ("NON000000", True),
        ("JUV000000", True),
        # Not BISAC:
        ("Fiction", False),
        ("Science Fiction", False),
        ("FIC", False),  # too short
        ("FIC045", False),  # too few digits
        ("FIC0450000", False),  # too many digits
        ("fic045000", False),  # lowercase — not the canonical form
        ("B00XYZ1234", False),  # Amazon ASIN shape
        ("9780141036144", False),  # ISBN
    ],
)
def test_bisac_code_regex(value: str, is_bisac: bool) -> None:
    assert bool(_BISAC_CODE_RE.match(value)) is is_bisac


def test_miner_run_lifecycle(db) -> None:
    run_id = _start_run(db)
    proposals.finish_run(
        db,
        run_id,
        books_scanned=10,
        books_with_epub=9,
        books_parsed=9,
        proposals_emitted=17,
        errors=0,
    )
    row = db.execute("SELECT * FROM miner_runs WHERE id = ?", (run_id,)).fetchone()
    assert row["books_scanned"] == 10
    assert row["proposals_emitted"] == 17
    assert row["completed_at"] is not None
