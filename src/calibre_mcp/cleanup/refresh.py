"""Orchestrate a propose-only refresh pass over the cleanup pipeline.

Re-runs Phase 1 (OPF miner), Phase 3b (tag normalizer), and Phase 4 (live
tag sweep) so that newly-added books generate fresh proposals and any
post-fix classifier improvements catch tags missed by earlier sweeps.
Optionally scope Phase 1 to ``book_id >= since_book_id`` so re-walking the
full library isn't required when only a handful of new books matter.

Phase 2 (Goodreads ISBN lookup) is **not** included here. It requires
ISBNs to be present in Calibre's ``identifiers`` table (not just proposed),
so the natural workflow is: refresh → review → apply → run Goodreads
separately. Including it here would silently skip the new books.

Apply is also intentionally separate. The library is read-only for refresh;
applying needs the Calibre-Web stop + rw remount dance, which the user
drives. ``refresh`` is the safe scouting pass.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from calibre_mcp.cleanup import (
    calibre_reader,
    live_tag_sweep,
    miner,
    opf_parser,
    proposals,
    tag_normalizer,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PhaseResult:
    """Counts from one phase of a refresh run.

    ``proposals_emitted`` is what the phase **decided** to write (e.g. the
    number of tag.delete + tag.merge classifications). It is **not** the
    count of newly-inserted DB rows — re-runs on the same state dedupe
    via the propose-queue's unique index, but the phase still classifies
    everything. To see how many rows were actually new, count proposals
    attached to the phase's run_id (or use ``list_new_proposals``)."""

    name: str
    elapsed_sec: float
    proposals_emitted: int
    notes: str = ""


@dataclass(frozen=True, slots=True)
class RefreshSummary:
    phase1: PhaseResult           # OPF miner
    phase3b: PhaseResult          # tag normalizer
    phase4: PhaseResult           # live tag sweep
    total_new_proposals: int      # sum across phases (de-duped via unique index)
    new_books_seen: int           # books with id >= since_book_id that were processed


def run(
    *,
    library_root: Path,
    metadata_db: Path,
    proposals_db: Path,
    since_book_id: int | None = None,
) -> RefreshSummary:
    """Execute Phases 1, 3b, 4 in order. All operations are read-only against
    the live library; only the propose-queue is mutated.

    ``since_book_id`` scopes Phase 1 to books with id >= the given value
    (typically the highest book_id from the previous refresh + 1). Phases
    3b and 4 always operate on the whole table — they're cheap (sub-second)
    and idempotent, and a re-run with improved rules picks up older tags
    that earlier sweeps missed."""

    library_root = Path(library_root).resolve()
    metadata_db = Path(metadata_db).resolve()
    proposals_db = Path(proposals_db).resolve()

    # ------------------------------------------------------------------
    # Phase 1: OPF miner (optionally scoped by since_book_id)
    # ------------------------------------------------------------------
    t0 = time.monotonic()
    p1_emitted = 0
    p1_errors = 0
    p1_books_parsed = 0

    with calibre_reader.open_library(metadata_db) as cal, proposals.connect(proposals_db) as pc:
        run_id = proposals.start_run(
            pc,
            source=miner.SOURCE,
            library_root=library_root,
            metadata_db_path=metadata_db,
            dry_run=False,
        )
        try:
            for book in calibre_reader.iter_books(cal):
                if since_book_id is not None and book.id < since_book_id:
                    continue
                epub_path = miner._find_epub(library_root, book.path)
                if epub_path is None:
                    continue
                try:
                    opf = opf_parser.parse_epub(epub_path)
                except Exception as exc:  # noqa: BLE001 — log + continue
                    p1_errors += 1
                    proposals.record_error(
                        pc, run_id,
                        book_id=book.id, book_path=str(epub_path), error=repr(exc),
                    )
                    continue
                if opf is None:
                    continue
                p1_books_parsed += 1
                for proposal in miner._diff(book, opf):
                    if proposals.insert_proposal(pc, run_id, proposal):
                        p1_emitted += 1
        finally:
            proposals.finish_run(
                pc, run_id,
                books_scanned=p1_books_parsed, books_with_epub=p1_books_parsed,
                books_parsed=p1_books_parsed, proposals_emitted=p1_emitted,
                errors=p1_errors,
                notes=f"refresh{f' since_book_id={since_book_id}' if since_book_id else ''}",
            )

    phase1 = PhaseResult(
        name="phase1_opf_miner",
        elapsed_sec=time.monotonic() - t0,
        proposals_emitted=p1_emitted,
        notes=f"parsed={p1_books_parsed} errors={p1_errors}",
    )

    # ------------------------------------------------------------------
    # Phase 3b: tag normalizer (auto-decide newly-proposed tag.add rows)
    # ------------------------------------------------------------------
    t0 = time.monotonic()
    n_summary = tag_normalizer.run(
        calibre_db=metadata_db, proposals_db=proposals_db, dry_run=False,
    )
    phase3b = PhaseResult(
        name="phase3b_tag_normalizer",
        elapsed_sec=time.monotonic() - t0,
        proposals_emitted=n_summary.approved + n_summary.rejected,
        notes=f"approved={n_summary.approved} rejected={n_summary.rejected} "
              f"deferred={n_summary.deferred}",
    )

    # ------------------------------------------------------------------
    # Phase 4: live tag sweep (re-classifies live tags table)
    # ------------------------------------------------------------------
    t0 = time.monotonic()
    s_summary, _ = live_tag_sweep.run(
        calibre_db=metadata_db, proposals_db=proposals_db, dry_run=False,
    )
    phase4 = PhaseResult(
        name="phase4_live_tag_sweep",
        elapsed_sec=time.monotonic() - t0,
        # Note: emitted reflects new proposals only; existing ones de-dupe
        # via the unique constraint in proposals.insert_proposal.
        proposals_emitted=s_summary.delete + s_summary.merge,
        notes=f"delete={s_summary.delete} merge={s_summary.merge} "
              f"keep={s_summary.keep}",
    )

    return RefreshSummary(
        phase1=phase1,
        phase3b=phase3b,
        phase4=phase4,
        total_new_proposals=phase1.proposals_emitted + phase3b.proposals_emitted + phase4.proposals_emitted,
        new_books_seen=p1_books_parsed,
    )


def list_new_proposals(
    proposals_db: Path,
    *,
    since_minutes: int = 10,
) -> list[dict]:
    """Pull proposals attached to runs that completed within the last
    ``since_minutes``. Used by the CLI to render a "what's new" summary
    after a refresh.

    NB: SQLite's ``datetime('now', '-N minutes')`` returns a string without
    a timezone suffix while propose-queue rows store ISO timestamps with
    ``+00:00``. To compare safely we filter on integer-seconds-since-epoch
    via ``strftime('%s', ...)``.
    """
    pc = proposals.connect(proposals_db)
    try:
        rows = pc.execute(
            """
            SELECT p.book_id, p.field, p.calibre_value, p.proposed_value,
                   p.status, p.source, r.id AS run_id
            FROM proposals p
            JOIN miner_runs r ON r.id = p.run_id
            WHERE r.completed_at IS NOT NULL
              AND CAST(strftime('%s', r.completed_at) AS INTEGER)
                > CAST(strftime('%s', 'now', ?) AS INTEGER)
            ORDER BY r.id, p.book_id, p.field
            """,
            (f"-{since_minutes} minutes",),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        pc.close()
