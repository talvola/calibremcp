"""End-to-end test for the OPF miner.

Builds a mini Calibre library on disk from the sampled EPUBs and runs the
miner against a copy of the snapshot metadata.db. Verifies that the miner
produces sensible proposals without crashing on real-world OPF variance.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from calibre_mcp.cleanup import miner, proposals


def _seed_mini_library(
    *,
    epub_samples_dir: Path,
    snapshot_db: Path,
    library: Path,
    metadata_db: Path,
) -> int:
    """Place each sample EPUB at its real Calibre-library path under ``library``.

    Returns the number of sample EPUBs successfully matched to a book in the
    snapshot metadata.db."""
    shutil.copy(snapshot_db, metadata_db)
    conn = sqlite3.connect(metadata_db)
    conn.row_factory = sqlite3.Row
    try:
        seeded = 0
        for sample in sorted(epub_samples_dir.glob("*.epub")):
            stem = sample.stem  # filename without .epub
            row = conn.execute(
                """
                SELECT b.id, b.path
                FROM books b
                JOIN data d ON d.book = b.id
                WHERE d.format = 'EPUB' AND d.name = ?
                LIMIT 1
                """,
                (stem,),
            ).fetchone()
            if row is None:
                continue
            target_dir = library / row["path"]
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(sample, target_dir / sample.name)
            seeded += 1
    finally:
        conn.close()
    return seeded


@pytest.mark.integration
def test_miner_end_to_end(tmp_path: Path, epub_samples_dir: Path, snapshot_db: Path) -> None:
    library = tmp_path / "library"
    library.mkdir()
    metadata_db = tmp_path / "metadata.db"

    seeded = _seed_mini_library(
        epub_samples_dir=epub_samples_dir,
        snapshot_db=snapshot_db,
        library=library,
        metadata_db=metadata_db,
    )
    assert seeded >= 5, (
        f"expected at least 5 sampled EPUBs to match the snapshot DB, got {seeded}. Samples in {epub_samples_dir}"
    )

    proposals_db = tmp_path / "proposals.db"
    summary = miner.run(
        library_root=library,
        metadata_db=metadata_db,
        proposals_db=proposals_db,
    )

    # Miner crossed the whole library; only our seeded books should have EPUBs.
    assert summary.books_with_epub == seeded
    assert summary.books_parsed >= seeded - 1  # allow one unparseable tolerance
    assert summary.errors == 0
    # Given the archetypes in our sample set (commercial + pulp), at least
    # *some* proposals should land.
    assert summary.proposals_emitted > 0

    # Re-running the miner is a no-op: same (book, field, source, value)
    # tuples already exist, so no new rows should get written.
    first_emit = summary.proposals_emitted
    second = miner.run(
        library_root=library,
        metadata_db=metadata_db,
        proposals_db=proposals_db,
    )
    assert second.proposals_emitted == 0, (
        f"re-run emitted {second.proposals_emitted} new proposals "
        f"(first run had {first_emit}); re-runs must be idempotent"
    )


@pytest.mark.integration
def test_dry_run_does_not_write(tmp_path: Path, epub_samples_dir: Path, snapshot_db: Path) -> None:
    library = tmp_path / "library"
    library.mkdir()
    metadata_db = tmp_path / "metadata.db"
    _seed_mini_library(
        epub_samples_dir=epub_samples_dir,
        snapshot_db=snapshot_db,
        library=library,
        metadata_db=metadata_db,
    )
    proposals_db = tmp_path / "proposals.db"
    summary = miner.run(
        library_root=library,
        metadata_db=metadata_db,
        proposals_db=proposals_db,
        dry_run=True,
    )
    # We still count what *would* be proposed during a dry run, so that
    # `miner report --dry-run` is useful.
    assert summary.proposals_emitted > 0
    # But the proposals table must remain empty.
    with proposals.connect(proposals_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM proposals").fetchone()[0] == 0
        # The miner_runs row should exist with dry_run=1.
        row = conn.execute("SELECT dry_run, proposals_emitted FROM miner_runs").fetchone()
        assert row["dry_run"] == 1
        assert row["proposals_emitted"] == summary.proposals_emitted
