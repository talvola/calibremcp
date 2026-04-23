"""Integration-ish tests that exercise the OPF parser against real sampled
EPUBs copied from Erik's library.

These tests are skipped if the snapshot isn't present, so they run for Erik
locally but won't break CI."""

from __future__ import annotations

from pathlib import Path

import pytest

from calibre_mcp.cleanup.opf_parser import parse_epub


@pytest.mark.integration
def test_every_sample_parses_without_crashing(epub_samples_dir: Path) -> None:
    """Every sampled EPUB should either produce an OpfMetadata or return None
    (never raise). This is the 'doesn't crash on real data' smoke test."""
    epubs = sorted(epub_samples_dir.glob("*.epub"))
    assert epubs, f"No EPUBs found in {epub_samples_dir}"
    failures: list[tuple[Path, Exception]] = []
    for f in epubs:
        try:
            parse_epub(f)  # return value may be None; we only check no-raise.
        except Exception as exc:  # noqa: BLE001 — we want to surface any parser crash
            failures.append((f, exc))
    assert not failures, f"parser raised on: {failures}"


@pytest.mark.integration
def test_at_least_one_sample_produces_isbn(epub_samples_dir: Path) -> None:
    """Our sampling includes books with ISBN=>1 identifiers, so we expect at
    least one valid ISBN extraction from the bundle."""
    isbn_hits = 0
    for f in sorted(epub_samples_dir.glob("*.epub")):
        opf = parse_epub(f)
        if opf is not None and opf.isbn:
            isbn_hits += 1
    assert isbn_hits >= 1, "expected at least one OPF with a validated ISBN"


@pytest.mark.integration
def test_at_least_one_sample_produces_real_publisher(epub_samples_dir: Path) -> None:
    """Same shape as above for real (non-sentinel) publishers."""
    for f in sorted(epub_samples_dir.glob("*.epub")):
        opf = parse_epub(f)
        if opf is not None and opf.publisher:
            assert opf.publisher.lower() not in {
                "manybooks.net", "project gutenberg", "unknown", "gutenberg",
            }
            return
    pytest.fail("expected at least one OPF with a real publisher")
