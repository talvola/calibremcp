"""Shared fixtures for cleanup unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

_SNAPSHOT_DIR = Path(__file__).resolve().parents[3] / ".cache" / "calibre-snapshot"


@pytest.fixture(scope="session")
def epub_samples_dir() -> Path:
    """Directory of sampled EPUBs from Erik's library. Skipped if absent."""
    path = _SNAPSHOT_DIR / "epub-samples"
    if not path.is_dir():
        pytest.skip(f"EPUB sample fixtures not present at {path}")
    return path


@pytest.fixture(scope="session")
def snapshot_db() -> Path:
    """Snapshot of Calibre's metadata.db. Skipped if absent."""
    path = _SNAPSHOT_DIR / "metadata.db"
    if not path.is_file():
        pytest.skip(f"metadata.db snapshot not present at {path}")
    return path
