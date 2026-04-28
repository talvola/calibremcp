"""Phase 7a: normalize Perry Rhodan titles and series indices.

Every book in the library follows the pattern::

    Perry Rhodan – 01 – Perry Rhodan ({NNN}[letter]) {Real Title}

where ``NNN`` is a zero-padded absolute book number and the optional
``letter`` (a, b, c …) marks a half-issue split.  All 133 books currently
have ``series_index=1.0`` (import artifact); the correct index is ``NNN``
(plus a fractional part for lettered variants: a→.1, b→.2, …).

This module emits two proposals per book:

* ``title``        — strip the ``Perry Rhodan – NN – Perry Rhodan (NNN) ``
                     prefix, leaving just the real title.
* ``series_index`` — set to the parsed numeric index.

The ``series`` field is already ``"Perry Rhodan"`` for every book and is
left untouched.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from calibre_mcp.cleanup import calibre_reader, proposals
from calibre_mcp.cleanup.proposals import Proposal

log = logging.getLogger(__name__)

SOURCE = "perry_rhodan"

# Matches: Perry Rhodan – 01 – Perry Rhodan (001) Enterprise Stardust
#          Perry Rhodan – 01 – Perry Rhodan (002a) The Radiant Dome
# Uses both em-dash (–) and plain hyphen as separators for robustness.
_TITLE_RE = re.compile(
    r"^Perry Rhodan\s*[–\-]\s*\d+\s*[–\-]\s*Perry Rhodan\s*\((\d+)([a-z]?)\)\s+(.+)$",
    re.IGNORECASE,
)

# Matches: Perry Rhodan – 01 – Perry Rhodan Atlan (01) Spider Desert
#          Perry Rhodan – 01 – Perry Rhodan Atlan (2b) Menace of Atomigeddon
_ATLAN_RE = re.compile(
    r"^Perry Rhodan\s*[–\-]\s*\d+\s*[–\-]\s*Perry Rhodan\s+Atlan\s*\((\d+)([a-z]?)\)\s+(.+)$",
    re.IGNORECASE,
)

SERIES_MAIN = "Perry Rhodan"
SERIES_ATLAN = "Perry Rhodan: Atlan"


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParsedBook:
    book_id: int
    raw_title: str
    current_index: float
    real_title: str
    book_number: int      # integer part of the series position
    letter: str           # "" or "a"/"b"/...
    new_index: float
    series: str = SERIES_MAIN  # "Perry Rhodan" or "Perry Rhodan: Atlan"


@dataclass(frozen=True, slots=True)
class RunSummary:
    examined: int
    skipped_no_match: int
    skipped_already_proposed: int
    proposals_emitted: int


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _letter_decimal(letter: str) -> float:
    """Convert a/b/c… to .1/.2/.3… — used for half-issue splits."""
    if not letter:
        return 0.0
    return (ord(letter.lower()) - ord("a") + 1) / 10.0


def _parse_with_re(
    pattern: re.Pattern[str], raw: str
) -> tuple[str, int, str, float] | None:
    m = pattern.match(raw.strip())
    if not m:
        return None
    num_str, letter = m.group(1), m.group(2).lower()
    real_title = m.group(3).strip()
    if real_title.startswith("- "):
        real_title = real_title[2:].strip()
    book_number = int(num_str)
    index = float(book_number) + _letter_decimal(letter)
    return real_title, book_number, letter, index


def parse_title(raw: str) -> tuple[str, int, str, float] | None:
    """Parse a main-series Perry Rhodan title.

    Returns ``(real_title, book_number, letter, series_index)`` or ``None``.
    """
    return _parse_with_re(_TITLE_RE, raw)


def parse_atlan_title(raw: str) -> tuple[str, int, str, float] | None:
    """Parse a Perry Rhodan Atlan sub-series title.

    Returns ``(real_title, book_number, letter, series_index)`` or ``None``.
    """
    return _parse_with_re(_ATLAN_RE, raw)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(
    *,
    calibre_db: Path,
    proposals_db: Path,
    dry_run: bool = False,
) -> tuple[RunSummary, list[ParsedBook]]:
    """Scan Perry Rhodan books, parse titles, emit title + series_index proposals."""
    calibre_db = Path(calibre_db).resolve()
    proposals_db = Path(proposals_db).resolve()

    cal = calibre_reader.open_readonly(calibre_db)
    pc = proposals.connect(proposals_db)

    run_id = proposals.start_run(
        pc,
        source=SOURCE,
        library_root=calibre_db.parent,
        metadata_db_path=calibre_db,
        dry_run=dry_run,
    )

    parsed_books: list[ParsedBook] = []
    skipped_no_match = 0
    skipped_already = 0
    emitted = 0
    rows: list = []

    try:
        rows = cal.execute("""
            SELECT b.id, b.title, COALESCE(b.series_index, 1.0) AS series_index
            FROM books b
            WHERE b.title LIKE 'Perry Rhodan%'
            ORDER BY b.title
        """).fetchall()

        for row in rows:
            book_id = int(row["id"])
            raw_title = row["title"]
            current_index = float(row["series_index"])

            parsed = parse_title(raw_title)
            target_series = SERIES_MAIN
            if parsed is None:
                parsed = parse_atlan_title(raw_title)
                if parsed is not None:
                    target_series = SERIES_ATLAN

            if parsed is None:
                log.debug("No match: %r", raw_title)
                skipped_no_match += 1
                continue

            real_title, book_number, letter, new_index = parsed
            pb = ParsedBook(
                book_id=book_id,
                raw_title=raw_title,
                current_index=current_index,
                real_title=real_title,
                book_number=book_number,
                letter=letter,
                new_index=new_index,
                series=target_series,
            )
            parsed_books.append(pb)

            if dry_run:
                continue

            notes = json.dumps({
                "book_number": book_number,
                "letter": letter or None,
                "series": target_series,
            })

            # title proposal
            title_inserted = proposals.insert_proposal(
                pc, run_id,
                Proposal(
                    book_id=book_id,
                    field="title",
                    calibre_value=raw_title,
                    proposed_value=real_title,
                    source=SOURCE,
                    confidence=1.0,
                    notes=notes,
                ),
            )

            # series proposal (only for Atlan sub-series)
            series_inserted = False
            if target_series != SERIES_MAIN:
                series_inserted = proposals.insert_proposal(
                    pc, run_id,
                    Proposal(
                        book_id=book_id,
                        field="series",
                        calibre_value=SERIES_MAIN,
                        proposed_value=target_series,
                        source=SOURCE,
                        confidence=1.0,
                        notes=notes,
                    ),
                )

            # series_index proposal (only if actually changing)
            index_inserted = False
            if abs(new_index - current_index) > 1e-9:
                index_inserted = proposals.insert_proposal(
                    pc, run_id,
                    Proposal(
                        book_id=book_id,
                        field="series_index",
                        calibre_value=str(current_index),
                        proposed_value=str(new_index),
                        source=SOURCE,
                        confidence=1.0,
                        notes=notes,
                    ),
                )

            if title_inserted or series_inserted or index_inserted:
                emitted += 1
            else:
                skipped_already += 1

    finally:
        proposals.finish_run(
            pc, run_id,
            books_scanned=len(rows),
            books_with_epub=0,
            books_parsed=len(parsed_books),
            proposals_emitted=emitted,
            errors=0,
            notes=f"matched={len(parsed_books)} no_match={skipped_no_match}",
        )
        cal.close()
        pc.close()

    return (
        RunSummary(
            examined=len(rows),
            skipped_no_match=skipped_no_match,
            skipped_already_proposed=skipped_already,
            proposals_emitted=emitted,
        ),
        parsed_books,
    )
