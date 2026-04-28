"""Phase 7b-1: normalize Star Trek titles and, for series-less books, assign series+index.

Titles follow the pattern::

    Star Trek: {Sub-series} - {NNN} - {Real Title}

with occasional nested sub-arcs::

    Star Trek: 24th Century Crossover - {NNN} - {Inner Series} - {MM} - {Real Title}

Most books already have the correct Calibre ``series`` and ``series_index``
(e.g. series="Star Trek: The Original Series", index=1-162).  The sole
problem is that the ``title`` field still carries the full ugly prefix.

This module emits:

* ``title`` — strip the prefix, leaving just the real title.
* ``series`` + ``series_index`` — only for the small cohort of books that
  have *no* series set in Calibre but do have a numbered title structure.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from calibre_mcp.cleanup import calibre_reader, proposals
from calibre_mcp.cleanup.proposals import Proposal

log = logging.getLogger(__name__)

SOURCE = "star_trek"
SOURCE_FIXUP = "star_trek_fixup"

# Matches any ' - one-or-more-digits - ' separator (with flexible surrounding
# whitespace).  Used to split the embedded prefix from the real title.
_SEP_RE = re.compile(r"\s+-\s+(\d+)\s+-\s+")

# Matches a leading in-universe year (or year-range) prefix such as
# "2298 - " or "2328-2346 - " left behind after the first title pass.
_YEAR_PREFIX_RE = re.compile(r"^\d{4}(?:-\d{4})?\s+-\s+")


@dataclass(frozen=True, slots=True)
class ParsedBook:
    book_id: int
    raw_title: str
    current_series: str    # "" if not set in Calibre
    current_index: float
    proposed_title: str    # real title (everything after the last separator)
    proposed_series: str   # prefix before the first separator
    proposed_index: int    # number from the first separator


@dataclass(frozen=True, slots=True)
class RunSummary:
    examined: int
    skipped_no_match: int
    skipped_already_proposed: int
    proposals_emitted: int


def parse_title(raw: str) -> tuple[str, int, str] | None:
    """Parse a Star Trek title.

    Returns ``(series_name, series_index, real_title)`` or ``None``.

    ``series_name``
        Prefix before the **first** ``' - NNN - '`` separator.
        e.g. ``"Star Trek: The Original Series"``.

    ``series_index``
        Integer from the **first** separator — the book's position in its
        own series (not in a crossover wrapper).

    ``real_title``
        Everything after the **last** ``' - NNN - '`` separator, so nested
        sub-arcs are handled automatically::

            "Star Trek: 24th Century Crossover - 001 - Destiny - 01 - Gods of Night"
            → series_name = "Star Trek: 24th Century Crossover"
            → series_index = 1
            → real_title   = "Gods of Night"
    """
    raw = raw.strip()
    matches = list(_SEP_RE.finditer(raw))
    if not matches:
        return None
    first = matches[0]
    last = matches[-1]
    series_name = raw[: first.start()].strip()
    if not series_name.startswith("Star Trek:"):
        return None
    series_index = int(first.group(1))
    real_title = raw[last.end() :].strip()
    if not real_title:
        return None
    return series_name, series_index, real_title


def run(
    *,
    calibre_db: Path,
    proposals_db: Path,
    dry_run: bool = False,
) -> tuple[RunSummary, list[ParsedBook]]:
    """Scan Star Trek books, parse titles, emit proposals."""
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
            SELECT
                b.id,
                b.title,
                COALESCE(b.series_index, 1.0) AS series_index,
                COALESCE(s.name, '')           AS series
            FROM books b
            LEFT JOIN books_series_link bsl ON bsl.book = b.id
            LEFT JOIN series s              ON s.id = bsl.series
            WHERE b.title LIKE 'Star Trek%'
            ORDER BY b.title
        """).fetchall()

        for row in rows:
            book_id = int(row["id"])
            raw_title = row["title"]
            current_series = row["series"]
            current_index = float(row["series_index"])

            parsed = parse_title(raw_title)
            if parsed is None:
                log.debug("No match: %r", raw_title)
                skipped_no_match += 1
                continue

            proposed_series, proposed_index, proposed_title = parsed

            pb = ParsedBook(
                book_id=book_id,
                raw_title=raw_title,
                current_series=current_series,
                current_index=current_index,
                proposed_title=proposed_title,
                proposed_series=proposed_series,
                proposed_index=proposed_index,
            )
            parsed_books.append(pb)

            if dry_run:
                continue

            notes = json.dumps({
                "proposed_series": proposed_series,
                "proposed_index": proposed_index,
            })
            any_inserted = False

            title_inserted = proposals.insert_proposal(
                pc, run_id,
                Proposal(
                    book_id=book_id,
                    field="title",
                    calibre_value=raw_title,
                    proposed_value=proposed_title,
                    source=SOURCE,
                    confidence=1.0,
                    notes=notes,
                ),
            )
            any_inserted = any_inserted or title_inserted

            # Books with no Calibre series: also propose series + series_index.
            if not current_series:
                series_inserted = proposals.insert_proposal(
                    pc, run_id,
                    Proposal(
                        book_id=book_id,
                        field="series",
                        calibre_value="",
                        proposed_value=proposed_series,
                        source=SOURCE,
                        confidence=1.0,
                        notes=notes,
                    ),
                )
                any_inserted = any_inserted or series_inserted

                if abs(proposed_index - current_index) > 1e-9:
                    idx_inserted = proposals.insert_proposal(
                        pc, run_id,
                        Proposal(
                            book_id=book_id,
                            field="series_index",
                            calibre_value=str(current_index),
                            proposed_value=str(float(proposed_index)),
                            source=SOURCE,
                            confidence=1.0,
                            notes=notes,
                        ),
                    )
                    any_inserted = any_inserted or idx_inserted

            if any_inserted:
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


def run_year_fixup(
    *,
    calibre_db: Path,
    proposals_db: Path,
    series_name: str = "Star Trek: The Lost Era",
    dry_run: bool = False,
) -> tuple[RunSummary, list[ParsedBook]]:
    """Strip in-universe year prefixes from titles in a series.

    Targets books whose current title starts with ``YYYY - `` or
    ``YYYY-YYYY - `` (left over from the initial title-strip pass).
    Only emits ``title`` proposals.
    """
    calibre_db = Path(calibre_db).resolve()
    proposals_db = Path(proposals_db).resolve()

    cal = calibre_reader.open_readonly(calibre_db)
    pc = proposals.connect(proposals_db)

    run_id = proposals.start_run(
        pc,
        source=SOURCE_FIXUP,
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
        rows = cal.execute(
            """
            SELECT b.id, b.title, COALESCE(b.series_index, 1.0) AS series_index,
                   COALESCE(s.name, '') AS series
            FROM books b
            LEFT JOIN books_series_link bsl ON bsl.book = b.id
            LEFT JOIN series s              ON s.id = bsl.series
            WHERE s.name = ?
            ORDER BY b.series_index
            """,
            (series_name,),
        ).fetchall()

        for row in rows:
            book_id = int(row["id"])
            raw_title = row["title"]
            current_series = row["series"]
            current_index = float(row["series_index"])

            m = _YEAR_PREFIX_RE.match(raw_title)
            if not m:
                log.debug("No year prefix: %r", raw_title)
                skipped_no_match += 1
                continue

            proposed_title = raw_title[m.end():].strip()
            if not proposed_title:
                skipped_no_match += 1
                continue

            pb = ParsedBook(
                book_id=book_id,
                raw_title=raw_title,
                current_series=current_series,
                current_index=current_index,
                proposed_title=proposed_title,
                proposed_series=current_series,
                proposed_index=int(current_index),
            )
            parsed_books.append(pb)

            if dry_run:
                continue

            inserted = proposals.insert_proposal(
                pc, run_id,
                Proposal(
                    book_id=book_id,
                    field="title",
                    calibre_value=raw_title,
                    proposed_value=proposed_title,
                    source=SOURCE_FIXUP,
                    confidence=1.0,
                    notes=None,
                ),
            )
            if inserted:
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
            notes=f"series={series_name!r} matched={len(parsed_books)} no_match={skipped_no_match}",
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
