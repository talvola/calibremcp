"""Orchestrator for the Science Fiction subgenre tagging pipeline.

Walks the Science Fiction bucket in Calibre, calls the LLM tagger per book,
emits ``tags.add`` proposals into the propose-queue with
``source='scifi_llm'``.  Idempotent — re-running on the same state
inserts no new rows.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import anthropic

from calibre_mcp.cleanup import calibre_reader, scifi_tagger, proposals
from calibre_mcp.cleanup.scifi_tagger import (
    CONFIDENCE_MAP,
    SOURCE,
    BookContext,
    SciFiTags,
    tag_book,
)

log = logging.getLogger(__name__)

_NO_TAGS_SENTINEL = "(no tags)"


@dataclass(frozen=True, slots=True)
class SciFiRunSummary:
    examined: int
    skipped_already_tagged: int
    tagged: int
    api_errors: int
    proposals_emitted: int
    elapsed_sec: float


# ---------------------------------------------------------------------------
# Bucket loader
# ---------------------------------------------------------------------------

def _iter_bucket(
    cal: sqlite3.Connection,
    *,
    bucket_tag: str = "Science Fiction",
    limit: int | None = None,
    book_ids: list[int] | None = None,
) -> Iterator[BookContext]:
    if book_ids is not None:
        placeholders = ",".join("?" * len(book_ids))
        query = f"""
            SELECT b.id, b.title, c.text AS description,
                   (SELECT GROUP_CONCAT(a.name, '|') FROM authors a
                    JOIN books_authors_link bal ON bal.author = a.id
                    WHERE bal.book = b.id) AS authors
            FROM books b
            LEFT JOIN comments c ON c.book = b.id
            WHERE b.id IN ({placeholders})
            ORDER BY b.id
        """
        params: tuple = tuple(book_ids)
    else:
        query = """
            SELECT b.id, b.title, c.text AS description,
                   (SELECT GROUP_CONCAT(a.name, '|') FROM authors a
                    JOIN books_authors_link bal ON bal.author = a.id
                    WHERE bal.book = b.id) AS authors
            FROM books b
            LEFT JOIN comments c ON c.book = b.id
            JOIN books_tags_link btl ON btl.book = b.id
            JOIN tags t ON t.id = btl.tag
            WHERE t.name = ?
            ORDER BY b.id
        """
        params = (bucket_tag,)
    if limit is not None:
        query += f" LIMIT {int(limit)}"

    for row in cal.execute(query, params):
        yield BookContext(
            book_id=int(row["id"]),
            title=row["title"] or "",
            authors=tuple(a for a in (row["authors"] or "").split("|") if a),
            description=row["description"],
        )


def _already_tagged(pc: sqlite3.Connection, book_id: int) -> bool:
    row = pc.execute(
        "SELECT 1 FROM proposals WHERE source = ? AND field = 'tags.add' AND book_id = ? LIMIT 1",
        (SOURCE, book_id),
    ).fetchone()
    return row is not None


def _emit_proposals(
    pc: sqlite3.Connection,
    run_id: int,
    book: BookContext,
    tags: SciFiTags,
) -> int:
    confidence = CONFIDENCE_MAP[tags.confidence]
    notes = (
        f"scifi_llm: confidence={tags.confidence} "
        f"subgenre={tags.subgenre} theme={tags.theme}"
    )
    all_tags = (*tags.subgenre, *tags.theme)
    if not all_tags:
        try:
            pc.execute(
                """
                INSERT INTO proposals
                    (book_id, field, calibre_value, proposed_value, source,
                     confidence, status, notes, run_id)
                VALUES (?, 'tags.add', NULL, ?, ?, ?, 'rejected', ?, ?)
                """,
                (book.book_id, _NO_TAGS_SENTINEL, SOURCE, confidence,
                 f"{notes}; LLM returned no tags", run_id),
            )
            return 1
        except sqlite3.IntegrityError:
            return 0

    inserted = 0
    for tag in all_tags:
        proposal = proposals.Proposal(
            book_id=book.book_id,
            field="tags.add",
            calibre_value=None,
            proposed_value=tag,
            source=SOURCE,
            confidence=confidence,
            notes=notes,
        )
        if proposals.insert_proposal(pc, run_id, proposal):
            inserted += 1
    return inserted


# ---------------------------------------------------------------------------
# Retag helpers
# ---------------------------------------------------------------------------

def _resolve_retag_book_ids(
    pc: sqlite3.Connection, retag_tags: list[str],
) -> list[int]:
    if not retag_tags:
        return []
    placeholders = ",".join("?" * len(retag_tags))
    rows = pc.execute(
        f"""
        SELECT DISTINCT book_id FROM proposals
        WHERE source = ? AND field = 'tags.add'
          AND proposed_value IN ({placeholders})
          AND book_id != 0
        ORDER BY book_id
        """,
        (scifi_tagger.SOURCE, *retag_tags),
    ).fetchall()
    return [int(r["book_id"]) for r in rows]


def _clear_book_proposals(pc: sqlite3.Connection, book_id: int) -> int:
    cur = pc.execute(
        """
        DELETE FROM proposals
        WHERE source = ? AND field = 'tags.add' AND book_id = ?
          AND status IN ('proposed', 'rejected')
        """,
        (scifi_tagger.SOURCE, book_id),
    )
    return cur.rowcount


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(
    *,
    metadata_db: Path,
    proposals_db: Path,
    bucket_tag: str = "Science Fiction",
    limit: int | None = None,
    book_ids: list[int] | None = None,
    retag_tags: list[str] | None = None,
    skip_already_tagged: bool = True,
    progress_every: int = 25,
) -> SciFiRunSummary:
    metadata_db  = Path(metadata_db).resolve()
    proposals_db = Path(proposals_db).resolve()

    cal = calibre_reader.open_readonly(metadata_db)
    pc  = proposals.connect(proposals_db)
    client = anthropic.Anthropic()

    run_id = proposals.start_run(
        pc,
        source=SOURCE,
        library_root=metadata_db.parent,
        metadata_db_path=metadata_db,
        dry_run=False,
    )

    if retag_tags:
        retag_book_ids = _resolve_retag_book_ids(pc, retag_tags)
        book_ids = (
            retag_book_ids if book_ids is None
            else sorted(set(book_ids) | set(retag_book_ids))
        )
        skip_already_tagged = False
        log.info("retag-tags=%s → %d books", retag_tags, len(retag_book_ids))

    examined = skipped = tagged_count = api_errors = emitted = 0
    t0 = time.monotonic()

    try:
        for book in _iter_bucket(
            cal, bucket_tag=bucket_tag, limit=limit, book_ids=book_ids,
        ):
            examined += 1
            if skip_already_tagged and _already_tagged(pc, book.book_id):
                skipped += 1
                continue
            if not skip_already_tagged:
                _clear_book_proposals(pc, book.book_id)

            tags = tag_book(client, book)
            if tags is None:
                api_errors += 1
                continue

            tagged_count += 1
            emitted += _emit_proposals(pc, run_id, book, tags)

            if examined % progress_every == 0:
                elapsed = time.monotonic() - t0
                rate = tagged_count / elapsed if elapsed > 0 else 0
                log.info(
                    "progress: examined=%d tagged=%d skipped=%d errors=%d "
                    "emitted=%d elapsed=%.0fs rate=%.2f/s",
                    examined, tagged_count, skipped, api_errors, emitted,
                    elapsed, rate,
                )
    finally:
        proposals.finish_run(
            pc, run_id,
            books_scanned=examined,
            books_with_epub=0,
            books_parsed=tagged_count,
            proposals_emitted=emitted,
            errors=api_errors,
            notes=(
                f"bucket={bucket_tag!r} examined={examined} tagged={tagged_count} "
                f"skipped={skipped} errors={api_errors}"
            ),
        )
        cal.close()
        pc.close()

    return SciFiRunSummary(
        examined=examined,
        skipped_already_tagged=skipped,
        tagged=tagged_count,
        api_errors=api_errors,
        proposals_emitted=emitted,
        elapsed_sec=time.monotonic() - t0,
    )


# ---------------------------------------------------------------------------
# Review report
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TagGroup:
    tag: str
    n_books: int
    sample_titles: tuple[str, ...]
    proposal_ids: tuple[int, ...]


def report_by_tag(
    proposals_db: Path,
    metadata_db: Path,
    *,
    status: str = "proposed",
) -> list[TagGroup]:
    pc  = proposals.connect(proposals_db)
    cal = calibre_reader.open_readonly(metadata_db)
    try:
        rows = pc.execute(
            """
            SELECT proposed_value AS tag, book_id, id
            FROM proposals
            WHERE source = ? AND field = 'tags.add' AND status = ?
              AND proposed_value != ?
            ORDER BY proposed_value, book_id
            """,
            (SOURCE, status, _NO_TAGS_SENTINEL),
        ).fetchall()
        by_tag: dict[str, list[tuple[int, int]]] = {}
        for r in rows:
            by_tag.setdefault(r["tag"], []).append((r["book_id"], r["id"]))

        out: list[TagGroup] = []
        for tag, items in by_tag.items():
            book_ids = [bid for bid, _ in items[:5]]
            placeholders = ",".join("?" * len(book_ids))
            titles = [
                t["title"] for t in cal.execute(
                    f"SELECT id, title FROM books WHERE id IN ({placeholders})",
                    book_ids,
                )
            ] if book_ids else []
            out.append(TagGroup(
                tag=tag,
                n_books=len(items),
                sample_titles=tuple(titles),
                proposal_ids=tuple(pid for _, pid in items),
            ))
        out.sort(key=lambda g: -g.n_books)
        return out
    finally:
        pc.close()
        cal.close()
