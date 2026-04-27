"""Orchestrator for the cookbook subgenre tagging pipeline.

Walks the Cookbooks bucket in Calibre, calls the LLM tagger per book,
emits ``tags.add`` proposals into the propose-queue with
``source='cookbook_llm'``. Idempotent at the propose-queue layer via
the existing unique index — re-running on the same state inserts no
new rows.

The pipeline is intentionally sequential, not batched. Anthropic's batch
API would halve the cost but takes hours to complete; sequential at
~2s/call gets the full bucket done in ~40 min with prompt caching, with
live progress and the ability to abort cleanly.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import anthropic

from calibre_mcp.cleanup import calibre_reader, proposals
from calibre_mcp.cleanup.cookbook_tagger import (
    CONFIDENCE_MAP,
    SOURCE,
    BookContext,
    CookbookTags,
    tag_book,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CookbookRunSummary:
    examined: int           # books in the bucket considered
    skipped_already_tagged: int  # already had cookbook_llm proposals
    tagged: int             # books we actually called the LLM on
    api_errors: int         # LLM call returned None
    proposals_emitted: int  # new tags.add rows inserted (post-dedupe)
    elapsed_sec: float


# ---------------------------------------------------------------------------
# Bucket loader: pulls (BookContext, run-skip-flag) per cookbook
# ---------------------------------------------------------------------------


def _iter_bucket(
    cal: sqlite3.Connection,
    library_root: Path,
    *,
    bucket_tag: str = "Cookbooks",
    limit: int | None = None,
    book_ids: list[int] | None = None,
) -> Iterator[BookContext]:
    """Yield ``BookContext`` for every book carrying ``bucket_tag``.

    ``book_ids`` overrides the bucket query — used by the CLI to scope a
    proof-of-concept run to specific books (the 5 new cookbooks Erik
    just added, for example) without filtering by tag presence.
    """
    if book_ids is not None:
        placeholders = ",".join("?" * len(book_ids))
        query = f"""
            SELECT b.id, b.title, b.path, c.text AS description,
                   (SELECT GROUP_CONCAT(a.name, '|') FROM authors a
                    JOIN books_authors_link bal ON bal.author = a.id
                    WHERE bal.book = b.id) AS authors
            FROM books b
            LEFT JOIN comments c ON c.book = b.id
            WHERE b.id IN ({placeholders})
            ORDER BY b.id
        """  # noqa: S608 — placeholders are '?' chars; values bind via params
        params: tuple = tuple(book_ids)
    else:
        query = """
            SELECT b.id, b.title, b.path, c.text AS description,
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
        cover = library_root / (row["path"] or "") / "cover.jpg"
        yield BookContext(
            book_id=int(row["id"]),
            title=row["title"] or "",
            authors=tuple(
                a for a in (row["authors"] or "").split("|") if a
            ),
            description=row["description"],
            cover_path=cover if cover.exists() else None,
        )


def _already_tagged(pc: sqlite3.Connection, book_id: int) -> bool:
    """True if the book already has any cookbook_llm tags.add proposal —
    used to skip on re-runs even before the unique-index dedupe."""
    row = pc.execute(
        "SELECT 1 FROM proposals WHERE source = ? AND field = 'tags.add' AND book_id = ? LIMIT 1",
        (SOURCE, book_id),
    ).fetchone()
    return row is not None


_NO_TAGS_SENTINEL = "(no tags)"


def _emit_proposals(
    pc: sqlite3.Connection,
    run_id: int,
    book: BookContext,
    tags: CookbookTags,
) -> int:
    """Insert one tags.add proposal per facet tag. Returns the number of
    new rows persisted (deduped via the propose-queue unique index).

    For books the model returned zero tags for, insert a single
    ``status='rejected'`` sentinel row so re-runs skip the book — same
    pattern Goodreads lookup uses for not-found ISBNs."""
    confidence = CONFIDENCE_MAP[tags.confidence]
    notes = (
        f"cookbook_llm: confidence={tags.confidence} "
        f"cuisine={tags.cuisine} technique={tags.technique} dietary={tags.dietary}"
    )
    inserted = 0
    all_tags = (*tags.cuisine, *tags.technique, *tags.dietary)
    if not all_tags:
        # Sentinel row at status='rejected' — never picked up by apply,
        # but presence on the book means _already_tagged() short-circuits.
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
            inserted += 1
        except sqlite3.IntegrityError:
            pass  # already recorded on a prior run
        return inserted

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
# Main runner
# ---------------------------------------------------------------------------


def run(
    *,
    library_root: Path,
    metadata_db: Path,
    proposals_db: Path,
    bucket_tag: str = "Cookbooks",
    limit: int | None = None,
    book_ids: list[int] | None = None,
    skip_already_tagged: bool = True,
    progress_every: int = 10,
) -> CookbookRunSummary:
    """Run the cookbook tagger over the bucket; emit tags.add proposals.

    ``skip_already_tagged`` (default True) avoids re-calling the LLM for
    books that already have cookbook_llm proposals — re-runs are cheap.
    Set False to force a re-tagging pass (e.g. after taxonomy edits).
    """
    library_root = Path(library_root).resolve()
    metadata_db = Path(metadata_db).resolve()
    proposals_db = Path(proposals_db).resolve()

    cal = calibre_reader.open_readonly(metadata_db)
    pc = proposals.connect(proposals_db)
    client = anthropic.Anthropic()

    run_id = proposals.start_run(
        pc,
        source=SOURCE,
        library_root=library_root,
        metadata_db_path=metadata_db,
        dry_run=False,
    )

    examined = skipped = tagged_count = api_errors = emitted = 0
    t0 = time.monotonic()

    try:
        for book in _iter_bucket(
            cal, library_root,
            bucket_tag=bucket_tag, limit=limit, book_ids=book_ids,
        ):
            examined += 1
            if skip_already_tagged and _already_tagged(pc, book.book_id):
                skipped += 1
                continue

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

    return CookbookRunSummary(
        examined=examined,
        skipped_already_tagged=skipped,
        tagged=tagged_count,
        api_errors=api_errors,
        proposals_emitted=emitted,
        elapsed_sec=time.monotonic() - t0,
    )


# ---------------------------------------------------------------------------
# Per-cuisine review report
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TagGroup:
    tag: str
    n_books: int
    sample_titles: tuple[str, ...]  # up to 5
    proposal_ids: tuple[int, ...]


def report_by_tag(
    proposals_db: Path,
    metadata_db: Path,
    *,
    status: str = "proposed",
) -> list[TagGroup]:
    """Group cookbook_llm proposals by proposed_value (the facet tag) so
    Erik can scan each cuisine/technique/dietary cohort independently
    and bulk-approve per tag."""
    pc = proposals.connect(proposals_db)
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
                    f"SELECT id, title FROM books WHERE id IN ({placeholders})",  # noqa: S608 — placeholders are '?' chars
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
