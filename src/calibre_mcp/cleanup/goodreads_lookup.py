"""Look up Goodreads book IDs for books that have an ISBN but no Goodreads ID.

Mechanics: ``GET https://www.goodreads.com/book/isbn/<ISBN>`` responds with
a ``301`` redirect to ``/book/show/<numeric_id>-<slug>``; we parse the ID
out of the Location header. A non-redirect response (200 with a search
page, 404, etc.) means Goodreads has no book for that ISBN — recorded as
a ``rejected`` proposal with reason, so re-runs skip the book.

Polite by default: one request per second, realistic User-Agent, honours
429 with exponential backoff. Not meant for bulk harvesting; this is
personal-library metadata cleanup."""

from __future__ import annotations

import contextlib
import logging
import re
import sqlite3
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx

from calibre_mcp.cleanup import calibre_reader, proposals
from calibre_mcp.cleanup.proposals import Proposal

log = logging.getLogger(__name__)

SOURCE = "goodreads_isbn"

# Realistic UA. Goodreads has historically served different content (or
# captcha walls) to obvious scrapers; a plain Firefox string is both polite
# (a real user would see the same page) and stable.
_DEFAULT_UA = "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"

_BOOK_SHOW_PATH_RE = re.compile(r"^/book/show/(\d+)(?:-.*)?$")
_BOOK_SHOW_URL_RE = re.compile(r"^https?://(?:www\.)?goodreads\.com/book/show/(\d+)(?:-.*)?$")

# Sentinel proposed_value for the "we tried, Goodreads had no match" marker.
# Lives at status='rejected', never reaches the apply step (which only
# processes status='approved'), and makes re-runs skip the book.
_NOT_FOUND_SENTINEL = "(not found)"


def _record_not_found(conn: sqlite3.Connection, run_id: int, book_id: int, isbn: str, note: str | None) -> None:
    """Insert a definitive-rejection marker for a book Goodreads couldn't find.

    Bypasses ``insert_proposal`` because this row needs to land at
    ``status='rejected'`` directly — it's a decision, not a proposal. The
    IntegrityError suppression covers the belt-and-suspenders case where
    the skip-list in _iter_candidates misses a race (shouldn't normally
    happen with a single-threaded run)."""
    with contextlib.suppress(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO proposals
                (book_id, field, calibre_value, proposed_value, source,
                 confidence, status, conflict_reason, notes, run_id)
            VALUES (?, 'identifier.goodreads', NULL, ?, ?, 0.0,
                    'rejected', 'not found on Goodreads', ?, ?)
            """,
            (book_id, _NOT_FOUND_SENTINEL, SOURCE, f"isbn={isbn}" + (f"; {note}" if note else ""), run_id),
        )


@dataclass(frozen=True, slots=True)
class LookupResult:
    book_id: int
    isbn: str
    goodreads_id: str | None
    status: str  # 'found' | 'not_found' | 'rate_limited' | 'error'
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class LookupSummary:
    run_id: int
    attempted: int
    found: int
    not_found: int
    errors: int
    rate_limited: int


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------


def _iter_candidates(
    calibre_conn: sqlite3.Connection,
    proposals_conn: sqlite3.Connection,
    *,
    limit: int | None = None,
) -> Iterator[tuple[int, str]]:
    """Yield ``(book_id, isbn)`` for books eligible for Goodreads lookup.

    Eligible = has ISBN in Calibre AND does not have Goodreads ID in Calibre
    AND has not already been tried in this proposals DB (any status)."""
    # Already-tried books: any Goodreads proposal from the ISBN lookup source,
    # regardless of status. Status 'rejected' with our not-found reason means
    # "we tried, got nothing" — don't re-hit.
    tried = {
        r["book_id"]
        for r in proposals_conn.execute(
            "SELECT book_id FROM proposals WHERE source = ? OR field = 'identifier.goodreads'",
            (SOURCE,),
        )
    }

    # Also skip books that already have a Goodreads ID in Calibre (from Phase
    # 1 OPF mining or Erik's manual entries).
    already_have = {
        r["book"] for r in calibre_conn.execute("SELECT book FROM identifiers WHERE type IN ('goodreads', 'Goodreads')")
    }

    query = """
        SELECT i.book AS book_id, i.val AS isbn
        FROM identifiers i
        WHERE i.type = 'isbn'
        ORDER BY i.book
    """
    if limit is not None:
        query += f" LIMIT {int(limit) * 3}"  # over-fetch to account for skips

    yielded = 0
    for row in calibre_conn.execute(query):
        book_id = int(row["book_id"])
        if book_id in tried or book_id in already_have:
            continue
        isbn = (row["isbn"] or "").strip()
        if not isbn:
            continue
        yield (book_id, isbn)
        yielded += 1
        if limit is not None and yielded >= limit:
            return


# ---------------------------------------------------------------------------
# Single-book lookup
# ---------------------------------------------------------------------------


def _extract_goodreads_id(url_or_path: str) -> str | None:
    """Pull the numeric ID out of a /book/show/<id>-<slug> URL or path."""
    if not url_or_path:
        return None
    m = _BOOK_SHOW_URL_RE.match(url_or_path.strip()) or _BOOK_SHOW_PATH_RE.match(url_or_path.strip())
    return m.group(1) if m else None


def lookup_one(client: httpx.Client, isbn: str) -> tuple[str | None, str, str | None]:
    """Single-ISBN lookup. Returns ``(goodreads_id, status, note)``.

    status: 'found' | 'not_found' | 'rate_limited' | 'error'"""
    url = f"https://www.goodreads.com/book/isbn/{isbn}"
    try:
        r = client.get(url)
    except httpx.HTTPError as exc:
        return None, "error", f"{type(exc).__name__}: {exc}"

    if r.status_code in (301, 302, 303, 307, 308):
        loc = r.headers.get("location", "")
        gid = _extract_goodreads_id(loc)
        if gid:
            return gid, "found", loc
        # Redirected but not to a /book/show page — usually means search
        # results or a disambiguation. Treat as not found so we don't
        # re-query; the user can fix manually if they care.
        return None, "not_found", f"redirect to {loc!r}"

    if r.status_code == 200:
        # Some ISBNs that don't map to a single Goodreads book land on a
        # search or 'book not found' page without a redirect.
        return None, "not_found", "200 without redirect"

    if r.status_code == 404:
        return None, "not_found", "404"

    if r.status_code == 429:
        return None, "rate_limited", "429 from Goodreads"

    return None, "error", f"unexpected status {r.status_code}"


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def run(
    *,
    calibre_db: Path,
    proposals_db: Path,
    rate_sec: float = 1.0,
    limit: int | None = None,
    user_agent: str = _DEFAULT_UA,
    backoff_base: float = 30.0,
    max_backoff: float = 600.0,
    progress_every: int = 50,
) -> LookupSummary:
    """Walk eligible books, fetch Goodreads IDs, write proposals.

    Rate-limits to one request per ``rate_sec`` seconds (default 1.0). On
    ``429`` responses, sleeps ``backoff_base`` seconds then retries — doubles
    up to ``max_backoff``. Records ``rejected`` proposals for definitive
    ``not_found`` responses so re-runs skip them."""
    calibre_db = Path(calibre_db).resolve()
    proposals_db = Path(proposals_db).resolve()

    cal = calibre_reader.open_readonly(calibre_db)
    pc = proposals.connect(proposals_db)

    run_id = proposals.start_run(
        pc,
        source=SOURCE,
        library_root=calibre_db.parent,
        metadata_db_path=calibre_db,
        dry_run=False,
    )

    attempted = found = not_found = errors = rate_limited = 0
    backoff = backoff_base

    try:
        candidates = list(_iter_candidates(cal, pc, limit=limit))
        log.info("lookup candidates: %d", len(candidates))

        with httpx.Client(
            follow_redirects=False,
            headers={"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml"},
            timeout=20.0,
        ) as client:
            for i, (book_id, isbn) in enumerate(candidates, start=1):
                attempted += 1
                gid, status, note = lookup_one(client, isbn)

                if status == "found" and gid is not None:
                    found += 1
                    proposals.insert_proposal(
                        pc,
                        run_id,
                        Proposal(
                            book_id=book_id,
                            field="identifier.goodreads",
                            proposed_value=gid,
                            source=SOURCE,
                            confidence=1.0,
                            notes=f"isbn={isbn}" + (f"; {note}" if note else ""),
                        ),
                    )
                elif status == "not_found":
                    not_found += 1
                    _record_not_found(pc, run_id, book_id, isbn, note)
                elif status == "rate_limited":
                    rate_limited += 1
                    log.warning("rate limited; backing off %.0fs", backoff)
                    time.sleep(backoff)
                    backoff = min(backoff * 2, max_backoff)
                    # Retry this one immediately (don't count as attempted again).
                    attempted -= 1
                    gid, status, note = lookup_one(client, isbn)
                    if status == "found" and gid:
                        found += 1
                        attempted += 1
                        proposals.insert_proposal(
                            pc,
                            run_id,
                            Proposal(
                                book_id=book_id,
                                field="identifier.goodreads",
                                proposed_value=gid,
                                source=SOURCE,
                                confidence=1.0,
                                notes=f"isbn={isbn}; retried after backoff",
                            ),
                        )
                    else:
                        errors += 1
                        attempted += 1
                else:  # 'error'
                    errors += 1
                    proposals.record_error(
                        pc,
                        run_id,
                        book_id=book_id,
                        book_path=None,
                        error=f"lookup {isbn}: {note or 'unknown'}",
                    )

                # Reset backoff after a successful non-429 response.
                if status != "rate_limited":
                    backoff = backoff_base

                if i % progress_every == 0 or i == len(candidates):
                    log.info(
                        "progress %d/%d found=%d not_found=%d errors=%d",
                        i,
                        len(candidates),
                        found,
                        not_found,
                        errors,
                    )

                # Polite pacing between requests.
                if i < len(candidates):
                    time.sleep(rate_sec)
    finally:
        proposals.finish_run(
            pc,
            run_id,
            books_scanned=attempted,
            books_with_epub=0,  # N/A for this phase
            books_parsed=found + not_found,
            proposals_emitted=found,
            errors=errors,
            notes=f"not_found={not_found} rate_limited={rate_limited}",
        )
        cal.close()
        pc.close()

    return LookupSummary(
        run_id=run_id,
        attempted=attempted,
        found=found,
        not_found=not_found,
        errors=errors,
        rate_limited=rate_limited,
    )
