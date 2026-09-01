"""Per-book gold-standard audit and lookup orchestration.

For a single Calibre book, compute the gap between current metadata and
the gold-standard target (cover present, series consistent with
series_index, ISBN + Goodreads + Amazon identifiers populated, junk
identifiers cleaned). Then optionally hit external sources to fill the
fillable gaps as proposals.

Lookup order (polite-first):
  1. OpenLibrary   — ISBN, cover, publisher
  2. Google Books  — ISBN cross-check, Amazon ASIN (in OTHER identifiers)
  3. Goodreads ISBN redirect — Goodreads ID (uses existing goodreads_lookup)

Series resolution always consults the library first (see
``series_resolver``) so we don't introduce a near-variant of an existing
series name. This module does NOT auto-propose series names: with no ISBN
applied yet we can't reach Goodreads (Erik's authoritative source), so we
surface library candidates and let the human decide.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from calibre_mcp.cleanup import (
    calibre_reader,
    goodreads_lookup,
    goodreads_page,
    google_books_lookup,
    openlibrary_lookup,
    proposals,
    series_resolver,
)
from calibre_mcp.cleanup.proposals import Proposal

log = logging.getLogger(__name__)

SOURCE = "book_audit"

# Calibre's "no series" default for series_index is 1.0. A bare 1.0 without
# a series row is therefore the ambiguous case (might be deliberate, might
# be untouched). We treat anything else (1.5, 2, 3.0…) as a definite orphan.
_DEFAULT_SERIES_INDEX = "1"

# Identifier 'type' values we consider garbage; proposed for removal.
_JUNK_IDENTIFIER_TYPES = frozenset({"pub-id"})


@dataclass(frozen=True, slots=True)
class BookSnapshot:
    """Current state of one book — every field the audit cares about."""

    id: int
    uuid: str | None
    title: str
    author_sort: str | None
    primary_author: str | None
    primary_author_id: int | None
    publisher: str | None
    publisher_id: int | None
    series: str | None
    series_index: str | None
    has_cover: bool
    isbn: str | None
    identifiers: dict[str, str]
    description_chars: int


@dataclass
class GapItem:
    field: str  # e.g. 'cover', 'series', 'identifier.isbn'
    status: str  # 'missing' | 'orphan' | 'junk' | 'ok'
    current: str | None
    proposed: str | None = None
    source: str | None = None
    confidence: float = 0.0
    notes: str = ""


@dataclass
class AuditReport:
    book: BookSnapshot
    gaps: list[GapItem] = field(default_factory=list)
    series_candidates: list[series_resolver.SeriesCandidate] = field(default_factory=list)
    goodreads_series: list[goodreads_page.SeriesRef] = field(default_factory=list)
    cover_url: str | None = None  # remote URL surfaced by lookups
    cover_local_path: Path | None = None  # filled by download_cover()
    lookup_notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def snapshot(conn: sqlite3.Connection, book_id: int) -> BookSnapshot | None:
    """Build a BookSnapshot from Calibre's metadata.db.

    Returns None if no such book.
    """
    row = conn.execute(
        """
        SELECT b.id, b.uuid, b.title, b.author_sort, b.series_index, b.has_cover,
               s.name  AS series_name,
               p.name  AS publisher_name,
               p.id    AS publisher_id,
               LENGTH(c.text) AS comment_len
        FROM books b
        LEFT JOIN books_series_link bsl ON bsl.book = b.id
        LEFT JOIN series s ON s.id = bsl.series
        LEFT JOIN books_publishers_link bpl ON bpl.book = b.id
        LEFT JOIN publishers p ON p.id = bpl.publisher
        LEFT JOIN comments c ON c.book = b.id
        WHERE b.id = ?
        """,
        (book_id,),
    ).fetchone()
    if row is None:
        return None

    ar = conn.execute(
        """
        SELECT a.id, a.name FROM authors a
        JOIN books_authors_link bal ON bal.author = a.id
        WHERE bal.book = ? ORDER BY bal.id LIMIT 1
        """,
        (book_id,),
    ).fetchone()

    idents: dict[str, str] = {}
    for r in conn.execute("SELECT type, val FROM identifiers WHERE book = ?", (book_id,)):
        t = (r["type"] or "").strip().lower()
        v = (r["val"] or "").strip()
        if t and v:
            idents[t] = v

    return BookSnapshot(
        id=int(row["id"]),
        uuid=row["uuid"],
        title=row["title"] or "",
        author_sort=row["author_sort"],
        primary_author=ar["name"] if ar else None,
        primary_author_id=int(ar["id"]) if ar else None,
        publisher=row["publisher_name"],
        publisher_id=int(row["publisher_id"]) if row["publisher_id"] else None,
        series=row["series_name"],
        series_index=_format_series_index(row["series_index"]),
        has_cover=bool(row["has_cover"]),
        isbn=idents.get("isbn"),
        identifiers=idents,
        description_chars=int(row["comment_len"] or 0),
    )


def _format_series_index(value: float | int | str | None) -> str | None:
    """Render Calibre's REAL series_index without trailing zeros."""
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.4f}".rstrip("0").rstrip(".") or None


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------


def detect_gaps(b: BookSnapshot) -> list[GapItem]:
    """Identify what's missing or inconsistent in the snapshot.

    Lookups will later populate ``proposed`` / ``source`` / ``confidence``
    on the fillable items.
    """
    gaps: list[GapItem] = []

    if not b.has_cover:
        gaps.append(GapItem(field="cover", status="missing", current=None))

    # Series. Calibre defaults series_index=1.0 for "no series", so we treat
    # 1 + no series as 'missing' (might be a series, might not) and anything
    # else (1.5, 2, 3.0…) as a real 'orphan' — someone deliberately set an
    # index without naming the series.
    if b.series is None:
        if b.series_index and b.series_index != _DEFAULT_SERIES_INDEX:
            gaps.append(GapItem(
                field="series", status="orphan",
                current=f"series_index={b.series_index}, no series",
                notes="series_index set without a series name",
            ))
        else:
            gaps.append(GapItem(field="series", status="missing", current=None))

    if not b.isbn:
        gaps.append(GapItem(field="identifier.isbn", status="missing", current=None))

    if "goodreads" not in b.identifiers:
        gaps.append(GapItem(field="identifier.goodreads", status="missing", current=None))

    if "amazon" not in b.identifiers and "asin" not in b.identifiers:
        gaps.append(GapItem(field="identifier.amazon", status="missing", current=None))

    for jt in _JUNK_IDENTIFIER_TYPES:
        if jt in b.identifiers:
            gaps.append(GapItem(
                field=f"identifier.{jt}", status="junk",
                current=b.identifiers[jt],
                notes="EPUB-internal UUID; not a real cross-reference",
            ))

    return gaps


# ---------------------------------------------------------------------------
# Lookup orchestration
# ---------------------------------------------------------------------------


def fill_lookups(report: AuditReport, calibre_conn: sqlite3.Connection) -> None:
    """Hit OpenLibrary, Google Books, Goodreads ISBN redirect, and Goodreads
    page. Populates ``proposed`` on the relevant gaps.

    Goodreads needs its own clients: the ISBN lookup wants
    ``follow_redirects=False`` to capture the Location header; the page
    scrape wants follow-redirects on. Both use a browser UA.
    """
    needs_isbn = _needs(report, "identifier.isbn")
    needs_cover = _needs(report, "cover")
    needs_goodreads = _needs(report, "identifier.goodreads")
    needs_amazon = _needs(report, "identifier.amazon")
    needs_series = any(g.field == "series" for g in report.gaps)

    if not (needs_isbn or needs_cover or needs_goodreads or needs_amazon or needs_series):
        return

    isbn: str | None = None
    with httpx.Client(timeout=20.0) as client:
        ol_hit = _best_openlibrary_hit(report, client)
        gb_hit = _best_google_books_hit(report, client)

        isbn = _pick_isbn(ol_hit, gb_hit)
        if needs_isbn and isbn:
            _set_gap(
                report, "identifier.isbn",
                proposed=isbn,
                source=_isbn_source(isbn, ol_hit, gb_hit),
                confidence=_isbn_confidence(isbn, ol_hit, gb_hit),
            )

        if needs_cover:
            cover = (ol_hit.cover_url if ol_hit else None) or (gb_hit.thumbnail_url if gb_hit else None)
            if cover:
                report.cover_url = cover
                _set_gap(
                    report, "cover",
                    proposed=cover,
                    source=openlibrary_lookup.SOURCE if (ol_hit and ol_hit.cover_url) else google_books_lookup.SOURCE,
                    confidence=0.8,
                )

        if needs_amazon and gb_hit:
            asin = _extract_asin(gb_hit)
            if asin:
                _set_gap(
                    report, "identifier.amazon",
                    proposed=asin,
                    source=google_books_lookup.SOURCE,
                    confidence=0.5,
                    notes="from Google Books industryIdentifiers (OTHER)",
                )

    # Goodreads ISBN→ID redirect (separate client: no redirects, browser UA).
    goodreads_id: str | None = None
    if isbn and (needs_goodreads or needs_cover or needs_amazon or needs_series):
        with httpx.Client(
            follow_redirects=False,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
                "Accept": "text/html,application/xhtml+xml",
            },
            timeout=20.0,
        ) as gr_client:
            gid, status, note = goodreads_lookup.lookup_one(gr_client, isbn)
            if gid:
                goodreads_id = gid
                if needs_goodreads:
                    _set_gap(
                        report, "identifier.goodreads",
                        proposed=gid,
                        source=goodreads_lookup.SOURCE,
                        confidence=1.0,
                        notes=f"via isbn={isbn}",
                    )
            else:
                report.lookup_notes.append(f"goodreads: {status} ({note})")

    # Goodreads page scrape: backfills cover (priority), ASIN, and series.
    if goodreads_id:
        with goodreads_page.new_client() as page_client:
            page = goodreads_page.fetch(page_client, goodreads_id)
        if page is None:
            report.lookup_notes.append(f"goodreads page {goodreads_id}: fetch failed")
        else:
            _apply_goodreads_page(report, page, calibre_conn)


def _apply_goodreads_page(
    report: AuditReport,
    page: goodreads_page.GoodreadsPageData,
    calibre_conn: sqlite3.Connection,
) -> None:
    """Fill cover / amazon / series gaps from a parsed Goodreads page.

    Goodreads wins over OpenLibrary/Google Books for cover (canonical edition
    art) and is our only structured source for series.
    """
    # Cover — overwrite OL/GB if present; Goodreads is canonical for cover art.
    cover_replaceable = (
        _needs(report, "cover")
        or _filled_by(report, "cover", openlibrary_lookup.SOURCE)
        or _filled_by(report, "cover", google_books_lookup.SOURCE)
    )
    if cover_replaceable and page.cover_url:
        report.cover_url = page.cover_url
        _set_gap(
            report, "cover",
            proposed=page.cover_url,
            source=goodreads_page.SOURCE,
            confidence=0.9,
        )

    # Amazon ASIN — Goodreads beats the Google Books OTHER-identifier heuristic.
    if page.asin and (_needs(report, "identifier.amazon") or _filled_by(report, "identifier.amazon", google_books_lookup.SOURCE)):
        note = "from Goodreads page"
        if len(page.asins_all) > 1:
            others = ", ".join(page.asins_all[1:])
            note += f" (other ASINs: {others})"
        _set_gap(
            report, "identifier.amazon",
            proposed=page.asin,
            source=goodreads_page.SOURCE,
            confidence=0.9,
            notes=note,
        )

    # Series — library-consult-first.
    if any(g.field == "series" for g in report.gaps):
        _resolve_series_from_page(report, page, calibre_conn)


def _resolve_series_from_page(
    report: AuditReport,
    page: goodreads_page.GoodreadsPageData,
    calibre_conn: sqlite3.Connection,
) -> None:
    """Pick a canonical series name with library-consistency guard.

    Logic:
      1. Surface all structured candidates so the human can see options.
      2. If any structured Goodreads series exact-matches a library series
         for this author/publisher, propose it verbatim + Goodreads index.
      3. Else if a single structured candidate exists with no library
         conflict, propose it as new canonical + Goodreads index.
      4. Else if structured candidates differ from library variants
         (fuzzy hit), surface the conflict — do NOT propose.
      5. Else fall back to the parenthetical heuristic, low confidence.
    """
    candidates = list(page.series)
    report.goodreads_series = candidates

    if not candidates and page.series_heuristic:
        candidates = [page.series_heuristic]

    if not candidates:
        return

    for cand in candidates:
        lib_hits = series_resolver.find_library_candidates(
            calibre_conn,
            author_id=report.book.primary_author_id,
            publisher_id=report.book.publisher_id,
            proposed_name=cand.title,
        )
        exact = [c for c in lib_hits if c.name == cand.title]
        if exact:
            _propose_series(
                report, cand, name=exact[0].name,
                confidence=0.95,
                notes=f"matches library series ({exact[0].book_count} books, {exact[0].reason})",
            )
            return
        fuzzy = [c for c in lib_hits if c.name != cand.title]
        if fuzzy:
            report.lookup_notes.append(
                f"series conflict: Goodreads {cand.title!r} (#{cand.user_position or '?'}) vs "
                f"library {fuzzy[0].name!r} ({fuzzy[0].book_count} books, {fuzzy[0].reason}) — "
                f"normalize manually before proposing"
            )
            # Surface in series_candidates so the human sees both options.
            report.series_candidates.extend(
                c for c in fuzzy if c.name not in {x.name for x in report.series_candidates}
            )
            return

    # No library hits at all — propose the FIRST structured candidate (or
    # the heuristic) as a new canonical. The human knows future books in
    # this series will inherit this name.
    primary = candidates[0]
    is_heuristic = primary is page.series_heuristic
    _propose_series(
        report, primary, name=primary.title,
        confidence=0.6 if is_heuristic else 0.85,
        notes="new canonical (no library match); " + (
            "from title parenthetical heuristic" if is_heuristic else "from Goodreads structured series"
        ),
    )


def _propose_series(
    report: AuditReport,
    cand: goodreads_page.SeriesRef,
    *,
    name: str,
    confidence: float,
    notes: str,
) -> None:
    """Fill the series gap and (if we have an index) emit a series_index gap."""
    _set_gap(
        report, "series",
        proposed=name,
        source=goodreads_page.SOURCE,
        confidence=confidence,
        notes=notes,
    )
    if cand.user_position:
        # Compare against current series_index; if it differs, add a gap.
        current = report.book.series_index
        if current != cand.user_position:
            # Append a fresh GapItem since detect_gaps doesn't pre-emit series_index.
            report.gaps.append(GapItem(
                field="series_index",
                status="orphan" if current and current != "1" else "missing",
                current=current,
                proposed=cand.user_position,
                source=goodreads_page.SOURCE,
                confidence=confidence,
                notes=notes,
            ))


def _filled_by(report: AuditReport, field_name: str, source: str) -> bool:
    return any(g.field == field_name and g.source == source for g in report.gaps)


# ---------------------------------------------------------------------------
# Cover download
# ---------------------------------------------------------------------------

# Reject anything smaller than this — Goodreads occasionally returns a tiny
# placeholder JPEG (~1KB) for books without uploaded cover art.
_MIN_COVER_BYTES = 2048


def download_cover(report: AuditReport, *, covers_dir: Path) -> Path | None:
    """Download ``report.cover_url`` into ``covers_dir/<book_id>.<ext>``.

    Sets ``report.cover_local_path`` on success. Returns the path or None.
    Validates size and content-type so we don't save placeholder/error pages.
    """
    if not report.cover_url:
        return None

    covers_dir = Path(covers_dir)
    covers_dir.mkdir(parents=True, exist_ok=True)

    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            r = client.get(report.cover_url)
    except httpx.HTTPError as exc:
        report.lookup_notes.append(f"cover download failed: {exc}")
        return None
    if r.status_code != 200:
        report.lookup_notes.append(f"cover download HTTP {r.status_code}")
        return None
    if len(r.content) < _MIN_COVER_BYTES:
        report.lookup_notes.append(
            f"cover download too small ({len(r.content)} bytes — likely placeholder)"
        )
        return None

    ctype = (r.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
    ext = _ext_for_image(ctype, r.content)
    # Absolute path so apply.py (which may run with a different cwd) can
    # resolve it; calibredb is run as a subprocess.
    dest = (covers_dir / f"{report.book.id}{ext}").resolve()
    dest.write_bytes(r.content)
    report.cover_local_path = dest
    # Re-point the cover gap at the local path for the proposal pass.
    _set_gap(
        report, "cover",
        proposed=str(dest),
        source=(_gap_source(report, "cover") or "downloaded"),
        confidence=_gap_confidence(report, "cover") or 0.9,
        notes=f"downloaded from {report.cover_url}",
    )
    return dest


def _gap_source(report: AuditReport, field_name: str) -> str | None:
    g = next((g for g in report.gaps if g.field == field_name), None)
    return g.source if g else None


def _gap_confidence(report: AuditReport, field_name: str) -> float:
    g = next((g for g in report.gaps if g.field == field_name), None)
    return g.confidence if g else 0.0


def _ext_for_image(content_type: str, body: bytes) -> str:
    """Pick a file extension from Content-Type, falling back to magic bytes."""
    by_type = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }
    if content_type in by_type:
        return by_type[content_type]
    # Magic-byte sniff for the common cases.
    if body[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if body[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if body[:4] == b"RIFF" and body[8:12] == b"WEBP":
        return ".webp"
    return ".jpg"  # Calibre accepts mis-named .jpg for most formats


def _needs(report: AuditReport, field_name: str) -> bool:
    return any(g.field == field_name and g.status == "missing" for g in report.gaps)


def _best_openlibrary_hit(
    report: AuditReport, client: httpx.Client
) -> openlibrary_lookup.OpenLibraryHit | None:
    b = report.book
    if not (b.title and b.primary_author):
        return None
    hits = openlibrary_lookup.search(title=b.title, author=b.primary_author, client=client)
    if not hits:
        report.lookup_notes.append("openlibrary: no hits")
        return None
    return hits[0]


def _best_google_books_hit(
    report: AuditReport, client: httpx.Client
) -> google_books_lookup.GoogleBooksHit | None:
    b = report.book
    if not (b.title and b.primary_author):
        return None
    hits = google_books_lookup.search(title=b.title, author=b.primary_author, client=client)
    if not hits:
        report.lookup_notes.append("google_books: no hits")
        return None
    return hits[0]


def _pick_isbn(
    ol: openlibrary_lookup.OpenLibraryHit | None,
    gb: google_books_lookup.GoogleBooksHit | None,
) -> str | None:
    """Prefer ISBN-13. OpenLibrary first, then Google Books."""
    if ol and ol.best_isbn:
        return ol.best_isbn
    if gb and gb.isbn_13:
        return gb.isbn_13
    if gb and gb.isbn_10:
        return gb.isbn_10
    return None


def _isbn_source(
    isbn: str,
    ol: openlibrary_lookup.OpenLibraryHit | None,
    gb: google_books_lookup.GoogleBooksHit | None,
) -> str:
    if ol and isbn in ol.isbn_list:
        return openlibrary_lookup.SOURCE
    return google_books_lookup.SOURCE


def _isbn_confidence(
    isbn: str,
    ol: openlibrary_lookup.OpenLibraryHit | None,
    gb: google_books_lookup.GoogleBooksHit | None,
) -> float:
    """Higher when both sources independently agree."""
    ol_isbns = set(ol.isbn_list) if ol else set()
    gb_isbns = {i for i in (gb.isbn_13, gb.isbn_10) if i} if gb else set()
    if isbn in ol_isbns and isbn in gb_isbns:
        return 0.95
    if isbn in ol_isbns or isbn in gb_isbns:
        return 0.75
    return 0.5


def _extract_asin(gb: google_books_lookup.GoogleBooksHit) -> str | None:
    """ASINs sometimes appear under an OTHER industryIdentifier (10-char, B0...).

    Heuristic; Google doesn't label these as Amazon explicitly.
    """
    for v in (gb.other_identifiers or {}).values():
        if v and len(v) == 10 and v.startswith("B"):
            return v
    return None


def _set_gap(
    report: AuditReport, field_name: str, *,
    proposed: str, source: str, confidence: float, notes: str = "",
) -> None:
    for g in report.gaps:
        if g.field == field_name:
            g.proposed = proposed
            g.source = source
            g.confidence = confidence
            if notes:
                g.notes = f"{g.notes}; {notes}".strip("; ") if g.notes else notes
            return


# ---------------------------------------------------------------------------
# Series — library-consult-first, never auto-propose
# ---------------------------------------------------------------------------


def fill_series(report: AuditReport, calibre_conn: sqlite3.Connection) -> None:
    """Populate ``series_candidates`` so the human can pick.

    OpenLibrary/Google Books carry series weakly; Erik prefers Goodreads as
    authoritative and Goodreads needs an applied ISBN to reach the right
    page. So we don't auto-propose here — we just surface what the library
    already knows.
    """
    if not any(g.field == "series" for g in report.gaps):
        return
    report.series_candidates = series_resolver.find_library_candidates(
        calibre_conn,
        author_id=report.book.primary_author_id,
        publisher_id=report.book.publisher_id,
    )


# ---------------------------------------------------------------------------
# Proposals
# ---------------------------------------------------------------------------


def to_proposals(report: AuditReport) -> list[Proposal]:
    """Turn filled GapItems into Proposal rows.

    Cover gaps are skipped (binary, doesn't fit the queue — handled by a
    separate apply path). Junk identifiers become ``identifier.delete``
    proposals carrying the scheme to remove as their proposed_value.
    Everything else maps directly.
    """
    out: list[Proposal] = []
    for g in report.gaps:
        if g.field == "cover":
            # Cover proposals only go into the queue once we've downloaded
            # the image locally — apply.py needs a filesystem path for
            # `calibredb set_cover`.
            if report.cover_local_path and report.cover_local_path.exists():
                out.append(Proposal(
                    book_id=report.book.id,
                    book_uuid=report.book.uuid,
                    field="cover",
                    calibre_value=None,
                    proposed_value=str(report.cover_local_path),
                    source=g.source or SOURCE,
                    confidence=g.confidence or 0.9,
                    notes=g.notes or None,
                ))
            continue

        if g.status == "junk" and g.field.startswith("identifier."):
            scheme = g.field.split(".", 1)[1]
            out.append(Proposal(
                book_id=report.book.id,
                book_uuid=report.book.uuid,
                field="identifier.delete",
                calibre_value=g.current,
                proposed_value=scheme,
                source=SOURCE,
                confidence=0.95,
                notes=g.notes or f"remove junk identifier {scheme}",
            ))
            continue

        if g.proposed is None:
            continue
        out.append(Proposal(
            book_id=report.book.id,
            book_uuid=report.book.uuid,
            field=g.field,
            calibre_value=g.current,
            proposed_value=g.proposed,
            source=g.source or SOURCE,
            confidence=g.confidence,
            notes=g.notes or None,
        ))
    return out


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------


def audit(
    *,
    metadata_db: Path,
    book_id: int,
    do_lookup: bool = False,
) -> AuditReport | None:
    """Run an audit for one book. Returns None if the book doesn't exist."""
    cal = calibre_reader.open_readonly(metadata_db)
    try:
        snap = snapshot(cal, book_id)
        if snap is None:
            return None
        report = AuditReport(book=snap, gaps=detect_gaps(snap))
        fill_series(report, cal)
        if do_lookup:
            fill_lookups(report, cal)
        return report
    finally:
        cal.close()


def persist(
    report: AuditReport, *,
    proposals_db: Path, metadata_db: Path,
) -> tuple[int, int]:
    """Insert proposals into the queue. Returns ``(inserted, deduped)``."""
    pc = proposals.connect(proposals_db)
    try:
        run_id = proposals.start_run(
            pc,
            source=SOURCE,
            library_root=Path(metadata_db).parent,
            metadata_db_path=metadata_db,
            dry_run=False,
        )
        inserted = deduped = 0
        for p in to_proposals(report):
            if proposals.insert_proposal(pc, run_id, p):
                inserted += 1
            else:
                deduped += 1
        proposals.finish_run(
            pc, run_id,
            books_scanned=1,
            books_with_epub=0,
            books_parsed=1,
            proposals_emitted=inserted,
            errors=0,
            notes=f"book_id={report.book.id}",
        )
    finally:
        pc.close()
    return inserted, deduped
