"""OPF miner — Phase 1 of the cleanup pipeline.

For each book in Calibre's library:
  1. Locate the ``.epub`` in the book's folder.
  2. Parse its OPF manifest.
  3. Diff against Calibre's current metadata.
  4. Emit proposals for empty Calibre fields the OPF can fill, and
     ``conflict`` proposals when both sides have values and they differ.

Only additive/fill-empty operations reach ``status='proposed'`` here;
disagreements land as ``status='conflict'`` for explicit review.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from calibre_mcp.cleanup import calibre_reader, opf_parser, proposals

log = logging.getLogger(__name__)

SOURCE = "opf"

# Identifier schemes that are Calibre/EPUB bookkeeping, not real catalog
# references — we never propose these as backfill.
_INTERNAL_IDENTIFIER_SCHEMES = frozenset({"uuid", "calibre", "guid", "book-id", "bookid", "id", "epub", "urn"})

# Non-ISBN identifier schemes we're willing to propose as backfill. Kept as
# an allow-list rather than a deny-list because the noise is open-ended —
# real-world OPFs include one-off junk like ``<dc:identifier opf:scheme=
# "9780061743900">9780061743900</dc:identifier>`` (bare ISBN used as a
# scheme name) that no deny-list can keep up with.
_ALLOWED_IDENTIFIER_SCHEMES = frozenset(
    {
        "amazon",
        "amazon_uk",
        "amazon_de",
        "amazon_fr",
        "amazon_it",
        "amazon_es",
        "amazon_jp",
        "asin",
        "mobi-asin",
        "mobi_asin",
        "goodreads",
        "google",
        "google-books",
        "google_books",
        "gbooks",
        "openlibrary",
        "ol",
        "isfdb",
        "librarything",
        "lt",
        "fictiondb",
        "barnesnoble",
        "bn",
        "kobo",
        "sonybookid",
        "sony",
        "doi",
        "ean",
        "lccn",  # Library of Congress
        "oclc",
        "dnb",  # Deutsche Nationalbibliothek
    }
)

# Tag values that are never worth proposing — useless generics and publisher
# marketing copy that aren't genres. Observed polluting the top of the
# tags.add distribution during the Phase 1 pilot run on Erik's library.
_NOISE_TAG_VALUES = frozenset(
    s.casefold()
    for s in (
        "fiction",
        "nonfiction",
        "non-fiction",
        "non fiction",
        "general",
        "general fiction",
        "general interest",
        "unknown",
        "adult",
        "ebook",
        "book",
        "null",
        "none",
        # Publisher marketing copy (not genres):
        "twists & turns",
        "twists and turns",
        "page-turner",
        "page turner",
        "intrigue",
        "gripping",
        "fast-paced",
        "fast paced",
        "suspenseful",
        # Encoding noise:
        "&#160;",
    )
)


@dataclass(frozen=True, slots=True)
class RunSummary:
    run_id: int
    books_scanned: int
    books_with_epub: int
    books_parsed: int
    proposals_emitted: int
    errors: int


def run(
    *,
    library_root: Path,
    metadata_db: Path,
    proposals_db: Path,
    sample: int | None = None,
    dry_run: bool = False,
) -> RunSummary:
    """Execute the miner. Returns counts + the persisted run_id."""
    library_root = Path(library_root).resolve()
    metadata_db = Path(metadata_db).resolve()

    books_scanned = books_with_epub = books_parsed = emitted = errors = 0

    with calibre_reader.open_library(metadata_db) as cal, proposals.connect(proposals_db) as pc:
        run_id = proposals.start_run(
            pc,
            source=SOURCE,
            library_root=library_root,
            metadata_db_path=metadata_db,
            dry_run=dry_run,
        )
        try:
            for book in calibre_reader.iter_books(cal, limit=sample):
                books_scanned += 1
                if books_scanned % 1000 == 0:
                    log.info(
                        "progress: scanned=%d with_epub=%d parsed=%d proposals=%d errors=%d",
                        books_scanned,
                        books_with_epub,
                        books_parsed,
                        emitted,
                        errors,
                    )
                epub_path = _find_epub(library_root, book.path)
                if epub_path is None:
                    continue
                books_with_epub += 1
                try:
                    opf = opf_parser.parse_epub(epub_path)
                except Exception as exc:  # noqa: BLE001 — we log and continue
                    errors += 1
                    proposals.record_error(
                        pc,
                        run_id,
                        book_id=book.id,
                        book_path=str(epub_path),
                        error=repr(exc),
                    )
                    continue
                if opf is None:
                    continue
                books_parsed += 1
                for proposal in _diff(book, opf):
                    if dry_run:
                        emitted += 1
                        log.debug("would emit proposal: %s", proposal)
                    else:
                        if proposals.insert_proposal(pc, run_id, proposal):
                            emitted += 1
        finally:
            proposals.finish_run(
                pc,
                run_id,
                books_scanned=books_scanned,
                books_with_epub=books_with_epub,
                books_parsed=books_parsed,
                proposals_emitted=emitted,
                errors=errors,
            )

    return RunSummary(
        run_id=run_id,
        books_scanned=books_scanned,
        books_with_epub=books_with_epub,
        books_parsed=books_parsed,
        proposals_emitted=emitted,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Filesystem lookup
# ---------------------------------------------------------------------------


def _find_epub(library_root: Path, relative_path: str) -> Path | None:
    """Return the (first) .epub in a book's folder, preferring exact format match."""
    folder = library_root / relative_path
    if not folder.is_dir():
        return None
    # Prefer the canonical format Calibre emits per-book. ``data`` table could
    # tell us the exact filename, but a simple glob is reliable enough and
    # avoids another query per book.
    candidates = sorted(folder.glob("*.epub"))
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Field diffing
# ---------------------------------------------------------------------------


def _diff(book: calibre_reader.BookRecord, opf: opf_parser.OpfMetadata) -> Iterator[proposals.Proposal]:
    """Emit zero or more proposals comparing Calibre's book to the OPF metadata."""
    # ISBN — canonical identifier, high-value backfill. Use digit-only
    # comparison so '978-1-59017-595-8' and '9781590175958' are recognised
    # as the same ISBN rather than a formatting-noise conflict.
    yield from _scalar("isbn", book.isbn, opf.isbn, book, confidence=1.0, equality=_isbn_equal)

    # Publisher, pubdate: lower-stakes but still valuable.
    yield from _scalar(
        "publisher",
        book.publisher,
        opf.publisher,
        book,
        confidence=0.9,
        equality=_calibre_richer_or_equal,
    )
    yield from _scalar("pubdate", book.pubdate, opf.pubdate, book, confidence=0.9)

    # Description: only fill if Calibre has none. Never conflict-overwrite — if
    # the user wrote a custom description, we don't second-guess it.
    if opf.description and not book.description:
        yield proposals.Proposal(
            book_id=book.id,
            book_uuid=book.uuid,
            field="description",
            calibre_value=None,
            proposed_value=opf.description,
            source=SOURCE,
            confidence=0.85,
            notes=f"length={len(opf.description)}",
        )

    # Series: propose name when Calibre has none; conflict when both populated
    # and differ. Treat as equivalent when Calibre's name already contains the
    # OPF's (e.g. 'Star Wars: Rebel Force' subsumes 'Rebel Force').
    yield from _scalar(
        "series",
        book.series,
        opf.series,
        book,
        confidence=0.9,
        equality=_calibre_richer_or_equal,
    )

    # series_index only meaningful alongside a series. Propose when Calibre has
    # none AND the OPF carries both a name and an index. Keep as text.
    if opf.series and opf.series_index and not book.series_index:
        yield proposals.Proposal(
            book_id=book.id,
            book_uuid=book.uuid,
            field="series_index",
            calibre_value=None,
            proposed_value=opf.series_index,
            source=SOURCE,
            confidence=0.9,
            notes=f"series={opf.series!r}",
        )

    # Tags — additive only. Suppress known-noise values (generic like 'Fiction',
    # publisher marketing like 'Page-Turner', encoding junk) since they'd
    # pollute Erik's curated vocabulary. Remaining proposals still need
    # Phase 3 tag-normalization before they're applied.
    for subject in opf.subjects:
        cf = subject.casefold()
        if cf in _NOISE_TAG_VALUES:
            continue
        if cf in book.tags_ci:
            continue
        yield proposals.Proposal(
            book_id=book.id,
            book_uuid=book.uuid,
            field="tags.add",
            calibre_value=None,
            proposed_value=subject,
            source=SOURCE,
            confidence=0.6,
        )

    # Non-ISBN identifiers (amazon, goodreads, google, openlibrary, etc.) —
    # additive per scheme. Catch all ISBN-like schemes (``eisbn``, ``e-isbn``,
    # ``isbn-paperback`` etc.) via the substring check so their values get
    # routed through the ISBN pipeline rather than emitted as separate
    # ``identifier.eisbn`` rows. Unknown schemes are skipped entirely —
    # better to miss a rare valid source than to pollute the queue with
    # garbage like ``identifier.9780061743900`` (bare-ISBN-as-scheme-name).
    for scheme, value in opf.identifiers.items():
        if "isbn" in scheme:
            continue  # covered by the ISBN picker above
        if scheme in _INTERNAL_IDENTIFIER_SCHEMES or scheme.startswith("urn"):
            continue
        if scheme not in _ALLOWED_IDENTIFIER_SCHEMES:
            continue
        field = f"identifier.{scheme}"
        if scheme in book.identifiers:
            if book.identifiers[scheme] == value:
                continue
            yield proposals.Proposal(
                book_id=book.id,
                book_uuid=book.uuid,
                field=field,
                calibre_value=book.identifiers[scheme],
                proposed_value=value,
                source=SOURCE,
                confidence=0.7,
                conflict_reason="identifier already set to different value",
            )
        else:
            yield proposals.Proposal(
                book_id=book.id,
                book_uuid=book.uuid,
                field=field,
                calibre_value=None,
                proposed_value=value,
                source=SOURCE,
                confidence=0.8,
            )


def _case_equal(a: str, b: str) -> bool:
    return a.strip().casefold() == b.strip().casefold()


def _strict_equal(a: str, b: str) -> bool:
    return a.strip() == b.strip()


def _isbn_equal(calibre_value: str, opf_value: str) -> bool:
    """Compare ISBNs by digit content, ignoring dash/space formatting.

    Calibre often stores ISBNs with dashes (``978-1-59017-595-8``) while the
    OPF parser has already stripped them (``9781590175958``). These are the
    same ISBN; flagging them as a conflict is noise.

    Genuinely different ISBNs (including ISBN-10 vs ISBN-13 of the *same*
    book — those share the root digits but not the check digit) remain
    flagged so a human can decide which edition to keep."""
    ca = opf_parser._isbn_digits(calibre_value)
    cb = opf_parser._isbn_digits(opf_value)
    return bool(ca and cb and ca == cb)


def _calibre_richer_or_equal(calibre_value: str, opf_value: str) -> bool:
    """Treat Calibre's value as 'already good enough' when it case-equals the
    OPF value or already contains it.

    Handles common formatting variance where Calibre's already-curated string
    is a superset of what the OPF carries — not a real disagreement:
      - 'Scholastic Inc.' vs 'Scholastic'            (Calibre has a suffix)
      - 'Star Wars: Rebel Force' vs 'Rebel Force'    (Calibre has a prefix)
      - 'Tor Books' vs 'Tor'                          (Calibre has a suffix)

    Require the OPF token to be at least 3 chars so we don't treat 'Inc' or
    'NY' alone as sufficient evidence of subsumption."""
    a = calibre_value.strip().casefold()
    b = opf_value.strip().casefold()
    if not a or not b:
        return False
    if a == b:
        return True
    return len(b) >= 3 and b in a


def _scalar(
    field: str,
    calibre_value: str | None,
    proposed_value: str | None,
    book: calibre_reader.BookRecord,
    *,
    confidence: float,
    equality: Callable[[str, str], bool] = _strict_equal,
) -> Iterator[proposals.Proposal]:
    """Diff a scalar field. Fill-if-empty, conflict-on-mismatch."""
    if not proposed_value:
        return
    if not calibre_value:
        yield proposals.Proposal(
            book_id=book.id,
            book_uuid=book.uuid,
            field=field,
            calibre_value=None,
            proposed_value=proposed_value,
            source=SOURCE,
            confidence=confidence,
        )
        return
    if equality(calibre_value, proposed_value):
        return
    yield proposals.Proposal(
        book_id=book.id,
        book_uuid=book.uuid,
        field=field,
        calibre_value=calibre_value,
        proposed_value=proposed_value,
        source=SOURCE,
        confidence=confidence,
        conflict_reason=f"calibre has {calibre_value!r}, opf has {proposed_value!r}",
    )
