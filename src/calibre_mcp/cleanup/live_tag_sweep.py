"""Classify tags currently in Calibre's live ``tags`` table for delete/merge.

Phase 3b's ``tag_normalizer`` looked at *proposed* tag.add rows mined from
OPFs. This module is its sibling for tags **already in the library**: pattern
sweeps to retire bibliographic / format / source / character noise, plus a
curated merge map to fold variant spellings into Erik's canonical genre
vocabulary.

Two new proposal field types are produced and stored in the same
``cleanup_proposals.db`` propose-queue:

* ``tag.delete`` — kill the tag everywhere; ``calibre_value`` = ``proposed_value``
  = the source tag name. ``book_id`` = 0 sentinel (these proposals are
  tag-scoped, not book-scoped).
* ``tag.merge``  — fold the source tag into a target tag. ``calibre_value`` =
  source name, ``proposed_value`` = ``"<src> -> <target>"`` so two different
  sources merging into the same target don't collide on the
  ``(book_id, field, source, proposed_value)`` uniqueness index.

The ``apply`` step (in ``apply.py``) expands each approved tag.* proposal
into per-book ``calibredb set_metadata --field tags:...`` calls.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from calibre_mcp.cleanup import calibre_reader, proposals

log = logging.getLogger(__name__)

SOURCE = "live_tag_sweep"
TAG_PROPOSAL_BOOK_ID = 0  # sentinel — these proposals are not book-scoped


# ---------------------------------------------------------------------------
# Canonical vocabulary (Erik's curated genre tags)
# ---------------------------------------------------------------------------
# Single-concept Title-Case genre labels Erik treats as canonical. Used as
# the ALLOWED set of merge targets — patterns like "X - Y" only fold into Y
# when Y is in here, so we never accidentally promote junk like ``General``
# or ``Boston`` to a canonical tag.

CANONICAL_GENRES: frozenset[str] = frozenset({
    "Fiction",
    "Nonfiction",
    "Science Fiction",
    "Fantasy",
    "Mystery",
    "Cozy Mystery",
    "Cookbooks",
    "Horror",
    "Adventure",
    "Space Opera",
    "Thriller",
    "Crime",
    "Romance",
    "Historical Fiction",
    "Star Wars",
    "Star Trek Fiction",
    "Suspense",
    "Cocktails",
    "Urban Fantasy",
    "Young Adult",
    "Anthologies",
    "Classics",
    "Humor",
    "Biography",
    "Cthulhu Mythos",
    "Sci-Fi Short",
    "Music",
    "Reference",
    "Contemporary",
    "Short Stories",
    "Steampunk",
    "Alternate History",
    "Magical Realism",
    "Media Tie-In",
})

_CANONICAL_CF: dict[str, str] = {c.casefold(): c for c in CANONICAL_GENRES}


def _resolve_canonical(part: str, explicit: dict[str, str]) -> str | None:
    """Map ``part`` to the canonical tag it stands for, treating both the
    ``CANONICAL_GENRES`` set and the source names of ``explicit`` merges
    as canonical-equivalents. Used by the dash-collapse rule so e.g.
    ``Mystery & Detective - General`` folds correctly via the explicit
    ``"mystery & detective" → "Mystery"`` mapping."""
    cf = part.casefold()
    if cf in _CANONICAL_CF:
        return _CANONICAL_CF[cf]
    if cf in explicit:
        return explicit[cf]
    return None


# ---------------------------------------------------------------------------
# Curated explicit merges
# ---------------------------------------------------------------------------
# Hand-mapped folds for cases where the source name is too divergent from
# the target for a pattern to find — and for the Fold M&D / A&A / Star Trek
# / Cthulhu Mythos consolidations Erik specifically called out.

EXPLICIT_MERGES: dict[str, str] = {
    # Cthulhu Mythos consolidation — both lowercase variants fold to the new
    # canonical. The new canonical doesn't exist in the library yet; the
    # apply step will create it on first write.
    "cthulhu": "Cthulhu Mythos",
    "mythos": "Cthulhu Mythos",

    # Drop ampersand-compound canonicals into their simpler forms.
    "mystery & detective": "Mystery",
    "action & adventure": "Adventure",

    # SF stylistic variants.
    "sf": "Science Fiction",
    "sci-fi": "Science Fiction",
    "sci fi": "Science Fiction",
    "fantasy fiction": "Fantasy",

    # Pluralization drift.
    "thrillers": "Thriller",
    "anthology": "Anthologies",
    "cookbook": "Cookbooks",
    "cookery": "Cookbooks",
    "short story": "Short Stories",

    # Historical Fiction dash-shapes. NB: bare 'Historical' (299 books)
    # is *not* mapped — it could be nonfiction history; ambiguous case
    # left for human review.
    "fiction - historical": "Historical Fiction",
    "historical - general": "Historical Fiction",
    "movie-tv tie-in": "Media Tie-In",

    # Star-Trek-character tags fold into the existing Star Trek Fiction
    # canonical (Erik's note: series tag is enough, character tags add no
    # discovery value for closed franchises).
    "kirk; james t. (fictitious character)": "Star Trek Fiction",
    "picard; jean luc (fictitious character)": "Star Trek Fiction",
    "picard; jean-luc (fictitious character)": "Star Trek Fiction",
    "spock (fictitious character)": "Star Trek Fiction",
    "mccoy; leonard (fictitious character)": "Star Trek Fiction",
    "riker; william t. (fictitious character)": "Star Trek Fiction",
    "janeway; kathryn (fictitious character)": "Star Trek Fiction",
    "archer; jonathan (fictitious character)": "Star Trek Fiction",
    "q (fictitious character)": "Star Trek Fiction",

    # Star Wars characters → Star Wars (closed franchise; series covers it).
    "skywalker; luke (fictitious character)": "Star Wars",
    "skywalker; luke (fictitious character) - fiction": "Star Wars",
    "solo; han (fictitious character)": "Star Wars",
    "solo; han (fictitious character) - fiction": "Star Wars",
    "leia; princess (fictitious character)": "Star Wars",
    "leia; princess (fictitious character) - fiction": "Star Wars",

    # Multi-author character tags → keep, fold the LCSH variants into a
    # clean canonical.
    "holmes; sherlock (fictitious character)": "Sherlock Holmes",
    "holmes": "Sherlock Holmes",
    "sherlock holmes fiction": "Sherlock Holmes",
    "sherlock holmes novels": "Sherlock Holmes",
    "sherlock holmes novel": "Sherlock Holmes",
    "sherlock holmes short fiction": "Sherlock Holmes",
    "sherlock holmes short stories": "Sherlock Holmes",
    "sherlock holmes collections": "Sherlock Holmes",
    "sherlock holmes novella": "Sherlock Holmes",
    "sherlock holmes rivals": "Sherlock Holmes",
    "sherlock (fictitious character) -- fiction": "Sherlock Holmes",
    "bond; james (fictitious character)": "James Bond",
    "bond; james (fictitious character) - fiction": "James Bond",
    "wolfe; nero (fictitious character)": "Nero Wolfe",
    "conan (fictitious character)": "Conan",
    "conan": "Conan",
    "marlowe; philip (fictitious character)": "Philip Marlowe",
    "marlowe; philip (fictitious character) - fiction": "Philip Marlowe",
}

# Multi-author character canonicals not in CANONICAL_GENRES but that we
# still allow as merge targets (they're discovery facets, not genres).
EXPLICIT_TARGETS: frozenset[str] = frozenset(EXPLICIT_MERGES.values())

# Whitelist of short / year-shaped tags that look like noise but are real
# things (book / series titles, world names, etc.) Keep these as-is.
NOISE_PATTERN_WHITELIST: frozenset[str] = frozenset(s.casefold() for s in (
    "Oz", "Tv", "AI", "1632", "2001", "300", "1984",
))


# ---------------------------------------------------------------------------
# Noise patterns (deletion candidates)
# ---------------------------------------------------------------------------

_LOC_DASH_RE = re.compile(r"\s--\s")
_LCSH_FICTITIOUS_RE = re.compile(r"\(Fictitious character\)", re.IGNORECASE)
_LCSH_AUTHOR_LIFESPAN_RE = re.compile(r",\s+\d{4}[-–]\d{0,4}\s*$")
_BISAC_PATH_RE = re.compile(
    r"^\s*(FICTION|NONFICTION|JUVENILE FICTION|JUVENILE NONFICTION|YOUNG ADULT|HISTORY|"
    r"BIOGRAPHY|BUSINESS|COOKING|COMPUTERS|EDUCATION|HEALTH|HUMOR|MUSIC|"
    r"PHILOSOPHY|POLITICAL SCIENCE|PSYCHOLOGY|RELIGION|SCIENCE|SOCIAL SCIENCE|"
    r"SPORTS|TRAVEL|TRUE CRIME)\s*/",
    re.IGNORECASE,
)
_BISAC_CODE_RE = re.compile(r"^[A-Z]{3}\d{6}\b")
_UNDERSCORE_PREFIX_RE = re.compile(r"^_")
_SNAKE_CASE_RE = re.compile(r"^[a-z]+(_[a-z0-9]+)+$")
_DATE_LIKE_RE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")
_VERSION_MARKER_RE = re.compile(r"^\s*\(v[\s.]*\d", re.IGNORECASE)
_BARE_PUNCT_RE = re.compile(r"^[-_+.,;:!?\s]+$")
_TYPO_PREFIX_RE = re.compile(r"^[a-z][A-Z][a-z]+$")
_BARE_SHORT_RE = re.compile(r"^[A-Za-z0-9]{1,2}$")
_YEAR_RANGE_RE = re.compile(r"^\d{4}([-–]\d{2,4})?(\s*--\s*.+)?$")
_CENTURY_RE = re.compile(r"^\d{1,2}(st|nd|rd|th)\s+Century\b", re.IGNORECASE)
_DECADE_RE = re.compile(r"^\d{4}s$")
_BARE_ISBN_RE = re.compile(r"^\d{9,13}[Xx]?(\s+\d{9,13}[Xx]?)*$")

# Single-dash LCSH-shaped: ``X - Fiction`` / ``X - Nonfiction``. These are
# bibliographic catalogue strings (places, occupations, conflicts +
# "Fiction" suffix) — treated as deletion candidates UNLESS the prefix is
# itself a canonical genre (then we'd want to fold, not delete).
_LCSH_FICTION_SUFFIX_RE = re.compile(r"\s-\s+(Fiction|Nonfiction)\s*$", re.IGNORECASE)

# Junk modifier suffixes after "Canonical - X" that we treat as droppable
# qualifiers — fold to the canonical and discard the modifier.
_JUNK_MODIFIERS: frozenset[str] = frozenset(s.casefold() for s in (
    "general", "series", "unknown", "n/a", "none", "uncategorized",
    "-", "--",
))

_FORMAT_OR_SOURCE_NOISE: frozenset[str] = frozenset(s.casefold() for s in (
    "epub", "epub2", "epub3", "epubbud", "amazon", "kindle", "mobi",
    "azw", "azw3", "pdf", "calibre", "isfdb", "openlibrary", "goodreads",
    "ebook", "books",
))
_USELESS_GENERIC: frozenset[str] = frozenset(s.casefold() for s in (
    "general", "adult", "unknown", "misc", "miscellaneous", "other",
    "uncategorized", "none", "n/a",
))


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

Verdict = Literal["delete", "merge", "keep"]


@dataclass(frozen=True, slots=True)
class TagDecision:
    src_name: str
    src_tag_id: int
    book_count: int
    verdict: Verdict
    target: str | None  # the canonical name for merge; None for delete/keep
    reason: str


@dataclass(frozen=True, slots=True)
class SweepSummary:
    examined: int
    delete: int
    merge: int
    keep: int
    delete_book_links: int
    merge_book_links: int


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify(name: str, *, count: int) -> tuple[Verdict, str | None, str]:
    """Decide ``(verdict, target, reason)`` for one live tag.

    Order matters:
      1. Whitelist short check (so ``Oz`` / ``1632`` survive bare/date rules).
      2. Explicit merge map (most specific wins).
      3. Pattern-based deletes.
      4. Pattern-based merges (dash-collapse → canonical, ``& Y`` → canonical).
      5. Default keep.
    """
    v = (name or "").strip()
    if not v:
        return "delete", None, "empty name"
    cf = v.casefold()

    # 1. Whitelist
    if cf in NOISE_PATTERN_WHITELIST:
        return "keep", None, "whitelisted (real thing shaped like noise)"

    # 2. Explicit curated merges (highest priority). Skip if the source
    # already IS the canonical (exact string match) — happens when an
    # apply pass has already created the canonical Title-Case variant
    # and a subsequent sweep re-classifies it via the same map. A
    # different-case variant (e.g. lowercase 'conan') still merges so
    # the case-rename happens.
    if cf in EXPLICIT_MERGES:
        target = EXPLICIT_MERGES[cf]
        if v == target:
            return "keep", None, ""
        return "merge", target, f"explicit map → {target!r}"

    # 3. Deletion patterns
    if cf in _USELESS_GENERIC:
        return "delete", None, "useless generic"
    if cf in _FORMAT_OR_SOURCE_NOISE:
        return "delete", None, "format/source marker"
    if _LOC_DASH_RE.search(v):
        return "delete", None, "LCSH bibliographic ('--' separators)"
    if _LCSH_FICTITIOUS_RE.search(v):
        return "delete", None, "fictitious-character LCSH (no canonical match)"
    if _LCSH_AUTHOR_LIFESPAN_RE.search(v):
        return "delete", None, "LCSH author with lifespan"
    if _BISAC_PATH_RE.search(v):
        return "delete", None, "BISAC slash-path"
    if _BISAC_CODE_RE.match(v):
        return "delete", None, "BISAC numeric code"
    if _UNDERSCORE_PREFIX_RE.match(v):
        return "delete", None, "underscore-prefixed source marker"
    if _SNAKE_CASE_RE.match(v):
        return "delete", None, "snake_case (machine-tagged)"
    if _DATE_LIKE_RE.match(v):
        return "delete", None, "date-shaped"
    if _VERSION_MARKER_RE.match(v):
        return "delete", None, "version marker"
    if _BARE_PUNCT_RE.match(v):
        return "delete", None, "bare punctuation"
    if _TYPO_PREFIX_RE.match(v):
        return "delete", None, "typo prefix (lowercase on Capitalized)"
    if _BARE_SHORT_RE.match(v):
        return "delete", None, "bare 1-2 char (not whitelisted)"
    if _YEAR_RANGE_RE.match(v):
        return "delete", None, "year-range"
    if _CENTURY_RE.match(v):
        return "delete", None, "century descriptor"
    if _DECADE_RE.match(v):
        return "delete", None, "decade descriptor"
    if _BARE_ISBN_RE.match(v):
        return "delete", None, "bare ISBN"

    # 4. Pattern-based merges into canonical
    # 4a. Case-rename: 'cozy mystery' → 'Cozy Mystery'.
    if cf in _CANONICAL_CF and v != _CANONICAL_CF[cf]:
        target = _CANONICAL_CF[cf]
        return "merge", target, f"case-rename → {target!r}"

    # 4b. Dash-collapse: 'Fiction - Science Fiction' → 'Science Fiction'.
    # Prefer specific (non-Fiction/Nonfiction) canonical matches over the
    # bare-genre parents — folding 'Fiction - Espionage' to 'Fiction' would
    # silently destroy the Espionage signal. EXPLICIT_MERGES source names
    # are also treated as canonical-equivalents (so 'Mystery & Detective -
    # General' folds to 'Mystery' via the explicit map).
    if " - " in v:
        parts = [p.strip() for p in v.split(" - ")]
        canonical_matches = [
            t for t in (_resolve_canonical(p, EXPLICIT_MERGES) for p in parts)
            if t is not None
        ]
        specific = [m for m in canonical_matches if m not in ("Fiction", "Nonfiction")]
        if specific:
            target = specific[-1]  # rightmost specific
            return "merge", target, f"dash-collapse → {target!r}"
        # Only Fiction/Nonfiction matched (or nothing did). Drop the parent
        # only if the *non-canonical* parts are recognisable junk modifiers
        # — otherwise they may carry signal worth preserving.
        if canonical_matches:
            non_canonical = [
                p for p in parts
                if _resolve_canonical(p, EXPLICIT_MERGES) is None
            ]
            if all(p.casefold() in _JUNK_MODIFIERS for p in non_canonical):
                target = canonical_matches[-1]
                return "merge", target, f"dash-collapse (drop junk modifier) → {target!r}"
            # Specific genre info we don't have a canonical for — keep
            # the tag instead of collapsing into Fiction/Nonfiction.

        # No canonical match anywhere: bibliographic LCSH-shaped patterns
        # like 'Orphans - Fiction', 'Boston (Mass.) - Fiction', '1950-1953
        # - Fiction' are catalogue noise — delete.
        if _LCSH_FICTION_SUFFIX_RE.search(v):
            return "delete", None, "LCSH bibliographic ('X - Fiction' shape)"

    # 4c. Trailing '; Adjective' qualifier ('Science Fiction; American').
    # Resolve the base via canonical OR explicit-merge map so e.g.
    # 'Fantasy fiction; English' folds via 'fantasy fiction' → 'Fantasy'.
    m = re.match(r"^(.+?);\s+[A-Z][a-z]+\s*$", v)
    if m:
        base = m.group(1).strip()
        target = _resolve_canonical(base, EXPLICIT_MERGES)
        if target is not None:
            return "merge", target, f"strip nationality qualifier → {target!r}"

    # 5. Default: keep.
    return "keep", None, ""


def sweep(
    rows: Iterable[tuple[int, str, int]],
) -> list[TagDecision]:
    """Classify a batch of ``(tag_id, name, book_count)`` tuples.

    Caller supplies the rows so this is testable without a real Calibre DB.
    """
    out: list[TagDecision] = []
    for tag_id, name, count in rows:
        verdict, target, reason = classify(name, count=count)
        out.append(TagDecision(
            src_name=name,
            src_tag_id=tag_id,
            book_count=count,
            verdict=verdict,
            target=target,
            reason=reason,
        ))
    return out


# ---------------------------------------------------------------------------
# Live-DB driver
# ---------------------------------------------------------------------------


def load_live_tags(calibre_conn: sqlite3.Connection) -> list[tuple[int, str, int]]:
    """Read the ``tags`` table joined with ``books_tags_link`` for counts."""
    rows = calibre_conn.execute("""
        SELECT t.id, t.name, COUNT(btl.book) AS n
        FROM tags t
        LEFT JOIN books_tags_link btl ON btl.tag = t.id
        GROUP BY t.id
        ORDER BY n DESC, t.name
    """).fetchall()
    return [(int(r["id"]), r["name"] or "", int(r["n"])) for r in rows]


def run(
    *,
    calibre_db: Path,
    proposals_db: Path,
    dry_run: bool = False,
) -> tuple[SweepSummary, list[TagDecision]]:
    """Walk the live tags table, classify each, and (unless ``dry_run``)
    persist tag.delete / tag.merge proposals.

    Returns the summary plus the full decision list — caller renders it.
    """
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

    try:
        tags = load_live_tags(cal)
        decisions = sweep(tags)

        delete_count = merge_count = keep_count = 0
        delete_links = merge_links = 0
        for d in decisions:
            if d.verdict == "delete":
                delete_count += 1
                delete_links += d.book_count
                if not dry_run:
                    _persist_delete(pc, run_id, d)
            elif d.verdict == "merge":
                merge_count += 1
                merge_links += d.book_count
                if not dry_run:
                    _persist_merge(pc, run_id, d)
            else:
                keep_count += 1
    finally:
        proposals.finish_run(
            pc,
            run_id,
            books_scanned=0,
            books_with_epub=0,
            books_parsed=len(tags),
            proposals_emitted=delete_count + merge_count,
            errors=0,
            notes=f"delete={delete_count} merge={merge_count} keep={keep_count}",
        )
        cal.close()
        pc.close()

    return (
        SweepSummary(
            examined=len(decisions),
            delete=delete_count,
            merge=merge_count,
            keep=keep_count,
            delete_book_links=delete_links,
            merge_book_links=merge_links,
        ),
        decisions,
    )


def _persist_delete(pc: sqlite3.Connection, run_id: int, d: TagDecision) -> None:
    notes = json.dumps({"src_tag_id": d.src_tag_id, "book_count": d.book_count, "reason": d.reason})
    proposals.insert_proposal(
        pc,
        run_id,
        proposals.Proposal(
            book_id=TAG_PROPOSAL_BOOK_ID,
            field="tag.delete",
            calibre_value=d.src_name,
            proposed_value=d.src_name,
            source=SOURCE,
            confidence=1.0,
            notes=notes,
        ),
    )


def _persist_merge(pc: sqlite3.Connection, run_id: int, d: TagDecision) -> None:
    assert d.target is not None
    notes = json.dumps({"src_tag_id": d.src_tag_id, "book_count": d.book_count, "reason": d.reason})
    # Encode src→target in proposed_value so multiple sources merging into
    # the same target don't collide on the (book_id, field, source,
    # proposed_value) unique index.
    encoded = f"{d.src_name} -> {d.target}"
    proposals.insert_proposal(
        pc,
        run_id,
        proposals.Proposal(
            book_id=TAG_PROPOSAL_BOOK_ID,
            field="tag.merge",
            calibre_value=d.src_name,
            proposed_value=encoded,
            source=SOURCE,
            confidence=1.0,
            notes=notes,
        ),
    )


# ---------------------------------------------------------------------------
# Helpers for downstream consumers (apply step, CLI report)
# ---------------------------------------------------------------------------


def parse_merge_proposed_value(proposed_value: str) -> tuple[str, str]:
    """Pull (src, target) back out of the encoded ``"<src> -> <target>"``."""
    if " -> " not in proposed_value:
        raise ValueError(f"merge proposed_value missing ' -> ': {proposed_value!r}")
    src, target = proposed_value.rsplit(" -> ", 1)
    return src.strip(), target.strip()
