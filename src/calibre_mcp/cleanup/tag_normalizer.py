"""Classify ``tags.add`` proposals as approve/reject/defer based on Erik's
canonical vocabulary plus pattern-based noise detection.

The normalizer is **conservative**: it only auto-decides cases where the
verdict is unambiguous. Everything else stays at ``status='proposed'`` for
human or LLM review later. Specifically:

- **Auto-approve** when the proposed tag case-folds to a tag already in
  Calibre's vocabulary (an "established" canonical match).
- **Auto-reject** when the proposed tag matches a known-noise pattern:
  * Library of Congress subject headings (``Horror tales, American``,
    ``France -- History -- Louis XIV, 1643-1715 -- Fiction``).
  * BISAC slash-paths (``FICTION / Christian / Futuristic``).
  * Underscore-prefixed source markers (``_isfdb``, ``_calibre_uuid``).
  * snake_case technical tags (``prose_contemporary``).
  * Format/source noise (``Epub2``, ``Epub3``, ``epubbud``, ``amazon``,
    ``kindle``, ``mobi``).
- **Defer** (leave proposed) for everything else — new genuine genre
  candidates, stylistic variants needing judgment (``sf``, ``cookbook``),
  and the long tail of one-offs.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from calibre_mcp.cleanup import calibre_reader, proposals

log = logging.getLogger(__name__)

SOURCE = "tag_normalizer"

Verdict = Literal["approve", "reject", "defer"]


# ---------------------------------------------------------------------------
# Noise patterns
# ---------------------------------------------------------------------------

# LoC-style subject heading: contains ` -- ` separators. These are
# bibliographic catalogue strings, not genre tags as Erik uses them.
_LOC_HEADING_RE = re.compile(r"\s--\s")

# LCSH-without-dashes: comma-separated nationality/language/period qualifier
# at the end. Examples: 'Horror tales, American', 'Detective and mystery
# stories, English', 'Fiction, Italian'. The trailing word is a Capitalized
# adjective.
_LCSH_QUALIFIER_RE = re.compile(r",\s+[A-Z][a-z]+\s*$")

# BISAC slash-path: starts with one of the BISAC top-level codes followed
# by ` / `. Picks up things like 'FICTION / Christian / Futuristic'.
_BISAC_PATH_RE = re.compile(
    r"^\s*(FICTION|NONFICTION|JUVENILE FICTION|JUVENILE NONFICTION|YOUNG ADULT|HISTORY|"
    r"BIOGRAPHY|BUSINESS|COOKING|COMPUTERS|EDUCATION|HEALTH|HUMOR|MUSIC|"
    r"PHILOSOPHY|POLITICAL SCIENCE|PSYCHOLOGY|RELIGION|SCIENCE|SOCIAL SCIENCE|"
    r"SPORTS|TRAVEL|TRUE CRIME)\s*/",
    re.IGNORECASE,
)

# Underscore prefix: `_isfdb`, `_calibre_uuid`, etc. — convention for
# source-system internal markers in some EPUB tooling.
_UNDERSCORE_PREFIX_RE = re.compile(r"^_")

# snake_case (lowercase letters + underscore between words). `prose_contemporary`,
# `media_tie_in`, etc. Almost always sourced from automated tagging tools.
_SNAKE_CASE_RE = re.compile(r"^[a-z]+(_[a-z0-9]+)+$")

# File-format / source markers that aren't genres. Match exact (case-folded).
_FORMAT_OR_SOURCE_NOISE: frozenset[str] = frozenset(
    s.casefold()
    for s in (
        "epub",
        "epub2",
        "epub3",
        "epubbud",
        "amazon",
        "kindle",
        "mobi",
        "azw",
        "azw3",
        "pdf",
        "calibre",
        "isfdb",
        "openlibrary",
    )
)

# Date-shaped tag values like '2014' or '2014-11-10' — sometimes leak into
# subjects from publisher feeds. Not a genre.
_DATE_LIKE_RE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")

# Version markers leaked from EPUB tooling: '(v5.0)', '(v 1.2)'.
_VERSION_MARKER_RE = re.compile(r"^\s*\(v[\s.]*\d", re.IGNORECASE)

# Bare punctuation like '-', '--', '.'. Not a tag.
_BARE_PUNCT_RE = re.compile(r"^[-_+.,;:!?\s]+$")

# Typo shape: single lowercase prefix glued onto a Capitalized real word —
# 'cThriller', 'aFiction', etc. (Observed in Erik's library at 8x for
# 'cThriller'; almost certainly a publisher-feed corruption.)
_TYPO_PREFIX_RE = re.compile(r"^[a-z][A-Z][a-z]+$")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TagDecision:
    proposal_id: int
    book_id: int
    proposed_value: str
    verdict: Verdict
    reason: str  # e.g. 'matches canonical Cthulhu', 'LoC heading'


@dataclass(frozen=True, slots=True)
class NormalizeSummary:
    run_id: int
    examined: int
    approved: int
    rejected: int
    deferred: int


# ---------------------------------------------------------------------------
# Canonical vocabulary
# ---------------------------------------------------------------------------


def load_canonical(calibre_conn: sqlite3.Connection) -> dict[str, str]:
    """Build ``{casefolded_name: canonical_name}`` from Calibre's tags table."""
    out: dict[str, str] = {}
    for row in calibre_conn.execute("SELECT name FROM tags WHERE name IS NOT NULL"):
        name = (row["name"] or "").strip()
        if not name:
            continue
        out[name.casefold()] = name
    return out


def load_author_and_series_names(calibre_conn: sqlite3.Connection) -> set[str]:
    """Build a casefolded set of author + series names. Used to reject tag
    proposals that are really just person/series names — those belong in
    the dedicated authors/series fields, not as genre tags."""
    out: set[str] = set()
    for row in calibre_conn.execute("SELECT name FROM authors WHERE name IS NOT NULL"):
        name = (row["name"] or "").strip()
        if name:
            out.add(name.casefold())
    for row in calibre_conn.execute("SELECT name FROM series WHERE name IS NOT NULL"):
        name = (row["name"] or "").strip()
        if name:
            out.add(name.casefold())
    return out


# ---------------------------------------------------------------------------
# Single-tag classification
# ---------------------------------------------------------------------------


def classify(
    proposed_value: str,
    canonical: dict[str, str],
    *,
    author_or_series_names: set[str] | None = None,
) -> tuple[Verdict, str]:
    """Decide approve/reject/defer for a single proposed tag value.

    Order matters: **noise patterns are checked before the canonical-match
    approval**. Calibre libraries that have been through prior auto-imports
    often contain pre-existing junk tags (``Epub3``, ``prose_contemporary``,
    ``amazon``, ``cThriller``); approving more proposals to those tags just
    propagates pollution. Reject by pattern, then look up canonical.
    """
    v = (proposed_value or "").strip()
    if not v:
        return "reject", "empty"
    cf = v.casefold()

    # 1. Noise patterns (run BEFORE canonical-match so existing junk doesn't propagate).
    if _LOC_HEADING_RE.search(v):
        return "reject", "LoC subject heading (' -- ' separators)"
    if _LCSH_QUALIFIER_RE.search(v):
        return "reject", "LCSH-style nationality/language qualifier"
    if _BISAC_PATH_RE.search(v):
        return "reject", "BISAC slash-path"
    if _UNDERSCORE_PREFIX_RE.match(v):
        return "reject", "underscore-prefixed (source marker)"
    if _SNAKE_CASE_RE.match(v):
        return "reject", "snake_case (machine-tagged)"
    if cf in _FORMAT_OR_SOURCE_NOISE:
        return "reject", "format/source marker, not a genre"
    if _DATE_LIKE_RE.match(v):
        return "reject", "date-shaped, not a genre"
    if _VERSION_MARKER_RE.match(v):
        return "reject", "version marker"
    if _BARE_PUNCT_RE.match(v):
        return "reject", "bare punctuation"
    if _TYPO_PREFIX_RE.match(v):
        return "reject", "typo: lowercase prefix on Capitalized word"
    if author_or_series_names and cf in author_or_series_names:
        return "reject", "author or series name (belongs in dedicated field)"

    # 2. Already in Erik's canonical vocabulary (case-fold match) → approve.
    if cf in canonical:
        return "approve", f"matches canonical {canonical[cf]!r}"

    # 3. Anything else: defer to human/LLM review.
    return "defer", "novel value — needs review"


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def run(
    *,
    calibre_db: Path,
    proposals_db: Path,
    dry_run: bool = False,
) -> NormalizeSummary:
    """Walk all ``status='proposed'`` ``tags.add`` rows and classify each.

    ``dry_run=True`` returns counts without modifying any rows — useful for
    previewing how the rule set behaves."""
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

    examined = approved = rejected = deferred = 0

    try:
        canonical = load_canonical(cal)
        author_series = load_author_and_series_names(cal)
        log.info(
            "loaded canonical tags=%d, author/series names=%d",
            len(canonical), len(author_series),
        )

        rows = pc.execute(
            "SELECT id, book_id, proposed_value FROM proposals WHERE field='tags.add' AND status='proposed'"
        ).fetchall()

        for row in rows:
            examined += 1
            verdict, reason = classify(
                row["proposed_value"], canonical, author_or_series_names=author_series,
            )
            if verdict == "approve":
                approved += 1
                if not dry_run:
                    pc.execute(
                        "UPDATE proposals SET status='approved', reviewed_at=CURRENT_TIMESTAMP, "
                        "notes=COALESCE(notes || '; ', '') || ? WHERE id=?",
                        (f"normalizer: {reason}", row["id"]),
                    )
            elif verdict == "reject":
                rejected += 1
                if not dry_run:
                    pc.execute(
                        "UPDATE proposals SET status='rejected', reviewed_at=CURRENT_TIMESTAMP, "
                        "notes=COALESCE(notes || '; ', '') || ? WHERE id=?",
                        (f"normalizer: {reason}", row["id"]),
                    )
            else:
                deferred += 1
    finally:
        proposals.finish_run(
            pc,
            run_id,
            books_scanned=examined,
            books_with_epub=0,
            books_parsed=examined,
            proposals_emitted=approved,
            errors=0,
            notes=f"deferred={deferred} rejected={rejected}",
        )
        cal.close()
        pc.close()

    return NormalizeSummary(
        run_id=run_id,
        examined=examined,
        approved=approved,
        rejected=rejected,
        deferred=deferred,
    )
