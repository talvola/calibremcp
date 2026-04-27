"""Detect and propose merges for author-name variants that differ only by
punctuation or whitespace — Phase 6a of the cleanup pipeline.

Two names that differ only in the presence/absence of periods after initials
(e.g. ``"Arthur C Clarke"`` vs ``"Arthur C. Clarke"``) normalise to the same
key.  This module:

1. Queries Calibre's ``authors`` table.
2. Groups names by their normalised key.
3. Within each cluster of variants, picks a canonical form (most periods wins;
   book count is the tiebreak).
4. Emits one ``author.merge`` proposal per non-canonical variant.

``author.merge`` proposals use the same sentinel / encoding conventions as
``tag.merge`` — ``book_id=0``, ``calibre_value=src``,
``proposed_value="<src> -> <canonical>"``.  The ``apply`` step expands each
approved proposal into per-book ``calibredb set_metadata --field authors:…``
calls.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from calibre_mcp.cleanup import calibre_reader, proposals

log = logging.getLogger(__name__)

SOURCE = "author_normalizer"
AUTHOR_PROPOSAL_BOOK_ID = 0  # sentinel — author-scoped, not book-scoped


# ---------------------------------------------------------------------------
# Explicit overrides — cases the punctuation algorithm gets wrong
# ---------------------------------------------------------------------------
# Maps a variant name to the correct canonical, bypassing the cluster
# algorithm.  Two reasons to add entries here:
#
# 1. *Typo periods* — a period appears after a full name, not an initial
#    (e.g. ``"Henry. Petroski"``). The algorithm mistakenly treats this as
#    a "has initials" signal and picks the wrong direction.
#
# 2. *Cross-cluster unification* — two normalisation keys that should
#    collapse to one canonical (e.g. ``"R. R."`` vs ``"R.R."`` both strip to
#    different keys: ``"r r"`` vs ``"rr"``).  Add every variant in both
#    clusters here pointing to the agreed canonical.

EXPLICIT_OVERRIDES: dict[str, str] = {
    # Typo periods — period lands on a full name, not an initial.
    "Henry. Petroski": "Henry Petroski",
    "Tyler. Cowen": "Tyler Cowen",

    # George R.R. Martin — "R.R." and "R. R." produce different cluster keys
    # ("rr" vs "r r").  Merge everything to the compact "R.R." form.
    "George R. R. Martin": "George R.R. Martin",
    "George R R Martin": "George R.R. Martin",
    "George R .R. Martin": "George R.R. Martin",

    # P.C. Cast — same issue: "P. C." vs "P.C." are separate clusters.
    "P. C. Cast": "P.C. Cast",
    "P C Cast": "P.C. Cast",

    # Keith R.A. DeCandido — cluster algorithm lowercases "De" → "Decandido".
    # Preserve the capital; the canonical doesn't yet exist in the DB but
    # calibredb will create it on first write.
    "Keith R A DeCandido": "Keith R.A. DeCandido",
    "Keith R. A. Decandido": "Keith R.A. DeCandido",

    # Ursula K. Le Guin — the cluster algorithm lowercases the "Le" particle.
    # Her preferred styling has a capital L.
    "Ursula K Le Guin": "Ursula K. Le Guin",
    "Ursula K. le Guin": "Ursula K. Le Guin",

    # H.P. Lovecraft — "hp" and "h p" are different normalisation keys so the
    # cluster algorithm misses these.  Full name variant also merges here.
    "H.P. Lovecraft": "H. P. Lovecraft",
    "Howard Phillips Lovecraft": "H. P. Lovecraft",
}


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def _normalize_key(name: str) -> str:
    """Strip trailing periods, remove internal periods, collapse whitespace, casefold.
    Also replaces underscores with spaces (e.g. ``MICHAEL_P.``) and collapses
    consecutive single-letter tokens so ``"B. V. Larson"`` and ``"B.V. Larson"``
    both become ``"bv larson"``.

    Examples::

        "Robert A. Heinlein"  → "robert a heinlein"
        "Brian M. Sammons."   → "brian m sammons"   (trailing period stripped)
        "B. V. Larson"        → "bv larson"
        "B.V. Larson"         → "bv larson"
        "MICHAEL_P. KUBE-MCDOWELL" → "michael p kube-mcdowell"
        "J.R.R. Tolkien"      → "jrr tolkien"
        "George R. R. Martin" → "george rr martin"
    """
    s = (name or "").strip().rstrip(".")
    s = s.replace("_", " ")   # MICHAEL_P → MICHAEL P
    s = s.replace(".", "")    # remove all periods
    s = re.sub(r"\s+", " ", s).strip()
    s = s.casefold()
    # Collapse consecutive single-letter tokens: "b v" → "bv", "r r" → "rr"
    tokens = s.split()
    merged: list[str] = []
    i = 0
    while i < len(tokens):
        if len(tokens[i]) == 1:
            run = tokens[i]
            while i + 1 < len(tokens) and len(tokens[i + 1]) == 1:
                i += 1
                run += tokens[i]
            merged.append(run)
        else:
            merged.append(tokens[i])
        i += 1
    return " ".join(merged)


def _pick_canonical(variants: list[tuple[str, int]]) -> str:
    """Pick the best canonical form from a list of ``(name, book_count)`` pairs.

    Prefers the form with the most periods (proper initial punctuation).
    Tiebreaks by book count — most-used wins.  Trailing periods are stripped
    from the winner before returning so ``"Aristotle."`` → ``"Aristotle"``.
    """
    def _score(item: tuple[str, int]) -> tuple[int, int]:
        name, count = item
        return (name.strip().rstrip(".").count("."), count)

    best_name, _ = max(variants, key=_score)
    return best_name.strip().rstrip(".")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AuthorDecision:
    src_name: str    # the variant name as it exists in the DB
    canonical: str   # the chosen canonical to merge into
    book_count: int  # how many books currently carry src_name


@dataclass(frozen=True, slots=True)
class SweepSummary:
    examined: int   # distinct author names scanned
    clusters: int   # clusters with >1 variant found
    merges: int     # author.merge proposals emitted


# ---------------------------------------------------------------------------
# Cluster detection
# ---------------------------------------------------------------------------


def find_missing_initial_pairs(
    calibre_conn: sqlite3.Connection,
) -> list[tuple[tuple[str, int], tuple[str, int]]]:
    """Find author pairs where one name is the other with a middle initial added.

    Matches names that share the same first and last token (case-insensitive)
    and where the longer name has exactly one extra middle token that looks like
    an initial (single letter or single letter + period).  Returns pairs as
    ``((shorter_name, shorter_count), (longer_name, longer_count))``.

    Example: ``("Joe Lansdale", 3), ("Joe R. Lansdale", 40)``.
    """
    rows = calibre_conn.execute("""
        SELECT a.name, COUNT(bal.book) AS n
        FROM authors a
        LEFT JOIN books_authors_link bal ON bal.author = a.id
        GROUP BY a.id
        ORDER BY a.name
    """).fetchall()

    # Build index: (first_cf, last_cf) → list of (name, token_count, book_count)
    by_boundary: dict[tuple[str, str], list[tuple[str, int, int]]] = defaultdict(list)
    for r in rows:
        name = (r["name"] or "").strip()
        tokens = name.split()
        if len(tokens) < 2:
            continue
        key = (tokens[0].casefold().rstrip("."), tokens[-1].casefold().rstrip("."))
        by_boundary[key].append((name, len(tokens), int(r["n"])))

    pairs = []
    for entries in by_boundary.values():
        if len(entries) < 2:
            continue
        # Compare every pair within the same first+last bucket.
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                a_name, a_toks, a_n = entries[i]
                b_name, b_toks, b_n = entries[j]
                # Exactly one token difference.
                if abs(a_toks - b_toks) != 1:
                    continue
                short, short_n = (a_name, a_n) if a_toks < b_toks else (b_name, b_n)
                long_, long_n  = (b_name, b_n) if a_toks < b_toks else (a_name, a_n)
                # The extra token must look like an initial (1-2 chars).
                long_tokens = long_.split()
                # Middle tokens = everything except first and last.
                middle = long_tokens[1:-1]
                extra = [t for t in middle if t.rstrip(".") not in short.split()]
                if len(extra) == 1 and len(extra[0].rstrip(".")) <= 2:
                    pairs.append(((short, short_n), (long_, long_n)))
    return pairs


def find_variant_clusters(
    calibre_conn: sqlite3.Connection,
) -> list[list[tuple[str, int]]]:
    """Return variant clusters from Calibre's ``authors`` table.

    Each cluster is a list of ``(name, book_count)`` pairs that share the same
    normalised key.  Only clusters with >1 distinct variant are returned.
    """
    rows = calibre_conn.execute("""
        SELECT a.name, COUNT(bal.book) AS n
        FROM authors a
        LEFT JOIN books_authors_link bal ON bal.author = a.id
        GROUP BY a.id
        ORDER BY a.name
    """).fetchall()

    by_key: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for r in rows:
        name = (r["name"] or "").strip()
        if not name:
            continue
        key = _normalize_key(name)
        if not key:
            continue
        by_key[key].append((name, int(r["n"])))

    return [vs for vs in by_key.values() if len(vs) > 1]


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def sweep(
    clusters: list[list[tuple[str, int]]],
    overrides: dict[str, str] | None = None,
) -> list[AuthorDecision]:
    """Produce one ``AuthorDecision`` per non-canonical variant per cluster.

    ``overrides`` (defaults to ``EXPLICIT_OVERRIDES``) takes priority over the
    cluster algorithm.  If the cluster-derived canonical is itself overridden,
    all other variants in that cluster inherit the override target.
    """
    if overrides is None:
        overrides = EXPLICIT_OVERRIDES
    decisions: list[AuthorDecision] = []
    for variants in clusters:
        cluster_canonical = _pick_canonical(variants)
        # If the cluster's canonical is itself overridden, all variants in
        # the cluster should target the override instead.
        effective_canonical = overrides.get(cluster_canonical, cluster_canonical)
        for name, count in variants:
            # Per-name override takes priority; fall back to effective_canonical.
            target = overrides.get(name, effective_canonical)
            if name == target:
                continue
            decisions.append(AuthorDecision(
                src_name=name,
                canonical=target,
                book_count=count,
            ))
    return decisions


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(
    *,
    calibre_db: Path,
    proposals_db: Path,
    dry_run: bool = False,
) -> tuple[SweepSummary, list[AuthorDecision]]:
    """Walk Calibre's authors table, find punctuation-variant clusters, and
    (unless ``dry_run``) persist ``author.merge`` proposals.
    """
    calibre_db = Path(calibre_db).resolve()
    proposals_db = Path(proposals_db).resolve()

    cal = calibre_reader.open_readonly(calibre_db)
    pc = proposals.connect(proposals_db)

    total_authors: int = cal.execute("SELECT COUNT(*) FROM authors").fetchone()[0]

    run_id = proposals.start_run(
        pc,
        source=SOURCE,
        library_root=calibre_db.parent,
        metadata_db_path=calibre_db,
        dry_run=dry_run,
    )

    try:
        clusters = find_variant_clusters(cal)
        decisions = sweep(clusters)

        if not dry_run:
            for d in decisions:
                _persist_merge(pc, run_id, d)
    finally:
        proposals.finish_run(
            pc,
            run_id,
            books_scanned=0,
            books_with_epub=0,
            books_parsed=total_authors,
            proposals_emitted=len(decisions) if not dry_run else 0,
            errors=0,
            notes=f"clusters={len(clusters)} merges={len(decisions)}",
        )
        cal.close()
        pc.close()

    return (
        SweepSummary(
            examined=total_authors,
            clusters=len(clusters),
            merges=len(decisions),
        ),
        decisions,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _persist_merge(
    pc: sqlite3.Connection, run_id: int, d: AuthorDecision,
) -> None:
    notes = json.dumps({"book_count": d.book_count})
    encoded = f"{d.src_name} -> {d.canonical}"
    proposals.insert_proposal(
        pc,
        run_id,
        proposals.Proposal(
            book_id=AUTHOR_PROPOSAL_BOOK_ID,
            field="author.merge",
            calibre_value=d.src_name,
            proposed_value=encoded,
            source=SOURCE,
            confidence=1.0,
            notes=notes,
        ),
    )


# ---------------------------------------------------------------------------
# Helpers for downstream consumers (apply step)
# ---------------------------------------------------------------------------


def parse_merge_proposed_value(proposed_value: str) -> tuple[str, str]:
    """Pull ``(src, canonical)`` from ``"<src> -> <canonical>"``."""
    if " -> " not in proposed_value:
        raise ValueError(
            f"author.merge proposed_value missing ' -> ': {proposed_value!r}"
        )
    src, target = proposed_value.rsplit(" -> ", 1)
    return src.strip(), target.strip()
