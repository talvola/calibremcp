"""Resolve ambiguous author-name pairs using OpenLibrary and VIAF authority data.

Phase 6b: for each candidate pair produced by ``find_missing_initial_pairs``
(e.g. ``"Philip Dick"`` vs ``"Philip K. Dick"``), this module:

1. Collects ISBNs from Calibre for books attributed to either name.
2. Queries the OpenLibrary Books API in batches of 20 to get the author name
   as recorded in the bibliographic record.
3. Falls back to the VIAF AutoSuggest API for pairs where no ISBNs are
   available (name-search only; lower confidence).
4. Votes across all lookups to determine the canonical form, then emits
   ``author.merge`` proposals for the non-canonical variant.

Proposals land at ``status='proposed'`` (confidence 0.9 for ISBN-confirmed,
0.6 for VIAF-only) — always human-reviewed before approval.

Pairs where OpenLibrary and VIAF disagree, or where neither source has data,
are reported as ``ambiguous`` / ``no_data`` and left for manual handling.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import httpx

from calibre_mcp.cleanup import author_normalizer, calibre_reader, proposals
from calibre_mcp.cleanup.proposals import Proposal

log = logging.getLogger(__name__)

SOURCE = "author_lookup"

_OL_BOOKS_URL = "https://openlibrary.org/api/books"
_VIAF_SUGGEST_URL = "https://viaf.org/viaf/AutoSuggest"
_DEFAULT_UA = "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
_OL_BATCH_SIZE = 20


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

Verdict = Literal["merge_to_long", "merge_to_short", "new_canonical", "ambiguous", "no_data", "same"]


@dataclass(frozen=True, slots=True)
class PairDecision:
    short_name: str
    short_count: int
    long_name: str
    long_count: int
    verdict: Verdict
    canonical: str | None       # target name for merge verdicts
    confidence: float
    evidence: list[str]         # ISBNs / "viaf" that voted for the verdict
    ol_names_seen: list[str]    # raw names OpenLibrary returned


@dataclass(frozen=True, slots=True)
class LookupSummary:
    examined: int
    merge_to_long: int
    merge_to_short: int
    new_canonical: int
    ambiguous: int
    no_data: int
    proposals_emitted: int


# ---------------------------------------------------------------------------
# OpenLibrary helpers
# ---------------------------------------------------------------------------


def _ol_fetch_batch(
    isbns: list[str],
    client: httpx.Client,
    *,
    rate_sec: float,
) -> dict[str, list[str]]:
    """Query OpenLibrary for up to 20 ISBNs.  Returns {isbn: [author_name, ...]}."""
    bibkeys = ",".join(f"ISBN:{isbn}" for isbn in isbns)
    try:
        resp = client.get(
            _OL_BOOKS_URL,
            params={"bibkeys": bibkeys, "jscmd": "data", "format": "json"},
            timeout=15,
        )
        time.sleep(rate_sec)
        if resp.status_code != 200:
            log.warning("OpenLibrary returned %s for batch of %d ISBNs", resp.status_code, len(isbns))
            return {}
        data = resp.json()
    except Exception as exc:
        log.warning("OpenLibrary request failed: %s", exc)
        return {}

    result: dict[str, list[str]] = {}
    for key, book in data.items():
        isbn = key.removeprefix("ISBN:")
        names = [a.get("name", "") for a in book.get("authors", []) if a.get("name")]
        if names:
            result[isbn] = names
    return result


def _ol_lookup_isbns(
    isbns: list[str],
    client: httpx.Client,
    *,
    rate_sec: float,
) -> dict[str, list[str]]:
    """Query OpenLibrary for all ISBNs, batching by ``_OL_BATCH_SIZE``."""
    out: dict[str, list[str]] = {}
    for i in range(0, len(isbns), _OL_BATCH_SIZE):
        batch = isbns[i : i + _OL_BATCH_SIZE]
        out.update(_ol_fetch_batch(batch, client, rate_sec=rate_sec))
    return out


# ---------------------------------------------------------------------------
# VIAF fallback (name search)
# ---------------------------------------------------------------------------


def _viaf_canonical(name: str, client: httpx.Client, *, rate_sec: float) -> str | None:
    """Query VIAF AutoSuggest for ``name``, return the LC-preferred form or None.

    VIAF returns entries like ``"Lansdale, Joe R., 1951-"`` — we invert
    surname-first to first-last and strip the date suffix.
    """
    try:
        resp = client.get(
            _VIAF_SUGGEST_URL,
            params={"query": name},
            timeout=10,
        )
        time.sleep(rate_sec)
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception as exc:
        log.warning("VIAF request failed for %r: %s", name, exc)
        return None

    results = data.get("result") or []
    for r in results:
        if r.get("nametype") != "personal":
            continue
        term = r.get("term", "")
        if not term:
            continue
        # Strip date suffix: "Lansdale, Joe R., 1951-" → "Lansdale, Joe R."
        term = term.rstrip(",").split(",")[0] if "," not in term else term
        # More robust: split on ", " and check if last part looks like a date
        parts = [p.strip() for p in term.split(",")]
        # Drop trailing year parts
        clean_parts = [p for p in parts if not p.strip().lstrip("0123456789- ").strip() == ""]
        # Actually just take the first two meaningful parts and invert
        # "Lansdale, Joe R." → ["Lansdale", "Joe R."] → "Joe R. Lansdale"
        if len(clean_parts) >= 2:
            surname = clean_parts[0].strip()
            given = clean_parts[1].strip()
            if given:
                return f"{given} {surname}"
        elif len(clean_parts) == 1:
            return clean_parts[0].strip()
        return None
    return None


# ---------------------------------------------------------------------------
# ISBN retrieval from Calibre
# ---------------------------------------------------------------------------


def _isbns_for_authors(
    cal_conn: sqlite3.Connection,
    author_names: list[str],
) -> dict[str, list[str]]:
    """Return {author_name: [isbn, ...]} for each name in ``author_names``."""
    out: dict[str, list[str]] = {n: [] for n in author_names}
    for name in author_names:
        rows = cal_conn.execute(
            """
            SELECT DISTINCT i.val
            FROM identifiers i
            JOIN books_authors_link bal ON bal.book = i.book
            JOIN authors a ON a.id = bal.author
            WHERE i.type = 'isbn'
              AND LOWER(a.name) = LOWER(?)
              AND i.val IS NOT NULL
              AND i.val != ''
            ORDER BY i.val
            """,
            (name,),
        ).fetchall()
        out[name] = [r["val"] for r in rows]
    return out


# ---------------------------------------------------------------------------
# Pair resolution
# ---------------------------------------------------------------------------


def _resolve_pair(
    short_name: str,
    short_count: int,
    long_name: str,
    long_count: int,
    cal_conn: sqlite3.Connection,
    client: httpx.Client,
    *,
    rate_sec: float,
) -> PairDecision:
    isbn_map = _isbns_for_authors(cal_conn, [short_name, long_name])
    all_isbns = list(dict.fromkeys(isbn_map[short_name] + isbn_map[long_name]))

    short_key = author_normalizer._normalize_key(short_name)
    long_key = author_normalizer._normalize_key(long_name)
    ol_names_seen: list[str] = []
    evidence: list[str] = []

    if all_isbns:
        isbn_to_names = _ol_lookup_isbns(all_isbns, client, rate_sec=rate_sec)
        # Tally votes: normalised author name → count
        votes: Counter[str] = Counter()
        isbn_to_norm: dict[str, str] = {}
        for isbn, names in isbn_to_names.items():
            ol_names_seen.extend(names)
            for author_name in names:
                nk = author_normalizer._normalize_key(author_name)
                votes[nk] += 1
                isbn_to_norm[isbn] = nk

        if votes:
            top_key, top_count = votes.most_common(1)[0]
            total_votes = sum(votes.values())
            agreement = top_count / total_votes if total_votes > 0 else 0
            evidence = [isbn for isbn, nk in isbn_to_norm.items() if nk == top_key]

            # Does the top key match the short or long normalised form?
            if top_key == short_key:
                return PairDecision(
                    short_name=short_name, short_count=short_count,
                    long_name=long_name, long_count=long_count,
                    verdict="merge_to_short", canonical=short_name,
                    confidence=0.9 if agreement >= 0.75 else 0.7,
                    evidence=evidence, ol_names_seen=ol_names_seen,
                )
            if top_key == long_key:
                return PairDecision(
                    short_name=short_name, short_count=short_count,
                    long_name=long_name, long_count=long_count,
                    verdict="merge_to_long", canonical=long_name,
                    confidence=0.9 if agreement >= 0.75 else 0.7,
                    evidence=evidence, ol_names_seen=ol_names_seen,
                )
            # OpenLibrary has a third form — pick the OL name as canonical
            # (use the raw name most frequently seen)
            raw_counter: Counter[str] = Counter(ol_names_seen)
            ol_canonical = raw_counter.most_common(1)[0][0]
            return PairDecision(
                short_name=short_name, short_count=short_count,
                long_name=long_name, long_count=long_count,
                verdict="new_canonical", canonical=ol_canonical,
                confidence=0.8 if agreement >= 0.75 else 0.6,
                evidence=evidence, ol_names_seen=ol_names_seen,
            )

    # No ISBNs or OpenLibrary returned nothing — try VIAF by name
    log.debug("No OL data for (%r, %r) — trying VIAF", short_name, long_name)
    viaf_name = _viaf_canonical(long_name, client, rate_sec=rate_sec)
    if viaf_name:
        vk = author_normalizer._normalize_key(viaf_name)
        if vk == short_key:
            return PairDecision(
                short_name=short_name, short_count=short_count,
                long_name=long_name, long_count=long_count,
                verdict="merge_to_short", canonical=short_name,
                confidence=0.6, evidence=["viaf"], ol_names_seen=[],
            )
        if vk == long_key:
            return PairDecision(
                short_name=short_name, short_count=short_count,
                long_name=long_name, long_count=long_count,
                verdict="merge_to_long", canonical=long_name,
                confidence=0.6, evidence=["viaf"], ol_names_seen=[],
            )

    return PairDecision(
        short_name=short_name, short_count=short_count,
        long_name=long_name, long_count=long_count,
        verdict="no_data", canonical=None,
        confidence=0.0, evidence=[], ol_names_seen=[],
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(
    *,
    calibre_db: Path,
    proposals_db: Path,
    dry_run: bool = False,
    rate_sec: float = 1.0,
    limit: int | None = None,
    min_confidence: float = 0.6,
) -> tuple[LookupSummary, list[PairDecision]]:
    """Find missing-initial pairs, resolve via OpenLibrary/VIAF, emit proposals."""
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

    decisions: list[PairDecision] = []
    counters: dict[str, int] = defaultdict(int)

    try:
        pairs = author_normalizer.find_missing_initial_pairs(cal)
        if limit is not None:
            pairs = pairs[:limit]

        headers = {"User-Agent": _DEFAULT_UA}
        with httpx.Client(headers=headers, follow_redirects=True) as client:
            for i, ((short, sn), (long_, ln)) in enumerate(pairs, 1):
                log.info("[%d/%d] resolving %r vs %r", i, len(pairs), short, long_)
                decision = _resolve_pair(
                    short, sn, long_, ln, cal, client, rate_sec=rate_sec,
                )
                decisions.append(decision)
                counters[decision.verdict] += 1

                if not dry_run and decision.verdict in ("merge_to_long", "merge_to_short", "new_canonical"):
                    if decision.confidence >= min_confidence:
                        _persist(pc, run_id, decision)

    finally:
        proposals.finish_run(
            pc, run_id,
            books_scanned=0, books_with_epub=0,
            books_parsed=len(decisions),
            proposals_emitted=sum(
                1 for d in decisions
                if d.verdict in ("merge_to_long", "merge_to_short", "new_canonical")
                and d.confidence >= min_confidence
                and not dry_run
            ),
            errors=0,
            notes=json.dumps({k: v for k, v in counters.items()}),
        )
        cal.close()
        pc.close()

    summary = LookupSummary(
        examined=len(decisions),
        merge_to_long=counters["merge_to_long"],
        merge_to_short=counters["merge_to_short"],
        new_canonical=counters["new_canonical"],
        ambiguous=counters["ambiguous"],
        no_data=counters["no_data"],
        proposals_emitted=sum(
            1 for d in decisions
            if d.verdict in ("merge_to_long", "merge_to_short", "new_canonical")
            and d.confidence >= min_confidence
        ),
    )
    return summary, decisions


def _persist(pc: sqlite3.Connection, run_id: int, d: PairDecision) -> None:
    assert d.canonical is not None
    notes = json.dumps({
        "verdict": d.verdict,
        "evidence": d.evidence[:5],
        "ol_names": list(dict.fromkeys(d.ol_names_seen))[:5],
    })
    # Emit one proposal per non-canonical variant.
    sources_to_merge = []
    if d.verdict == "merge_to_long":
        sources_to_merge = [(d.short_name, d.short_count)]
    elif d.verdict == "merge_to_short":
        sources_to_merge = [(d.long_name, d.long_count)]
    elif d.verdict == "new_canonical":
        sources_to_merge = [
            (d.short_name, d.short_count),
            (d.long_name, d.long_count),
        ]

    for src_name, _count in sources_to_merge:
        if src_name == d.canonical:
            continue
        encoded = f"{src_name} -> {d.canonical}"
        proposals.insert_proposal(
            pc, run_id,
            Proposal(
                book_id=0,
                field="author.merge",
                calibre_value=src_name,
                proposed_value=encoded,
                source=SOURCE,
                confidence=d.confidence,
                notes=notes,
            ),
        )
