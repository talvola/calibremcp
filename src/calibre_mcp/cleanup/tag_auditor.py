"""Phase B: LLM tag auditor — batch-classify singleton/small tags.

Sends batches of low-book-count tags to Claude Haiku for classification:
  DELETE  — noise/junk/import artifact; safe to remove
  MERGE   — near-duplicate of a canonical tag; target provided
  KEEP    — legitimate browse category; preserve as-is

Results are saved as tag_ops rows (pass=7) in proposals.db.  KEEP decisions
are not written to tag_ops (nothing to do); only DELETE and MERGE are stored.
Run ``tag-clean apply`` afterwards to execute them.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import anthropic

from calibre_mcp.cleanup.tag_cleaner import TagOp, save_ops

log = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"
BATCH_SIZE = 100
MAX_BOOKS = 4   # audit tags with this many books or fewer
PASS_NUM = 7

# ---------------------------------------------------------------------------
# System prompt  (shared + cached across all batches)
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """\
You are auditing tags in a personal book library of ~21,000 books. Your goal: \
reduce tag noise so readers can browse by meaningful categories.

CANONICAL TAGS — established categories with many books.  Use these exact \
strings as MERGE targets:
{canonical_tags}

CLASSIFICATION RULES

DELETE — use for any of these:
• Keyword-stuffed import artifacts  ("authentic Japanese recipes", "keto diet for beginners")
• Typos or garbled strings  ("Scicnce Fiction", "animalled", "nepalifiction")
• LCSH catalog junk  ("Cookery; Chinese.", "1934-", "Authors; American")
• Publisher / series / blog names used as tags  ("RecipeTin Eats", "apex book company")
• Sentence-length tags  ("how to be a better parent", "great for weekend baking")
• Overly specific one-off settings  ("1930s San Francisco", "Bangkok (Thailand)")
• Nationality + genre qualifiers when the base genre already exists  ("American Horror Fiction")
• Amazon sub-category codes  ("FICTION_FANTASY_GENERAL")
• Tags that describe a single book rather than categorise it

MERGE:<target> — use when:
• The tag is a clear variant / synonym of one of the canonical tags above
• Use the EXACT canonical tag name as the target string

KEEP — use for:
• Recognised literary subgenres  ("Locked Room Mystery", "Solarpunk", "Grimdark", "Kaiju")
• Specific historical events / eras  ("American Civil War", "Cold War", "Prohibition Era")
• Distinct cultural or intellectual topics  ("Alchemy", "Norse Mythology", "Kabbalah")
• Cooking craft terms  ("Fermentation", "Sourdough", "Charcuterie")
• Named award categories  ("Hugo Award", "Edgar Award Winner")
• Character archetypes with an established readership  ("Chosen One", "Reluctant Hero")
• Any tag you would actually click on when browsing a library

When uncertain between DELETE and KEEP: prefer KEEP for recognised subject terms.
When uncertain between MERGE and KEEP: use MERGE only if the tag is clearly a variant, \
not a meaningful sub-category.

OUTPUT — respond with a JSON array ONLY, no markdown fences, no other text:
[{{"tag":"exact name","action":"DELETE"}},{{"tag":"exact name","action":"MERGE","target":"Canonical Name"}},{{"tag":"exact name","action":"KEEP"}}]
"""


def _build_system_prompt(canonical: list[tuple[str, int]]) -> str:
    tag_str = "\n".join(f"  {name} ({cnt:,})" for name, cnt in canonical)
    return _SYSTEM_TEMPLATE.format(canonical_tags=tag_str)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _canonical_tags(cal: sqlite3.Connection, min_books: int = 20) -> list[tuple[str, int]]:
    rows = cal.execute("""
        SELECT t.name, COUNT(btl.book) as cnt
        FROM tags t LEFT JOIN books_tags_link btl ON btl.tag = t.id
        GROUP BY t.id HAVING cnt >= ?
        ORDER BY cnt DESC
    """, (min_books,)).fetchall()
    return [(r["name"], r["cnt"]) for r in rows]


def _audit_tags(
    cal: sqlite3.Connection,
    ops_conn: sqlite3.Connection,
    max_books: int,
) -> list[tuple[str, int]]:
    """Tags with ≤ max_books books that are not already queued in tag_ops."""
    already_queued: set[str] = {
        r[0] for r in ops_conn.execute(
            "SELECT source_tag FROM tag_ops WHERE status IN ('pending','applied')"
        ).fetchall()
    }
    rows = cal.execute("""
        SELECT t.name, COUNT(btl.book) as cnt
        FROM tags t LEFT JOIN books_tags_link btl ON btl.tag = t.id
        GROUP BY t.id HAVING cnt BETWEEN 1 AND ?
        ORDER BY t.name COLLATE NOCASE
    """, (max_books,)).fetchall()
    return [
        (r["name"], r["cnt"])
        for r in rows
        if r["name"] not in already_queued
    ]


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _classify_batch(
    client: anthropic.Anthropic,
    system_prompt: str,
    tags: list[str],
    *,
    retries: int = 2,
) -> list[dict]:
    """Call Haiku; return parsed list of classification dicts.

    Each dict has ``action`` (DELETE/MERGE/KEEP) and optionally ``target``.
    Falls back to KEEP for any tag the model omits or that fails to parse.
    """
    tag_list_str = "\n".join(f"- {t}" for t in tags)
    user_msg = f"Classify these {len(tags)} tags:\n\n{tag_list_str}"

    for attempt in range(retries + 1):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=2048,
                system=[{
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }],
                messages=[{"role": "user", "content": user_msg}],
            )
            text = resp.content[0].text.strip()
            # Strip markdown fences if model adds them
            if text.startswith("```"):
                text = text[text.index("["):]
                text = text[:text.rindex("]") + 1]
            parsed: list[dict] = json.loads(text)

            # Index by tag name for fast lookup
            result_map = {d["tag"]: d for d in parsed if "tag" in d and "action" in d}

            # Ensure every submitted tag has a result (fall back to KEEP)
            out = []
            for t in tags:
                if t in result_map:
                    out.append(result_map[t])
                else:
                    log.warning("model omitted tag %r — defaulting to KEEP", t)
                    out.append({"tag": t, "action": "KEEP"})
            return out

        except (anthropic.APIError, json.JSONDecodeError, ValueError) as exc:
            log.warning("batch attempt %d/%d failed: %s", attempt + 1, retries + 1, exc)
            if attempt < retries:
                time.sleep(2 ** attempt)

    # Total failure — treat everything as KEEP (safe default)
    log.error("all retries exhausted for batch; defaulting to KEEP")
    return [{"tag": t, "action": "KEEP"} for t in tags]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@dataclass
class AuditResult:
    total_audited: int = 0
    deletes: int = 0
    merges: int = 0
    keeps: int = 0
    errors: int = 0
    ops_saved: int = 0
    kept_tags: list[str] = field(default_factory=list)


def run(
    *,
    calibre_db: Path,
    proposals_db: Path,
    max_books: int = MAX_BOOKS,
    batch_size: int = BATCH_SIZE,
    dry_run: bool = False,
    limit: int | None = None,
) -> AuditResult:
    """Audit all low-book-count tags via LLM and save ops to proposals_db.

    Parameters
    ----------
    calibre_db:    path to Calibre's metadata.db (read-only is fine)
    proposals_db:  path to cleanup_proposals.db
    max_books:     audit tags with this many books or fewer
    batch_size:    tags per LLM call
    dry_run:       if True, call the LLM and log decisions but don't write ops
    limit:         cap total tags audited (useful for test runs)
    """
    calibre_db   = Path(calibre_db).resolve()
    proposals_db = Path(proposals_db).resolve()

    cal = sqlite3.connect(f"file:{calibre_db}?mode=ro", uri=True)
    cal.row_factory = sqlite3.Row
    ops_conn = sqlite3.connect(proposals_db)
    ops_conn.row_factory = sqlite3.Row
    ops_conn.execute("PRAGMA foreign_keys = ON")
    ops_conn.execute("BEGIN")

    result = AuditResult()

    try:
        canonical = _canonical_tags(cal)
        log.info("%d canonical tags (≥20 books)", len(canonical))

        to_audit = _audit_tags(cal, ops_conn, max_books)
        if limit is not None:
            to_audit = to_audit[:limit]
        log.info("%d tags to audit (≤%d books, not already queued)", len(to_audit), max_books)

        if not to_audit:
            log.info("nothing to audit")
            return result

        system_prompt = _build_system_prompt(canonical)
        client = anthropic.Anthropic()

        batches = [
            to_audit[i : i + batch_size]
            for i in range(0, len(to_audit), batch_size)
        ]

        for batch_idx, batch in enumerate(batches):
            tag_names = [t for t, _ in batch]
            tag_count_map = {t: c for t, c in batch}

            log.info(
                "batch %d/%d  (%d tags)", batch_idx + 1, len(batches), len(tag_names)
            )

            decisions = _classify_batch(client, system_prompt, tag_names)
            result.total_audited += len(decisions)

            new_ops: list[TagOp] = []
            for d in decisions:
                tag = d.get("tag", "")
                action = d.get("action", "KEEP").upper()
                books = tag_count_map.get(tag, 0)

                if action == "DELETE":
                    result.deletes += 1
                    new_ops.append(TagOp(
                        op="delete",
                        source_tag=tag,
                        target_tag=None,
                        reason="LLM audit: delete",
                        books_affected=books,
                        pass_num=PASS_NUM,
                    ))
                elif action == "MERGE":
                    target = d.get("target", "")
                    if not target:
                        log.warning("MERGE with no target for %r — treating as KEEP", tag)
                        result.keeps += 1
                        result.kept_tags.append(tag)
                        continue
                    result.merges += 1
                    new_ops.append(TagOp(
                        op="merge",
                        source_tag=tag,
                        target_tag=target,
                        reason=f"LLM audit: merge → {target!r}",
                        books_affected=books,
                        pass_num=PASS_NUM,
                    ))
                else:  # KEEP
                    result.keeps += 1
                    result.kept_tags.append(tag)

            if not dry_run and new_ops:
                saved = save_ops(ops_conn, new_ops)
                result.ops_saved += saved

        ops_conn.execute("COMMIT")

    finally:
        cal.close()
        ops_conn.close()

    return result
