# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Fork context

This is **Erik's fork** (`talvola/calibremcp`) of `sandraschi/calibremcp`. The README, justfile, `.cursorrules`, and most upstream docs describe Sandra's original surface: a FastMCP 3.2 server exposing Calibre over MCP, with LanceDB RAG, a webapp, a Calibre plugin, etc. **That surface is not the active work.** Erik's fork adds and iterates on the cleanup pipeline under `src/calibre_mcp/cleanup/`. Treat upstream files as reference unless explicitly asked to touch them.

- `origin` → `talvola/calibremcp` (push here)
- `upstream` → `sandraschi/calibremcp` (read-only)
- HTTPS via `gh`; default pushes go to origin.

## Cleanup pipeline (the active codebase)

A propose-and-apply system that mines metadata fixes for a Calibre library and writes them to a **separate SQLite DB** (`.cache/cleanup_proposals.db`) for human review. Only an explicit apply step touches Calibre's `metadata.db` — and only via the `calibredb` CLI, never by direct SQLite writes.

### Architecture

- **Two databases, deliberately separate.** Calibre's `metadata.db` is read-only to the pipeline except via `calibredb`; the propose-queue lives in its own SQLite DB (`src/calibre_mcp/cleanup/schema.sql`). The `proposals` table has a unique index on `(book_id, field, source, proposed_value)` — re-running any miner is a no-op for already-seen proposals, which is what makes `miner refresh` safe to run repeatedly.
- **Phases** (each is a separate miner subcommand, all idempotent):
  - Phase 1 — `miner run`: OPF parser diffs each book's `.epub` against Calibre, fills empty fields, marks disagreements as `conflict`.
  - Phase 2 — `miner goodreads`: ISBN-based Goodreads ID lookup (rate-limited, browser UA).
  - Phase 3b — `miner normalize-tags`: classifies `tags.add` proposals as approve/reject/defer using existing vocabulary.
  - Phase 4 — tag cleaner (`tag-clean scan|review|apply|audit`): library-level delete/merge/rename ops in a separate `tag_ops` table.
  - Phase 5a — `miner cookbooks` + `cookbooks-review`: LLM-tags Cookbooks bucket; 47-label taxonomy v3.
  - Phase 6a/6b — `miner author-sweep` / `author-lookup`: punctuation-variant cluster merges + OpenLibrary/VIAF resolution.
  - Phase 7a/7b — `miner perry-rhodan` / `star-trek` / `star-trek-fixup`: series-specific title normalizers.
  - Mystery / Sci-Fi / Fantasy taggers — same shape as cookbooks; bucket-tag → LLM subgenre proposals.
  - `miner audit-book` — per-book gold-standard audit (cover, series, ISBN/Goodreads/Amazon identifiers, junk-identifier removal). Library-consult-first for series names so we don't introduce variants of existing series. Lookup order: OpenLibrary → Google Books → Goodreads ISBN redirect → Goodreads page scrape.
- **`miner refresh`** runs Phases 1+3b+4 back-to-back in propose-only mode; the standard entry point after Erik adds books or changes a classifier. The `refresh-library` skill walks through the full mount-dance.
- **`apply` is the only writer to Calibre.** Always dry-run by default; requires `--execute`. Spawns `calibredb` once per book. Use `scripts/apply_with_progress.py` for large batches (>500 commands).
- Status states: `proposed` → `approved` → `applied`; or `rejected` / `superseded` / `conflict`.

### Common commands

```bash
# Setup
uv sync                          # install from lockfile
uv sync --all-extras             # + dev deps (pytest, etc.)

# Test / lint
uv run pytest                    # full suite
uv run pytest tests/unit -q      # fast unit-only
uv run pytest tests/path/to/test_foo.py::test_bar  # single test
uv run ruff check . --fix --unsafe-fixes
uv run ruff format .

# Cleanup pipeline (all subcommands under one CLI)
uv run python -m calibre_mcp.cleanup --help

# Typical refresh flow (propose-only, ro mount OK)
uv run python -m calibre_mcp.cleanup miner refresh \
  --library /mnt/calibre/books \
  --metadata-db /mnt/calibre/books/metadata.db \
  --proposals-db .cache/cleanup_proposals.db \
  --since-book-id <max_old_id+1> --show-new

# Review / approve / apply
uv run python -m calibre_mcp.cleanup miner report --proposals-db .cache/cleanup_proposals.db
uv run python -m calibre_mcp.cleanup miner review --proposals-db .cache/cleanup_proposals.db --field tags.add
uv run python -m calibre_mcp.cleanup miner approve --proposals-db ... --source <source> --yes
uv run python -m calibre_mcp.cleanup miner apply --proposals-db ... --library-path /mnt/calibre/books --execute

# Review webapp (FastAPI)
uv run python -m calibre_mcp.cleanup miner web \
  --proposals-db .cache/cleanup_proposals.db \
  --calibre-db /mnt/calibre/books/metadata.db

# Per-book audit (read-only inspection; add --lookup for network sources,
# --fetch-cover to download the cover, --propose to persist as proposals)
uv run python -m calibre_mcp.cleanup miner audit-book \
  --metadata-db /mnt/calibre/books/metadata.db \
  --proposals-db .cache/cleanup_proposals.db \
  --book-id 22488 --lookup --fetch-cover --propose
```

`justfile` is **pwsh-only** (Sandra's Windows fleet); ignore it on Linux/WSL — invoke `uv run` directly.

## Library mount & write workflow

The Calibre library lives on a NAS (AS6604 at `192.168.1.64`) mounted at `/mnt/calibre` via CIFS. Steady state is **read-only**. Writing to Calibre requires:

1. Stopping the **Calibre-Web** container (Portainer) — it holds an exclusive write lock on `metadata.db`.
2. Remounting **rw** (admin remount; never `sudo` from Claude's shell — ask Erik).
3. Backing up `metadata.db` to `.cache/backups/metadata-pre-<reason>-<timestamp>.db`.
4. Running the apply.
5. Erik remounts ro + restarts Calibre-Web.

Phase 1 OPF mining over CIFS is slow (~50–200ms/EPUB → 30+ min full-library). Always pass `--since-book-id` when only new books matter.

## Conventions

- **Series** sourced from Goodreads. `series_index = 1.3` = omnibus, `1.5` = in-between volume.
- **Tags are genres.** Anchor proposals in existing library vocabulary; the Phase 3b classifier defers anything it doesn't recognize.
- Confidence is `0.0–1.0`; under `0.6` typically defers rather than proposes.
- Date strings in memories/proposals are absolute (`2026-05-21`), never relative.
- Python ≥ 3.12; ruff line length 120; double-quoted strings; `from __future__ import annotations` is standard.

## What lives where

- `src/calibre_mcp/cleanup/` — the pipeline. All new work goes here.
- `src/calibre_mcp/mcp_server/` — the live Calibre MCP server (Erik's, built 2026-05). Stdio + streamable-HTTP, OAuth-protected at `https://erikbooks.duckdns.org/mcp`. Read tools wrap `calibre_reader` (without `immutable=1` so it sees fresh WAL writes); write tools post to Calibre-Web's `/admin/book/<id>` form. Deployed as Docker stack id=7 on the NAS — see [[reference_nas_deploy]] memory.
- `src/calibre_mcp/cleanup/__main__.py` — the unified `miner` / `tag-clean` CLI (argparse, Rich tables).
- `src/calibre_mcp/cleanup/schema.sql` — propose-queue + `tag_ops` schema; load idempotently on first connect.
- `src/calibre_mcp/cleanup/proposals.py` — the only module that writes to the propose-queue.
- `src/calibre_mcp/cleanup/calibre_reader.py` — read-only wrapper around Calibre's `metadata.db`.
- `src/calibre_mcp/cleanup/apply.py` — the only module that spawns `calibredb` (the actual writer).
- `src/calibre_mcp/cleanup/web.py` — FastAPI review UI; covers per-book and per-tag groupings.
- `scripts/apply_with_progress.py` — long-running apply with progress bar, intended for the large batches.
- `.cache/` — proposals DB, backups, snapshots; gitignored.
- Everything else under `src/calibre_mcp/` (server.py, tools/, rag/, webapp/, calibre_plugin/, mcpb/) — Sandra's upstream MCP server surface, unrelated to `mcp_server/`. Don't modify unless asked.

## Pre-existing pitfalls worth knowing

- Don't write directly to Calibre's `metadata.db`. Always go through `calibredb` (subprocess) so triggers, search indexes, and content-server caches stay consistent.
- **`calibredb` has no `set_cover` subcommand.** Covers go through `set_metadata --field cover:<absolute-path>` (calibre 5.37). Failures show as `rc=1` with empty stderr — calibredb dumps help text to stdout instead.
- **OpenLibrary placeholder cover is exactly 19,501 bytes** (image/jpeg). Seen when Calibre-Web auto-fetched a cover URL that resolved to OL's "no image" placeholder. The `calibre_view_cover` MCP tool's "under 2KB" heuristic doesn't catch it — check for this exact byte count when a book "has a cover" but the detail page shows none.
- **SQLite `mode=ro` is flaky over CIFS.** Always combine with `immutable=1` (see `calibre_reader.open_readonly`); without it, ~10% of opens fail with "unable to open database file". `immutable=1` is safe because the canonical workflow keeps the mount ro and stops Calibre-Web before any apply.
- **`src/calibre_mcp/__init__.py` calls `basicConfig(level=DEBUG)` at import** — preempts every downstream CLI's `--log-level`. The cleanup CLI re-applies via `getLogger().setLevel(args.log_level)` after its own `basicConfig`. Mirror that pattern in any new CLI under this package or expect debug spam.
- The unique index on proposals means a "same field, same source, same proposed value" insert is silently dropped. Tag-level ops with `book_id=0` use this too.
- When a classifier produces a no-op (e.g., `Conan → Conan`), reject the bogus rows rather than leaving them queued — the queue should reflect intent.
- For new bulk operations, back up `cleanup_proposals.db` too, not just `metadata.db`.
- **MCP `calibre_update_book`:** `title`/`authors`/`tags`/etc are plain text — do NOT HTML-escape `&`. Only `comments` is HTML. Authors uses `&` as separator, so escaping `&` to `&amp;` splits one author into two.
- **Author display order** keys off `books.author_sort`, which Calibre-Web re-derives from `authors` on every form save **in the order given** (verified in 0.6.26 `handle_author_on_edit`; not alphabetical). So passing `authors="A & B"` alone reorders display. The `/admin/book/<id>` form has no `author_sort` input; the MCP `calibre_update_book` tool's `author_sort` parameter POSTs to `/ajax/editbooks/author_sort` (wire format `{"pk":[<int>], "value":"<Last, First & Last, First>"}`) for explicit overrides. `books_authors_link` row order is never rewritten by Calibre-Web, so it goes stale — never read display order from it.
- **Goodreads lookups:** `WebFetch` on goodreads.com is bot-blocked (returns empty). Use `WebSearch "site:goodreads.com <query>"` to get IDs via Google's index. Books in the current/coming year often aren't indexed yet — accept partial coverage.
- **Local stdio MCP serves stale reads.** The `mcp__calibre__*` tools (configured in `.mcp.json`) read from `.cache/calibre-snapshot/metadata.db` (frozen), but writes go to live Calibre-Web. So `calibre_get_book` may say "no book with id X" or return pre-edit values even when the live data is fresh. For live reads use the remote MCP (`mcp__claude_ai_Calibre__*`, NAS deploy) or authenticated curl against `http://192.168.1.64:38083/admin/book/<id>` and grep on `name="..." value="..."`.
- **MCP OAuth token store is in-memory** — rebuilding/recreating the `calibre-mcp` container wipes all sessions. After any redeploy, Erik must re-authorize the Calibre connector in Claude Desktop (and any other clients) before tool calls work. If you see `MCP server requires re-authorization (token expired)` right after a deploy, that's why.
- **Per-book metadata review of newly-added books** is typically done interactively via the live remote MCP write tools (`mcp__claude_ai_Calibre__*`), not the pipeline's `miner audit-book`. Given only book IDs, always `calibre_get_book` each ID first and have the result in hand **before** any `calibre_update_book`. Never put a write in the same tool batch as the `get_book`/`WebSearch` lookup that feeds its values (title, goodreads ID, ISBN, pubdate) — you'll write guesses. Deterministic ASIN-from-ISBN-10 is the only safe in-batch derive. If a value (e.g. a Goodreads ID for a forthcoming title) can't be confirmed in a source, omit it rather than invent one.
- **Imports bury the real ISBN/ASIN under a junk identifier key.** Seen: `bookid`, `BookId=URN:ISBN:<isbn>`, `uid`, `eisbn`/`ebookisbn`, `pub-id`, `mobi-asin`, `disn_<isbn>`, `p<isbn>`. Convert to a proper `isbn` key (or `amazon` for `mobi-asin`/ASINs) and drop the junk key. `calibre_update_book identifiers=` is a **full replacement** — include every key you want to keep, or you'll wipe the others.
- **Covers via `calibre_set_cover_from_url`:** OpenLibrary by ISBN (`covers.openlibrary.org/b/isbn/<isbn>-L.jpg`) for catalogued books; Amazon by ASIN (`images-na.ssl-images-amazon.com/images/P/<ASIN>.01.LZZZZZZZ.jpg`) for Kindle-only/self-pub titles. The tool reports the stored byte count — confirm it isn't the 19,501-byte OL placeholder before trusting it.
- **Genre tags are stored Title Case** (`Cookbooks`, `Baking`, `Asian`, `Crime Fiction`) — but a few canonical tags are lowercase (`cookbooks`, `nonfiction`, `thriller`, `short stories`, `history`). Match the stored form exactly; `calibre_list_tags` is the authority. (Before 2026-09-01 `calibre_get_book` casefolded tags on output — fixed; it now reports stored case.)
- **MCP read path must never transform case or order.** `calibre_get_book` reports authors in Calibre-Web display order (driven by `books.author_sort`, which is shown on its own line — *not* `books_authors_link` order, which goes stale when authors are reordered via Calibre-Web), tags in stored case, and identifier types exactly as stored (Calibre-Web keeps import case, so `ISBN`/`FICTIONDB` junk really is uppercase in the DB). The cleanup pipeline's `BookRecord.tags_ci` / lowercase `identifiers` are for its own matching only; the MCP uses `tags` / `_identifiers_raw_for`. If get_book and the update diff ever disagree on the same data, the read path is the bug — writes have been correct in every case tested.
- **`calibre_update_book` with `series` but no `series_index`** stores index `1.0` (Calibre's default; there is no "unset"). The response now flags this. Goodreads list rank is not a usable index source for reprint series like British Library Crime Classics.
