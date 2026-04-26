---
name: refresh-library
description: Run the cleanup pipeline against newly-added Calibre books. Walks Erik through the rw-mount + Calibre-Web stop dance, runs `miner refresh` (Phases 1+3b+4 propose-only), reviews the new proposals, optionally applies them, then puts the library back to ro. Use whenever Erik says he added books and wants them processed, or when a classifier improvement should be back-applied to the existing library.
---

# Refresh the Calibre library

Use this skill when Erik says something like "I added some books", "let's run the refresh", or after we change a classifier rule and want to re-sweep. It coordinates the propose-and-apply cycle against his NAS-hosted library.

## Pre-flight

1. **Confirm with Erik** what triggered the refresh (new books? classifier fix? both?). If new books, ask him for the highest old book id (or grab it from the propose-queue's previous max) so we can pass `--since-book-id`.
2. **Check current mount state**: `mount | grep calibre`. We need rw for the apply step. If it's ro, Erik needs to remount manually (admin password) — do NOT attempt sudo from this shell.
3. **Check Calibre-Web container state**. If the apply step will run, it must be stopped (Portainer console). If we're only doing the propose-only scout pass (read-only), Calibre-Web can stay up.

## Standard flow

The pipeline is idempotent — re-running on the same state is a no-op via the propose-queue's unique index. Don't worry about "did this already get applied"; just run it.

### Step 1 — Scout (propose-only, ro mount is fine)

Always start here. No library writes; tells us what's new.

```bash
uv run python -m calibre_mcp.cleanup miner refresh \
  --library /mnt/calibre/books \
  --metadata-db /mnt/calibre/books/metadata.db \
  --proposals-db .cache/cleanup_proposals.db \
  --since-book-id <max_old_id+1> \
  --show-new
```

Skip `--since-book-id` to scan the whole library (slower; fine when classifier rules changed and we want to back-apply to old books).

Read the summary table to Erik. The `proposals_emitted` numbers are *classifier decisions*, not actual new DB inserts (idempotency dedupes); to see what's truly new, look at `--show-new` output or query:

```sql
SELECT r.id, r.source, COUNT(p.id) AS n
FROM miner_runs r LEFT JOIN proposals p ON p.run_id = r.id
WHERE r.completed_at > datetime('now', '-10 minutes')
GROUP BY r.id;
```

### Step 2 — Review with Erik

Show him the new proposals. He decides what to approve. Common patterns:

- **Tag-level (book_id=0)** delete/merge proposals — bulk-approve usually fine, since the Phase 4 classifier is conservative. Review any unfamiliar source tag names with him first.
- **Per-book proposals** (ISBN, identifiers, tags.add) — usually safe to bulk-approve. Spot-check a few unusual ones.
- **`prose_contemporary`-style noise on a specific book** — if any new books carry junk tags, those will show as `tag.delete` proposals; safe to approve.

Approve via:

```bash
uv run python -m calibre_mcp.cleanup miner approve \
  --proposals-db .cache/cleanup_proposals.db \
  --status proposed --source <source>
```

### Step 3 — Apply (rw mount + Calibre-Web stopped)

If there's anything to apply, walk Erik through:

1. Stop Calibre-Web container (Portainer)
2. Remount rw: `sudo umount /mnt/calibre && sudo mount -t cifs //192.168.1.64/Calibre /mnt/calibre -o credentials=/etc/samba/calibre.cred,uid=$(id -u),gid=$(id -g),iocharset=utf8,vers=3.0,nobrl`
3. **Backup** before apply:
   ```bash
   sqlite3 /mnt/calibre/books/metadata.db ".backup .cache/backups/metadata-pre-refresh-$(date +%Y%m%d-%H%M%S).db"
   ```
4. Run apply:
   ```bash
   uv run python -m calibre_mcp.cleanup miner apply \
     --proposals-db .cache/cleanup_proposals.db \
     --library-path /mnt/calibre/books \
     --timeout 180 --execute
   ```
   Use background+Monitor pattern (see Phase 4 apply session) if the queue is large (>500 commands). For small batches (<50), run in foreground.
5. Erik remounts ro and restarts Calibre-Web.

### Step 4 — Optionally chase Goodreads IDs for the new books

Phase 2 needs ISBNs to be applied (in Calibre's `identifiers` table) before it can find candidates. So this only makes sense **after** Step 3 if any ISBNs were applied. Run on its own — rate-limited, can take a while:

```bash
uv run python -m calibre_mcp.cleanup miner goodreads \
  --calibre-db /mnt/calibre/books/metadata.db \
  --proposals-db .cache/cleanup_proposals.db \
  --rate 1.0
```

This generates `identifier.goodreads` proposals → review/approve → apply (back to Step 3 if executing).

## Cautions

- **Never sudo from this shell** for mount changes — Erik does that manually.
- **Always backup metadata.db** before an apply pass. The backup-naming convention is `.cache/backups/metadata-pre-<reason>-<timestamp>.db`.
- **The propose-queue is also worth backing up** before any large bulk-approve operation that might be hard to undo: `.cache/backups/cleanup_proposals-pre-<reason>-<timestamp>.db`.
- If a **classifier change** introduces a no-op self-merge (like the original `Conan → Conan` bug), reject the bogus proposals with a notes annotation rather than just leaving them in the queue. The queue should accurately reflect Erik's intent.
- **Phase 1 OPF mining over CIFS is slow** (~50-200ms per EPUB). On the full library that's 30+ minutes. Always pass `--since-book-id` when only new books matter.

## When to skip this skill

- For a one-off classifier rule change you want to test → just run `uv run pytest` against the relevant test file.
- For a **single specific book** that needs metadata fixed → use `calibredb set_metadata` directly or the existing `miner` apply with `--id` filter; no need for the orchestration.
- For data-quality investigation only (no writes intended) → use `miner report` / `miner review` instead.
