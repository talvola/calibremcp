"""Run the cleanup-pipeline apply step with streaming progress + ETA.

Wraps ``calibre_mcp.cleanup.apply.execute`` so that long apply passes
(thousands of books, tens of minutes) print a progress checkpoint every
N commands instead of going dark. Used in past sessions for the live-tag
sweep apply (Phase 4) and the cookbook-tagger apply (Phase 5).

Why a script and not just the existing CLI: ``miner apply`` exists and
works, but its console output is tuned for short batches. For a
2,000-command pass over slow CIFS we want explicit checkpoints, an ETA,
and an exit code that reflects success/failure — easy to wire into the
Bash background-task + Monitor pattern the project's apply skill uses.

Usage:

    uv run python scripts/apply_with_progress.py \\
        --source cookbook_llm \\
        --proposals-db .cache/cleanup_proposals.db \\
        --library-path /mnt/calibre/books

Pre-conditions: ``/mnt/calibre`` mounted rw, Calibre-Web stopped.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from calibre_mcp.cleanup import apply, proposals


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--proposals-db", type=Path, required=True)
    p.add_argument("--library-path", type=Path, required=True,
                   help="Calibre library root (the dir containing metadata.db)")
    p.add_argument("--source", default=None,
                   help="Restrict to one proposal source (e.g. 'cookbook_llm'). "
                        "Default: all approved proposals.")
    p.add_argument("--field", default=None,
                   help="Restrict to one field (e.g. 'tags.add')")
    p.add_argument("--timeout", type=float, default=180.0,
                   help="Per-book calibredb timeout in seconds (default 180; "
                        "bump for slow CIFS moments)")
    p.add_argument("--checkpoint-every", type=int, default=25,
                   help="Print progress every N commands (default 25)")
    args = p.parse_args(argv)

    pc = proposals.connect(args.proposals_db)
    print("Planning…", flush=True)
    cmds = list(apply.plan(
        pc, library_path=args.library_path,
        source=args.source, field=args.field,
    ))
    print(f"Plan: {len(cmds)} per-book commands "
          f"(source={args.source or 'any'}, field={args.field or 'any'})",
          flush=True)
    if not cmds:
        pc.close()
        return 0

    start = time.monotonic()
    ok = fail = 0
    for i, result in enumerate(apply.execute(pc, cmds, timeout=args.timeout), start=1):
        if result.ok:
            ok += 1
        else:
            fail += 1
        if i % args.checkpoint_every == 0 or i == len(cmds):
            elapsed = time.monotonic() - start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(cmds) - i) / rate if rate > 0 else 0
            print(
                f"[{i:>4}/{len(cmds)}] ok={ok} fail={fail}  "
                f"elapsed={elapsed:.0f}s  rate={rate:.1f}/s  eta={eta:.0f}s",
                flush=True,
            )
        # First few failures get a one-line stderr snippet so root cause is
        # visible from the streaming log without grepping.
        if not result.ok and fail <= 5:
            print(f"  FAIL book={result.command.book_id} "
                  f"stderr={result.stderr.strip()[:200]}", flush=True)

    elapsed = time.monotonic() - start
    print(f"\nDONE: {ok} ok, {fail} fail in {elapsed:.0f}s", flush=True)

    # Final per-source status snapshot. Helpful when the apply was scoped
    # to a single source to confirm no rows were left in 'approved' state.
    if args.source:
        for status in ("applied", "approved", "conflict"):
            n = pc.execute(
                "SELECT COUNT(*) FROM proposals WHERE source=? AND status=?",
                (args.source, status),
            ).fetchone()[0]
            if n:
                print(f"  {args.source}: status={status} count={n}", flush=True)

    pc.close()
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
