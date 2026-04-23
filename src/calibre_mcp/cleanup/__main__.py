"""Command-line entry point for the cleanup pipeline.

Usage (from repo root, after ``uv sync`` / pip install -e .):

    python -m calibre_mcp.cleanup miner run \\
        --library  .cache/calibre-snapshot \\
        --metadata-db .cache/calibre-snapshot/metadata.db \\
        --proposals-db .cache/cleanup_proposals.db \\
        [--sample 100] [--dry-run]

    python -m calibre_mcp.cleanup miner report \\
        --proposals-db .cache/cleanup_proposals.db

    python -m calibre_mcp.cleanup miner review \\
        --proposals-db .cache/cleanup_proposals.db \\
        --field isbn --limit 20
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from calibre_mcp.cleanup import miner, proposals


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-5s %(name)s  %(message)s",
    )

    if args.cmd == "miner" and args.subcmd == "run":
        return _cmd_run(args)
    if args.cmd == "miner" and args.subcmd == "report":
        return _cmd_report(args)
    if args.cmd == "miner" and args.subcmd == "review":
        return _cmd_review(args)
    parser.print_help()
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="calibre_mcp.cleanup")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("miner", help="OPF miner — Phase 1")
    msub = m.add_subparsers(dest="subcmd", required=True)

    run = msub.add_parser("run", help="Execute the miner against a library")
    run.add_argument("--library", type=Path, required=True, help="Calibre library root")
    run.add_argument("--metadata-db", type=Path, required=True, help="Path to metadata.db")
    run.add_argument(
        "--proposals-db", type=Path, required=True, help="Path to the propose-queue DB (created if missing)"
    )
    run.add_argument("--sample", type=int, default=None, help="Only scan N books (debug/dev)")
    run.add_argument("--dry-run", action="store_true", help="Count proposals without persisting them")

    report = msub.add_parser("report", help="Summarize proposals by field and status")
    report.add_argument("--proposals-db", type=Path, required=True)

    review = msub.add_parser("review", help="List individual proposals for review")
    review.add_argument("--proposals-db", type=Path, required=True)
    review.add_argument("--field", default=None, help="Filter by field (e.g. 'isbn')")
    review.add_argument("--status", default=None, help="Filter by status")
    review.add_argument("--source", default=None, help="Filter by source (default: any)")
    review.add_argument("--limit", type=int, default=25)

    return parser


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    console = Console()
    console.print(
        f"[bold]OPF miner[/bold] library=[cyan]{args.library}[/cyan] "
        f"sample=[yellow]{args.sample or 'all'}[/yellow] "
        f"dry_run=[yellow]{args.dry_run}[/yellow]"
    )
    summary = miner.run(
        library_root=args.library,
        metadata_db=args.metadata_db,
        proposals_db=args.proposals_db,
        sample=args.sample,
        dry_run=args.dry_run,
    )
    table = Table(title=f"Run #{summary.run_id}")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("books scanned", str(summary.books_scanned))
    table.add_row("books with EPUB", str(summary.books_with_epub))
    table.add_row("OPFs parsed", str(summary.books_parsed))
    table.add_row("proposals emitted", str(summary.proposals_emitted))
    table.add_row("errors", str(summary.errors))
    console.print(table)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    console = Console()
    with proposals.connect(args.proposals_db) as conn:
        counts = proposals.summary_counts(conn)
    if not counts:
        console.print("[yellow]No proposals yet.[/yellow]")
        return 0
    all_statuses = sorted({s for field_counts in counts.values() for s in field_counts})
    table = Table(title="Proposals by field × status")
    table.add_column("field")
    for st in all_statuses:
        table.add_column(st, justify="right")
    for field in sorted(counts):
        row = [field]
        for st in all_statuses:
            row.append(str(counts[field].get(st, 0)))
        table.add_row(*row)
    console.print(table)
    return 0


def _cmd_review(args: argparse.Namespace) -> int:
    console = Console()
    with proposals.connect(args.proposals_db) as conn:
        rows = proposals.list_proposals(
            conn,
            field=args.field,
            status=args.status,
            source=args.source,
            limit=args.limit,
        )
    if not rows:
        console.print("[yellow]No matching proposals.[/yellow]")
        return 0
    table = Table(
        title=(
            f"Proposals "
            f"(field={args.field or '*'}, "
            f"status={args.status or '*'}, "
            f"source={args.source or '*'}, "
            f"limit={args.limit})"
        ),
        show_lines=True,
    )
    table.add_column("id", justify="right")
    table.add_column("book", justify="right")
    table.add_column("field")
    table.add_column("calibre", max_width=30, overflow="fold")
    table.add_column("proposed", max_width=40, overflow="fold")
    table.add_column("status")
    table.add_column("src")
    table.add_column("conf", justify="right")
    for r in rows:
        table.add_row(
            str(r["id"]),
            str(r["book_id"]),
            r["field"],
            r["calibre_value"] or "",
            r["proposed_value"] or "",
            r["status"],
            r["source"],
            f"{r['confidence']:.2f}",
        )
    console.print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
