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

from calibre_mcp.cleanup import apply as apply_mod
from calibre_mcp.cleanup import miner, proposals
from calibre_mcp.cleanup import refresh as refresh_mod


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
    if args.cmd == "miner" and args.subcmd == "approve":
        return _cmd_set_status(args, "approved")
    if args.cmd == "miner" and args.subcmd == "reject":
        return _cmd_set_status(args, "rejected")
    if args.cmd == "miner" and args.subcmd == "apply":
        return _cmd_apply(args)
    if args.cmd == "miner" and args.subcmd == "web":
        return _cmd_web(args)
    if args.cmd == "miner" and args.subcmd == "goodreads":
        return _cmd_goodreads(args)
    if args.cmd == "miner" and args.subcmd == "normalize-tags":
        return _cmd_normalize_tags(args)
    if args.cmd == "miner" and args.subcmd == "refresh":
        return _cmd_refresh(args)
    if args.cmd == "miner" and args.subcmd == "cookbooks":
        return _cmd_cookbooks(args)
    if args.cmd == "miner" and args.subcmd == "cookbooks-review":
        return _cmd_cookbooks_review(args)
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

    ap = msub.add_parser(
        "apply",
        help="Write approved proposals to Calibre via calibredb (dry-run by default)",
    )
    ap.add_argument("--proposals-db", type=Path, required=True)
    ap.add_argument("--library-path", type=Path, required=True, help="Path calibredb should use for --library-path")
    ap.add_argument("--execute", action="store_true", help="Actually run calibredb (default is dry-run)")
    ap.add_argument("--field", default=None, help="Only apply proposals for this field")
    ap.add_argument(
        "--id",
        type=int,
        action="append",
        dest="ids",
        default=None,
        help="Only apply specific proposal IDs (repeat for multiple)",
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="Stop after applying N commands (useful for a cautious first batch)"
    )
    ap.add_argument("--calibredb", default="calibredb", help="Path to calibredb binary (default: found on PATH)")
    ap.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-book calibredb timeout in seconds (default: 30; bump for slow CIFS moments)",
    )

    web = msub.add_parser("web", help="Launch the proposal-review webapp")
    web.add_argument("--proposals-db", type=Path, required=True)
    web.add_argument(
        "--calibre-db",
        type=Path,
        default=None,
        help="Calibre metadata.db (optional — enables book title/author in the UI)",
    )
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8090)

    gr = msub.add_parser(
        "goodreads",
        help="Phase 2: look up Goodreads IDs via ISBN redirect for books that lack one",
    )
    gr.add_argument("--calibre-db", type=Path, required=True, help="Calibre metadata.db (read-only)")
    gr.add_argument("--proposals-db", type=Path, required=True)
    gr.add_argument(
        "--rate",
        type=float,
        default=1.0,
        help="Seconds between requests (default: 1.0; Goodreads is touchy about bulk)",
    )
    gr.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after N books (useful for a small sample run first)",
    )
    gr.add_argument("--user-agent", default=None, help="Override the default browser UA")

    nt = msub.add_parser(
        "normalize-tags",
        help="Phase 3b: classify tags.add proposals as approve/reject/defer",
    )
    nt.add_argument(
        "--calibre-db", type=Path, required=True, help="Calibre metadata.db (read-only — for canonical vocabulary)"
    )
    nt.add_argument("--proposals-db", type=Path, required=True)
    nt.add_argument("--dry-run", action="store_true", help="Show counts only; don't update proposal statuses")

    rf = msub.add_parser(
        "refresh",
        help="Re-run Phases 1+3b+4 to pick up new books and post-fix classifier improvements (propose-only)",
    )
    rf.add_argument("--library", type=Path, required=True, help="Calibre library root")
    rf.add_argument("--metadata-db", type=Path, required=True, help="Path to metadata.db")
    rf.add_argument("--proposals-db", type=Path, required=True)
    rf.add_argument(
        "--since-book-id",
        type=int,
        default=None,
        help="Scope Phase 1 OPF mining to books with id >= this value "
             "(typically <previous-max-id>+1). Phases 3b/4 always run on the whole DB.",
    )
    rf.add_argument(
        "--show-new",
        action="store_true",
        help="After the refresh, list every proposal attached to the runs just executed.",
    )

    cb = msub.add_parser(
        "cookbooks",
        help="Phase 5a: LLM-tag the Cookbooks bucket with cuisine/technique/dietary facets",
    )
    cb.add_argument("--library", type=Path, required=True, help="Calibre library root")
    cb.add_argument("--metadata-db", type=Path, required=True, help="Path to metadata.db")
    cb.add_argument("--proposals-db", type=Path, required=True)
    cb.add_argument(
        "--bucket-tag", default="Cookbooks",
        help="Tag identifying the bucket to walk (default: Cookbooks)",
    )
    cb.add_argument(
        "--limit", type=int, default=None,
        help="Stop after N books (useful for proof-of-concept / cost cap)",
    )
    cb.add_argument(
        "--book-id", type=int, action="append", dest="book_ids", default=None,
        help="Restrict to specific book IDs (repeat for multiple); skips bucket-tag filter",
    )
    cb.add_argument(
        "--retag", action="store_true",
        help="Re-tag books that already have cookbook_llm proposals (default: skip)",
    )

    cbr = msub.add_parser(
        "cookbooks-review",
        help="Group cookbook_llm proposals by tag for per-cuisine review + bulk approve",
    )
    cbr.add_argument("--proposals-db", type=Path, required=True)
    cbr.add_argument("--metadata-db", type=Path, required=True)
    cbr.add_argument(
        "--status", default="proposed",
        help="Proposal status to review (default: proposed)",
    )

    for verb, help_text in [
        ("approve", "Mark matching proposals as approved (ready to apply)"),
        ("reject", "Mark matching proposals as rejected (ignored by apply)"),
    ]:
        ap = msub.add_parser(verb, help=help_text)
        ap.add_argument("--proposals-db", type=Path, required=True)
        ap.add_argument(
            "--id",
            type=int,
            action="append",
            dest="ids",
            default=None,
            help="Specific proposal ID (repeat for multiple)",
        )
        ap.add_argument("--field", default=None, help="Filter by field")
        ap.add_argument(
            "--status",
            default=None,
            help="Filter by current status (default: 'proposed' for approve, any for reject)",
        )
        ap.add_argument("--source", default=None, help="Filter by source")
        ap.add_argument("--min-confidence", type=float, default=None)
        ap.add_argument("--max-confidence", type=float, default=None)
        ap.add_argument(
            "--proposed-value", default=None,
            help="Filter by exact proposed_value (e.g. 'Italian' to bulk-approve all "
                 "Italian cookbook proposals)",
        )
        ap.add_argument("--notes", default=None, help="Reviewer note stored on the row")
        ap.add_argument("--yes", "-y", action="store_true", help="Skip the count-and-confirm prompt")

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


def _cmd_normalize_tags(args: argparse.Namespace) -> int:
    from calibre_mcp.cleanup import tag_normalizer

    console = Console()
    console.print(
        f"[bold]Tag normalizer[/bold]  calibre=[cyan]{args.calibre_db}[/cyan]  dry_run=[yellow]{args.dry_run}[/yellow]"
    )
    summary = tag_normalizer.run(
        calibre_db=args.calibre_db,
        proposals_db=args.proposals_db,
        dry_run=args.dry_run,
    )
    table = Table(title=f"Run #{summary.run_id}")
    table.add_column("verdict")
    table.add_column("count", justify="right")
    table.add_row("examined", str(summary.examined))
    table.add_row("[green]approved[/green]", str(summary.approved))
    table.add_row("[red]rejected[/red]", str(summary.rejected))
    table.add_row("deferred (left at proposed)", str(summary.deferred))
    console.print(table)
    return 0


def _cmd_goodreads(args: argparse.Namespace) -> int:
    from calibre_mcp.cleanup import goodreads_lookup

    console = Console()
    console.print(
        f"[bold]Goodreads ISBN lookup[/bold]  "
        f"calibre=[cyan]{args.calibre_db}[/cyan]  "
        f"rate=[yellow]{args.rate}[/yellow]s  "
        f"limit=[yellow]{args.limit or 'all'}[/yellow]"
    )
    kwargs = {
        "calibre_db": args.calibre_db,
        "proposals_db": args.proposals_db,
        "rate_sec": args.rate,
        "limit": args.limit,
    }
    if args.user_agent:
        kwargs["user_agent"] = args.user_agent
    summary = goodreads_lookup.run(**kwargs)

    table = Table(title=f"Run #{summary.run_id}")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("attempted", str(summary.attempted))
    table.add_row("found", str(summary.found))
    table.add_row("not found", str(summary.not_found))
    table.add_row("rate limited", str(summary.rate_limited))
    table.add_row("errors", str(summary.errors))
    console.print(table)
    return 0


def _cmd_web(args: argparse.Namespace) -> int:
    # Import lazily so the other CLI commands don't pay for uvicorn/FastAPI
    # import time.
    from calibre_mcp.cleanup import web as web_mod

    console = Console()
    console.print(
        f"[bold]Cleanup review webapp[/bold]  "
        f"[cyan]http://{args.host}:{args.port}[/cyan]  "
        f"proposals=[yellow]{args.proposals_db}[/yellow]"
        + (f"  calibre=[yellow]{args.calibre_db}[/yellow]" if args.calibre_db else "")
    )
    web_mod.serve(
        proposals_db=args.proposals_db,
        calibre_db=args.calibre_db,
        host=args.host,
        port=args.port,
    )
    return 0


def _cmd_apply(args: argparse.Namespace) -> int:
    console = Console()
    with proposals.connect(args.proposals_db) as conn:
        skipped_summary = apply_mod.plan_report(conn)
        commands = list(
            apply_mod.plan(
                conn,
                library_path=args.library_path,
                ids=args.ids,
                field=args.field,
                limit=args.limit,
                calibredb=args.calibredb,
            )
        )

        # Preview: count what'll run + what'll be skipped.
        preview = Table(title="Apply plan")
        preview.add_column("category")
        preview.add_column("count", justify="right")
        for key, n in sorted(skipped_summary.items()):
            preview.add_row(key, str(n))
        preview.add_row("[bold]commands to run (grouped by book)[/bold]", str(len(commands)))
        console.print(preview)

        if not commands:
            console.print("[yellow]Nothing to apply.[/yellow]")
            return 0

        if not args.execute:
            console.print("\n[bold]Dry-run — commands that WOULD run:[/bold]\n")
            for cmd in commands[:20]:
                console.print(f"  [dim]#{cmd.book_id}[/dim] {apply_mod.render(cmd)}")
            if len(commands) > 20:
                console.print(f"  [dim]... and {len(commands) - 20} more[/dim]")
            console.print(
                "\n[cyan]To actually apply, re-run with [bold]--execute[/bold]. "
                "Runs calibredb once per book — expect ~0.5s each.[/cyan]"
            )
            return 0

        # Real execution.
        total = len(commands)
        console.print(f"\n[bold]Executing {total} calibredb commands...[/bold]")
        import time as _time

        started = _time.monotonic()
        ok_count = fail_count = 0
        progress_every = 100 if total > 500 else 25
        for i, result in enumerate(apply_mod.execute(conn, commands, timeout=args.timeout), start=1):
            if result.ok:
                ok_count += 1
            else:
                fail_count += 1
                console.print(
                    f"  [red]FAIL[/red] book=#{result.command.book_id} "
                    f"rc={result.returncode} stderr={result.stderr.strip()!r}"
                )
            if i % progress_every == 0 or i == total:
                elapsed = _time.monotonic() - started
                rate = i / elapsed if elapsed else 0
                remaining = (total - i) / rate if rate else 0
                console.print(
                    f"  [cyan]progress[/cyan] {i}/{total} "
                    f"(ok={ok_count} fail={fail_count}) "
                    f"rate={rate:.1f}/s eta={remaining / 60:.1f} min"
                )
        console.print(f"[green]Applied:[/green] {ok_count}  [red]Failed:[/red] {fail_count}")
        return 0 if fail_count == 0 else 1


def _cmd_set_status(args: argparse.Namespace, new_status: str) -> int:
    """Shared implementation for ``approve`` and ``reject``."""
    console = Console()
    # For approve, default to status='proposed' if none given, so 'approve all
    # fill-empty ISBN' is a one-liner. For reject, default to 'any' so the
    # user can explicitly target conflicts.
    status_filter = args.status
    if status_filter is None and new_status == "approved":
        status_filter = "proposed"

    filter_kwargs = {
        "field": args.field,
        "status": status_filter,
        "source": args.source,
        "ids": args.ids,
        "min_confidence": args.min_confidence,
        "max_confidence": args.max_confidence,
        "proposed_value": args.proposed_value,
    }

    # Refuse unfiltered bulk updates — the DB layer also guards, but give
    # a friendlier message here.
    if not any(filter_kwargs.values()):
        console.print(
            "[red]No filter specified.[/red] Pass --id, --field, --status, --source, "
            "--proposed-value, or --min/max-confidence.",
        )
        return 2

    with proposals.connect(args.proposals_db) as conn:
        n = proposals.count_proposals(conn, **filter_kwargs)
        if n == 0:
            console.print("[yellow]No matching proposals.[/yellow]")
            return 0
        verb = "approve" if new_status == "approved" else "reject"
        console.print(
            f"[bold]{n}[/bold] proposals match (status → [cyan]{new_status}[/cyan])"
            + (f' with note "{args.notes}"' if args.notes else "")
        )
        if not args.yes:
            try:
                reply = input(f"  {verb.capitalize()} them? [y/N] ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                console.print("[yellow]Aborted.[/yellow]")
                return 1
            if reply not in {"y", "yes"}:
                console.print("[yellow]Aborted.[/yellow]")
                return 1
        changed = proposals.set_status_where(conn, new_status, notes=args.notes, **filter_kwargs)
        console.print(f"[green]{new_status}[/green] {changed} proposal(s).")
    return 0


def _cmd_cookbooks(args: argparse.Namespace) -> int:
    from calibre_mcp.cleanup import cookbooks as cookbooks_mod
    console = Console()
    scope_desc = (
        f"book_ids={args.book_ids}" if args.book_ids
        else f"tag={args.bucket_tag!r}"
        + (f" limit={args.limit}" if args.limit else "")
    )
    console.print(
        f"[bold]Cookbook tagger[/bold]  {scope_desc}  "
        f"retag=[yellow]{args.retag}[/yellow]"
    )
    summary = cookbooks_mod.run(
        library_root=args.library,
        metadata_db=args.metadata_db,
        proposals_db=args.proposals_db,
        bucket_tag=args.bucket_tag,
        limit=args.limit,
        book_ids=args.book_ids,
        skip_already_tagged=not args.retag,
    )
    table = Table(title="Cookbook tagger summary")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("examined", str(summary.examined))
    table.add_row("[dim]skipped (already tagged)[/dim]", str(summary.skipped_already_tagged))
    table.add_row("[green]tagged[/green]", str(summary.tagged))
    table.add_row("[red]API errors[/red]", str(summary.api_errors))
    table.add_row("proposals emitted", str(summary.proposals_emitted))
    table.add_row("elapsed", f"{summary.elapsed_sec:.0f}s")
    console.print(table)
    console.print(
        "\n[dim]Next: [bold]miner cookbooks-review[/bold] for per-cuisine grouping, "
        "then [bold]miner approve --field tags.add --source cookbook_llm --id ...[/bold].[/dim]"
    )
    return 0


def _cmd_cookbooks_review(args: argparse.Namespace) -> int:
    from calibre_mcp.cleanup import cookbooks as cookbooks_mod
    console = Console()
    groups = cookbooks_mod.report_by_tag(
        proposals_db=args.proposals_db,
        metadata_db=args.metadata_db,
        status=args.status,
    )
    if not groups:
        console.print(f"[dim]No cookbook_llm proposals at status={args.status!r}.[/dim]")
        return 0

    console.print(f"[bold]Cookbook proposals by tag[/bold] (status={args.status})")
    table = Table()
    table.add_column("tag")
    table.add_column("books", justify="right")
    table.add_column("samples", overflow="fold", max_width=70)
    table.add_column("approve cmd", overflow="fold", max_width=80)
    for g in groups:
        # Bulk-approve hint: --proposed-value scales better than --id when
        # the cohort is large.
        if len(g.proposal_ids) <= 3:
            ids_str = " ".join(f"--id {i}" for i in g.proposal_ids)
        else:
            ids_str = f"--source cookbook_llm --proposed-value '{g.tag}'"
        samples = "; ".join(g.sample_titles[:3])
        if len(g.sample_titles) < g.n_books:
            samples += f"; … ({g.n_books - len(g.sample_titles)} more)"
        table.add_row(g.tag, str(g.n_books), samples, ids_str)
    console.print(table)
    console.print(
        f"\n[dim]Total: {len(groups)} distinct tags across "
        f"{sum(g.n_books for g in groups)} proposals.[/dim]"
    )
    return 0


def _cmd_refresh(args: argparse.Namespace) -> int:
    console = Console()
    scope = f"book_id >= {args.since_book_id}" if args.since_book_id is not None else "all books"
    console.print(
        f"[bold]Refresh[/bold]  library=[cyan]{args.library}[/cyan]  "
        f"scope=[yellow]{scope}[/yellow]  proposals=[cyan]{args.proposals_db}[/cyan]"
    )

    summary = refresh_mod.run(
        library_root=args.library,
        metadata_db=args.metadata_db,
        proposals_db=args.proposals_db,
        since_book_id=args.since_book_id,
    )

    table = Table(title="Refresh summary")
    table.add_column("phase")
    table.add_column("emitted", justify="right")
    table.add_column("elapsed", justify="right")
    table.add_column("notes")
    for ph in (summary.phase1, summary.phase3b, summary.phase4):
        table.add_row(ph.name, str(ph.proposals_emitted), f"{ph.elapsed_sec:.1f}s", ph.notes)
    console.print(table)
    console.print(
        f"[bold]Total new proposals:[/bold] {summary.total_new_proposals}  "
        f"(books touched in Phase 1: {summary.new_books_seen})"
    )

    if args.show_new:
        rows = refresh_mod.list_new_proposals(args.proposals_db, since_minutes=10)
        if not rows:
            console.print("[dim]No proposals in the last 10 min.[/dim]")
        else:
            t2 = Table(title=f"New proposals ({len(rows)})")
            t2.add_column("book", justify="right")
            t2.add_column("field")
            t2.add_column("calibre", max_width=30, overflow="fold")
            t2.add_column("proposed", max_width=40, overflow="fold")
            t2.add_column("status")
            t2.add_column("source")
            for r in rows:
                t2.add_row(
                    str(r["book_id"]) if r["book_id"] != 0 else "—",
                    r["field"],
                    r["calibre_value"] or "",
                    r["proposed_value"] or "",
                    r["status"],
                    r["source"],
                )
            console.print(t2)

    console.print(
        "\n[dim]Next: review with [bold]miner review[/bold] / approve, then "
        "[bold]miner apply --execute[/bold] (rw mount + Calibre-Web stop required for apply).[/dim]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
