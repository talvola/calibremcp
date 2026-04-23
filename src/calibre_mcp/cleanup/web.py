"""Lightweight webapp for browsing and actioning cleanup proposals.

A minimal FastAPI app served via ``miner serve --port 8090``. Intentionally
self-contained and isolated from the project's existing ``webapp/`` — this
is purely for the cleanup review workflow.

Features:
  - Dashboard: proposals by field × status.
  - Proposal list with filters (field/status/source/book_id), pagination.
  - Inline approve/reject per row; bulk approve/reject by filter.
  - Book context (title + authors + current tags) pulled from Calibre's
    metadata.db when ``--calibre-db`` is configured; falls back to book_id.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from jinja2 import BaseLoader, Environment, select_autoescape

from calibre_mcp.cleanup import proposals

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


_STYLE = """
<style>
  body { font: 14px/1.45 -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; padding: 20px; color: #222; }
  h1 { font-size: 20px; margin: 0 0 12px; }
  h2 { font-size: 16px; margin: 18px 0 8px; color: #555; }
  nav a { margin-right: 14px; color: #0366d6; text-decoration: none; }
  nav a.active { font-weight: 600; color: #222; }
  table { border-collapse: collapse; width: 100%; margin-top: 8px; font-size: 13px; }
  th, td { border: 1px solid #ddd; padding: 6px 9px; text-align: left; vertical-align: top; }
  th { background: #f6f8fa; font-weight: 600; }
  tr:nth-child(even) td { background: #fafbfc; }
  td.n { text-align: right; font-variant-numeric: tabular-nums; }
  .status-proposed { color: #1e7e34; }
  .status-approved { color: #0366d6; font-weight: 600; }
  .status-rejected { color: #6a737d; text-decoration: line-through; }
  .status-conflict { color: #b31d28; font-weight: 600; }
  .status-applied { color: #28a745; font-weight: 600; }
  form.inline { display: inline; }
  button { font: inherit; padding: 3px 9px; cursor: pointer; border: 1px solid #ccc;
           background: #f6f8fa; border-radius: 3px; }
  button.approve { background: #e7f3ea; border-color: #88cf9c; }
  button.reject  { background: #fbeaea; border-color: #e88; }
  .filters { background: #f6f8fa; padding: 10px; border-radius: 6px; margin: 8px 0; }
  .filters label { margin-right: 16px; }
  .filters select, .filters input { font: inherit; padding: 3px 6px; }
  .pager { margin: 12px 0; font-size: 13px; color: #555; }
  .pager a { margin: 0 4px; }
  .bulk { background: #fff4e6; padding: 10px; border-radius: 6px; margin: 12px 0;
          border: 1px solid #f2c97a; }
  .bulk strong { color: #b36200; }
  code { background: #f6f8fa; padding: 1px 5px; border-radius: 3px; font-size: 12.5px; }
  .val { max-width: 380px; word-wrap: break-word; white-space: pre-wrap; }
  .dim { color: #888; }
  .book { font-size: 12.5px; }
  .book .title { font-weight: 600; }
  .book .authors { color: #555; }
</style>
"""

_NAV = """
<nav>
  <a href="/" class="{{ 'active' if route == 'dashboard' else '' }}">Dashboard</a>
  <a href="/proposals" class="{{ 'active' if route == 'list' else '' }}">Proposals</a>
  {% if current_filter %}<span class="dim">({{ current_filter }})</span>{% endif %}
</nav>
<hr>
"""

TEMPLATE_DASHBOARD = (
    "<!doctype html><html><head><title>Calibre Cleanup</title>"
    + _STYLE
    + "</head><body>"
    + _NAV
    + """
<h1>Calibre Cleanup — Phase 1 review</h1>
<p class="dim">Proposals DB: <code>{{ db_path }}</code>
   {% if calibre_db %} · Calibre: <code>{{ calibre_db }}</code>{% endif %}</p>

<h2>Proposals by field × status</h2>
<table>
  <tr>
    <th>field</th>
    {% for st in statuses %}<th class="n">{{ st }}</th>{% endfor %}
    <th class="n">total</th>
  </tr>
  {% for field in fields %}
  <tr>
    <td><a href="/proposals?field={{ field|urlencode }}">{{ field }}</a></td>
    {% for st in statuses %}
      {% set n = counts.get(field, {}).get(st, 0) %}
      <td class="n">
        {% if n %}<a href="/proposals?field={{ field|urlencode }}&status={{ st }}">{{ n }}</a>{% else %}0{% endif %}
      </td>
    {% endfor %}
    <td class="n"><strong>{{ row_totals[field] }}</strong></td>
  </tr>
  {% endfor %}
  <tr>
    <th>total</th>
    {% for st in statuses %}<th class="n">{{ col_totals.get(st, 0) }}</th>{% endfor %}
    <th class="n">{{ grand_total }}</th>
  </tr>
</table>

{% if not fields %}
<p class="dim">No proposals yet. Run <code>miner run</code> to populate the queue.</p>
{% endif %}
</body></html>
"""
)


TEMPLATE_LIST = (
    "<!doctype html><html><head><title>Proposals — Calibre Cleanup</title>"
    + _STYLE
    + "</head><body>"
    + _NAV
    + """
<h1>Proposals</h1>

<form class="filters" method="get" action="/proposals">
  <label>field
    <select name="field">
      <option value="">(any)</option>
      {% for f in all_fields %}
      <option value="{{ f }}" {% if f == filter_field %}selected{% endif %}>{{ f }}</option>
      {% endfor %}
    </select>
  </label>
  <label>status
    <select name="status">
      <option value="">(any)</option>
      {% for s in all_statuses %}
      <option value="{{ s }}" {% if s == filter_status %}selected{% endif %}>{{ s }}</option>
      {% endfor %}
    </select>
  </label>
  <label>source
    <input type="text" name="source" value="{{ filter_source or '' }}" size="8">
  </label>
  <label>book_id
    <input type="number" name="book_id" value="{{ filter_book_id or '' }}" size="8">
  </label>
  <label>page size
    <select name="size">
      {% for s in [25, 50, 100, 200, 500] %}
      <option value="{{ s }}" {% if s == size %}selected{% endif %}>{{ s }}</option>
      {% endfor %}
    </select>
  </label>
  <button type="submit">Apply</button>
  <a href="/proposals" class="dim">clear</a>
</form>

<p class="dim">
  {{ total }} match · page {{ page }} of {{ total_pages }}
  {% if total == 0 %}— nothing to review.{% endif %}
</p>

{% if filter_field or filter_status or filter_source or filter_book_id %}
<form class="bulk" method="post" action="/bulk">
  <input type="hidden" name="field"   value="{{ filter_field or '' }}">
  <input type="hidden" name="status"  value="{{ filter_status or '' }}">
  <input type="hidden" name="source"  value="{{ filter_source or '' }}">
  <input type="hidden" name="book_id" value="{{ filter_book_id or '' }}">
  <input type="hidden" name="return_to" value="{{ request_url }}">
  <strong>Bulk</strong> for the {{ total }} matching proposals:
  <button type="submit" class="approve" name="action" value="approved">Approve all</button>
  <button type="submit" class="reject"  name="action" value="rejected">Reject all</button>
  <label>note (optional): <input type="text" name="notes" size="30"></label>
</form>
{% endif %}

<table>
  <tr>
    <th>id</th>
    <th>book</th>
    <th>field</th>
    <th>calibre value</th>
    <th>proposed value</th>
    <th class="n">conf</th>
    <th>status</th>
    <th>actions</th>
  </tr>
  {% for r in rows %}
  <tr>
    <td class="n">{{ r.id }}</td>
    <td class="book">
      <a href="/proposals?book_id={{ r.book_id }}">#{{ r.book_id }}</a>
      {% if r.book_title %}
        <div class="title">{{ r.book_title }}</div>
        <div class="authors">{{ r.book_authors }}</div>
      {% endif %}
    </td>
    <td>{{ r.field }}</td>
    <td class="val">{{ r.calibre_value or '' }}</td>
    <td class="val">{{ r.proposed_value }}</td>
    <td class="n">{{ '%.2f' % r.confidence }}</td>
    <td class="status-{{ r.status }}">{{ r.status }}</td>
    <td>
      {% if r.status in ('proposed', 'conflict') %}
      <form class="inline" method="post" action="/proposals/{{ r.id }}/status">
        <input type="hidden" name="action" value="approved">
        <input type="hidden" name="return_to" value="{{ request_url }}">
        <button type="submit" class="approve">approve</button>
      </form>
      <form class="inline" method="post" action="/proposals/{{ r.id }}/status">
        <input type="hidden" name="action" value="rejected">
        <input type="hidden" name="return_to" value="{{ request_url }}">
        <button type="submit" class="reject">reject</button>
      </form>
      {% elif r.status == 'approved' %}
      <form class="inline" method="post" action="/proposals/{{ r.id }}/status">
        <input type="hidden" name="action" value="proposed">
        <input type="hidden" name="return_to" value="{{ request_url }}">
        <button type="submit">un-approve</button>
      </form>
      {% endif %}
    </td>
  </tr>
  {% endfor %}
</table>

<div class="pager">
  {% if page > 1 %}<a href="{{ page_url(page-1) }}">« prev</a>{% endif %}
  {% for p in page_range %}
    {% if p == page %}<strong>{{ p }}</strong>{% else %}<a href="{{ page_url(p) }}">{{ p }}</a>{% endif %}
  {% endfor %}
  {% if page < total_pages %}<a href="{{ page_url(page+1) }}">next »</a>{% endif %}
</div>
</body></html>
"""
)


_JINJA = Environment(loader=BaseLoader(), autoescape=select_autoescape(["html"]))
_DASHBOARD_TMPL = _JINJA.from_string(TEMPLATE_DASHBOARD)
_LIST_TMPL = _JINJA.from_string(TEMPLATE_LIST)


# ---------------------------------------------------------------------------
# Book-context loader — optional read from Calibre's metadata.db
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BookContext:
    title: str
    authors: str  # joined with " & "


class _BookContextLoader:
    """Lazily fetches (title, authors) for Calibre books referenced by
    proposals. Cached in-process for the lifetime of the app.

    When no Calibre DB is configured, acts as a no-op so the webapp still
    works against a bare proposals DB."""

    def __init__(self, calibre_db: Path | None) -> None:
        self._calibre_db = calibre_db
        self._cache: dict[int, BookContext | None] = {}

    def get(self, book_ids: Iterable[int]) -> dict[int, BookContext]:
        if self._calibre_db is None:
            return {}
        needed = [b for b in book_ids if b not in self._cache]
        if needed:
            uri = f"file:{self._calibre_db.resolve()}?mode=ro"
            with sqlite3.connect(uri, uri=True) as conn:
                conn.row_factory = sqlite3.Row
                placeholders = ",".join(["?"] * len(needed))
                rows = conn.execute(
                    f"""
                    SELECT b.id, b.title,
                           GROUP_CONCAT(a.name, ' & ') AS authors
                    FROM books b
                    LEFT JOIN books_authors_link bal ON bal.book = b.id
                    LEFT JOIN authors a               ON a.id = bal.author
                    WHERE b.id IN ({placeholders})
                    GROUP BY b.id
                    """,  # noqa: S608 — placeholders is '?,?,?'; ids bind via params
                    needed,
                ).fetchall()
                for row in rows:
                    self._cache[row["id"]] = BookContext(
                        title=row["title"] or "",
                        authors=row["authors"] or "",
                    )
                # Cache misses (deleted books) as None so we don't re-query.
                for b in needed:
                    self._cache.setdefault(b, None)
        return {b: ctx for b in book_ids if (ctx := self._cache.get(b)) is not None}


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(*, proposals_db: Path, calibre_db: Path | None = None) -> FastAPI:
    app = FastAPI(title="Calibre Cleanup Review", docs_url=None, redoc_url=None)
    loader = _BookContextLoader(calibre_db)

    def _conn() -> sqlite3.Connection:
        return proposals.connect(proposals_db)

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        with _conn() as conn:
            counts = proposals.summary_counts(conn)
        statuses = sorted({s for field_counts in counts.values() for s in field_counts})
        fields = sorted(counts)
        row_totals = {f: sum(counts[f].values()) for f in fields}
        col_totals = {s: sum(counts.get(f, {}).get(s, 0) for f in fields) for s in statuses}
        grand_total = sum(row_totals.values())
        return _DASHBOARD_TMPL.render(
            route="dashboard",
            db_path=str(proposals_db),
            calibre_db=str(calibre_db) if calibre_db else None,
            counts=counts,
            statuses=statuses,
            fields=fields,
            row_totals=row_totals,
            col_totals=col_totals,
            grand_total=grand_total,
            current_filter=None,
        )

    @app.get("/proposals", response_class=HTMLResponse)
    def list_view(
        request: Request,
        field: str | None = Query(None),
        status: str | None = Query(None),
        source: str | None = Query(None),
        book_id: int | None = Query(None),
        page: int = Query(1, ge=1),
        size: int = Query(50, ge=1, le=500),
    ) -> str:
        filters = {
            k: v
            for k, v in {
                "field": field or None,
                "status": status or None,
                "source": source or None,
            }.items()
            if v
        }
        with _conn() as conn:
            # Count + page-fetch. We paginate via OFFSET/LIMIT, small enough
            # at this scale that it's fine; if the table grows huge we can
            # move to keyset pagination.
            where_clauses: list[str] = ["1=1"]
            params: list[Any] = []
            for key in ("field", "status", "source"):
                if filters.get(key):
                    where_clauses.append(f"{key} = ?")
                    params.append(filters[key])
            if book_id is not None:
                where_clauses.append("book_id = ?")
                params.append(book_id)
            where = " AND ".join(where_clauses)
            total = conn.execute(
                f"SELECT COUNT(*) FROM proposals WHERE {where}",  # noqa: S608 — column names are fixed
                params,
            ).fetchone()[0]
            total_pages = max(1, (total + size - 1) // size)
            page = min(page, total_pages)
            offset = (page - 1) * size
            rows = conn.execute(
                f"""
                SELECT * FROM proposals WHERE {where}
                ORDER BY id DESC LIMIT ? OFFSET ?
                """,  # noqa: S608
                [*params, size, offset],
            ).fetchall()
            # Distinct field/status lists for dropdowns (small sets).
            all_fields = [r[0] for r in conn.execute("SELECT DISTINCT field FROM proposals ORDER BY field")]
            all_statuses = [r[0] for r in conn.execute("SELECT DISTINCT status FROM proposals ORDER BY status")]

        # Attach book context to each row.
        ctx_map = loader.get([r["book_id"] for r in rows])
        enriched = []
        for r in rows:
            d = dict(r)
            ctx = ctx_map.get(r["book_id"])
            d["book_title"] = ctx.title if ctx else ""
            d["book_authors"] = ctx.authors if ctx else ""
            enriched.append(d)

        # Pager: show up to 10 page links around current.
        start = max(1, page - 5)
        end = min(total_pages, start + 9)
        page_range = list(range(start, end + 1))

        def page_url(p: int) -> str:
            return str(request.url.include_query_params(page=p))

        filter_label_parts = [f"{k}={v}" for k, v in filters.items()]
        if book_id is not None:
            filter_label_parts.append(f"book_id={book_id}")

        return _LIST_TMPL.render(
            route="list",
            rows=enriched,
            total=total,
            page=page,
            total_pages=total_pages,
            size=size,
            filter_field=field,
            filter_status=status,
            filter_source=source,
            filter_book_id=book_id,
            all_fields=all_fields,
            all_statuses=all_statuses,
            page_range=page_range,
            page_url=page_url,
            request_url=str(request.url),
            current_filter=", ".join(filter_label_parts) or None,
        )

    @app.post("/proposals/{proposal_id}/status")
    def set_status(
        proposal_id: int,
        action: str = Form(...),
        return_to: str = Form("/proposals"),
    ) -> RedirectResponse:
        if action not in {"approved", "rejected", "proposed"}:
            raise HTTPException(status_code=400, detail=f"invalid action: {action!r}")
        with _conn() as conn:
            row = conn.execute("SELECT id FROM proposals WHERE id = ?", (proposal_id,)).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail=f"no proposal with id {proposal_id}")
            proposals.set_status(conn, proposal_id, action)
        return RedirectResponse(return_to, status_code=303)

    @app.post("/bulk")
    def bulk(
        action: str = Form(...),
        field: str = Form(""),
        status: str = Form(""),
        source: str = Form(""),
        book_id: str = Form(""),
        notes: str = Form(""),
        return_to: str = Form("/proposals"),
    ) -> RedirectResponse:
        if action not in {"approved", "rejected"}:
            raise HTTPException(status_code=400, detail=f"invalid action: {action!r}")
        # book_id filter isn't supported by set_status_where; translate via ids.
        with _conn() as conn:
            ids: list[int] | None = None
            if book_id:
                try:
                    bid = int(book_id)
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail="bad book_id") from exc
                ids = [r["id"] for r in conn.execute("SELECT id FROM proposals WHERE book_id = ?", (bid,))]
                if not ids:
                    return RedirectResponse(return_to, status_code=303)
            proposals.set_status_where(
                conn,
                action,
                field=field or None,
                status=status or None,
                source=source or None,
                ids=ids,
                notes=notes or None,
            )
        return RedirectResponse(return_to, status_code=303)

    return app


def serve(
    *,
    proposals_db: Path,
    calibre_db: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8090,
) -> None:
    """Run the webapp with uvicorn (blocks until interrupted)."""
    import uvicorn

    app = create_app(proposals_db=proposals_db, calibre_db=calibre_db)
    uvicorn.run(app, host=host, port=port, log_level="info")
