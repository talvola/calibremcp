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
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from jinja2 import BaseLoader, Environment, select_autoescape

from calibre_mcp.cleanup import cookbook_tagger, proposals

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
  /* Cookbook dashboard */
  .facet-section { margin: 16px 0 28px; }
  .facet-section h2 { color: #b36200; }
  .tag-group { background: #fff; border: 1px solid #ddd; border-radius: 6px;
               padding: 12px 14px; margin: 10px 0; }
  .tag-group h3 { margin: 0 0 8px; font-size: 15px; }
  .tag-group h3 a { color: #0366d6; text-decoration: none; }
  .tag-group .count { color: #888; font-weight: normal; font-size: 13px;
                       margin-left: 6px; }
  .tag-group .actions { float: right; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));
          gap: 10px; margin-top: 8px; }
  .card { border: 1px solid #e1e4e8; border-radius: 4px; padding: 6px;
          background: #fafbfc; text-align: center; font-size: 11.5px; }
  .card img { display: block; max-width: 100px; max-height: 140px;
              margin: 0 auto 4px; border-radius: 2px; }
  .card .ttitle { font-weight: 600; line-height: 1.3;
                   overflow: hidden; max-height: 2.6em;
                   text-overflow: ellipsis; }
  .card .conf { color: #888; }
  .card.confidence-low { border-color: #f2c97a; background: #fff8e6; }
  .row-cover { width: 60px; max-height: 90px; display: inline-block;
               vertical-align: middle; margin-right: 8px; }
</style>
"""

_NAV = """
<nav>
  <a href="/" class="{{ 'active' if route == 'dashboard' else '' }}">Dashboard</a>
  <a href="/proposals" class="{{ 'active' if route == 'list' else '' }}">Proposals</a>
  <a href="/cookbooks" class="{{ 'active' if route == 'cookbooks' else '' }}">Cookbooks</a>
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
  <label>tag (proposed value)
    <input type="text" name="proposed_value" value="{{ filter_proposed_value or '' }}" size="14">
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

{% if filter_field or filter_status or filter_source or filter_book_id or filter_proposed_value %}
<form class="bulk" method="post" action="/bulk">
  <input type="hidden" name="field"   value="{{ filter_field or '' }}">
  <input type="hidden" name="status"  value="{{ filter_status or '' }}">
  <input type="hidden" name="source"  value="{{ filter_source or '' }}">
  <input type="hidden" name="book_id" value="{{ filter_book_id or '' }}">
  <input type="hidden" name="proposed_value" value="{{ filter_proposed_value or '' }}">
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


TEMPLATE_COOKBOOKS = (
    "<!doctype html><html><head><title>Cookbooks — Calibre Cleanup</title>"
    + _STYLE
    + "</head><body>"
    + _NAV
    + """
<h1>Cookbook proposals by tag <span class="dim">(status={{ status }})</span></h1>

<form class="filters" method="get" action="/cookbooks">
  <label>status
    <select name="status">
      {% for s in ['proposed', 'approved', 'rejected', 'applied'] %}
      <option value="{{ s }}" {% if s == status %}selected{% endif %}>{{ s }}</option>
      {% endfor %}
    </select>
  </label>
  <button type="submit">Apply</button>
</form>

{% if not facets %}
<p class="dim">No cookbook_llm proposals at status={{ status }} yet. Run
<code>miner cookbooks</code> first.</p>
{% endif %}

{% for facet_name, groups in facets.items() %}
{% if groups %}
<div class="facet-section">
<h2>{{ facet_name }} <span class="dim">({{ groups|length }} tags,
    {{ groups|sum(attribute='n_books') }} proposals)</span></h2>

{% for g in groups %}
<div class="tag-group">
  {% if status == 'proposed' %}
  <form class="actions" method="post" action="/bulk">
    <input type="hidden" name="source" value="cookbook_llm">
    <input type="hidden" name="proposed_value" value="{{ g.tag }}">
    <input type="hidden" name="status" value="proposed">
    <input type="hidden" name="return_to" value="{{ request_url }}">
    <button type="submit" class="approve" name="action" value="approved">
      Approve all {{ g.n_books }}
    </button>
    <button type="submit" class="reject" name="action" value="rejected">
      Reject all
    </button>
  </form>
  {% endif %}
  <h3>
    <a href="/proposals?source=cookbook_llm&proposed_value={{ g.tag|urlencode }}&status={{ status }}">
      {{ g.tag }}</a>
    <span class="count">{{ g.n_books }} books</span>
  </h3>
  <div class="grid">
    {% for book in g.books %}
    <div class="card{% if book.confidence < 0.7 %} confidence-low{% endif %}">
      {% if has_covers %}
      <a href="/proposals?book_id={{ book.book_id }}">
        <img src="/cover/{{ book.book_id }}.jpg" loading="lazy"
             alt="{{ book.title }}">
      </a>
      {% endif %}
      <div class="ttitle" title="{{ book.title }}">{{ book.title }}</div>
      <div class="conf">{{ '%.1f' % book.confidence }}</div>
    </div>
    {% endfor %}
  </div>
</div>
{% endfor %}
</div>
{% endif %}
{% endfor %}
</body></html>
"""
)

_JINJA = Environment(loader=BaseLoader(), autoescape=select_autoescape(["html"]))
_DASHBOARD_TMPL = _JINJA.from_string(TEMPLATE_DASHBOARD)
_LIST_TMPL = _JINJA.from_string(TEMPLATE_LIST)
_COOKBOOKS_TMPL = _JINJA.from_string(TEMPLATE_COOKBOOKS)


# ---------------------------------------------------------------------------
# Book-context loader — optional read from Calibre's metadata.db
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BookContext:
    title: str
    authors: str  # joined with " & "


class _BookPathLoader:
    """Lazily resolve the ``books.path`` column for cover-image lookup. The
    cover for book ``B`` lives at ``library_root / B.path / cover.jpg``."""

    def __init__(self, calibre_db: Path | None) -> None:
        self._calibre_db = calibre_db
        self._cache: dict[int, str] = {}

    def get(self, book_id: int) -> str | None:
        if self._calibre_db is None:
            return None
        if book_id in self._cache:
            return self._cache[book_id] or None
        uri = f"file:{self._calibre_db.resolve()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            row = conn.execute("SELECT path FROM books WHERE id=?", (book_id,)).fetchone()
        path = (row[0] if row else "") or ""
        self._cache[book_id] = path
        return path or None


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


def create_app(
    *,
    proposals_db: Path,
    calibre_db: Path | None = None,
    library_root: Path | None = None,
) -> FastAPI:
    """Construct the FastAPI app.

    ``library_root`` is the directory containing the Calibre book folders
    (where each book has a ``cover.jpg``). When provided, the cookbook
    dashboard renders cover thumbnails. If omitted but ``calibre_db`` is
    set, defaults to ``calibre_db.parent`` — that's the standard Calibre
    layout (metadata.db lives at the library root)."""
    app = FastAPI(title="Calibre Cleanup Review", docs_url=None, redoc_url=None)
    loader = _BookContextLoader(calibre_db)
    if library_root is None and calibre_db is not None:
        library_root = calibre_db.parent
    book_paths = _BookPathLoader(calibre_db) if library_root else None

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
        proposed_value: str | None = Query(None),
        page: int = Query(1, ge=1),
        size: int = Query(50, ge=1, le=500),
    ) -> str:
        filters = {
            k: v
            for k, v in {
                "field": field or None,
                "status": status or None,
                "source": source or None,
                "proposed_value": proposed_value or None,
            }.items()
            if v
        }
        with _conn() as conn:
            # Count + page-fetch. We paginate via OFFSET/LIMIT, small enough
            # at this scale that it's fine; if the table grows huge we can
            # move to keyset pagination.
            where_clauses: list[str] = ["1=1"]
            params: list[Any] = []
            for key in ("field", "status", "source", "proposed_value"):
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
            filter_proposed_value=proposed_value,
            all_fields=all_fields,
            all_statuses=all_statuses,
            page_range=page_range,
            page_url=page_url,
            request_url=str(request.url),
            current_filter=", ".join(filter_label_parts) or None,
        )

    @app.get("/cookbooks", response_class=HTMLResponse)
    def cookbooks_view(request: Request, status: str = Query("proposed")) -> str:
        """Per-tag dashboard for cookbook_llm proposals. Groups by facet
        (cuisine / technique / dietary) using the taxonomy module's enums,
        then by individual tag within each facet."""
        with _conn() as conn:
            rows = conn.execute(
                """
                SELECT proposed_value AS tag, book_id, confidence
                FROM proposals
                WHERE source = 'cookbook_llm' AND field = 'tags.add'
                  AND status = ? AND proposed_value != '(no tags)'
                ORDER BY proposed_value, confidence DESC
                """,
                (status,),
            ).fetchall()

        # Build facet-aware groupings using the taxonomy enums.
        cuisines = set(cookbook_tagger.CuisineTag.__args__)  # type: ignore[attr-defined]
        techniques = set(cookbook_tagger.TechniqueTag.__args__)  # type: ignore[attr-defined]
        dietaries = set(cookbook_tagger.DietaryTag.__args__)  # type: ignore[attr-defined]

        # Aggregate: tag → list of (book_id, confidence)
        per_tag: dict[str, list[tuple[int, float]]] = {}
        for r in rows:
            per_tag.setdefault(r["tag"], []).append((r["book_id"], r["confidence"]))

        # Pull book titles in one batch for cards.
        all_book_ids = {bid for items in per_tag.values() for bid, _ in items}
        ctx_map = loader.get(all_book_ids)

        def _build_groups(tag_set: set[str]) -> list[dict]:
            out = []
            for tag, items in per_tag.items():
                if tag not in tag_set:
                    continue
                # Sort books within a group by confidence desc — high-conf
                # first lets Erik scan the obvious ones quickly.
                items_sorted = sorted(items, key=lambda x: -x[1])
                books = [
                    {
                        "book_id": bid,
                        "title": ctx_map[bid].title if bid in ctx_map else f"#{bid}",
                        "confidence": conf,
                    }
                    for bid, conf in items_sorted
                ]
                out.append({"tag": tag, "n_books": len(books), "books": books})
            out.sort(key=lambda g: -g["n_books"])
            return out

        facets = {
            "Cuisine": _build_groups(cuisines),
            "Technique": _build_groups(techniques),
            "Dietary": _build_groups(dietaries),
        }

        return _COOKBOOKS_TMPL.render(
            route="cookbooks",
            status=status,
            facets=facets,
            has_covers=book_paths is not None,
            request_url=str(request.url),
            current_filter=f"status={status}",
        )

    @app.get("/cover/{book_id}.jpg")
    def cover(book_id: int) -> FileResponse:
        """Stream the cover.jpg for a book. Returns 404 if missing."""
        if library_root is None or book_paths is None:
            raise HTTPException(status_code=404, detail="library_root not configured")
        book_dir = book_paths.get(book_id)
        if not book_dir:
            raise HTTPException(status_code=404, detail=f"no path for book {book_id}")
        cover_file = library_root / book_dir / "cover.jpg"
        if not cover_file.is_file():
            raise HTTPException(status_code=404, detail="no cover")
        return FileResponse(cover_file, media_type="image/jpeg")

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
        proposed_value: str = Form(""),
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
                proposed_value=proposed_value or None,
                ids=ids,
                notes=notes or None,
            )
        return RedirectResponse(return_to, status_code=303)

    return app


def serve(
    *,
    proposals_db: Path,
    calibre_db: Path | None = None,
    library_root: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8090,
) -> None:
    """Run the webapp with uvicorn (blocks until interrupted)."""
    import uvicorn

    app = create_app(
        proposals_db=proposals_db, calibre_db=calibre_db, library_root=library_root,
    )
    uvicorn.run(app, host=host, port=port, log_level="info")
