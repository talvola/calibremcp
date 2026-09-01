FROM python:3.12-slim AS base

WORKDIR /app

# Install only the runtime deps the MCP server actually uses. We deliberately
# do NOT `pip install .` because the project's pyproject.toml carries the full
# cleanup-pipeline dep set (lancedb, fastembed, pymupdf, anthropic, ...) which
# the MCP server never touches at runtime and which would bloat the image by
# ~2 GB. The MCP server's hard runtime deps are minimal.
RUN pip install --no-cache-dir \
    "mcp>=1.0.0" \
    "httpx>=0.27.0" \
    "beautifulsoup4>=4.12.0"

# Copy only the source files the MCP server needs.
COPY src/calibre_mcp/mcp_server/ /app/calibre_mcp/mcp_server/
COPY src/calibre_mcp/cleanup/calibre_reader.py /app/calibre_mcp/cleanup/calibre_reader.py

# Empty __init__.py files override the heavy upstream ones so we don't drag in
# Sandra's framework or the cleanup-pipeline imports.
RUN : > /app/calibre_mcp/__init__.py \
 && : > /app/calibre_mcp/cleanup/__init__.py

ENV PYTHONPATH=/app

# HTTP transport mode env vars (override in docker-compose):
#   CALIBRE_METADATA_DB   path to mounted metadata.db (must be readable)
#   CALIBRE_WEB_URL       Calibre-Web base URL (used for OAuth login backend)
#   MCP_ISSUER_URL        externally-resolvable URL of THIS MCP server
ENV CALIBRE_METADATA_DB=""
ENV CALIBRE_WEB_URL=""
ENV MCP_ISSUER_URL=""

EXPOSE 8101

CMD ["python", "-m", "calibre_mcp.mcp_server", "--transport", "streamable-http"]
