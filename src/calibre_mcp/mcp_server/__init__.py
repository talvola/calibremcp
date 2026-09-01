"""Calibre MCP server.

Exposes Erik's Calibre library over MCP so Claude clients (desktop, mobile)
can browse and edit metadata without a terminal. Reads come straight from
``metadata.db`` (lock-free); writes go through Calibre-Web's HTTP API so
Calibre-Web remains the single writer.
"""
