"""Metadata cleanup pipeline for the Calibre library.

Phase 1: OPF miner. Walks books in a Calibre library, extracts metadata
from each EPUB's internal OPF manifest, and proposes field backfills where
Calibre's metadata.db is empty.

This subpackage is intentionally isolated from the main calibre_mcp server:
it has its own standalone SQLite propose-queue (``cleanup_proposals.db``)
and never writes to Calibre's ``metadata.db`` directly. Approved proposals
are later applied via a separate ``apply`` step (not in Phase 1).
"""

from calibre_mcp.cleanup.opf_parser import OpfMetadata, parse_epub, parse_opf_bytes
from calibre_mcp.cleanup.proposals import Proposal, connect, insert_proposal

__all__ = [
    "OpfMetadata",
    "Proposal",
    "connect",
    "insert_proposal",
    "parse_epub",
    "parse_opf_bytes",
]
