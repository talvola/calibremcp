-- cleanup_proposals.db — standalone SQLite database for the cleanup pipeline.
--
-- The miner writes proposed metadata edits here for human review. This DB is
-- intentionally separate from Calibre's metadata.db; proposals only reach
-- Calibre via an explicit, separate apply step once approved.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS proposals (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id          INTEGER NOT NULL,             -- Calibre books.id at proposal time
    book_uuid        TEXT,                          -- Calibre's UUID — stable across rebuilds
    field            TEXT NOT NULL,                 -- see field catalogue in opf_parser.py
    calibre_value    TEXT,                          -- current value in Calibre (NULL if empty/sentinel)
    proposed_value   TEXT NOT NULL,                 -- TEXT preserves fractional series_index like '1.3'
    source           TEXT NOT NULL,                 -- 'opf', 'title_pattern', 'goodreads_isbn', 'openlibrary', 'google_books', 'llm'
    confidence       REAL NOT NULL,                 -- 0.0 .. 1.0
    status           TEXT NOT NULL DEFAULT 'proposed',
    conflict_reason  TEXT,                          -- populated when status='conflict'
    notes            TEXT,                          -- free-form: extraction details or reviewer notes
    run_id           INTEGER,                       -- which miner_runs row produced this
    created_at       TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    reviewed_at      TEXT,
    applied_at       TEXT,
    CHECK (status IN ('proposed','approved','rejected','applied','superseded','conflict')),
    CHECK (confidence >= 0.0 AND confidence <= 1.0)
);

-- Idempotent re-runs: same (book, field, source, value) tuple is a no-op.
-- List-valued fields like 'tags.add' rely on proposed_value to distinguish rows.
CREATE UNIQUE INDEX IF NOT EXISTS ix_proposals_identity
    ON proposals(book_id, field, source, proposed_value);

CREATE INDEX IF NOT EXISTS ix_proposals_book          ON proposals(book_id);
CREATE INDEX IF NOT EXISTS ix_proposals_status_field  ON proposals(status, field);
CREATE INDEX IF NOT EXISTS ix_proposals_field_source  ON proposals(field, source);
CREATE INDEX IF NOT EXISTS ix_proposals_run           ON proposals(run_id);


CREATE TABLE IF NOT EXISTS miner_runs (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at         TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at       TEXT,
    source             TEXT NOT NULL,              -- matches proposals.source
    library_root       TEXT NOT NULL,
    metadata_db_path   TEXT NOT NULL,
    books_scanned      INTEGER NOT NULL DEFAULT 0,
    books_with_epub    INTEGER NOT NULL DEFAULT 0,
    books_parsed       INTEGER NOT NULL DEFAULT 0,
    proposals_emitted  INTEGER NOT NULL DEFAULT 0,
    errors             INTEGER NOT NULL DEFAULT 0,
    dry_run            INTEGER NOT NULL DEFAULT 0, -- 0=persisted, 1=dry-run (nothing written)
    notes              TEXT
);


CREATE TABLE IF NOT EXISTS miner_errors (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     INTEGER NOT NULL,
    book_id    INTEGER,
    book_path  TEXT,
    error      TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES miner_runs(id)
);
CREATE INDEX IF NOT EXISTS ix_miner_errors_run ON miner_errors(run_id);
