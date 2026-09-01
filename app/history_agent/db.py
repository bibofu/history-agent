from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    creators_json TEXT NOT NULL,
    source_type TEXT NOT NULL,
    edition TEXT,
    volume TEXT,
    source_series_json TEXT NOT NULL,
    verification_status TEXT NOT NULL,
    enabled INTEGER NOT NULL,
    ocr_strategy TEXT NOT NULL,
    expected_page_count INTEGER,
    notes TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS document_files (
    file_id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL REFERENCES documents(document_id),
    relative_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    page_count INTEGER NOT NULL,
    modified_at TEXT NOT NULL,
    pdf_title TEXT,
    pdf_author TEXT,
    is_encrypted INTEGER NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    is_current INTEGER NOT NULL DEFAULT 1,
    is_present INTEGER NOT NULL DEFAULT 1,
    UNIQUE(document_id, sha256)
);

CREATE INDEX IF NOT EXISTS idx_document_files_current
    ON document_files(document_id, is_current, is_present);
CREATE INDEX IF NOT EXISTS idx_document_files_sha256
    ON document_files(sha256);

CREATE TABLE IF NOT EXISTS corpus_scan_runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    status TEXT NOT NULL,
    summary_json TEXT NOT NULL
);
"""


class Database:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)
