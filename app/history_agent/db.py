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

CREATE TABLE IF NOT EXISTS persons (
    person_id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL UNIQUE,
    description TEXT,
    status TEXT NOT NULL CHECK(status IN ('active', 'merged')),
    merged_into_person_id TEXT REFERENCES persons(person_id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK(person_id != merged_into_person_id),
    CHECK(
        (status = 'active' AND merged_into_person_id IS NULL)
        OR (status = 'merged' AND merged_into_person_id IS NOT NULL)
    )
);

CREATE TABLE IF NOT EXISTS person_aliases (
    alias_id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL REFERENCES persons(person_id),
    alias_text TEXT NOT NULL,
    normalized_alias TEXT NOT NULL,
    alias_type TEXT NOT NULL,
    notes TEXT,
    source TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(person_id, normalized_alias)
);

CREATE INDEX IF NOT EXISTS idx_person_aliases_normalized
    ON person_aliases(normalized_alias, is_active);

CREATE TABLE IF NOT EXISTS person_ambiguities (
    ambiguity_id TEXT PRIMARY KEY,
    mention_text TEXT NOT NULL,
    normalized_mention TEXT NOT NULL,
    candidate_person_ids_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('unresolved', 'resolved')),
    resolved_person_id TEXT REFERENCES persons(person_id),
    notes TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_person_ambiguities_mention
    ON person_ambiguities(normalized_mention, status);

CREATE TABLE IF NOT EXISTS person_merge_proposals (
    proposal_id TEXT PRIMARY KEY,
    source_person_id TEXT NOT NULL REFERENCES persons(person_id),
    target_person_id TEXT NOT NULL REFERENCES persons(person_id),
    reason TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending', 'accepted', 'rejected')),
    proposed_by TEXT NOT NULL,
    proposed_at TEXT NOT NULL,
    reviewed_by TEXT,
    reviewed_at TEXT,
    review_note TEXT,
    CHECK(source_person_id != target_person_id)
);

CREATE TABLE IF NOT EXISTS relation_types (
    relation_type TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    object_kind TEXT NOT NULL,
    extraction_policy TEXT NOT NULL,
    requires_human_review INTEGER NOT NULL,
    requires_event INTEGER NOT NULL,
    description TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence_records (
    evidence_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id),
    chunk_id TEXT,
    pdf_page_start INTEGER NOT NULL CHECK(pdf_page_start >= 1),
    pdf_page_end INTEGER NOT NULL CHECK(pdf_page_end >= pdf_page_start),
    quote TEXT NOT NULL,
    extraction_methods_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_evidence_document_page
    ON evidence_records(document_id, pdf_page_start);

CREATE TABLE IF NOT EXISTS historical_events (
    event_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    event_type TEXT NOT NULL,
    start_value TEXT,
    start_precision TEXT NOT NULL,
    start_certainty TEXT NOT NULL,
    start_original_text TEXT,
    end_value TEXT,
    end_precision TEXT,
    end_certainty TEXT,
    end_original_text TEXT,
    location_text TEXT,
    organization_names_json TEXT NOT NULL,
    description TEXT NOT NULL,
    extraction_method TEXT NOT NULL,
    extraction_confidence REAL NOT NULL CHECK(
        extraction_confidence >= 0 AND extraction_confidence <= 1
    ),
    review_status TEXT NOT NULL,
    extractor_version TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS event_participants (
    event_id TEXT NOT NULL REFERENCES historical_events(event_id),
    person_id TEXT NOT NULL REFERENCES persons(person_id),
    role TEXT,
    mention_text TEXT NOT NULL,
    mention_source TEXT NOT NULL DEFAULT 'explicit',
    PRIMARY KEY(event_id, person_id, mention_text)
);

CREATE INDEX IF NOT EXISTS idx_event_participants_person
    ON event_participants(person_id, event_id);

CREATE TABLE IF NOT EXISTS event_evidence (
    event_id TEXT NOT NULL REFERENCES historical_events(event_id),
    evidence_id TEXT NOT NULL REFERENCES evidence_records(evidence_id),
    PRIMARY KEY(event_id, evidence_id)
);

CREATE TABLE IF NOT EXISTS person_relationships (
    relationship_id TEXT PRIMARY KEY,
    relation_type TEXT NOT NULL REFERENCES relation_types(relation_type),
    subject_person_id TEXT NOT NULL REFERENCES persons(person_id),
    subject_mention_text TEXT NOT NULL,
    object_person_id TEXT REFERENCES persons(person_id),
    object_mention_text TEXT,
    organization_name TEXT,
    role_title TEXT,
    start_value TEXT,
    start_precision TEXT NOT NULL,
    start_certainty TEXT NOT NULL,
    start_original_text TEXT,
    end_value TEXT,
    end_precision TEXT,
    end_certainty TEXT,
    end_original_text TEXT,
    event_id TEXT REFERENCES historical_events(event_id),
    description TEXT,
    extraction_method TEXT NOT NULL,
    extraction_confidence REAL NOT NULL CHECK(
        extraction_confidence >= 0 AND extraction_confidence <= 1
    ),
    review_status TEXT NOT NULL,
    reviewed_by TEXT,
    extractor_version TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK(object_person_id IS NOT NULL OR organization_name IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS idx_relationship_subject
    ON person_relationships(subject_person_id, relation_type);
CREATE INDEX IF NOT EXISTS idx_relationship_object
    ON person_relationships(object_person_id, relation_type);

CREATE TABLE IF NOT EXISTS relationship_evidence (
    relationship_id TEXT NOT NULL REFERENCES person_relationships(relationship_id),
    evidence_id TEXT NOT NULL REFERENCES evidence_records(evidence_id),
    PRIMARY KEY(relationship_id, evidence_id)
);

CREATE TABLE IF NOT EXISTS research_revisions (
    revision_id TEXT PRIMARY KEY,
    record_type TEXT NOT NULL CHECK(record_type IN ('event', 'relationship')),
    record_id TEXT NOT NULL,
    previous_json TEXT NOT NULL,
    revised_json TEXT NOT NULL,
    reason TEXT NOT NULL,
    revised_by TEXT NOT NULL,
    revised_at TEXT NOT NULL
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
            relationship_columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(person_relationships)"
                ).fetchall()
            }
            if "subject_mention_text" not in relationship_columns:
                connection.execute(
                    "ALTER TABLE person_relationships ADD COLUMN subject_mention_text TEXT"
                )
            if "object_mention_text" not in relationship_columns:
                connection.execute(
                    "ALTER TABLE person_relationships ADD COLUMN object_mention_text TEXT"
                )
            participant_columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(event_participants)"
                ).fetchall()
            }
            if "mention_source" not in participant_columns:
                connection.execute(
                    "ALTER TABLE event_participants "
                    "ADD COLUMN mention_source TEXT NOT NULL DEFAULT 'explicit'"
                )
            connection.execute("PRAGMA user_version = 4")
