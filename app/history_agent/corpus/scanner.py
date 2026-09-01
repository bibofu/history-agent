from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from history_agent.corpus.models import (
    CatalogDocument,
    CorpusCatalog,
    CorpusScanSummary,
    FileObservation,
    FileStatus,
)
from history_agent.db import Database
from history_agent.errors import CorpusScanError
from history_agent.log import log_event

logger = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_pdf(path: Path) -> dict[str, Any]:
    reader = PdfReader(str(path), strict=False)
    is_encrypted = bool(reader.is_encrypted)
    if is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:  # pragma: no cover - depends on encrypted fixtures
            raise CorpusScanError(f"Cannot open encrypted PDF {path.name}: {exc}") from exc
    metadata: Any = reader.metadata or {}
    return {
        "page_count": len(reader.pages),
        "pdf_title": str(metadata.get("/Title") or "") or None,
        "pdf_author": str(metadata.get("/Author") or "") or None,
        "is_encrypted": is_encrypted,
    }


def _upsert_document(database: Database, document: CatalogDocument, timestamp: str) -> None:
    with database.connect() as connection:
        connection.execute(
            """
            INSERT INTO documents (
                document_id, title, creators_json, source_type, edition, volume,
                source_series_json, verification_status, enabled, ocr_strategy,
                expected_page_count, notes, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                title = excluded.title,
                creators_json = excluded.creators_json,
                source_type = excluded.source_type,
                edition = excluded.edition,
                volume = excluded.volume,
                source_series_json = excluded.source_series_json,
                verification_status = excluded.verification_status,
                enabled = excluded.enabled,
                ocr_strategy = excluded.ocr_strategy,
                expected_page_count = excluded.expected_page_count,
                notes = excluded.notes,
                updated_at = excluded.updated_at
            """,
            (
                document.document_id,
                document.title,
                json.dumps(document.creators, ensure_ascii=False),
                document.source_type,
                document.edition,
                document.volume,
                json.dumps(document.source_series, ensure_ascii=False),
                document.verification_status,
                int(document.enabled),
                document.ocr_strategy,
                document.expected_page_count,
                document.notes,
                timestamp,
                timestamp,
            ),
        )


def _current_file(database: Database, document_id: str) -> Any | None:
    with database.connect() as connection:
        return connection.execute(
            """
            SELECT * FROM document_files
            WHERE document_id = ? AND is_current = 1
            ORDER BY file_id DESC LIMIT 1
            """,
            (document_id,),
        ).fetchone()


def _save_file(
    database: Database,
    document_id: str,
    path: Path,
    relative_path: str,
    sha256: str,
    metadata: dict[str, Any],
    timestamp: str,
) -> None:
    modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()
    with database.connect() as connection:
        connection.execute(
            "UPDATE document_files SET is_current = 0 WHERE document_id = ?",
            (document_id,),
        )
        connection.execute(
            """
            INSERT INTO document_files (
                document_id, relative_path, filename, sha256, size_bytes, page_count,
                modified_at, pdf_title, pdf_author, is_encrypted, first_seen_at,
                last_seen_at, is_current, is_present
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1)
            ON CONFLICT(document_id, sha256) DO UPDATE SET
                relative_path = excluded.relative_path,
                filename = excluded.filename,
                size_bytes = excluded.size_bytes,
                page_count = excluded.page_count,
                modified_at = excluded.modified_at,
                pdf_title = excluded.pdf_title,
                pdf_author = excluded.pdf_author,
                is_encrypted = excluded.is_encrypted,
                last_seen_at = excluded.last_seen_at,
                is_current = 1,
                is_present = 1
            """,
            (
                document_id,
                relative_path,
                path.name,
                sha256,
                path.stat().st_size,
                metadata["page_count"],
                modified_at,
                metadata["pdf_title"],
                metadata["pdf_author"],
                int(metadata["is_encrypted"]),
                timestamp,
                timestamp,
            ),
        )


def _mark_missing(database: Database, document_id: str, timestamp: str) -> None:
    with database.connect() as connection:
        connection.execute(
            """
            UPDATE document_files
            SET is_present = 0, last_seen_at = ?
            WHERE document_id = ? AND is_current = 1
            """,
            (timestamp, document_id),
        )


def _record_scan(database: Database, summary: CorpusScanSummary) -> None:
    with database.connect() as connection:
        connection.execute(
            """
            INSERT INTO corpus_scan_runs (
                run_id, started_at, finished_at, status, summary_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                summary.run_id,
                summary.started_at,
                summary.finished_at,
                summary.status,
                summary.model_dump_json(),
            ),
        )


def scan_corpus(
    *,
    database: Database,
    catalog: CorpusCatalog,
    docs_dir: Path,
    project_root: Path,
    run_id: str,
    accept_changes: bool = False,
) -> CorpusScanSummary:
    if not docs_dir.is_dir():
        raise CorpusScanError(f"Corpus directory not found: {docs_dir}")

    started_at = utc_now()
    timestamp = started_at
    database.initialize()

    discovered_paths = sorted(
        (path for path in docs_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: path.name.casefold(),
    )
    discovered: dict[Path, str] = {path: sha256_file(path) for path in discovered_paths}
    by_name = {path.name.casefold(): path for path in discovered_paths}
    unassigned = set(discovered_paths)
    observations: list[FileObservation] = []

    for document in catalog.documents:
        _upsert_document(database, document, timestamp)
        current = _current_file(database, document.document_id)
        path = by_name.get(document.filename.casefold())
        status: FileStatus = "new"

        if path is None and current is not None:
            same_hash = [
                candidate for candidate in unassigned if discovered[candidate] == current["sha256"]
            ]
            if len(same_hash) == 1:
                path = same_hash[0]
                status = "renamed"

        if path is None:
            _mark_missing(database, document.document_id, timestamp)
            observations.append(
                FileObservation(
                    document_id=document.document_id,
                    filename=document.filename,
                    status="missing",
                    expected_page_count=document.expected_page_count,
                    message="Catalog entry has no matching PDF in docs/.",
                )
            )
            continue

        unassigned.discard(path)
        sha256 = discovered[path]
        if status != "renamed":
            if current is None:
                status = "new"
            elif current["sha256"] == sha256:
                status = "unchanged"
            else:
                status = "changed"

        metadata = inspect_pdf(path)
        relative_path = path.relative_to(project_root).as_posix()
        if status == "changed" and not accept_changes:
            page_count_matches = (
                document.expected_page_count is None
                or metadata["page_count"] == document.expected_page_count
            )
            observations.append(
                FileObservation(
                    document_id=document.document_id,
                    filename=path.name,
                    status=status,
                    sha256=sha256,
                    size_bytes=path.stat().st_size,
                    page_count=metadata["page_count"],
                    expected_page_count=document.expected_page_count,
                    page_count_matches=page_count_matches,
                    relative_path=relative_path,
                    message=(
                        "Content hash changed; existing registered version was preserved. "
                        "Review the diff and rerun with --accept-changes."
                    ),
                )
            )
            continue
        _save_file(
            database,
            document.document_id,
            path,
            relative_path,
            sha256,
            metadata,
            timestamp,
        )
        page_count_matches = (
            document.expected_page_count is None
            or metadata["page_count"] == document.expected_page_count
        )
        message = None
        if not page_count_matches:
            message = (
                f"Expected {document.expected_page_count} pages, found {metadata['page_count']}."
            )
        observations.append(
            FileObservation(
                document_id=document.document_id,
                filename=path.name,
                status=status,
                sha256=sha256,
                size_bytes=path.stat().st_size,
                page_count=metadata["page_count"],
                expected_page_count=document.expected_page_count,
                page_count_matches=page_count_matches,
                relative_path=relative_path,
                message=message,
            )
        )
        log_event(
            logger,
            logging.INFO,
            "corpus_file_scanned",
            document_id=document.document_id,
            status=status,
            page_count=metadata["page_count"],
        )

    for path in sorted(unassigned, key=lambda candidate: candidate.name.casefold()):
        observations.append(
            FileObservation(
                filename=path.name,
                status="unregistered",
                sha256=discovered[path],
                size_bytes=path.stat().st_size,
                relative_path=path.relative_to(project_root).as_posix(),
                message="PDF is present in docs/ but absent from the corpus catalog.",
            )
        )

    summary = CorpusScanSummary(
        run_id=run_id,
        started_at=started_at,
        finished_at=utc_now(),
        status="succeeded",
        observations=observations,
    )
    _record_scan(database, summary)
    return summary
