from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import pymupdf
from pypdf import PdfReader

from history_agent.db import Database
from history_agent.errors import ExtractionError
from history_agent.extraction.models import (
    DocumentExtractionResult,
    ExtractionMethod,
    FullExtractionSummary,
    PageRecord,
)
from history_agent.extraction.text import (
    build_failed_page_record,
    build_page_record,
    count_page_images,
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


ParserName = Literal["pymupdf", "pypdf"]


def parser_pipeline_version(parser_name: ParserName) -> str:
    if parser_name == "pymupdf":
        return f"pymupdf {version('pymupdf')}; fallback pypdf {version('pypdf')}"
    return f"pypdf {version('pypdf')}"


def current_files(database: Database) -> dict[str, dict[str, Any]]:
    with database.connect() as connection:
        rows = connection.execute(
            """
            SELECT d.document_id, d.title, d.ocr_strategy, f.filename, f.relative_path,
                   f.sha256, f.page_count
            FROM documents d
            JOIN document_files f ON f.document_id = d.document_id
            WHERE f.is_current = 1 AND f.is_present = 1 AND d.enabled = 1
            ORDER BY d.document_id
            """
        ).fetchall()
    return {row["document_id"]: dict(row) for row in rows}


def _load_records(
    paths: list[Path],
    *,
    file_sha256: str,
    extractor_version: str,
) -> dict[int, PageRecord]:
    records: dict[int, PageRecord] = {}
    for path in paths:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = PageRecord.model_validate_json(line)
                except Exception as exc:
                    raise ExtractionError(
                        f"Invalid page record in {path} line {line_number}: {exc}"
                    ) from exc
                if (
                    record.file_sha256 == file_sha256
                    and record.extractor_version == extractor_version
                    and record.status != "failed"
                ):
                    records[record.pdf_page] = record
    return records


def _write_final_records(path: Path, records: dict[int, PageRecord]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
        for pdf_page in sorted(records):
            stream.write(records[pdf_page].model_dump_json() + "\n")
    os.replace(temporary_path, path)


def _apply_ocr_policy(record: PageRecord, file_record: dict[str, Any]) -> None:
    from history_agent.extraction.text import classify_extracted_text

    if record.status == "failed":
        return
    policy_flags = {"catalog_full_ocr_required", "catalog_page_ocr_required"}
    if policy_flags.intersection(record.quality_flags):
        record.status, _ = classify_extracted_text(
            record.normalized_text, image_object_count=record.image_object_count
        )
        record.quality_flags = [f for f in record.quality_flags if f not in policy_flags]
    if file_record.get("ocr_strategy") == "full_required":
        record.status = "ocr_required"
        record.quality_flags.append("catalog_full_ocr_required")
    elif record.pdf_page in file_record.get("ocr_pages", []):
        record.status = "ocr_required"
        record.quality_flags.append("catalog_page_ocr_required")


def extract_document(
    *,
    file_record: dict[str, Any],
    project_root: Path,
    output_dir: Path,
    run_id: str,
    rebuild: bool,
    parser_name: ParserName,
) -> DocumentExtractionResult:
    started = perf_counter()
    document_id = str(file_record["document_id"])
    source_path = project_root / str(file_record["relative_path"])
    output_path = output_dir / f"{document_id}.jsonl"
    partial_path = output_dir / f"{document_id}.partial.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    extractor_version = parser_pipeline_version(parser_name)
    file_sha256 = str(file_record["sha256"])

    records: dict[int, PageRecord] = {}
    if not rebuild:
        records = _load_records(
            [output_path, partial_path],
            file_sha256=file_sha256,
            extractor_version=extractor_version,
        )
    else:
        partial_path.write_text("", encoding="utf-8")

    pymupdf_document: Any | None = None
    pypdf_reader: PdfReader | None = None
    if parser_name == "pymupdf":
        pymupdf_document = pymupdf.open(source_path)
        total_pages = pymupdf_document.page_count
    else:
        pypdf_reader = PdfReader(str(source_path), strict=False)
        total_pages = len(pypdf_reader.pages)
    reused_pages = len(records)
    processed_pages = 0

    with partial_path.open("a", encoding="utf-8", newline="\n") as partial_stream:
        for pdf_page in range(1, total_pages + 1):
            if pdf_page in records:
                _apply_ocr_policy(records[pdf_page], file_record)
                continue
            try:
                extraction_method: ExtractionMethod
                if parser_name == "pymupdf":
                    assert pymupdf_document is not None
                    page = pymupdf_document[pdf_page - 1]
                    text = page.get_text("text") or ""
                    image_object_count = len(page.get_images(full=True))
                    extraction_method = "pymupdf_text_layer"
                else:
                    assert pypdf_reader is not None
                    page = pypdf_reader.pages[pdf_page - 1]
                    text = page.extract_text() or ""
                    image_object_count = count_page_images(page)
                    extraction_method = "pypdf_text_layer"
                record = build_page_record(
                    document_id=document_id,
                    file_sha256=file_sha256,
                    pdf_page=pdf_page,
                    text=text,
                    image_object_count=image_object_count,
                    run_id=run_id,
                    extraction_method=extraction_method,
                    extractor_version=extractor_version,
                )
            except Exception as primary_exc:
                try:
                    if parser_name == "pypdf":
                        raise primary_exc
                    if pypdf_reader is None:
                        pypdf_reader = PdfReader(str(source_path), strict=False)
                    page = pypdf_reader.pages[pdf_page - 1]
                    text = page.extract_text() or ""
                    record = build_page_record(
                        document_id=document_id,
                        file_sha256=file_sha256,
                        pdf_page=pdf_page,
                        text=text,
                        image_object_count=count_page_images(page),
                        run_id=run_id,
                        extraction_method="pypdf_text_layer",
                        extractor_version=extractor_version,
                    )
                    record.quality_flags.append("primary_parser_failed_used_pypdf")
                except Exception as fallback_exc:
                    combined_exc = ExtractionError(
                        f"Primary parser failed: {primary_exc}; fallback failed: {fallback_exc}"
                    )
                    record = build_failed_page_record(
                        document_id=document_id,
                        file_sha256=file_sha256,
                        pdf_page=pdf_page,
                        run_id=run_id,
                        exc=combined_exc,
                        extractor_version=extractor_version,
                    )
            _apply_ocr_policy(record, file_record)
            records[pdf_page] = record
            partial_stream.write(record.model_dump_json() + "\n")
            processed_pages += 1
            if processed_pages % 100 == 0:
                partial_stream.flush()

    if len(records) != total_pages:
        raise ExtractionError(
            f"Page count mismatch for {document_id}: "
            f"{len(records)} records for {total_pages} pages."
        )
    _write_final_records(output_path, records)
    partial_path.unlink(missing_ok=True)
    if pymupdf_document is not None:
        pymupdf_document.close()

    status_counts: dict[str, int] = {}
    total_characters = 0
    for record in records.values():
        status_counts[record.status] = status_counts.get(record.status, 0) + 1
        total_characters += record.character_count
    return DocumentExtractionResult(
        document_id=document_id,
        filename=str(file_record["filename"]),
        total_pages=total_pages,
        processed_pages=processed_pages,
        reused_pages=reused_pages,
        status_counts=status_counts,
        total_characters=total_characters,
        output_path=str(output_path),
        elapsed_seconds=round(perf_counter() - started, 3),
    )


def extract_all_text(
    *,
    database: Database,
    project_root: Path,
    output_dir: Path,
    reports_dir: Path,
    run_id: str,
    document_ids: list[str] | None = None,
    rebuild: bool = False,
    parser_name: ParserName = "pymupdf",
    ocr_pages_by_document: dict[str, list[int]] | None = None,
) -> FullExtractionSummary:
    started_at = utc_now()
    files = current_files(database)
    selected_ids = document_ids or list(files)
    unknown_ids = sorted(set(selected_ids) - set(files))
    if unknown_ids:
        raise ExtractionError(f"Unknown or unavailable document IDs: {', '.join(unknown_ids)}")

    results: list[DocumentExtractionResult] = []
    for document_id in selected_ids:
        files[document_id]["ocr_pages"] = (ocr_pages_by_document or {}).get(document_id, [])
        if any(
            not 1 <= page <= int(files[document_id]["page_count"])
            for page in files[document_id]["ocr_pages"]
        ):
            raise ExtractionError(f"OCR page outside PDF bounds for {document_id}")
        results.append(
            extract_document(
                file_record=files[document_id],
                project_root=project_root,
                output_dir=output_dir,
                run_id=run_id,
                rebuild=rebuild,
                parser_name=parser_name,
            )
        )

    summary = FullExtractionSummary(
        run_id=run_id,
        parser=parser_name,
        started_at=started_at,
        finished_at=utc_now(),
        documents=results,
    )
    payload = summary.model_dump()
    payload["totals"] = summary.totals
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"text_extraction_{run_id}.json").write_text(rendered, encoding="utf-8")
    (reports_dir / "text_extraction_latest.json").write_text(rendered, encoding="utf-8")
    return summary
