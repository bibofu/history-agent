from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Protocol, cast

import pymupdf

from history_agent.db import Database
from history_agent.errors import ExtractionError
from history_agent.extraction.full import current_files
from history_agent.extraction.models import (
    OcrDocumentResult,
    OcrExtractionSummary,
    PageRecord,
)
from history_agent.extraction.report import load_page_records
from history_agent.extraction.text import build_failed_page_record, build_page_record

DEFAULT_DPI = 110
DEFAULT_DETECTION_MODEL = "PP-OCRv5_mobile_det"
DEFAULT_RECOGNITION_MODEL = "PP-OCRv5_mobile_rec"


class OcrEngine(Protocol):
    def predict(self, input_data: Any) -> Iterable[Any]: ...


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def ocr_engine_version(dpi: int = DEFAULT_DPI) -> str:
    try:
        paddleocr_version = version("paddleocr")
        paddle_version = version("paddlepaddle")
    except PackageNotFoundError as exc:
        raise ExtractionError(
            "OCR dependencies are missing. Run: uv sync --extra ocr --group dev"
        ) from exc
    return (
        f"paddleocr {paddleocr_version}; paddlepaddle {paddle_version}; "
        f"det={DEFAULT_DETECTION_MODEL}; rec={DEFAULT_RECOGNITION_MODEL}; "
        f"dpi={dpi}; mkldnn=true"
    )


def create_paddle_ocr_engine() -> OcrEngine:
    try:
        from paddleocr import PaddleOCR  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ExtractionError(
            "OCR dependencies are missing. Run: uv sync --extra ocr --group dev"
        ) from exc
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    return cast(
        OcrEngine,
        PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            engine="paddle",
            enable_mkldnn=True,
            text_detection_model_name=DEFAULT_DETECTION_MODEL,
            text_recognition_model_name=DEFAULT_RECOGNITION_MODEL,
        ),
    )


def parse_paddle_ocr_result(results: Iterable[Any]) -> tuple[str, float | None]:
    texts: list[str] = []
    scores: list[float] = []
    for result in results:
        result_json = getattr(result, "json", result)
        if callable(result_json):
            result_json = result_json()
        if not isinstance(result_json, dict):
            raise ExtractionError(f"Unexpected PaddleOCR result type: {type(result_json)!r}")
        payload = result_json.get("res", result_json)
        if not isinstance(payload, dict):
            raise ExtractionError("PaddleOCR result does not contain a mapping payload.")
        raw_texts = list(payload.get("rec_texts", []))
        raw_scores = list(payload.get("rec_scores", []))
        for index, raw_text in enumerate(raw_texts):
            text = str(raw_text).strip()
            if not text:
                continue
            texts.append(text)
            if index < len(raw_scores):
                scores.append(float(raw_scores[index]))
    confidence = round(mean(scores), 6) if scores else None
    return "\n".join(texts), confidence


def render_page_for_ocr(document: pymupdf.Document, pdf_page: int, dpi: int) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise ExtractionError(
            "OCR image dependencies are missing. Run: uv sync --extra ocr --group dev"
        ) from exc
    page = document[pdf_page - 1]
    pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
    return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )


def _load_reusable_records(
    paths: list[Path], *, file_sha256: str, engine_version: str
) -> dict[int, PageRecord]:
    records: dict[int, PageRecord] = {}
    for path in paths:
        if not path.is_file():
            continue
        for pdf_page, record in load_page_records(path).items():
            if (
                record.file_sha256 == file_sha256
                and record.extractor_version == engine_version
                and record.status != "failed"
            ):
                records[pdf_page] = record
    return records


def _write_final_records(path: Path, records: dict[int, PageRecord]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
        for pdf_page in sorted(records):
            stream.write(records[pdf_page].model_dump_json() + "\n")
    os.replace(temporary_path, path)


def extract_document_ocr(
    *,
    file_record: dict[str, Any],
    source_records: dict[int, PageRecord],
    project_root: Path,
    output_dir: Path,
    run_id: str,
    engine_version: str,
    engine: OcrEngine,
    dpi: int,
    rebuild: bool,
    selected_pages: set[int] | None,
    max_pages: int | None,
) -> OcrDocumentResult:
    started = perf_counter()
    document_id = str(file_record["document_id"])
    filename = str(file_record["filename"])
    source_path = project_root / str(file_record["relative_path"])
    output_path = output_dir / f"{document_id}.jsonl"
    partial_path = output_dir / f"{document_id}.partial.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    file_sha256 = str(file_record["sha256"])
    candidates = sorted(
        record.pdf_page for record in source_records.values() if record.status == "ocr_required"
    )
    if selected_pages is not None:
        unknown_pages = sorted(selected_pages - set(candidates))
        if unknown_pages:
            raise ExtractionError(
                f"Pages are not OCR candidates for {document_id}: "
                + ", ".join(map(str, unknown_pages))
            )
        candidates = [page for page in candidates if page in selected_pages]

    records: dict[int, PageRecord] = {}
    if not rebuild:
        records = _load_reusable_records(
            [output_path, partial_path],
            file_sha256=file_sha256,
            engine_version=engine_version,
        )
    else:
        partial_path.write_text("", encoding="utf-8")
    records = {page: record for page, record in records.items() if page in candidates}
    pending = [page for page in candidates if page not in records]
    if max_pages is not None:
        pending = pending[:max_pages]

    processed_pages = 0
    document = pymupdf.open(source_path)
    try:
        with partial_path.open("a", encoding="utf-8", newline="\n") as partial_stream:
            for pdf_page in pending:
                source_record = source_records[pdf_page]
                try:
                    image = render_page_for_ocr(document, pdf_page, dpi)
                    text, confidence = parse_paddle_ocr_result(engine.predict(image))
                    record = build_page_record(
                        document_id=document_id,
                        file_sha256=file_sha256,
                        pdf_page=pdf_page,
                        text=text,
                        image_object_count=0,
                        run_id=run_id,
                        extraction_method="ocr",
                        extractor_version=engine_version,
                    )
                    record.image_object_count = source_record.image_object_count
                    record.ocr_confidence = confidence
                    record.quality_flags.extend(
                        flag
                        for flag in source_record.quality_flags
                        if flag not in record.quality_flags
                    )
                    record.quality_flags.append("ocr_from_image_only_page")
                    if record.character_count < 20 or record.cjk_character_count < 5:
                        record.quality_flags.append("ocr_low_substantive_text")
                except Exception as exc:
                    record = build_failed_page_record(
                        document_id=document_id,
                        file_sha256=file_sha256,
                        pdf_page=pdf_page,
                        run_id=run_id,
                        exc=exc,
                        extractor_version=engine_version,
                    )
                records[pdf_page] = record
                partial_stream.write(record.model_dump_json() + "\n")
                partial_stream.flush()
                processed_pages += 1
    finally:
        document.close()

    _write_final_records(output_path, records)
    partial_path.unlink(missing_ok=True)
    status_counts: dict[str, int] = {}
    confidences: list[float] = []
    for record in records.values():
        status_counts[record.status] = status_counts.get(record.status, 0) + 1
        if record.ocr_confidence is not None:
            confidences.append(record.ocr_confidence)
    return OcrDocumentResult(
        document_id=document_id,
        filename=filename,
        candidate_pages=len(candidates),
        processed_pages=processed_pages,
        reused_pages=len(records) - processed_pages,
        remaining_pages=len(candidates) - len(records),
        status_counts=status_counts,
        total_characters=sum(record.character_count for record in records.values()),
        mean_confidence=round(mean(confidences), 6) if confidences else None,
        output_path=str(output_path),
        elapsed_seconds=round(perf_counter() - started, 3),
    )


def extract_all_ocr(
    *,
    database: Database,
    project_root: Path,
    pages_dir: Path,
    output_dir: Path,
    reports_dir: Path,
    run_id: str,
    document_ids: list[str] | None = None,
    selected_pages: set[int] | None = None,
    max_pages: int | None = None,
    dpi: int = DEFAULT_DPI,
    rebuild: bool = False,
    engine_factory: Callable[[], OcrEngine] = create_paddle_ocr_engine,
) -> OcrExtractionSummary:
    if selected_pages is not None and (document_ids is None or len(document_ids) != 1):
        raise ExtractionError("--page requires exactly one --document-id.")
    files = current_files(database)
    selected_ids = document_ids or list(files)
    unknown_ids = sorted(set(selected_ids) - set(files))
    if unknown_ids:
        raise ExtractionError(f"Unknown or unavailable document IDs: {', '.join(unknown_ids)}")

    source_by_document: dict[str, dict[int, PageRecord]] = {}
    has_work = False
    for document_id in selected_ids:
        source_records = load_page_records(pages_dir / f"{document_id}.jsonl")
        source_by_document[document_id] = source_records
        if any(record.status == "ocr_required" for record in source_records.values()):
            has_work = True
    engine_version = ocr_engine_version(dpi)
    engine = engine_factory() if has_work else None
    if engine is None:
        raise ExtractionError("No OCR candidates were found in the selected documents.")

    started_at = utc_now()
    results: list[OcrDocumentResult] = []
    remaining_limit = max_pages
    for document_id in selected_ids:
        document_limit = remaining_limit
        result = extract_document_ocr(
            file_record=files[document_id],
            source_records=source_by_document[document_id],
            project_root=project_root,
            output_dir=output_dir,
            run_id=run_id,
            engine_version=engine_version,
            engine=engine,
            dpi=dpi,
            rebuild=rebuild,
            selected_pages=selected_pages,
            max_pages=document_limit,
        )
        results.append(result)
        if remaining_limit is not None:
            remaining_limit = max(0, remaining_limit - result.processed_pages)

    summary = OcrExtractionSummary(
        run_id=run_id,
        engine_version=engine_version,
        started_at=started_at,
        finished_at=utc_now(),
        documents=results,
    )
    payload = summary.model_dump()
    payload["totals"] = summary.totals
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"ocr_extraction_{run_id}.json").write_text(rendered, encoding="utf-8")
    (reports_dir / "ocr_extraction_latest.json").write_text(rendered, encoding="utf-8")
    return summary
