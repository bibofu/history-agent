from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from history_agent.db import Database
from history_agent.errors import ExtractionError
from history_agent.extraction.full import current_files
from history_agent.extraction.models import PageRecord


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def load_page_records(path: Path) -> dict[int, PageRecord]:
    if not path.is_file():
        raise ExtractionError(f"Page data does not exist: {path}")
    records: dict[int, PageRecord] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = PageRecord.model_validate_json(line)
            except Exception as exc:
                message = f"Invalid record in {path} line {line_number}: {exc}"
                raise ExtractionError(message) from exc
            records[record.pdf_page] = record
    return records


def compact_page_ranges(pages: list[int]) -> str:
    if not pages:
        return "—"
    ordered = sorted(set(pages))
    ranges: list[str] = []
    start = previous = ordered[0]
    for page in ordered[1:]:
        if page == previous + 1:
            previous = page
            continue
        ranges.append(str(start) if start == previous else f"{start}–{previous}")
        start = previous = page
    ranges.append(str(start) if start == previous else f"{start}–{previous}")
    return ", ".join(ranges)


def _ocr_route(ocr_record: PageRecord | None) -> str:
    if ocr_record is None:
        return "ocr_pending"
    if ocr_record.status == "failed":
        return "ocr_failed"
    if ocr_record.status == "empty" or ocr_record.character_count == 0:
        return "visual_or_blank"
    if ocr_record.character_count < 20 or ocr_record.cjk_character_count < 5:
        return "ocr_low_text"
    return "ocr_completed"


def _document_report(
    file_record: dict[str, Any],
    page_records: dict[int, PageRecord],
    ocr_records: dict[int, PageRecord],
) -> dict[str, Any]:
    status_counts = Counter(record.status for record in page_records.values())
    flag_counts = Counter(flag for record in page_records.values() for flag in record.quality_flags)
    extracted_lengths = [
        record.character_count for record in page_records.values() if record.status == "extracted"
    ]
    candidates: list[dict[str, Any]] = []
    for record in page_records.values():
        if record.status != "ocr_required":
            continue
        ocr_record = ocr_records.get(record.pdf_page)
        candidates.append(
            {
                "pdf_page": record.pdf_page,
                "image_object_count": record.image_object_count,
                "route": _ocr_route(ocr_record),
                "route_reasons": record.quality_flags,
                "ocr_character_count": ocr_record.character_count if ocr_record else None,
                "ocr_cjk_character_count": ocr_record.cjk_character_count if ocr_record else None,
                "ocr_confidence": ocr_record.ocr_confidence if ocr_record else None,
            }
        )
    route_counts = Counter(candidate["route"] for candidate in candidates)
    empty_pages = sorted(
        record.pdf_page for record in page_records.values() if record.status == "empty"
    )
    failed_pages = sorted(
        record.pdf_page for record in page_records.values() if record.status == "failed"
    )
    return {
        "document_id": file_record["document_id"],
        "filename": file_record["filename"],
        "ocr_strategy": file_record["ocr_strategy"],
        "total_pages": len(page_records),
        "status_counts": dict(sorted(status_counts.items())),
        "quality_flag_counts": dict(sorted(flag_counts.items())),
        "text_character_stats": {
            "total": sum(extracted_lengths),
            "minimum": min(extracted_lengths, default=0),
            "median": round(float(median(extracted_lengths)), 1) if extracted_lengths else 0,
            "maximum": max(extracted_lengths, default=0),
        },
        "ocr_route_counts": dict(sorted(route_counts.items())),
        "ocr_candidates": candidates,
        "empty_pages": empty_pages,
        "failed_pages": failed_pages,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# 文本提取与 OCR 质量报告",
        "",
        f"- 生成时间：{payload['generated_at']}",
        f"- 文献数：{payload['totals']['documents']}",
        f"- PDF 页数：{payload['totals']['pages']}",
        f"- 已提取文字页：{payload['totals']['text_extracted']}",
        f"- OCR 候选页：{payload['totals']['ocr_candidates']}",
        f"- OCR 已完成页：{payload['totals']['ocr_completed']}",
        f"- OCR 待处理页：{payload['totals']['ocr_pending']}",
        f"- 空白或纯矢量页：{payload['totals']['empty']}",
        f"- 失败页：{payload['totals']['failed']}",
        "",
        "| 文献 | 总页数 | 文字页 | OCR 候选 | 已完成 | 待处理 | 低文字/视觉页 | 失败 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for document in payload["documents"]:
        status = document["status_counts"]
        routes = document["ocr_route_counts"]
        visual = routes.get("ocr_low_text", 0) + routes.get("visual_or_blank", 0)
        lines.append(
            f"| {document['filename']} | {document['total_pages']} | "
            f"{status.get('extracted', 0)} | {status.get('ocr_required', 0)} | "
            f"{routes.get('ocr_completed', 0)} | {routes.get('ocr_pending', 0)} | "
            f"{visual} | {status.get('failed', 0) + routes.get('ocr_failed', 0)} |"
        )
    lines.extend(["", "## 待处理页", ""])
    for document in payload["documents"]:
        pending = [
            candidate["pdf_page"]
            for candidate in document["ocr_candidates"]
            if candidate["route"] == "ocr_pending"
        ]
        if pending:
            lines.append(f"- **{document['filename']}**：{compact_page_ranges(pending)}")
    lines.extend(
        [
            "",
            "> PDF 页码均为从 1 开始的物理页码。OCR 低文字页会保留审计记录，"
            "后续切分时默认不作为正文入库。",
            "",
        ]
    )
    return "\n".join(lines)


def build_quality_report(
    *,
    database: Database,
    pages_dir: Path,
    ocr_dir: Path,
    reports_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    files = current_files(database)
    documents: list[dict[str, Any]] = []
    for document_id, file_record in files.items():
        page_records = load_page_records(pages_dir / f"{document_id}.jsonl")
        ocr_path = ocr_dir / f"{document_id}.jsonl"
        ocr_records = load_page_records(ocr_path) if ocr_path.is_file() else {}
        documents.append(_document_report(file_record, page_records, ocr_records))

    totals = {
        "documents": len(documents),
        "pages": sum(document["total_pages"] for document in documents),
        "text_extracted": sum(
            document["status_counts"].get("extracted", 0) for document in documents
        ),
        "ocr_candidates": sum(
            document["status_counts"].get("ocr_required", 0) for document in documents
        ),
        "ocr_completed": sum(
            document["ocr_route_counts"].get("ocr_completed", 0) for document in documents
        ),
        "ocr_pending": sum(
            document["ocr_route_counts"].get("ocr_pending", 0) for document in documents
        ),
        "ocr_low_text": sum(
            document["ocr_route_counts"].get("ocr_low_text", 0) for document in documents
        ),
        "visual_or_blank": sum(
            document["ocr_route_counts"].get("visual_or_blank", 0) for document in documents
        ),
        "empty": sum(document["status_counts"].get("empty", 0) for document in documents),
        "failed": sum(
            document["status_counts"].get("failed", 0)
            + document["ocr_route_counts"].get("ocr_failed", 0)
            for document in documents
        ),
    }
    payload = {
        "run_id": run_id,
        "generated_at": utc_now(),
        "totals": totals,
        "documents": documents,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    rendered_json = json.dumps(payload, ensure_ascii=False, indent=2)
    rendered_markdown = _render_markdown(payload)
    (reports_dir / f"extraction_quality_{run_id}.json").write_text(
        rendered_json, encoding="utf-8"
    )
    (reports_dir / f"extraction_quality_{run_id}.md").write_text(
        rendered_markdown, encoding="utf-8"
    )
    (reports_dir / "extraction_quality_latest.json").write_text(
        rendered_json, encoding="utf-8"
    )
    (reports_dir / "extraction_quality_latest.md").write_text(
        rendered_markdown, encoding="utf-8"
    )
    return payload
