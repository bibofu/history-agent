from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from importlib.metadata import version
from typing import Any

from history_agent.extraction.models import ExtractionMethod, PageRecord, PageStatus

CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def normalize_for_storage(text: str) -> str:
    """Apply only lossless-enough whitespace normalization at the page stage."""

    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")


def count_page_images(page: Any) -> int:
    """Count top-level PDF image XObjects without decoding image bytes."""

    try:
        resources = page.get("/Resources")
        if resources is None:
            return 0
        resources = resources.get_object()
        xobjects = resources.get("/XObject")
        if xobjects is None:
            return 0
        xobjects = xobjects.get_object()
        count = 0
        for reference in xobjects.values():
            obj = reference.get_object()
            if obj.get("/Subtype") == "/Image":
                count += 1
        return count
    except Exception:
        return 0


def classify_extracted_text(
    text: str,
    *,
    image_object_count: int = 0,
) -> tuple[PageStatus, list[str]]:
    compact = "".join(text.split())
    flags: list[str] = []
    if not compact:
        flags.append("no_text_layer")
        if image_object_count > 0:
            flags.append("image_without_text")
            return "ocr_required", flags
        flags.append("blank_or_vector_page")
        return "empty", flags
    if len(compact) < 50:
        flags.append("very_short_text")
    replacement_count = compact.count("\ufffd")
    if replacement_count / max(len(compact), 1) > 0.01:
        flags.append("replacement_characters")
    cjk_count = len(CJK_PATTERN.findall(compact))
    if cjk_count == 0:
        flags.append("no_cjk_characters")
    if "replacement_characters" in flags:
        return "ocr_required", flags
    return "extracted", flags


def build_page_record(
    *,
    document_id: str,
    file_sha256: str,
    pdf_page: int,
    text: str,
    image_object_count: int,
    run_id: str,
    extraction_method: ExtractionMethod = "pypdf_text_layer",
    extractor_version: str | None = None,
) -> PageRecord:
    normalized = normalize_for_storage(text)
    status, flags = classify_extracted_text(
        normalized,
        image_object_count=image_object_count,
    )
    compact = "".join(normalized.split())
    return PageRecord(
        document_id=document_id,
        file_sha256=file_sha256,
        pdf_page=pdf_page,
        status=status,
        extraction_method=extraction_method if compact else "none",
        extractor_version=extractor_version or f"pypdf {version('pypdf')}",
        raw_text=text,
        normalized_text=normalized,
        text_sha256=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        character_count=len(compact),
        cjk_character_count=len(CJK_PATTERN.findall(compact)),
        replacement_character_count=compact.count("\ufffd"),
        image_object_count=image_object_count,
        quality_flags=flags,
        run_id=run_id,
        processed_at=utc_now(),
    )


def build_failed_page_record(
    *,
    document_id: str,
    file_sha256: str,
    pdf_page: int,
    run_id: str,
    exc: BaseException,
    extractor_version: str | None = None,
) -> PageRecord:
    return PageRecord(
        document_id=document_id,
        file_sha256=file_sha256,
        pdf_page=pdf_page,
        status="failed",
        extraction_method="none",
        extractor_version=extractor_version or f"pypdf {version('pypdf')}",
        raw_text="",
        normalized_text="",
        text_sha256=hashlib.sha256(b"").hexdigest(),
        character_count=0,
        cjk_character_count=0,
        replacement_character_count=0,
        image_object_count=0,
        quality_flags=["extraction_exception"],
        error=f"{type(exc).__name__}: {exc}",
        run_id=run_id,
        processed_at=utc_now(),
    )
