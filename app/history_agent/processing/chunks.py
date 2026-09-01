from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from history_agent.db import Database
from history_agent.errors import ExtractionError
from history_agent.extraction.models import PageRecord
from history_agent.extraction.report import load_page_records
from history_agent.processing.cleaning import (
    CLEANER_VERSION,
    clean_page_text,
    detect_repeated_marginal_lines,
    is_table_of_contents_page,
)
from history_agent.processing.models import (
    ChunkBuildSummary,
    ChunkRecord,
    DocumentChunkResult,
    StructureEntry,
)
from history_agent.processing.structure import section_path_for_page, structure_for_document
from history_agent.research.catalog import load_person_catalog

CHUNKER_VERSION = "page-sentence-chunker-v1"
ARABIC_DATE = re.compile(
    r"(?P<year>(?:18|19|20)\d{2})\s*年"
    r"(?:\s*(?P<month>\d{1,2})\s*月(?:\s*(?P<day>\d{1,2})\s*日)?)?"
)
CHINESE_YEAR = re.compile(r"([〇○零一二三四五六七八九]{4})年")
SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？；])")
CHINESE_DIGITS = {
    "〇": "0",
    "○": "0",
    "零": "0",
    "一": "1",
    "二": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def current_documents(database: Database) -> dict[str, dict[str, Any]]:
    with database.connect() as connection:
        rows = connection.execute(
            """
            SELECT d.document_id, d.title, d.creators_json, d.source_type, d.edition,
                   d.volume, d.verification_status, f.filename, f.relative_path,
                   f.sha256, f.page_count
            FROM documents d
            JOIN document_files f ON f.document_id = d.document_id
            WHERE f.is_current = 1 AND f.is_present = 1 AND d.enabled = 1
            ORDER BY d.document_id
            """
        ).fetchall()
    return {row["document_id"]: dict(row) for row in rows}


def load_person_aliases(path: Path) -> dict[str, list[str]]:
    try:
        return load_person_catalog(path).alias_map()
    except Exception as exc:
        raise ExtractionError(f"Invalid person alias catalog {path}: {exc}") from exc


def effective_pages(
    base_records: dict[int, PageRecord], ocr_records: dict[int, PageRecord]
) -> dict[int, PageRecord]:
    effective: dict[int, PageRecord] = {}
    for pdf_page, base_record in base_records.items():
        if base_record.status == "extracted" and base_record.normalized_text.strip():
            effective[pdf_page] = base_record
            continue
        ocr_record = ocr_records.get(pdf_page)
        if (
            base_record.status == "ocr_required"
            and ocr_record is not None
            and ocr_record.status == "extracted"
            and ocr_record.normalized_text.strip()
        ):
            effective[pdf_page] = ocr_record
    return effective


def split_text(text: str, *, target_chars: int = 650, max_chars: int = 900) -> list[str]:
    units: list[str] = []
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        pieces = [piece for piece in SENTENCE_BOUNDARY.split(paragraph) if piece]
        units.extend(pieces or [paragraph])
    chunks: list[str] = []
    buffer = ""
    for unit in units:
        if len(unit) > max_chars:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            for start in range(0, len(unit), max_chars):
                chunks.append(unit[start : start + max_chars])
            continue
        if buffer and len(buffer) + len(unit) > max_chars:
            chunks.append(buffer)
            buffer = unit
        else:
            buffer += unit
        if len(buffer) >= target_chars:
            chunks.append(buffer)
            buffer = ""
    if buffer:
        if chunks and len(buffer) < 80 and len(chunks[-1]) + len(buffer) <= max_chars:
            chunks[-1] += buffer
        else:
            chunks.append(buffer)
    return [chunk for chunk in chunks if len(chunk.strip()) >= 20]


def extract_dates(text: str) -> tuple[list[int], list[str]]:
    years: set[int] = set()
    dates: set[str] = set()
    for match in ARABIC_DATE.finditer(text):
        year = int(match.group("year"))
        years.add(year)
        month = match.group("month")
        day = match.group("day")
        rendered = f"{year:04d}"
        if month:
            rendered += f"-{int(month):02d}"
        if day:
            rendered += f"-{int(day):02d}"
        dates.add(rendered)
    for match in CHINESE_YEAR.finditer(text):
        year = int("".join(CHINESE_DIGITS[character] for character in match.group(1)))
        years.add(year)
        dates.add(f"{year:04d}")
    return sorted(years), sorted(dates)


def scope_status(
    years: list[int], research_start: int, research_end: int
) -> Literal["in_scope", "out_of_scope", "mixed", "unknown"]:
    if not years:
        return "unknown"
    inside = [research_start <= year <= research_end for year in years]
    if all(inside):
        return "in_scope"
    if not any(inside):
        return "out_of_scope"
    return "mixed"


def extract_people(text: str, aliases: dict[str, list[str]]) -> list[str]:
    return sorted(
        person
        for person, names in aliases.items()
        if person in text or any(alias in text for alias in names)
    )


def _chunk_id(
    document_id: str, file_sha256: str, pdf_page: int, page_index: int, text: str
) -> tuple[str, str]:
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    identity = (
        f"{document_id}:{file_sha256}:{pdf_page}:{page_index}:"
        f"{CLEANER_VERSION}:{CHUNKER_VERSION}:{content_hash}"
    )
    return hashlib.sha256(identity.encode()).hexdigest()[:32], content_hash


def _write_jsonl(path: Path, records: list[ChunkRecord]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(record.model_dump_json() + "\n")
    os.replace(temporary_path, path)


def _write_structure(path: Path, entries: list[StructureEntry]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    payload = [entry.model_dump() for entry in entries]
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary_path, path)


def build_document_chunks(
    *,
    document: dict[str, Any],
    project_root: Path,
    pages_dir: Path,
    ocr_dir: Path,
    chunks_dir: Path,
    structure_dir: Path,
    aliases: dict[str, list[str]],
    research_start: int,
    research_end: int,
) -> DocumentChunkResult:
    document_id = str(document["document_id"])
    base_records = load_page_records(pages_dir / f"{document_id}.jsonl")
    ocr_path = ocr_dir / f"{document_id}.jsonl"
    ocr_records = load_page_records(ocr_path) if ocr_path.is_file() else {}
    extracted_pages = effective_pages(base_records, ocr_records)
    pages = {
        pdf_page: record
        for pdf_page, record in extracted_pages.items()
        if not is_table_of_contents_page(record.normalized_text)
    }
    entries = structure_for_document(
        document_id=document_id,
        title=str(document["title"]),
        source_path=project_root / str(document["relative_path"]),
        pages=pages,
        page_count=int(document["page_count"]),
    )
    repeated_lines = detect_repeated_marginal_lines(pages)
    records: list[ChunkRecord] = []
    creators = json.loads(str(document["creators_json"]))
    for pdf_page, page_record in sorted(pages.items()):
        cleaned = clean_page_text(page_record.normalized_text, repeated_lines)
        for page_chunk_index, text in enumerate(split_text(cleaned)):
            section_path = section_path_for_page(entries, pdf_page)
            years, dates = extract_dates(text)
            section_years, section_dates = extract_dates(" ".join(section_path))
            years = sorted(set(years) | set(section_years))
            dates = sorted(set(dates) | set(section_dates))
            people = extract_people(text, aliases)
            chunk_id, content_hash = _chunk_id(
                document_id,
                str(document["sha256"]),
                pdf_page,
                page_chunk_index,
                text,
            )
            search_metadata = " ".join(
                [
                    str(document["title"]),
                    *section_path,
                    *people,
                    *(f"{year}年" for year in years),
                ]
            )
            records.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    file_sha256=str(document["sha256"]),
                    title=str(document["title"]),
                    filename=str(document["filename"]),
                    creators=list(creators),
                    source_type=str(document["source_type"]),
                    edition=document["edition"],
                    volume=document["volume"],
                    verification_status=str(document["verification_status"]),
                    chunk_index=len(records),
                    page_chunk_index=page_chunk_index,
                    pdf_page_start=pdf_page,
                    pdf_page_end=pdf_page,
                    printed_page_start=page_record.printed_page,
                    printed_page_end=page_record.printed_page,
                    section_path=section_path,
                    text=text,
                    search_text=f"{text} {search_metadata}".strip(),
                    character_count=len(text),
                    year_mentions=years,
                    date_mentions=dates,
                    scope_status=scope_status(years, research_start, research_end),
                    people=people,
                    extraction_methods=[page_record.extraction_method],
                    content_hash=content_hash,
                    cleaner_version=CLEANER_VERSION,
                    chunker_version=CHUNKER_VERSION,
                )
            )
    for index, record in enumerate(records):
        record.previous_chunk_id = records[index - 1].chunk_id if index else None
        record.next_chunk_id = records[index + 1].chunk_id if index + 1 < len(records) else None

    chunks_dir.mkdir(parents=True, exist_ok=True)
    structure_dir.mkdir(parents=True, exist_ok=True)
    output_path = chunks_dir / f"{document_id}.jsonl"
    _write_jsonl(output_path, records)
    _write_structure(structure_dir / f"{document_id}.json", entries)
    counts: dict[str, int] = {}
    for record in records:
        counts[record.scope_status] = counts.get(record.scope_status, 0) + 1
    return DocumentChunkResult(
        document_id=document_id,
        filename=str(document["filename"]),
        effective_pages=len(pages),
        skipped_pages=int(document["page_count"]) - len(pages),
        structure_entries=len(entries),
        chunks=len(records),
        characters=sum(record.character_count for record in records),
        scope_counts=counts,
        output_path=str(output_path),
    )


def build_all_chunks(
    *,
    database: Database,
    project_root: Path,
    pages_dir: Path,
    ocr_dir: Path,
    chunks_dir: Path,
    structure_dir: Path,
    reports_dir: Path,
    aliases_path: Path,
    run_id: str,
    research_start: int,
    research_end: int,
    document_ids: list[str] | None = None,
) -> ChunkBuildSummary:
    documents = current_documents(database)
    selected_ids = document_ids or list(documents)
    unknown_ids = sorted(set(selected_ids) - set(documents))
    if unknown_ids:
        raise ExtractionError(f"Unknown or unavailable document IDs: {', '.join(unknown_ids)}")
    aliases = load_person_aliases(aliases_path)
    started_at = utc_now()
    results = [
        build_document_chunks(
            document=documents[document_id],
            project_root=project_root,
            pages_dir=pages_dir,
            ocr_dir=ocr_dir,
            chunks_dir=chunks_dir,
            structure_dir=structure_dir,
            aliases=aliases,
            research_start=research_start,
            research_end=research_end,
        )
        for document_id in selected_ids
    ]
    summary = ChunkBuildSummary(
        run_id=run_id,
        started_at=started_at,
        finished_at=utc_now(),
        documents=results,
    )
    payload = summary.model_dump()
    payload["totals"] = summary.totals
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"chunk_build_{run_id}.json").write_text(rendered, encoding="utf-8")
    (reports_dir / "chunk_build_latest.json").write_text(rendered, encoding="utf-8")
    return summary
