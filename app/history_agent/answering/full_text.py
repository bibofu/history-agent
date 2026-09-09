"""Deterministic delivery of a complete locally indexed document section."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from pydantic import ValidationError

from history_agent.answering.models import AnswerResponse, Citation, QuestionRequest
from history_agent.config import Settings
from history_agent.processing.models import ChunkRecord, StructureEntry

QUOTED_TITLE = re.compile(r"《(?P<title>[^》\n]{1,80})》")
FULL_TEXT_MARKER = re.compile(r"全文|完整(?:原文|内容|文本)|原文全文")


def _normalized_title(value: str) -> str:
    return re.sub(r"[\s*＊]", "", value).strip("《》")


def _section_candidates(structure_dir: Path, title: str) -> list[StructureEntry]:
    expected = _normalized_title(title)
    candidates: list[StructureEntry] = []
    if not structure_dir.is_dir():
        return candidates
    for path in sorted(structure_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            entries = [StructureEntry.model_validate(item) for item in payload]
        except (OSError, json.JSONDecodeError, TypeError, ValidationError):
            continue
        candidates.extend(
            entry for entry in entries if _normalized_title(entry.title) == expected
        )
    return candidates


def _load_section_chunks(
    chunks_dir: Path, entry: StructureEntry, title: str
) -> list[ChunkRecord]:
    path = chunks_dir / f"{entry.document_id}.jsonl"
    if not path.is_file():
        return []
    expected = _normalized_title(title)
    records: list[ChunkRecord] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = ChunkRecord.model_validate_json(line)
                if (
                    entry.pdf_page_start <= record.pdf_page_start <= entry.pdf_page_end
                    and any(_normalized_title(part) == expected for part in record.section_path)
                ):
                    records.append(record)
    except (OSError, ValidationError):
        return []
    return sorted(records, key=lambda item: item.chunk_index)


def _section_text(records: list[ChunkRecord]) -> str:
    pages: dict[int, list[str]] = defaultdict(list)
    for record in records:
        pages[record.pdf_page_start].append(record.text)
    return "\n\n".join("".join(parts).strip() for _, parts in sorted(pages.items())).strip()


def _failure(request: QuestionRequest, title: str, message: str) -> AnswerResponse:
    return AnswerResponse(
        question=request.question,
        answer=message,
        evidence_status="no_evidence",
        generator_mode="extractive",
        llm_status="not_applicable",
        retrieval_mode="full_text_section",
        query_intent="document_full_text",
        citations=[],
        limitations=[f"本次按本地篇目结构查找《{title}》，没有改用零散 RAG 片段拼凑全文。"],
    )


def answer_full_text_question(
    settings: Settings, request: QuestionRequest
) -> AnswerResponse | None:
    """Return a complete section only for an explicit 《title》 full-text request."""

    title_match = QUOTED_TITLE.search(request.question)
    if title_match is None or FULL_TEXT_MARKER.search(request.question) is None:
        return None
    title = title_match.group("title").strip()
    candidates = _section_candidates(settings.structure_dir, title)
    loaded = [
        (entry, records)
        for entry in candidates
        if (records := _load_section_chunks(settings.chunks_dir, entry, title))
    ]
    if not loaded:
        return _failure(
            request,
            title,
            f"本地篇目索引中没有找到《{title}》的完整章节，请检查结构和 chunks 是否已构建。",
        )
    # Prefer an authored/selected-works source when the question names its creator.
    loaded.sort(
        key=lambda item: (
            any(creator in request.question for creator in item[1][0].creators),
            item[1][0].source_type == "selected_works",
            len(item[1]),
        ),
        reverse=True,
    )
    best_score = (
        any(creator in request.question for creator in loaded[0][1][0].creators),
        loaded[0][1][0].source_type == "selected_works",
        len(loaded[0][1]),
    )
    equally_ranked = [
        item
        for item in loaded
        if (
            any(creator in request.question for creator in item[1][0].creators),
            item[1][0].source_type == "selected_works",
            len(item[1]),
        )
        == best_score
    ]
    if len(equally_ranked) > 1:
        sources = "、".join(sorted({item[1][0].title for item in equally_ranked}))
        return _failure(
            request,
            title,
            f"本地有多个同名篇目《{title}》（{sources}），请补充作者或文献名。",
        )
    entry, records = loaded[0]
    body = _section_text(records)
    first = records[0]
    citation = Citation(
        evidence_id="E1",
        document_id=first.document_id,
        document=first.title,
        volume=first.volume,
        pdf_page=entry.pdf_page_start,
        pdf_page_end=entry.pdf_page_end,
        section=first.section_path,
        quote=body[:420] + ("……" if len(body) > 420 else ""),
        source_type=first.source_type,
        verification_status=first.verification_status,
        extraction_methods=list(
            dict.fromkeys(method for record in records for method in record.extraction_methods)
        ),
    )
    return AnswerResponse(
        question=request.question,
        answer=f"# 《{title}》全文\n\n{body}\n\n[E1]",
        evidence_status="supported",
        generator_mode="extractive",
        llm_status="not_applicable",
        retrieval_mode="full_text_section",
        query_intent="document_full_text",
        citations=[citation],
        retrieved_evidence_count=len(records),
        limitations=[
            "正文按本地预处理 chunks 和 PDF 页序直接拼接，未调用生成模型改写；"
            "分页、段落和脚注格式可能与纸质版不同。"
        ],
    )
