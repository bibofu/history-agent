from __future__ import annotations

import asyncio
import json
from pathlib import Path

from history_agent.answering.full_text import answer_full_text_question
from history_agent.answering.models import QuestionRequest
from history_agent.answering.service import answer_question
from history_agent.answering.streaming import AnswerStreamEvent, stream_answer_question
from history_agent.config import Settings
from history_agent.processing.models import ChunkRecord, StructureEntry


def _settings(root: Path) -> Settings:
    return Settings(
        _env_file=None,
        project_root=root,
        data_dir=root / "data",
        llm_api_key=None,
        llm_query_planning=False,
    )


def _chunk(index: int, page: int, text: str, *, section: str = "矛盾论") -> ChunkRecord:
    return ChunkRecord(
        chunk_id=f"chunk-{index}",
        document_id="mao-selected",
        file_sha256="hash",
        title="毛泽东选集",
        filename="毛泽东选集.pdf",
        creators=["毛泽东"],
        source_type="selected_works",
        edition="测试版",
        volume="第一卷",
        verification_status="edition_labeled",
        chunk_index=index,
        page_chunk_index=index % 2,
        pdf_page_start=page,
        pdf_page_end=page,
        section_path=["第一卷", section],
        text=text,
        search_text=text,
        character_count=len(text),
        scope_status="in_scope",
        extraction_methods=["text_layer"],
        content_hash=f"content-{index}",
        cleaner_version="test",
        chunker_version="test",
    )


def _write_section(root: Path) -> Settings:
    settings = _settings(root)
    settings.structure_dir.mkdir(parents=True)
    settings.chunks_dir.mkdir(parents=True)
    entry = StructureEntry(
        entry_id="entry",
        document_id="mao-selected",
        level=2,
        title="矛盾论",
        pdf_page_start=185,
        pdf_page_end=186,
        source="pdf_outline",
    )
    (settings.structure_dir / "mao-selected.json").write_text(
        json.dumps([entry.model_dump()], ensure_ascii=False), encoding="utf-8"
    )
    records = [
        _chunk(0, 185, "矛盾论（一九三七年八月）"),
        _chunk(1, 185, "第一页正文。"),
        _chunk(2, 186, "第二页正文。"),
        _chunk(3, 187, "下一篇正文。", section="反对自由主义"),
    ]
    (settings.chunks_dir / "mao-selected.jsonl").write_text(
        "\n".join(record.model_dump_json() for record in records) + "\n",
        encoding="utf-8",
    )
    return settings


def test_explicit_quoted_title_returns_complete_local_section(work_path: Path) -> None:
    settings = _write_section(work_path)

    result = answer_full_text_question(
        settings, QuestionRequest(question="毛泽东《矛盾论》的全文")
    )

    assert result is not None
    assert result.retrieval_mode == "full_text_section"
    assert result.llm_status == "not_applicable"
    assert "矛盾论（一九三七年八月）第一页正文。\n\n第二页正文。" in result.answer
    assert "下一篇正文" not in result.answer
    assert result.citations[0].document == "毛泽东选集"
    assert result.citations[0].pdf_page == 185
    assert result.citations[0].pdf_page_end == 186
    assert result.retrieved_evidence_count == 3


def test_full_text_route_bypasses_rag_and_llm(work_path: Path) -> None:
    settings = _write_section(work_path)

    result = answer_question(settings, QuestionRequest(question="请输出《矛盾论》完整原文"))

    assert result.evidence_status == "supported"
    assert result.generator_mode == "extractive"
    assert result.answer.endswith("[E1]")


def test_streaming_full_text_route_returns_complete_answer(work_path: Path) -> None:
    settings = _write_section(work_path)

    async def collect() -> list[AnswerStreamEvent]:
        return [
            event
            async for event in stream_answer_question(
                settings, QuestionRequest(question="请输出《矛盾论》全文")
            )
        ]

    events = asyncio.run(collect())

    assert [event.event for event in events] == ["status", "done"]
    assert events[-1].data["retrieval_mode"] == "full_text_section"
    assert "第一页正文" in events[-1].data["answer"]


def test_ordinary_title_question_does_not_trigger_full_text_route(work_path: Path) -> None:
    result = answer_full_text_question(
        _settings(work_path), QuestionRequest(question="《矛盾论》如何分析主要矛盾？")
    )

    assert result is None
