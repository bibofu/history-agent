from __future__ import annotations

import json
from typing import Any

import httpx
from history_agent.answering.timeline_evidence import filter_person_timeline_evidence
from history_agent.config import Settings
from history_agent.retrieval.models import SearchHit, SearchResponse


def _hit(rank: int, title: str, text: str) -> SearchHit:
    return SearchHit(
        rank=rank,
        chunk_id=f"chunk-{rank}",
        document_id=f"doc-{rank}",
        title=title,
        filename=f"doc-{rank}.pdf",
        source_type="chronology",
        verification_status="verified",
        pdf_page_start=100 + rank,
        pdf_page_end=100 + rank,
        section_path=[],
        text=text,
        year_mentions=[1935],
        people=["彭德怀"],
        extraction_methods=["text_layer"],
        score=1.0,
        matched_terms=["彭德怀"],
        keyword_rank=rank,
        vector_rank=rank,
    )


def _response() -> SearchResponse:
    return SearchResponse(
        query="彭德怀在1935年的主要经历",
        query_intent="timeline",
        query_terms=["彭德怀", "1935"],
        query_years=[1935],
        query_year_range=[1935, 1935],
        query_people=["彭德怀"],
        document_filters=[],
        include_out_of_scope=False,
        hits=[
            _hit(1, "毛泽东年谱", "毛泽东同彭德怀致电各纵队，布置下一步行动。"),
            _hit(2, "毛泽东年谱", "周恩来致电彭德怀，同意攻打甘泉。"),
            _hit(3, "周恩来年谱（1949—1976）", "1935年彭德怀率部到达陕北。"),
        ],
        retrieval_mode="planned_hybrid_rrf",
    )


def test_llm_filter_keeps_only_explicit_subject_evidence(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        captured.update(kwargs)
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "choices": [
                    {"message": {"content": json.dumps({"selected_ids": [1]})}}
                ],
                "usage": {"total_tokens": 80},
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = filter_person_timeline_evidence(
        Settings(_env_file=None, llm_api_key="sk-test"),
        "彭德怀在1935年的主要经历",
        _response(),
    )

    assert result.status == "applied"
    assert result.removed_count == 2
    assert [hit.chunk_id for hit in result.retrieval.hits] == ["chunk-1"]
    assert result.retrieval.hits[0].rank == 1
    assert result.retrieval.retrieval_mode.endswith("_subject_filtered")
    assert result.usage == {"total_tokens": 80}
    prompt = captured["json"]["messages"][0]["content"]
    assert "他人只是向目标人物发电" in prompt
    assert "文献标题明确标示的年代范围之外" in prompt
    candidates = captured["json"]["messages"][1]["content"]
    assert "周恩来年谱（1949—1976）" not in candidates


def test_invalid_filter_response_keeps_original_candidates(monkeypatch: Any) -> None:
    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": "not-json"}}]},
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    retrieval = _response()
    result = filter_person_timeline_evidence(
        Settings(_env_file=None, llm_api_key="sk-test"),
        "彭德怀在1935年的主要经历",
        retrieval,
    )

    assert result.status == "fallback"
    assert result.error_code == "invalid_selection"
    assert result.retrieval is retrieval
