from __future__ import annotations

import json
from typing import Any

import httpx
from history_agent.answering.models import QueryEntity, QueryPlan, QuestionRequest
from history_agent.answering.query_understanding import QueryExecution, QueryPlanningResult
from history_agent.answering.retrieval_reflection import (
    EvidenceAssessment,
    ReflectionResult,
    assess_retrieval,
    should_reflect,
)
from history_agent.answering.service import _retrieve_context
from history_agent.config import Settings
from history_agent.errors import RetrievalError
from history_agent.retrieval.models import RetrievalPlan, SearchHit, SearchResponse


def _hit(chunk_id: str, text: str, *, rank: int = 1) -> SearchHit:
    return SearchHit(
        rank=rank,
        chunk_id=chunk_id,
        document_id="history",
        title="中国共产党历史",
        filename="history.pdf",
        source_type="official_history",
        verification_status="verified",
        pdf_page_start=900 + rank,
        pdf_page_end=900 + rank,
        section_path=["淮海战役"],
        text=text,
        year_mentions=[1948],
        people=[],
        extraction_methods=["text_layer"],
        score=1.0,
        matched_terms=["淮海战役"],
        keyword_rank=rank,
        vector_rank=rank,
    )


def _response(hits: list[SearchHit], *, query: str = "详细介绍淮海战役") -> SearchResponse:
    return SearchResponse(
        query=query,
        query_intent="event_overview",
        query_terms=["淮海战役"],
        query_years=[],
        query_year_range=[],
        query_people=[],
        document_filters=[],
        include_out_of_scope=False,
        hits=hits,
        retrieval_mode="planned_hybrid_rrf",
    )


def _plan() -> QueryPlan:
    return QueryPlan(
        intent="event_overview",
        normalized_question="详细介绍淮海战役",
        entities=[QueryEntity(type="event", text="淮海战役", canonical="淮海战役")],
    )


def test_reflection_only_runs_for_complex_questions() -> None:
    settings = Settings(_env_file=None, llm_api_key="sk-test")
    simple = QueryPlan(intent="general", normalized_question="淮海战役是什么")

    assert not should_reflect(
        settings, QuestionRequest(question="淮海战役是什么"), simple
    )
    assert should_reflect(
        settings, QuestionRequest(question="详细介绍淮海战役"), _plan()
    )


def test_reflection_builds_bounded_safe_followup_queries(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}
    assessment = {
        "sufficient": False,
        "covered_aspects": ["战役背景"],
        "missing_aspects": ["战役结果", "1949年战役意义", "更多证据"],
        "reason": "缺少结果和意义材料",
    }

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        captured.update(kwargs)
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "choices": [
                    {"message": {"content": json.dumps(assessment, ensure_ascii=False)}}
                ],
                "usage": {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100},
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = assess_retrieval(
        Settings(_env_file=None, llm_api_key="sk-test"),
        QuestionRequest(question="详细介绍淮海战役"),
        _plan(),
        _response([_hit("first", "淮海战役是在一定背景下发起的。")]),
    )

    assert result.status == "retry"
    assert result.followup_queries == (
        "淮海战役 战役结果",
        "淮海战役 战役意义",
    )
    assert result.usage == {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100}
    body = captured["json"]
    assert body["model"] == "deepseek-v4-flash"
    assert body["temperature"] == 0
    assert "淮海战役是在一定背景下发起的" in body["messages"][-1]["content"]


def test_reflection_accepts_sufficient_first_pass(monkeypatch: Any) -> None:
    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "sufficient": True,
                                    "covered_aspects": ["背景", "经过", "结果"],
                                    "missing_aspects": [],
                                    "reason": "主要方面已有直接材料",
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = assess_retrieval(
        Settings(_env_file=None, llm_api_key="sk-test"),
        QuestionRequest(question="详细介绍淮海战役"),
        _plan(),
        _response([_hit("first", "背景、经过和结果均有记载。")]),
    )

    assert result.status == "sufficient"
    assert result.followup_queries == ()


def test_retrieve_context_merges_one_retry_without_discarding_first_pass(
    monkeypatch: Any,
) -> None:
    plan = _plan()
    planning = QueryPlanningResult(plan, "used")
    execution = QueryExecution(
        "详细介绍淮海战役",
        (),
        RetrievalPlan(query_intent="event_overview"),
    )
    initial = _response([_hit("initial", "首轮背景材料。")])
    retry = _response(
        [
            _hit("initial", "重复材料。"),
            _hit("result", "补充的战役结果材料。", rank=2),
        ],
        query="淮海战役 战役结果",
    )
    responses = iter([initial, retry])
    calls: list[str] = []

    monkeypatch.setattr(
        "history_agent.answering.service.query_execution", lambda *args: execution
    )

    def search(**kwargs: Any) -> SearchResponse:
        calls.append(kwargs["query"])
        return next(responses)

    monkeypatch.setattr("history_agent.answering.service.search_hybrid_index", search)
    monkeypatch.setattr(
        "history_agent.answering.service.assess_retrieval",
        lambda *args: ReflectionResult(
            "retry",
            EvidenceAssessment(
                sufficient=False,
                covered_aspects=["背景"],
                missing_aspects=["战役结果"],
                reason="缺少结果",
            ),
            ("淮海战役 战役结果",),
            {"total_tokens": 50},
        ),
    )

    context = _retrieve_context(
        Settings(_env_file=None),
        QuestionRequest(question="详细介绍淮海战役"),
        planning,
    )

    assert calls == ["详细介绍淮海战役", "淮海战役 战役结果"]
    assert [hit.chunk_id for hit in context.retrieval.hits] == ["initial", "result"]
    assert context.retrieval.retrieval_mode == "planned_hybrid_rrf_refined"
    assert context.reflection_status == "retried"
    assert context.retrieval_rounds == 2
    assert context.missing_aspects == ("战役结果",)
    assert context.reflection_usage == {"total_tokens": 50}
    assert [citation.evidence_id for citation in context.citations] == ["E1", "E2"]


def test_failed_retry_keeps_first_pass_evidence(monkeypatch: Any) -> None:
    planning = QueryPlanningResult(_plan(), "used")
    execution = QueryExecution(
        "详细介绍淮海战役",
        (),
        RetrievalPlan(query_intent="event_overview"),
    )
    calls = 0

    monkeypatch.setattr(
        "history_agent.answering.service.query_execution", lambda *args: execution
    )

    def search(**kwargs: Any) -> SearchResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _response([_hit("initial", "首轮可用材料。")])
        raise RetrievalError("retry failed")

    monkeypatch.setattr("history_agent.answering.service.search_hybrid_index", search)
    monkeypatch.setattr(
        "history_agent.answering.service.assess_retrieval",
        lambda *args: ReflectionResult(
            "retry",
            EvidenceAssessment(
                sufficient=False,
                covered_aspects=["背景"],
                missing_aspects=["战役结果"],
                reason="缺少结果",
            ),
            ("淮海战役 战役结果",),
        ),
    )

    context = _retrieve_context(
        Settings(_env_file=None),
        QuestionRequest(question="详细介绍淮海战役"),
        planning,
    )

    assert calls == 2
    assert [hit.chunk_id for hit in context.retrieval.hits] == ["initial"]
    assert context.reflection_status == "fallback"
    assert context.retrieval_rounds == 1
    assert context.reflection_error_code == "retry_retrieval_failed"
