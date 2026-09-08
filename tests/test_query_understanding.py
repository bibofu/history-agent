from __future__ import annotations

import json
from typing import Any

import httpx
from history_agent.answering.models import QueryEntity, QueryPlan, QuestionRequest
from history_agent.answering.query_understanding import (
    QueryPlanningResult,
    plan_question,
    retrieval_query,
)
from history_agent.answering.service import LLMResult, answer_question
from history_agent.config import Settings
from history_agent.retrieval.models import SearchResponse


def _plan_payload() -> dict[str, Any]:
    return {
        "intent": "event_overview",
        "normalized_question": "介绍中国共产党第一次全国代表大会的情况",
        "search_queries": [
            "中国共产党第一次全国代表大会 召开 地点 代表 决议",
            "中国共产党第一次全国代表大会 会议成果",
        ],
        "entities": [
            {
                "type": "event",
                "text": "党的头一次全国大会",
                "canonical": "中国共产党第一次全国代表大会",
            }
        ],
        "start_year": None,
        "end_year": None,
        "coverage": "relevance",
        "constraints": [],
        "needs_clarification": False,
        "clarification_question": None,
    }


def test_query_planner_normalizes_free_form_question(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        captured["url"] = url
        captured.update(kwargs)
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(_plan_payload(), ensure_ascii=False)
                        }
                    }
                ],
                "usage": {"prompt_tokens": 80, "completion_tokens": 40, "total_tokens": 120},
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    settings = Settings(_env_file=None, llm_api_key="sk-test")
    result = plan_question(
        settings, QuestionRequest(question="党的头一次全国大会讲了些什么？")
    )

    assert result.status == "used"
    assert result.plan is not None
    assert result.plan.intent == "event_overview"
    assert result.plan.entities[0].canonical == "中国共产党第一次全国代表大会"
    assert result.usage == {"prompt_tokens": 80, "completion_tokens": 40, "total_tokens": 120}
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    body = captured["json"]
    assert body["model"] == "deepseek-v4-flash"
    assert body["response_format"] == {"type": "json_object"}
    assert body["thinking"] == {"type": "disabled"}
    assert "不回答历史问题" in body["messages"][0]["content"]
    assert captured["timeout"] == 20


def test_invalid_query_plan_falls_back_without_using_model_text(monkeypatch: Any) -> None:
    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": "not-json"}}]},
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = plan_question(
        Settings(_env_file=None, llm_api_key="sk-test"),
        QuestionRequest(question="介绍党的头一次全国大会"),
    )

    assert result.status == "fallback"
    assert result.plan is None
    assert result.error_code == "invalid_plan"


def test_retrieval_query_keeps_original_constraints_and_adds_semantics() -> None:
    plan = QueryPlan(
        intent="timeline",
        normalized_question="梳理毛泽东1921年至1926年的活动",
        search_queries=["毛泽东 1921年至1926年 经历"],
        entities=[QueryEntity(type="person", text="润之", canonical="毛泽东")],
        start_year=1921,
        end_year=1926,
        coverage="per_year",
        constraints=["限定湖南地区"],
    )
    original = "逐年梳理润之1921—1926年在湖南的活动"
    query = retrieval_query(original, plan)

    assert query.startswith(original)
    assert "毛泽东" in query
    assert "经历 活动 时间线" in query
    assert "1921—1926" in query
    assert "湖南" in query


def test_query_planner_can_request_clarification(monkeypatch: Any) -> None:
    payload = _plan_payload()
    payload.update(
        {
            "normalized_question": "他后来采取的政策",
            "entities": [],
            "needs_clarification": True,
            "clarification_question": "请问“他”指的是哪位人物？",
        }
    )

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": json.dumps(payload, ensure_ascii=False)}}]},
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = plan_question(
        Settings(_env_file=None, llm_api_key="sk-test"),
        QuestionRequest(question="他在1949年做了什么？"),
    )

    assert result.plan is not None
    assert result.plan.needs_clarification is True
    assert result.plan.clarification_question == "请问“他”指的是哪位人物？"


def test_answer_question_uses_semantic_plan_for_hybrid_retrieval(monkeypatch: Any) -> None:
    plan = QueryPlan.model_validate(_plan_payload())
    planning = QueryPlanningResult(
        plan,
        "used",
        model_name="deepseek-v4-flash",
        usage={"total_tokens": 100},
    )
    captured: dict[str, str] = {}

    def search(**kwargs: Any) -> SearchResponse:
        captured["query"] = kwargs["query"]
        return SearchResponse(
            query=kwargs["query"],
            query_intent="general",
            query_terms=["中国共产党第一次全国代表大会"],
            query_years=[],
            query_year_range=[],
            query_people=[],
            document_filters=[],
            include_out_of_scope=False,
            hits=[],
            retrieval_mode="hybrid_rrf",
        )

    monkeypatch.setattr("history_agent.answering.service.plan_question", lambda *a: planning)
    monkeypatch.setattr("history_agent.answering.service.search_hybrid_index", search)
    monkeypatch.setattr(
        "history_agent.answering.service._llm_answer",
        lambda **kwargs: LLMResult(None, "no_evidence"),
    )
    original = "党的头一次全国大会讲了些什么？"
    result = answer_question(
        Settings(_env_file=None, llm_api_key=None), QuestionRequest(question=original)
    )

    assert captured["query"].startswith(original)
    assert "中国共产党第一次全国代表大会" in captured["query"]
    assert result.query_plan == plan
    assert result.query_planner_status == "used"
    assert result.query_planner_usage == {"total_tokens": 100}


def test_answer_question_returns_planner_clarification_without_retrieval(
    monkeypatch: Any,
) -> None:
    payload = _plan_payload()
    payload.update(
        {
            "normalized_question": "他在1949年的活动",
            "entities": [],
            "needs_clarification": True,
            "clarification_question": "请问“他”指的是哪位人物？",
        }
    )
    planning = QueryPlanningResult(QueryPlan.model_validate(payload), "used")
    monkeypatch.setattr("history_agent.answering.service.plan_question", lambda *a: planning)

    def unexpected(**kwargs: Any) -> None:
        raise AssertionError("ambiguous query must not start retrieval")

    monkeypatch.setattr("history_agent.answering.service.search_hybrid_index", unexpected)
    result = answer_question(
        Settings(_env_file=None, llm_api_key=None),
        QuestionRequest(question="他后来采取了什么政策？"),
    )

    assert result.retrieval_mode == "query_clarification"
    assert result.answer == "请问“他”指的是哪位人物？"
    assert result.citations == []
