from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
from history_agent.answering.models import (
    ConversationMessage,
    QueryEntity,
    QueryPlan,
    QuestionRequest,
)
from history_agent.answering.query_understanding import (
    QueryPlanningResult,
    plan_question,
    query_execution,
)
from history_agent.answering.service import LLMResult, answer_question
from history_agent.answering.time_ranges import parse_relative_year_range
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
                    {"message": {"content": json.dumps(_plan_payload(), ensure_ascii=False)}}
                ],
                "usage": {"prompt_tokens": 80, "completion_tokens": 40, "total_tokens": 120},
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    settings = Settings(_env_file=None, llm_api_key="sk-test")
    result = plan_question(settings, QuestionRequest(question="党的头一次全国大会讲了些什么？"))

    assert result.status == "used"
    assert result.plan is not None
    assert result.plan.intent == "event_overview"
    assert result.plan.entities[0].canonical == "中国共产党第一次全国代表大会"
    assert result.usage == {"prompt_tokens": 80, "completion_tokens": 40, "total_tokens": 120}
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    body = captured["json"]
    assert body["model"] == "deepseek-v4-flash"
    assert body["response_format"] == {"type": "json_object"}
    assert body["temperature"] == 0
    assert body["thinking"] == {"type": "disabled"}
    assert "不回答历史问题" in body["messages"][0]["content"]
    assert captured["timeout"] == 20


def test_standalone_event_isolated_from_history_and_unsafe_plan_fields(
    monkeypatch: Any,
) -> None:
    captured: dict[str, Any] = {}
    payload = {
        "intent": "event_overview",
        "normalized_question": "介绍林彪1930年至1949年的淮海战役经历",
        "search_queries": ["林彪 淮海战役 1930年至1949年"],
        "entities": [
            {"type": "event", "text": "淮海战役", "canonical": "淮海战役"},
            {"type": "person", "text": "林彪", "canonical": "林彪"},
        ],
        "start_year": 1930,
        "end_year": 1949,
        "coverage": "balanced_period",
        "constraints": ["限定1930年至1949年"],
        "needs_clarification": False,
        "clarification_question": None,
    }

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        captured.update(kwargs)
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": json.dumps(payload, ensure_ascii=False)}}]},
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    request = QuestionRequest(
        question="详细介绍淮海战役",
        history=[
            ConversationMessage(role="user", content="林彪在1930-1949年相继担任的职务"),
            ConversationMessage(role="assistant", content="林彪的有关职务……"),
        ],
    )
    result = plan_question(Settings(_env_file=None, llm_api_key="sk-test"), request)

    user_prompt = captured["json"]["messages"][-1]["content"]
    assert "conversation_history" not in user_prompt
    assert "林彪在1930-1949年" not in user_prompt
    assert result.plan is not None
    assert result.plan.normalized_question == "详细介绍淮海战役"
    assert result.plan.start_year is None
    assert result.plan.end_year is None
    assert result.plan.coverage == "relevance"
    assert [entity.canonical for entity in result.plan.entities] == ["淮海战役"]
    assert result.plan.search_queries == []
    assert result.plan.constraints == []


def test_referential_question_keeps_history_for_resolution(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}
    payload = _plan_payload()
    payload.update(
        {
            "normalized_question": "淮海战役的历史意义",
            "search_queries": ["淮海战役 历史意义"],
            "entities": [
                {"type": "event", "text": "淮海战役", "canonical": "淮海战役"},
                {"type": "person", "text": "林彪", "canonical": "林彪"},
            ],
            "start_year": 1930,
            "end_year": 1949,
            "coverage": "balanced_period",
        }
    )

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        captured.update(kwargs)
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": json.dumps(payload, ensure_ascii=False)}}]},
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    request = QuestionRequest(
        question="这场战役当时的意义呢？",
        history=[
            ConversationMessage(role="user", content="林彪在1930-1949年的职务"),
            ConversationMessage(role="assistant", content="林彪的有关职务……"),
            ConversationMessage(role="user", content="详细介绍淮海战役"),
        ],
    )
    result = plan_question(Settings(_env_file=None, llm_api_key="sk-test"), request)

    user_prompt = captured["json"]["messages"][-1]["content"]
    assert "<conversation_history>" in user_prompt
    assert "详细介绍淮海战役" in user_prompt
    assert result.plan is not None
    assert [entity.canonical for entity in result.plan.entities] == ["淮海战役"]
    assert result.plan.start_year is None
    assert result.plan.end_year is None
    assert result.plan.coverage == "relevance"


def test_current_question_year_overrides_planner_years(monkeypatch: Any) -> None:
    payload = _plan_payload()
    payload.update(
        {
            "normalized_question": "介绍淮海战役1930年至1949年的经过",
            "search_queries": ["淮海战役 1930年至1949年"],
            "entities": [
                {"type": "event", "text": "淮海战役", "canonical": "淮海战役"}
            ],
            "start_year": 1930,
            "end_year": 1949,
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
        QuestionRequest(question="介绍淮海战役1948年的经过"),
    )

    assert result.plan is not None
    assert (result.plan.start_year, result.plan.end_year) == (1948, 1948)
    assert result.plan.normalized_question == "介绍淮海战役1948年的经过"
    assert result.plan.search_queries == []


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


def test_query_execution_keeps_variants_independent_and_compiles_typed_filters() -> None:
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
    execution = query_execution(original, plan, Path("config/person_aliases.json"))

    assert execution.primary_query == original
    assert any("毛泽东" in query for query in execution.additional_queries)
    assert all("经历 活动 时间线" in query for query in execution.additional_queries)
    assert any("湖南" in query for query in execution.additional_queries)
    assert execution.retrieval_plan is not None
    assert execution.retrieval_plan.query_intent == "timeline"
    assert execution.retrieval_plan.query_year_range == [1921, 1926]
    assert execution.retrieval_plan.query_people == ["毛泽东"]
    assert execution.retrieval_plan.coverage == "per_year"
    assert all(
        any(f"{year}年" in query for query in execution.additional_queries)
        for year in range(1921, 1927)
    )


def test_balanced_period_builds_three_targeted_phase_queries() -> None:
    payload = _plan_payload()
    payload.update(
        {
            "normalized_question": "梳理毛泽东1921年至1949年的经历",
            "entities": [{"type": "person", "text": "毛泽东", "canonical": "毛泽东"}],
            "start_year": 1921,
            "end_year": 1949,
            "coverage": "balanced_period",
        }
    )

    execution = query_execution(
        "梳理毛泽东1921年至1949年的经历",
        QueryPlan.model_validate(payload),
        Path("config/person_aliases.json"),
    )

    assert len(execution.additional_queries) == 3
    assert "1921年至1929年" in execution.additional_queries[0]
    assert "1940年至1949年" in execution.additional_queries[-1]


def test_before_year_intersection_has_no_deterministic_preplan() -> None:
    settings = Settings(_env_file=None, llm_api_key=None)
    question = "周恩来和邓小平在1949年之前的交集"

    result = plan_question(settings, QuestionRequest(question=question))
    execution = query_execution(question, result.plan, settings.person_aliases_path)

    assert result.status == "disabled"
    assert result.plan is None
    assert execution.primary_query == question
    assert execution.additional_queries == ()
    assert execution.retrieval_plan is None


def test_relative_year_boundaries_respect_inclusive_marker_and_spaces() -> None:
    before = parse_relative_year_range("1949 年 之前", 1921, 1978)
    inclusive = parse_relative_year_range("1949年及以前", 1921, 1978)
    after = parse_relative_year_range("1949年之后", 1921, 1978)

    assert before is not None and (before.start, before.end) == (1921, 1948)
    assert inclusive is not None and (inclusive.start, inclusive.end) == (1921, 1949)
    assert after is not None and (after.start, after.end) == (1950, 1978)


def test_long_timeline_does_not_treat_phase_rewrites_as_per_item() -> None:
    payload = _plan_payload()
    payload.update(
        {
            "intent": "timeline",
            "normalized_question": "习仲勋在1921年至1949年党内职务的变化",
            "search_queries": ["习仲勋 土地革命时期 职务"],
            "entities": [{"type": "person", "text": "习仲勋", "canonical": "习仲勋"}],
            "start_year": 1921,
            "end_year": 1949,
            "coverage": "per_item",
        }
    )

    execution = query_execution(
        "习仲勋在1921—1949年党内职务的变化",
        QueryPlan.model_validate(payload),
        Path("config/person_aliases.json"),
    )

    assert execution.retrieval_plan is not None
    assert execution.retrieval_plan.coverage == "balanced_period"
    assert len(execution.additional_queries) == 3


def test_query_execution_caps_ordinary_queries_but_keeps_per_item_expansion() -> None:
    payload = _plan_payload()
    payload["search_queries"] = [f"检索表达式{index}" for index in range(1, 6)]
    ordinary = QueryPlan.model_validate(payload)
    aliases = Path("config/person_aliases.json")

    ordinary_execution = query_execution("原问题", ordinary, aliases)
    per_item_execution = query_execution(
        "原问题", ordinary.model_copy(update={"coverage": "per_item"}), aliases
    )

    assert ordinary_execution.additional_queries == ("检索表达式1", "检索表达式2")
    assert len(per_item_execution.additional_queries) == 6


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
    captured: dict[str, Any] = {}

    def search(**kwargs: Any) -> SearchResponse:
        captured["query"] = kwargs["query"]
        captured["additional_queries"] = kwargs["additional_queries"]
        captured["plan"] = kwargs["plan"]
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

    assert captured["query"] == original
    assert any("中国共产党第一次全国代表大会" in query for query in captured["additional_queries"])
    assert captured["plan"].query_intent == "event_overview"
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
