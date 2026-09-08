"""LLM-assisted semantic query understanding with deterministic safe fallback."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

import httpx
from pydantic import ValidationError

from history_agent.answering.models import QueryPlan, QuestionRequest
from history_agent.config import Settings

PROMPT_VERSION = "query-understanding-v1"
INTENT_HINTS = {
    "timeline": "经历 活动 时间线",
    "intersection": "交集 共同活动 人物关系",
    "viewpoint": "观点 论述 主张 策略",
    "observation": "记述 描述 印象",
}


@dataclass(frozen=True)
class QueryPlanningResult:
    plan: QueryPlan | None
    status: Literal["used", "disabled", "fallback", "not_applicable"]
    model_name: str | None = None
    usage: dict[str, int] | None = None
    error_code: str | None = None


def _error_code(exc: httpx.HTTPError) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return {
            401: "authentication_failed",
            402: "insufficient_balance",
            429: "rate_limited",
        }.get(exc.response.status_code, f"http_{exc.response.status_code}")
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    return "network_error"


def _messages(settings: Settings, request: QuestionRequest) -> list[dict[str, str]]:
    schema = {
        "intent": (
            "general|event_overview|timeline|intersection|viewpoint|observation|comparison|"
            "causal_analysis"
        ),
        "normalized_question": "完整保留约束并规范化简称、别名和指代后的问题",
        "search_queries": ["1至4条简短的本地史料检索表达式"],
        "entities": [
            {
                "type": "person|organization|event|place|document|other",
                "text": "原词",
                "canonical": "规范名称",
            }
        ],
        "start_year": None,
        "end_year": None,
        "coverage": "relevance|per_year|per_item|balanced_period",
        "constraints": ["不能在检索时丢弃的月份、地点、否定、来源等限制"],
        "needs_clarification": False,
        "clarification_question": None,
    }
    system = (
        "你是中国近现代史本地知识库的查询理解器，不回答历史问题。"
        "把用户的自然语言问题转换为严格JSON，禁止输出JSON之外的文字。"
        "识别简称、别名、隐含意图、时间范围、序数范围、逐年/逐项/分阶段覆盖要求和多轮指代。"
        "normalized_question必须保留用户所有明确约束，不能删除月份、地点、否定、比较对象或来源要求。"
        "search_queries用于召回，可以展开规范名称和同义表达，但不能加入用户没有要求的历史事实。"
        "start_year和end_year只填写用户明说的年份/范围，或用户明说的历史时期所对应的范围；"
        "不能根据事件常识自行补入发生年份。"
        "对“一大到六大”之类范围使用per_item；对“逐年”使用per_year；"
        "对跨度较长且要求整体梳理的时期使用balanced_period。"
        "只有人物、指代或限制条件确实无法确定时才needs_clarification；宽泛问题本身不需要澄清。"
        f"研究时间边界是{settings.research_start.year}—{settings.research_end.year}年。"
        f"输出字段示意：{json.dumps(schema, ensure_ascii=False)}"
    )
    history = "\n".join(
        f"{item.role}: {item.content}" for item in request.history[-6:]
    )
    content = f"当前问题：{request.question}"
    if history:
        content = f"最近对话：\n{history}\n\n{content}"
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def _validated_plan(content: str) -> QueryPlan:
    plan = QueryPlan.model_validate_json(content.strip())
    if (plan.start_year is None) != (plan.end_year is None):
        raise ValueError("incomplete_year_range")
    if (
        plan.start_year is not None
        and plan.end_year is not None
        and plan.start_year > plan.end_year
    ):
        raise ValueError("reversed_year_range")
    if plan.needs_clarification and not (plan.clarification_question or "").strip():
        raise ValueError("missing_clarification_question")
    if not plan.needs_clarification and plan.clarification_question:
        plan = plan.model_copy(update={"clarification_question": None})
    return plan


def plan_question(settings: Settings, request: QuestionRequest) -> QueryPlanningResult:
    if not settings.llm_query_planning or not settings.llm_enabled:
        return QueryPlanningResult(None, "disabled")
    assert settings.llm_api_key is not None
    try:
        response = httpx.post(
            settings.llm_base_url.rstrip("/") + "/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.llm_api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.llm_query_planner_model,
                "messages": _messages(settings, request),
                "response_format": {"type": "json_object"},
                "stream": False,
                "max_tokens": settings.llm_query_planner_max_tokens,
                "thinking": {"type": "disabled"},
            },
            timeout=settings.llm_query_planner_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        choice = payload["choices"][0]
        if choice.get("finish_reason") == "length":
            return QueryPlanningResult(
                None,
                "fallback",
                settings.llm_query_planner_model,
                error_code="max_tokens_exhausted",
            )
        plan = _validated_plan(str(choice["message"]["content"]))
        raw_usage = payload.get("usage", {})
        usage = {
            key: int(raw_usage[key])
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if key in raw_usage
        }
        return QueryPlanningResult(plan, "used", settings.llm_query_planner_model, usage or None)
    except httpx.HTTPError as exc:
        return QueryPlanningResult(
            None, "fallback", settings.llm_query_planner_model, error_code=_error_code(exc)
        )
    except (KeyError, IndexError, TypeError, ValueError, ValidationError):
        return QueryPlanningResult(
            None, "fallback", settings.llm_query_planner_model, error_code="invalid_plan"
        )


def retrieval_query(question: str, plan: QueryPlan | None) -> str:
    """Keep the original verbatim, then add only validated semantic expansions."""

    if plan is None:
        return question
    parts = [question, plan.normalized_question, *plan.search_queries]
    parts.extend(entity.canonical for entity in plan.entities if entity.canonical != entity.text)
    hint = INTENT_HINTS.get(plan.intent)
    if hint:
        parts.append(hint)
    unique: list[str] = []
    for part in parts:
        compact = " ".join(part.split()).strip()
        if compact and compact not in unique:
            unique.append(compact)
    return " ".join(unique)
