"""LLM-assisted semantic query understanding with deterministic safe fallback."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
from pydantic import ValidationError

from history_agent.answering.models import QueryPlan, QuestionRequest
from history_agent.answering.runtime import LLMRuntime, RequestBudget
from history_agent.config import Settings
from history_agent.processing.chunks import load_person_aliases
from history_agent.retrieval.models import RetrievalPlan

PROMPT_VERSION = "query-understanding-v1"
INTENT_HINTS = {
    "timeline": "经历 活动 时间线",
    "intersection": "交集 共同活动 人物关系",
    "viewpoint": "观点 论述 主张 策略",
    "observation": "记述 描述 印象",
}
MAX_COVERAGE_QUERIES = 8


@dataclass(frozen=True)
class QueryPlanningResult:
    plan: QueryPlan | None
    status: Literal["used", "disabled", "fallback", "not_applicable"]
    model_name: str | None = None
    usage: dict[str, int] | None = None
    error_code: str | None = None


@dataclass(frozen=True)
class QueryExecution:
    primary_query: str
    additional_queries: tuple[str, ...]
    retrieval_plan: RetrievalPlan | None


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
        "search_queries": ["1至8条可独立执行的简短本地史料检索表达式"],
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
    history = "\n".join(f"{item.role}: {item.content}" for item in request.history[-6:])
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


def plan_question(
    settings: Settings,
    request: QuestionRequest,
    runtime: LLMRuntime | None = None,
    budget: RequestBudget | None = None,
) -> QueryPlanningResult:
    if not settings.llm_query_planning or not settings.llm_enabled:
        return QueryPlanningResult(None, "disabled")
    assert settings.llm_api_key is not None
    try:
        timeout = (
            budget.timeout(settings.llm_query_planner_timeout_seconds)
            if budget is not None
            else settings.llm_query_planner_timeout_seconds
        )
        headers = {
            "Authorization": f"Bearer {settings.llm_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        request_payload: dict[str, object] = {
            "model": settings.llm_query_planner_model,
            "messages": _messages(settings, request),
            "response_format": {"type": "json_object"},
            "stream": False,
            "max_tokens": settings.llm_query_planner_max_tokens,
            "thinking": {"type": "disabled"},
        }
        url = settings.llm_base_url.rstrip("/") + "/chat/completions"
        response = (
            runtime.post(url, headers=headers, json=request_payload, timeout=timeout)
            if runtime is not None
            else httpx.post(url, headers=headers, json=request_payload, timeout=timeout)
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


def _known_people(plan: QueryPlan, aliases_path: Path) -> list[str]:
    aliases = load_person_aliases(aliases_path)
    by_form = {
        form: canonical for canonical, forms in aliases.items() for form in [canonical, *forms]
    }
    people: list[str] = []
    for entity in plan.entities:
        if entity.type != "person":
            continue
        canonical = by_form.get(entity.canonical) or by_form.get(entity.text)
        if canonical and canonical not in people:
            people.append(canonical)
    return people


def _coverage_queries(plan: QueryPlan, people: list[str]) -> list[str]:
    """Build deterministic group queries when completeness matters."""

    if plan.start_year is None or plan.end_year is None:
        return []
    entities = list(
        dict.fromkeys(
            [*people, *(entity.canonical for entity in plan.entities if entity.type != "person")]
        )
    )
    subject = " ".join(entities) or plan.normalized_question
    suffix = " ".join([*plan.constraints, INTENT_HINTS.get(plan.intent, "")]).strip()
    if plan.coverage == "per_year":
        years = list(range(plan.start_year, plan.end_year + 1))
        if len(years) > MAX_COVERAGE_QUERIES:
            years = [
                years[round(index * (len(years) - 1) / (MAX_COVERAGE_QUERIES - 1))]
                for index in range(MAX_COVERAGE_QUERIES)
            ]
        return [f"{subject} {year}年 {suffix}".strip() for year in years]
    if plan.coverage == "balanced_period" and plan.start_year < plan.end_year:
        span = plan.end_year - plan.start_year + 1
        ranges = []
        for index in range(3):
            start = plan.start_year + span * index // 3
            end = plan.start_year + span * (index + 1) // 3 - 1
            ranges.append((start, max(start, end)))
        return [
            f"{subject} {start}年至{end}年 {suffix}".strip() for start, end in ranges
        ]
    return []


def query_execution(question: str, plan: QueryPlan | None, aliases_path: Path) -> QueryExecution:
    """Compile an LLM plan into independently executable, locally validated retrieval input."""

    if plan is None:
        return QueryExecution(question, (), None)
    hint = INTENT_HINTS.get(plan.intent)
    # Retrieval-oriented rewrites are more valuable than the prose-normalized
    # question because the original question is always executed separately.
    parts = [*plan.search_queries, plan.normalized_question]
    for entity in plan.entities:
        if entity.canonical != entity.text and not any(
            entity.canonical in part for part in parts
        ):
            parts.append(entity.canonical)
    constraint_hint = " ".join(plan.constraints)
    if constraint_hint:
        parts = [f"{part} {constraint_hint}" for part in parts]
    if hint:
        parts = [f"{part} {hint}" for part in parts]
    people = _known_people(plan, aliases_path)
    unique: list[str] = []
    for part in [*_coverage_queries(plan, people), *parts]:
        compact = " ".join(part.split()).strip()
        if compact and compact != question and compact not in unique:
            unique.append(compact)
    year_range: list[int] = []
    if plan.start_year is not None and plan.end_year is not None:
        year_range = [plan.start_year, plan.end_year]
    years = year_range[:1] if year_range and year_range[0] == year_range[1] else []
    retrieval_plan = RetrievalPlan(
        query_intent=plan.intent,
        query_years=years,
        query_year_range=year_range,
        query_people=people,
        coverage=plan.coverage,
    )
    query_limit = (
        MAX_COVERAGE_QUERIES
        if plan.coverage in {"per_item", "per_year"}
        else 3
        if plan.coverage == "balanced_period"
        else 2
    )
    return QueryExecution(question, tuple(unique[:query_limit]), retrieval_plan)


def retrieval_query(question: str, plan: QueryPlan | None) -> str:
    """Compatibility helper returning the visible aggregate of planned query variants."""

    if plan is None:
        return question
    parts = [question, plan.normalized_question, *plan.search_queries]
    parts.extend(entity.canonical for entity in plan.entities if entity.canonical != entity.text)
    hint = INTENT_HINTS.get(plan.intent)
    if hint:
        parts.append(hint)
    return " ".join(dict.fromkeys(" ".join(part.split()) for part in parts if part.strip()))
