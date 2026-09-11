"""LLM-assisted semantic query understanding with deterministic safe fallback."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import httpx
from llama_index.core.output_parsers import PydanticOutputParser
from pydantic import ValidationError

from history_agent.answering.context import (
    requires_conversation_context,
    sanitize_history_content,
)
from history_agent.answering.models import QueryEntity, QueryPlan, QuestionRequest
from history_agent.answering.runtime import LLMRuntime, RequestBudget
from history_agent.answering.time_ranges import parse_relative_year_range
from history_agent.config import Settings
from history_agent.processing.chunks import load_person_aliases
from history_agent.retrieval.keyword import PERIOD_RANGES
from history_agent.retrieval.models import RetrievalPlan

PROMPT_VERSION = "query-understanding-v2"
INTENT_HINTS = {
    "timeline": "经历 活动 时间线",
    "intersection": "交集 共同活动 人物关系",
    "viewpoint": "观点 论述 主张 策略",
    "observation": "记述 描述 印象",
}
MAX_COVERAGE_QUERIES = 8
QUERY_PLAN_PARSER = PydanticOutputParser(QueryPlan)
CURRENT_QUESTION_YEAR = re.compile(r"(?<!\d)(?:18|19|20)\d{2}(?!\d)")
CONTEXTUAL_TIME_REFERENCE = re.compile(
    r"那年|当年|这一年|同年|次年|翌年|当时|同期|这一时期|那个时期|这段时期"
)
CONTEXTUAL_ENTITY_REFERENCE = re.compile(
    r"(?:^|[，,。！？?!；;\s])(?:他|她|他们|她们|它|其)(?:的|在|于|后来|当时|又|还|曾|是否|如何|为何|为什么|做|说|提出|经历|观点)"
    r"|这位|那位|这个人|那个人|两人|双方|前者|后者|上述人物"
    r"|这场|那场|该战役|这次|那次|这件事|那件事"
)


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
        "当前问题是人物、事件和检索限制的唯一来源；最近对话仅可用于解析当前问题中明确出现"
        "的指代或省略。当前问题已经说清人物或事件时，禁止从历史对话继承其他人物、年份、"
        "意图或限制条件。"
        "对“一大到六大”之类范围使用per_item；对“逐年”使用per_year；"
        "对跨度较长且要求整体梳理的时期使用balanced_period。"
        "只有人物、指代或限制条件确实无法确定时才needs_clarification；宽泛问题本身不需要澄清。"
        f"研究时间边界是{settings.research_start.year}—{settings.research_end.year}年。"
        f"输出字段示意：{json.dumps(schema, ensure_ascii=False)}"
    )
    history_items = request.history if requires_conversation_context(request.question) else []
    history = "\n".join(
        f"{item.role}: {sanitize_history_content(item.content)}" for item in history_items
    )
    content = f"<current_question>{request.question}</current_question>"
    if history:
        content = (
            "<conversation_history>\n"
            f"{history}\n"
            "</conversation_history>\n\n"
            f"{content}"
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def _current_question_year_range(
    settings: Settings, question: str
) -> tuple[int, int] | None:
    relative = parse_relative_year_range(
        question, settings.research_start.year, settings.research_end.year
    )
    if relative is not None:
        return relative.start, relative.end
    years = [int(match.group()) for match in CURRENT_QUESTION_YEAR.finditer(question)]
    if years:
        return min(years), max(years)
    ranges = [year_range for name, year_range in PERIOD_RANGES.items() if name in question]
    if ranges:
        return min(item[0] for item in ranges), max(item[1] for item in ranges)
    return None


def _compact(value: str) -> str:
    return "".join(value.casefold().split())


def _latest_user_question(request: QuestionRequest) -> str:
    return next(
        (item.content for item in reversed(request.history) if item.role == "user"), ""
    )


def _entity_is_grounded(
    entity: QueryEntity,
    request: QuestionRequest,
    *,
    allow_contextual_reference: bool,
) -> bool:
    compact_question = _compact(request.question)
    if _compact(entity.canonical) in compact_question:
        return True
    if (
        _compact(entity.text) in compact_question
        and not requires_conversation_context(entity.text)
    ):
        return True
    if not allow_contextual_reference:
        return False
    compact_previous = _compact(_latest_user_question(request))
    return any(
        _compact(value) in compact_previous for value in (entity.text, entity.canonical)
    )


def _guard_plan_against_history(
    settings: Settings,
    request: QuestionRequest,
    plan: QueryPlan,
) -> QueryPlan:
    """Keep history-only constraints from becoming hard retrieval filters."""

    uses_history = bool(request.history) and requires_conversation_context(request.question)
    current_year_range = _current_question_year_range(settings, request.question)
    allow_contextual_time = (
        uses_history and CONTEXTUAL_TIME_REFERENCE.search(request.question) is not None
    )
    allow_contextual_entities = (
        uses_history and CONTEXTUAL_ENTITY_REFERENCE.search(request.question) is not None
    )

    start_year, end_year = plan.start_year, plan.end_year
    disallowed_terms: set[str] = set()
    if current_year_range is None and allow_contextual_time:
        current_year_range = _current_question_year_range(
            settings, _latest_user_question(request)
        )

    if current_year_range is not None:
        start_year, end_year = current_year_range
        for year in (plan.start_year, plan.end_year):
            if year is not None and year not in current_year_range:
                disallowed_terms.add(str(year))
    else:
        disallowed_terms.update(
            str(year) for year in (plan.start_year, plan.end_year) if year is not None
        )
        start_year = end_year = None

    entities = [
        entity
        for entity in plan.entities
        if _entity_is_grounded(
            entity,
            request,
            allow_contextual_reference=allow_contextual_entities,
        )
    ]
    for entity in plan.entities:
        if entity not in entities:
            disallowed_terms.update((entity.text, entity.canonical))

    disallowed_terms = {term for term in disallowed_terms if len(term.strip()) >= 2}

    def contains_disallowed(value: str) -> bool:
        return any(term in value for term in disallowed_terms)

    normalized_question = plan.normalized_question
    if contains_disallowed(normalized_question):
        normalized_question = request.question
    search_queries = [
        query for query in plan.search_queries if not contains_disallowed(query)
    ]
    constraints = [
        constraint for constraint in plan.constraints if not contains_disallowed(constraint)
    ]
    coverage = plan.coverage
    if start_year is None and coverage in {"per_year", "balanced_period"}:
        coverage = "relevance"
    return plan.model_copy(
        update={
            "normalized_question": normalized_question,
            "search_queries": search_queries,
            "entities": entities,
            "start_year": start_year,
            "end_year": end_year,
            "coverage": coverage,
            "constraints": constraints,
        }
    )


def _validated_plan(content: str) -> QueryPlan:
    plan = cast(QueryPlan, QUERY_PLAN_PARSER.parse(content.strip()))
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


def _normalize_coverage(plan: QueryPlan) -> QueryPlan:
    if (
        plan.intent == "timeline"
        and plan.coverage == "per_item"
        and plan.start_year is not None
        and plan.end_year is not None
        and plan.end_year - plan.start_year >= 4
    ):
        # ``per_item`` is for an explicitly enumerated series such as party
        # congresses. A long career range needs temporal coverage instead.
        return plan.model_copy(update={"coverage": "balanced_period"})
    return plan


def _relative_intersection_plan(
    settings: Settings, request: QuestionRequest
) -> QueryPlan | None:
    if not any(marker in request.question for marker in ("交集", "共同")):
        return None
    relative = parse_relative_year_range(
        request.question, settings.research_start.year, settings.research_end.year
    )
    if relative is None or relative.start > relative.end:
        return None
    try:
        aliases = load_person_aliases(settings.person_aliases_path)
    except Exception:
        return None
    entities: list[QueryEntity] = []
    for canonical, forms in aliases.items():
        matched = next(
            (form for form in [canonical, *forms] if form in request.question),
            None,
        )
        if matched is not None:
            entities.append(QueryEntity(type="person", text=matched, canonical=canonical))
    if len(entities) != 2:
        return None
    normalized = request.question.replace(
        relative.raw, f"{relative.start}年至{relative.end}年"
    )
    return QueryPlan(
        intent="intersection",
        normalized_question=normalized,
        entities=entities,
        start_year=relative.start,
        end_year=relative.end,
        coverage="balanced_period",
    )


def plan_question(
    settings: Settings,
    request: QuestionRequest,
    runtime: LLMRuntime | None = None,
    budget: RequestBudget | None = None,
) -> QueryPlanningResult:
    relative_plan = _relative_intersection_plan(settings, request)
    if relative_plan is not None:
        return QueryPlanningResult(relative_plan, "not_applicable")
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
            "temperature": 0,
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
        plan = _normalize_coverage(
            _guard_plan_against_history(
                settings,
                request,
                _validated_plan(str(choice["message"]["content"])),
            )
        )
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
    plan = _normalize_coverage(plan)
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
