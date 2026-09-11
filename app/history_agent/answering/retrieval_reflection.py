"""Bounded LLM reflection over first-pass retrieval coverage."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Annotated, Literal

import httpx
from llama_index.core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, ValidationError

from history_agent.answering.models import QueryPlan, QuestionRequest
from history_agent.answering.runtime import LLMRuntime, RequestBudget
from history_agent.config import Settings
from history_agent.retrieval.models import SearchResponse

PROMPT_VERSION = "retrieval-reflection-v1"
COMPLEX_QUESTION = re.compile(
    r"详细|全面|系统|完整|经过|过程|前后|各阶段|主要经历|梳理|演变|变化|"
    r"原因|影响|意义|比较|异同"
)
YEAR = re.compile(r"(?<!\d)(?P<year>(?:18|19|20)\d{2})(?!\d)\s*年?")
VAGUE_ASPECTS = {"更多", "更多证据", "其他", "其余", "全部", "完整资料", "相关资料"}
Aspect = Annotated[str, Field(min_length=2, max_length=40)]


class EvidenceAssessment(BaseModel):
    """Strict semantic coverage judgment returned by the reflection model."""

    sufficient: bool
    covered_aspects: list[Aspect] = Field(default_factory=list, max_length=8)
    missing_aspects: list[Aspect] = Field(default_factory=list, max_length=4)
    reason: str = Field(min_length=2, max_length=240)


EVIDENCE_ASSESSMENT_PARSER = PydanticOutputParser(EvidenceAssessment)


@dataclass(frozen=True)
class ReflectionResult:
    status: Literal["disabled", "skipped", "sufficient", "retry", "fallback"]
    assessment: EvidenceAssessment | None = None
    followup_queries: tuple[str, ...] = ()
    usage: dict[str, int] | None = None
    error_code: str | None = None


def should_reflect(
    settings: Settings,
    request: QuestionRequest,
    plan: QueryPlan | None,
) -> bool:
    """Keep the extra model call off the latency path for ordinary questions."""

    if (
        not settings.llm_retrieval_reflection
        or settings.llm_retrieval_reflection_max_rounds <= 0
        or not settings.llm_enabled
        or plan is None
    ):
        return False
    return (
        plan.coverage != "relevance"
        or plan.intent in {"event_overview", "comparison", "causal_analysis"}
        or COMPLEX_QUESTION.search(request.question) is not None
    )


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


def _evidence_text(retrieval: SearchResponse) -> str:
    blocks = []
    for index, hit in enumerate(retrieval.hits[:12], start=1):
        text = " ".join(hit.text.split())
        blocks.append(
            f"[R{index}] {hit.title}；章节：{' > '.join(hit.section_path) or '未识别'}\n"
            f"{text[:420]}"
        )
    return "\n\n".join(blocks) or "（首轮没有检索到候选片段）"


def _messages(
    request: QuestionRequest,
    plan: QueryPlan,
    retrieval: SearchResponse,
) -> list[dict[str, str]]:
    system = (
        "你是本地史料检索的证据充分性评估器，不回答历史问题。"
        "只判断首轮候选片段是否足以覆盖用户明确要求的主要方面，输出严格JSON。"
        "不能用模型记忆补充史实，不能因为你知道答案就判定充分。"
        "missing_aspects只写用户问题要求但当前片段缺少直接材料的方面，每项是2至40字的"
        "检索主题短语；禁止加入用户没有要求的人物、年份、地点或具体史实，最多4项。"
        "如果已有材料能支撑有用但不一定百科全书式完整的回答，应判定sufficient=true。"
        "只有明显缺少核心阶段、比较对象、原因、结果或用户指定项目时才判定不足。"
        "输出字段：sufficient布尔值、covered_aspects字符串数组、missing_aspects字符串数组、"
        "reason简短理由。禁止输出JSON之外的文字。"
    )
    content = (
        f"当前问题：{request.question}\n"
        f"查询意图：{plan.intent}\n"
        f"规范问题：{plan.normalized_question}\n"
        f"覆盖方式：{plan.coverage}\n\n"
        f"首轮候选片段：\n{_evidence_text(retrieval)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def _request_payload(
    settings: Settings,
    request: QuestionRequest,
    plan: QueryPlan,
    retrieval: SearchResponse,
) -> dict[str, object]:
    return {
        "model": settings.llm_query_planner_model,
        "messages": _messages(request, plan, retrieval),
        "response_format": {"type": "json_object"},
        "stream": False,
        "max_tokens": settings.llm_retrieval_reflection_max_tokens,
        "temperature": 0,
        "thinking": {"type": "disabled"},
    }


def _followup_queries(
    settings: Settings,
    request: QuestionRequest,
    plan: QueryPlan,
    assessment: EvidenceAssessment,
) -> tuple[str, ...]:
    subject = " ".join(
        dict.fromkeys(entity.canonical for entity in plan.entities)
    ).strip() or plan.normalized_question
    constraints = " ".join(plan.constraints).strip()
    allowed_years = {match["year"] for match in YEAR.finditer(request.question)}
    queries: list[str] = []
    for raw_aspect in assessment.missing_aspects:
        aspect = " ".join(raw_aspect.split()).strip("，,。；;：:、- ")
        aspect = YEAR.sub(
            lambda match: match.group() if match["year"] in allowed_years else "",
            aspect,
        )
        aspect = " ".join(aspect.split())
        if len(aspect) < 2 or aspect in VAGUE_ASPECTS:
            continue
        query = " ".join(item for item in (subject, aspect, constraints) if item)
        if query != request.question and query not in queries:
            queries.append(query)
        if len(queries) >= settings.llm_retrieval_reflection_max_queries:
            break
    return tuple(queries)


def assess_retrieval(
    settings: Settings,
    request: QuestionRequest,
    plan: QueryPlan | None,
    retrieval: SearchResponse,
    runtime: LLMRuntime | None = None,
    budget: RequestBudget | None = None,
) -> ReflectionResult:
    """Assess first-pass evidence and return bounded, validated follow-up queries."""

    if (
        not settings.llm_retrieval_reflection
        or settings.llm_retrieval_reflection_max_rounds <= 0
        or not settings.llm_enabled
    ):
        return ReflectionResult("disabled")
    if not should_reflect(settings, request, plan):
        return ReflectionResult("skipped")
    assert plan is not None
    assert settings.llm_api_key is not None
    try:
        timeout = (
            budget.timeout(settings.llm_retrieval_reflection_timeout_seconds)
            if budget is not None
            else settings.llm_retrieval_reflection_timeout_seconds
        )
        url = settings.llm_base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.llm_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        request_payload = _request_payload(settings, request, plan, retrieval)
        response = (
            runtime.post(
                url,
                headers=headers,
                json=request_payload,
                timeout=timeout,
            )
            if runtime is not None
            else httpx.post(
                url,
                headers=headers,
                json=request_payload,
                timeout=timeout,
            )
        )
        response.raise_for_status()
        payload = response.json()
        choice = payload["choices"][0]
        if choice.get("finish_reason") == "length":
            return ReflectionResult("fallback", error_code="max_tokens_exhausted")
        assessment = EVIDENCE_ASSESSMENT_PARSER.parse(
            str(choice["message"]["content"]).strip()
        )
        raw_usage = payload.get("usage", {})
        usage = {
            key: int(raw_usage[key])
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if key in raw_usage
        }
        if assessment.sufficient:
            return ReflectionResult("sufficient", assessment, usage=usage or None)
        queries = _followup_queries(settings, request, plan, assessment)
        if not queries:
            return ReflectionResult(
                "fallback", assessment, usage=usage or None, error_code="no_safe_followup_query"
            )
        return ReflectionResult("retry", assessment, queries, usage or None)
    except httpx.HTTPError as exc:
        return ReflectionResult("fallback", error_code=_error_code(exc))
    except (KeyError, IndexError, TypeError, ValueError, ValidationError):
        return ReflectionResult("fallback", error_code="invalid_assessment")
