"""LLM selection of evidence that really describes a person's own activity."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Annotated, Literal

import httpx
from llama_index.core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, ValidationError

from history_agent.answering.llamaindex_llm import (
    DeepSeekLlamaIndexLLM,
    chat_messages,
    response_usage,
)
from history_agent.answering.runtime import LLMRuntime, RequestBudget
from history_agent.config import Settings
from history_agent.retrieval.models import SearchHit, SearchResponse

PROMPT_VERSION = "timeline-evidence-selection-v1"
MAX_CANDIDATES = 36
TITLE_YEAR_RANGE = re.compile(
    r"(?<!\d)(?P<start>(?:18|19|20)\d{2})(?:\.\d{1,2})?\s*"
    r"(?:—|–|-|至|~)\s*(?P<end>(?:18|19|20)\d{2})(?:\.\d{1,2})?(?!\d)"
)
CandidateId = Annotated[int, Field(ge=1, le=MAX_CANDIDATES)]


class TimelineEvidenceSelection(BaseModel):
    """Strict result returned by the evidence-selection model."""

    selected_ids: list[CandidateId] = Field(default_factory=list, max_length=MAX_CANDIDATES)


SELECTION_PARSER = PydanticOutputParser(TimelineEvidenceSelection)


@dataclass(frozen=True)
class TimelineEvidenceFilterResult:
    retrieval: SearchResponse
    status: Literal["disabled", "skipped", "applied", "fallback"]
    removed_count: int = 0
    usage: dict[str, int] | None = None
    error_code: str | None = None


def _compact(text: str) -> str:
    return " ".join(text.split())


def _excerpt(hit: SearchHit, person: str, limit: int = 560) -> str:
    text = _compact(hit.text)
    if len(text) <= limit:
        return text
    position = text.find(person)
    center = position if position >= 0 else 0
    start = max(0, center - 140)
    end = min(len(text), start + limit)
    return ("……" if start else "") + text[start:end] + ("……" if end < len(text) else "")


def _candidate_text(retrieval: SearchResponse, person: str) -> str:
    blocks = []
    for index, hit in enumerate(retrieval.hits[:MAX_CANDIDATES], start=1):
        blocks.append(
            f"[R{index}] 文献：《{hit.title}》；PDF第{hit.pdf_page_start}页；"
            f"片段标注年份：{hit.year_mentions or '未标注'}\n{_excerpt(hit, person)}"
        )
    return "\n\n".join(blocks)


def _source_period_matches_query(hit: SearchHit, retrieval: SearchResponse) -> bool:
    """Reject chronology volumes whose explicit title span cannot cover the query."""

    match = TITLE_YEAR_RANGE.search(hit.title)
    if match is None:
        return True
    source_start, source_end = int(match["start"]), int(match["end"])
    if retrieval.query_year_range:
        query_start, query_end = retrieval.query_year_range
    elif retrieval.query_years:
        query_start, query_end = min(retrieval.query_years), max(retrieval.query_years)
    else:
        return True
    return source_start <= query_end and query_start <= source_end


def _request_payload(
    settings: Settings,
    question: str,
    person: str,
    retrieval: SearchResponse,
) -> dict[str, object]:
    system = (
        "你是人物年表证据审查器，不回答历史问题，也不能使用模型记忆补充事实。"
        "逐条判断候选片段能否直接证明目标人物在问题所问时期的一项本人经历。"
        "只有以下情况可以选择：原文明示目标人物亲自行动、参与、表态、拒绝、建议、"
        "主持、指挥、会见、共同发电或承担明确职务；被任命或当选为明确职务也可选择。"
        "以下情况不能选择：他人只是向目标人物发电、致信、邀请、下令、评价或提到他；"
        "目标人物仅是收件人、被会见者、普通名单成员，且片段没有写出他的行动或明确职务；"
        "仅凭人名共现推测他参与；片段所述事件年份明显落在文献标题明确标示的年代范围之外，"
        "导致来源归属可疑。一个片段同时包含他人行为和目标人物明确行动时，可以选择。"
        "宁可少选，不要把动作主体不明的材料交给回答模型。输出严格JSON，只有selected_ids"
        "字段，值为可用候选的数字编号数组；没有可用证据就返回空数组。"
    )
    content = (
        f"问题：{question}\n目标人物：{person}\n\n"
        f"候选片段：\n{_candidate_text(retrieval, person)}"
    )
    return {
        "model": settings.llm_query_planner_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        "response_format": {"type": "json_object"},
        "stream": False,
        "max_tokens": min(1400, max(450, len(retrieval.hits) * 32)),
        "temperature": 0,
        "thinking": {"type": "disabled"},
    }


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


def filter_person_timeline_evidence(
    settings: Settings,
    question: str,
    retrieval: SearchResponse,
    runtime: LLMRuntime | None = None,
    budget: RequestBudget | None = None,
) -> TimelineEvidenceFilterResult:
    """Keep only passages the LLM judges to describe the target's own activity."""

    if retrieval.query_intent != "timeline" or len(retrieval.query_people) != 1:
        return TimelineEvidenceFilterResult(retrieval, "skipped")
    if not retrieval.hits:
        return TimelineEvidenceFilterResult(retrieval, "skipped")
    if not settings.llm_enabled:
        return TimelineEvidenceFilterResult(retrieval, "disabled")

    person = retrieval.query_people[0]
    eligible_source_hits = [
        hit for hit in retrieval.hits if _source_period_matches_query(hit, retrieval)
    ]
    source_filtered = retrieval.model_copy(update={"hits": eligible_source_hits})
    if not eligible_source_hits:
        return TimelineEvidenceFilterResult(
            retrieval.model_copy(
                update={
                    "hits": [],
                    "retrieval_mode": f"{retrieval.retrieval_mode}_subject_filtered",
                }
            ),
            "applied",
            removed_count=len(retrieval.hits),
        )
    assert settings.llm_api_key is not None
    try:
        timeout = (
            budget.timeout(settings.llm_query_planner_timeout_seconds)
            if budget is not None
            else settings.llm_query_planner_timeout_seconds
        )
        payload = _request_payload(settings, question, person, source_filtered)
        response = DeepSeekLlamaIndexLLM(
            settings=settings,
            runtime=runtime,
            budget=budget,
            model=settings.llm_query_planner_model,
            timeout_seconds=timeout,
        ).chat(chat_messages(payload), request_payload=payload)
        if response.additional_kwargs.get("finish_reason") == "length":
            return TimelineEvidenceFilterResult(
                retrieval, "fallback", error_code="max_tokens_exhausted"
            )
        selection = SELECTION_PARSER.parse((response.message.content or "").strip())
        selected = set(selection.selected_ids)
        hits = [
            hit.model_copy(update={"rank": rank})
            for rank, hit in enumerate(
                (
                    hit
                    for index, hit in enumerate(
                        eligible_source_hits[:MAX_CANDIDATES], start=1
                    )
                    if index in selected
                ),
                start=1,
            )
        ]
        filtered = retrieval.model_copy(
            update={
                "hits": hits,
                "retrieval_mode": f"{retrieval.retrieval_mode}_subject_filtered",
            }
        )
        return TimelineEvidenceFilterResult(
            filtered,
            "applied",
            removed_count=len(retrieval.hits) - len(hits),
            usage=response_usage(response),
        )
    except httpx.HTTPError as exc:
        return TimelineEvidenceFilterResult(retrieval, "fallback", error_code=_error_code(exc))
    except (KeyError, IndexError, TypeError, ValueError, ValidationError):
        return TimelineEvidenceFilterResult(
            retrieval, "fallback", error_code="invalid_selection"
        )
