from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, cast

import httpx

from history_agent.answering.context import (
    requires_conversation_context,
    sanitize_history_content,
)
from history_agent.answering.full_text import answer_full_text_question
from history_agent.answering.models import AnswerResponse, Citation, QuestionRequest
from history_agent.answering.query_understanding import (
    QueryExecution,
    QueryPlanningResult,
    plan_question,
    query_execution,
)
from history_agent.answering.retrieval_reflection import assess_retrieval
from history_agent.answering.runtime import LLMRuntime, RequestBudget
from history_agent.answering.structured import (
    answer_structured_question,
    requires_structured_generation,
)
from history_agent.answering.validation import remove_uncited_claim_blocks, validate_grounded_answer
from history_agent.config import Settings
from history_agent.errors import RetrievalError
from history_agent.retrieval.hybrid import search_hybrid_index
from history_agent.retrieval.models import SearchHit, SearchResponse

WHITESPACE = re.compile(r"\s+")
SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？；])")
LEADING_ENTITY = re.compile(
    r"^(?:请问|我想知道|想知道|帮我查)?(?P<entity>[\u3400-\u4dbf\u4e00-\u9fff·]{2,18})"
    r"(?:在|于)(?=(?:18|19|20)\d{2}年)"
)
ENTITY_SEPARATOR = re.compile(r"[、和与]")
PROMPT_VERSION = "grounded-answer-v14"
LLM_EVIDENCE_BATCH_SIZE = 12
MAX_COMPLEX_RETRIEVAL_CHUNKS = 36


def _compact(text: str) -> str:
    return WHITESPACE.sub(" ", text).strip()


def _quote_for_hit(
    hit: SearchHit,
    query_terms: list[str],
    limit: int = 420,
    *,
    query_people: list[str] | None = None,
) -> str:
    text = _compact(hit.text)
    people = set(query_people or [])
    if len(people) == 1:
        position = text.find(next(iter(people)))
        if position > 0:
            window_start = max(0, position - 110)
            boundary = max(text.rfind(mark, window_start, position) for mark in "。！？；")
            start = boundary + 1 if boundary >= window_start else window_start
            if start:
                text = "……" + text[start:]
    if len(text) <= limit:
        return text
    person_positions = [
        match.start()
        for person in people
        for match in re.finditer(re.escape(person), text)
    ]
    positions = person_positions or [text.find(term) for term in query_terms if len(term) >= 2]
    positions = [position for position in positions if position >= 0]
    center = min(positions) if positions else 0
    if len(people) == 2:
        mentions = sorted(
            (match.start(), match.end(), person)
            for person in people
            for match in re.finditer(re.escape(person), text)
        )
        # Center the quote on the closest pair of mentions, not the first occurrence
        # of a common keyword. Proximity selects context; it does not prove interaction.
        pairs = [
            (right[1] - left[0], left[0], right[1])
            for left, right in zip(mentions, mentions[1:], strict=False)
            if left[2] != right[2] and right[1] - left[0] <= limit
        ]
        if pairs:
            _, center, pair_end = min(pairs)
            # Keep the preceding event context, including when a roster is at the
            # end of the passage. A short tail alone can lose which meeting it names.
            window_start = max(0, pair_end - limit, min(center - 110, len(text) - limit))
            boundaries = [text.find(mark, window_start, center) for mark in "。！？；"]
            boundary = min((position for position in boundaries if position >= 0), default=-1)
            start = boundary + 1 if boundary >= 0 else window_start
            end = min(len(text), start + limit)
            return ("……" if start else "") + text[start:end] + ("……" if end < len(text) else "")
    window_start = max(0, center - 110)
    boundaries = [text.rfind(mark, window_start, center) for mark in "。！？；"]
    boundary = max(boundaries)
    start = boundary + 1 if boundary >= window_start else window_start
    end = min(len(text), start + limit)
    prefix = "……" if start else ""
    suffix = "……" if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}"


def _unsupported_leading_entity(question: str, hits: list[SearchHit]) -> str | None:
    match = LEADING_ENTITY.search(question.strip())
    if match is None:
        return None
    entities = [entity for entity in ENTITY_SEPARATOR.split(match.group("entity")) if entity]
    for entity in entities:
        if not any(
            entity in hit.title
            or entity in hit.text
            or any(entity in part for part in hit.section_path)
            for hit in hits
        ):
            return entity
    return None


def _citations(response: Any) -> list[Citation]:
    return [
        Citation(
            evidence_id=f"E{index}",
            document_id=hit.document_id,
            document=hit.title,
            volume=hit.volume,
            pdf_page=hit.pdf_page_start,
            section=hit.section_path,
            quote=_quote_for_hit(
                hit,
                response.query_terms,
                query_people=response.query_people,
            ),
            source_type=hit.source_type,
            verification_status=hit.verification_status,
            extraction_methods=hit.extraction_methods,
        )
        for index, hit in enumerate(response.hits, start=1)
    ]


def _citations_used_by_answer(answer: str, citations: list[Citation]) -> list[Citation]:
    """Expose only evidence the authoritative answer actually cites."""

    used_ids = set(validate_grounded_answer(answer, citations).used_evidence_ids)
    return [citation for citation in citations if citation.evidence_id in used_ids]


def _extractive_answer(intent: str, citations: list[Citation]) -> str:
    if not citations:
        return "现有本地资料中没有检索到足以回答这个问题的证据。"
    lead = {
        "timeline": "根据当前本地资料，可先按以下史料线索梳理：",
        "intersection": (
            "根据当前本地资料，检索到以下涉及两位人物的史料线索，可据原文核查具体交集："
        ),
        "viewpoint": "根据当前本地资料，相关观点主要见于以下原文：",
    }.get(intent, "根据当前本地资料，检索到以下可核验线索：")
    bullets: list[str] = []
    for citation in citations[:5]:
        sentences = [
            item.strip()
            for item in SENTENCE_BOUNDARY.split(citation.quote)
            if len(item.replace("…", "").strip().strip("。")) >= 12
        ]
        summary = (sentences[0] if sentences else citation.quote)[:180].strip()
        summary = re.sub(r"^[…，、\s]+", "", summary)
        bullets.append(f"- {summary} [{citation.evidence_id}]")
    return lead + "\n\n" + "\n".join(bullets)


def _chat_completions_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


@dataclass(frozen=True)
class LLMResult:
    answer: str | None
    error_code: str | None = None
    usage: dict[str, int] | None = None
    uncited_claims: tuple[str, ...] = ()


def _salvage_llm_result(
    answer: str,
    citations: list[Citation],
    uncited_claims: tuple[str, ...],
    usage: dict[str, int] | None,
) -> LLMResult | None:
    salvaged = remove_uncited_claim_blocks(answer, uncited_claims)
    if salvaged is None or not validate_grounded_answer(salvaged, citations).valid:
        return None
    return LLMResult(
        answer=salvaged,
        error_code="removed_uncited_claims",
        usage=usage,
        uncited_claims=uncited_claims,
    )


def _prefer_llm_result(
    first: LLMResult | None,
    second: LLMResult | None,
    usage: dict[str, int] | None,
) -> LLMResult | None:
    candidates = [item for item in (first, second) if item is not None and item.answer]
    if not candidates:
        return None
    preferred = max(candidates, key=lambda item: (len(item.answer or ""), item.error_code is None))
    return LLMResult(
        answer=preferred.answer,
        error_code=preferred.error_code,
        usage=usage,
        uncited_claims=preferred.uncited_claims,
    )


def _deepseek_error_code(exc: httpx.HTTPError) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return {
            401: "authentication_failed",
            402: "insufficient_balance",
            429: "rate_limited",
        }.get(exc.response.status_code, f"http_{exc.response.status_code}")
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    return "network_error"


def _merge_usage(*items: dict[str, int] | None) -> dict[str, int] | None:
    keys = {key for item in items if item for key in item}
    if not keys:
        return None
    return {key: sum(item.get(key, 0) for item in items if item) for key in keys}


def _request_deepseek_completion(
    settings: Settings,
    request_payload: dict[str, object],
    runtime: LLMRuntime | None = None,
    budget: RequestBudget | None = None,
) -> LLMResult:
    assert settings.llm_api_key is not None
    try:
        timeout = (
            budget.timeout(settings.llm_timeout_seconds)
            if budget is not None
            else settings.llm_timeout_seconds
        )
        url = _chat_completions_url(settings.llm_base_url)
        headers = {
            "Authorization": f"Bearer {settings.llm_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        result = (
            runtime.post(url, headers=headers, json=request_payload, timeout=timeout)
            if runtime is not None
            else httpx.post(url, headers=headers, json=request_payload, timeout=timeout)
        )
        result.raise_for_status()
        payload = result.json()
        answer = str(payload["choices"][0]["message"]["content"]).strip()
        finish_reason = payload["choices"][0].get("finish_reason")
        raw_usage = payload.get("usage", {})
        usage = {
            key: int(raw_usage[key])
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if key in raw_usage
        }
    except httpx.HTTPError as exc:
        return LLMResult(answer=None, error_code=_deepseek_error_code(exc))
    except (KeyError, IndexError, TypeError, ValueError):
        return LLMResult(answer=None, error_code="invalid_response")
    if finish_reason == "length":
        return LLMResult(answer=None, error_code="max_tokens_exhausted", usage=usage)
    return LLMResult(answer=answer, usage=usage)


def _llm_request_payload(
    *,
    settings: Settings,
    request: QuestionRequest,
    citations: list[Citation],
    evidence_override: str | None = None,
    max_tokens: int | None = None,
) -> dict[str, object]:
    evidence = evidence_override or "\n\n".join(
        f"[{item.evidence_id}] 《{item.document}》PDF第{item.pdf_page}页"
        f"；章节：{' > '.join(item.section) or '未识别'}\n{item.quote}"
        for item in citations
    )
    system = (
        "你是中国近现代史本地史料研究助手。只能根据用户提供的证据回答，不能用模型记忆"
        "补充事实。每个事实性结论后必须标注对应证据编号，如[E1]。区分原文观点、年谱记载"
        "和后人叙述；证据不足就明确说明。不要虚构页码或证据编号。先给简明结论，再按时间"
        "或主题组织要点，最后说明资料限制。不要输出证据包中不存在的知识。回答中必须至少"
        "出现一个本次证据编号；引用格式只能是[E1]、[E2]这种形式。每一条包含日期、职务、"
        "地点、行动或人物关系的事实必须在同一段或同一列表项给出证据编号；同一要点正常换行"
        "不必重复标注。如需写文献名或PDF页码，必须与证据包完全一致。"
        "对于人物交集问题，必须说明原文支持两人围绕哪一具体事件发生了什么互动，"
        "区分共同参与、意见支持与分工协作，不能把人名共现推断成共同参与。"
        "若问题指定长征等历史时期，只总结证据明确支持属于该时期的活动；"
        "检索年份范围只是召回线索，不能把同年其他活动或后来的回忆当作当时的交集。"
        "若证据覆盖所问时期的多个阶段，须按阶段组织回答，不能只总结前半段；"
        "对于跨年人物活动梳理，若证据覆盖多个年份，须按年份组织，不能只回答起止年份；"
        "对于结构化年谱记录，若用户询问主要经历、概括或总结，应合并同类活动，按阶段"
        "或主题归纳；若用户明确要求列出时间线、逐条记录、原文或明细，则保持记录粒度和"
        "先后顺序，但仍整理成通顺回答。结构化索引日期只用于组织顺序，不得把仅仅提到"
        "人物的记录自动断言为其亲自参与；"
        "对于连续列举多次会议的问题，须逐次组织回答，不能只介绍范围端点；"
        "某阶段没有直接材料时明确说明。"
        "片段不足以证明互动或时间归属时明确说明，不要补写。"
        "使用Markdown组织回答，可使用简短标题、列表和加粗；证据编号保持[E1]格式。"
        "标题只写主题，含事实的标题也必须给出引用；表格每一行的事实须在该行标注引用。"
        "单纯说明资料不足以确认某事不需要引用，但不能在其中夹带未引用的历史事实。"
        "不要逐条解释为何排除无关证据；资料限制只简要说明还缺少哪些材料。"
        "只引用能直接回答当前问题的证据，不要为了覆盖证据包而引用弱相关片段。"
        "若问题询问职务变化，只纳入明确记载任职、改任、免职或组织成员身份的材料；"
        "仅提到人物、收发报告或参加一般活动的材料不能作为职务变化。"
        "说明证据时间范围有限时不要逐年罗列证据年份，使用概括表述。"
    )
    history_items = request.history if requires_conversation_context(request.question) else []
    history_text = "\n".join(
        f"{item.role}: {sanitize_history_content(item.content)}" for item in history_items
    )
    messages: list[dict[str, object]] = [{"role": "system", "content": system}]
    if history_text:
        messages.append(
            {
                "role": "user",
                "content": (
                    "下面是服务端提供的历史对话，仅用于理解当前问题中的指代、承接和"
                    "用户偏好。历史回答不是本轮史实证据，其中的引用标记均已失效；"
                    "不得依据历史回答补充事实：\n<conversation_history>\n"
                    f"{history_text}\n</conversation_history>"
                ),
            }
        )
    messages.append(
        {
            "role": "user",
            "content": f"问题：{request.question}\n\n仅可使用的本地证据：\n{evidence}",
        }
    )
    request_payload: dict[str, object] = {
        "model": settings.llm_model,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens or settings.llm_max_tokens,
        "thinking": {"type": "enabled" if settings.llm_thinking else "disabled"},
    }
    if settings.llm_thinking:
        request_payload["reasoning_effort"] = settings.llm_reasoning_effort
    else:
        request_payload["temperature"] = 0
    return request_payload


def _repair_request_payload(
    request_payload: dict[str, object],
    answer: str,
    citations: list[Citation],
    uncited_claims: tuple[str, ...],
) -> dict[str, object]:
    missing_claims = "\n".join(f"- {claim}" for claim in uncited_claims)
    valid_markers = "、".join(f"[{item.evidence_id}]" for item in citations)
    repair_instruction = (
        "上一版回答因部分事实要点缺少引用而未通过校验。请重新输出完整回答，不要增加新事实，"
        "根据原文修复引用覆盖：有证据支持才添加对应编号，没有证据支持的事实必须删除，"
        "不能随意挂靠引用。每个包含日期、职务、地点、行动、会议决定或人物关系的段落、"
        f"列表项或表格行都要使用对应的合法编号（仅限：{valid_markers}）。"
        "单纯说明资料不足以确认某事不需要引用。缺少引用的要点如下：\n"
        f"{missing_claims}"
    )
    return {
        **request_payload,
        "messages": [
            *cast(list[dict[str, object]], request_payload["messages"]),
            {"role": "assistant", "content": answer},
            {"role": "user", "content": repair_instruction},
        ],
    }


def _validated_llm_answer(
    *,
    settings: Settings,
    request_payload: dict[str, object],
    citations: list[Citation],
    runtime: LLMRuntime | None = None,
    budget: RequestBudget | None = None,
) -> LLMResult:
    first = _request_deepseek_completion(settings, request_payload, runtime, budget)
    if first.answer is None:
        return first
    validation = validate_grounded_answer(first.answer, citations)
    if validation.valid:
        return first
    if validation.error_code != "uncited_core_claim":
        return LLMResult(
            answer=None,
            error_code=validation.error_code,
            usage=first.usage,
            uncited_claims=validation.uncited_claims,
        )
    safe_first = _salvage_llm_result(
        first.answer, citations, validation.uncited_claims, first.usage
    )
    # A validated salvage is already safe to return. A second LLM round trip only
    # tries to recover removed prose and was the dominant latency in common cases.
    if safe_first is not None:
        return safe_first
    repair_payload = _repair_request_payload(
        request_payload, first.answer, citations, validation.uncited_claims
    )
    repaired = _request_deepseek_completion(settings, repair_payload, runtime, budget)
    combined_usage = _merge_usage(first.usage, repaired.usage)
    if repaired.answer is None:
        preferred = _prefer_llm_result(safe_first, None, combined_usage)
        if preferred is not None:
            return preferred
        return LLMResult(
            answer=None,
            error_code=f"citation_repair_{repaired.error_code}",
            usage=combined_usage,
        )
    repaired_validation = validate_grounded_answer(repaired.answer, citations)
    if repaired_validation.valid:
        preferred = _prefer_llm_result(
            safe_first, LLMResult(answer=repaired.answer), combined_usage
        )
        assert preferred is not None
        return preferred
    safe_repaired = None
    if repaired_validation.error_code == "uncited_core_claim":
        safe_repaired = _salvage_llm_result(
            repaired.answer,
            citations,
            repaired_validation.uncited_claims,
            combined_usage,
        )
    preferred = _prefer_llm_result(safe_first, safe_repaired, combined_usage)
    if preferred is not None:
        return preferred
    return LLMResult(
        answer=None,
        error_code=f"citation_repair_{repaired_validation.error_code}",
        usage=combined_usage,
        uncited_claims=repaired_validation.uncited_claims,
    )


def _llm_answer_direct(
    *,
    settings: Settings,
    request: QuestionRequest,
    citations: list[Citation],
    runtime: LLMRuntime | None = None,
    budget: RequestBudget | None = None,
    max_tokens: int | None = None,
) -> LLMResult:
    payload = _llm_request_payload(
        settings=settings,
        request=request,
        citations=citations,
        max_tokens=max_tokens,
    )
    return _validated_llm_answer(
        settings=settings,
        request_payload=payload,
        citations=citations,
        runtime=runtime,
        budget=budget,
    )


@dataclass(frozen=True)
class HierarchicalPreparation:
    request_payload: dict[str, object]
    usage: dict[str, int] | None
    map_failed: bool


def _prepare_hierarchical_answer(
    *,
    settings: Settings,
    request: QuestionRequest,
    citations: list[Citation],
    runtime: LLMRuntime | None,
    budget: RequestBudget | None,
) -> HierarchicalPreparation:
    batches = [
        citations[index : index + LLM_EVIDENCE_BATCH_SIZE]
        for index in range(0, len(citations), LLM_EVIDENCE_BATCH_SIZE)
    ]

    def summarize(item: tuple[int, list[Citation]]) -> tuple[str, LLMResult]:
        index, batch = item
        instruction = (
            f"{request.question}\n\n这是覆盖检索的第 {index + 1}/{len(batches)} 组证据。"
            "请只提炼本组能支持的局部结论，保留每项原始证据编号，控制在600字以内；"
            "不要因为这是局部证据就下完整性结论。"
        )
        partial = _llm_answer_direct(
            settings=settings,
            request=request.model_copy(update={"question": instruction, "history": []}),
            citations=batch,
            runtime=runtime,
            budget=budget,
            max_tokens=min(settings.llm_max_tokens, 1200),
        )
        if partial.answer:
            return partial.answer, partial
        fallback = "\n".join(
            f"[{citation.evidence_id}] {citation.quote}" for citation in batch
        )
        return fallback, partial

    workers = min(len(batches), 3)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="evidence-map") as executor:
        mapped = list(executor.map(summarize, enumerate(batches)))
    summaries = "\n\n".join(
        f"第 {index + 1} 组已核查摘要：\n{summary}"
        for index, (summary, _) in enumerate(mapped)
    )
    request_payload = _llm_request_payload(
        settings=settings,
        request=request,
        citations=citations,
        evidence_override=(
            "以下是分组证据经逐组引用校验后的摘要。请合并重复内容，检查各分组覆盖，"
            "形成完整回答；只能使用摘要中出现的原始证据编号。\n\n" + summaries
        ),
    )
    return HierarchicalPreparation(
        request_payload=request_payload,
        usage=_merge_usage(*(result.usage for _, result in mapped)),
        map_failed=any(result.answer is None for _, result in mapped),
    )


def _hierarchical_llm_answer(
    *,
    settings: Settings,
    request: QuestionRequest,
    citations: list[Citation],
    runtime: LLMRuntime | None,
    budget: RequestBudget | None,
) -> LLMResult:
    preparation = _prepare_hierarchical_answer(
        settings=settings,
        request=request,
        citations=citations,
        runtime=runtime,
        budget=budget,
    )
    final = _validated_llm_answer(
        settings=settings,
        request_payload=preparation.request_payload,
        citations=citations,
        runtime=runtime,
        budget=budget,
    )
    usage = _merge_usage(preparation.usage, final.usage)
    error_code = final.error_code
    if final.answer and preparation.map_failed and error_code is None:
        error_code = "hierarchical_partial_map_fallback"
    return LLMResult(
        answer=final.answer,
        error_code=error_code,
        usage=usage,
        uncited_claims=final.uncited_claims,
    )


def _llm_answer(
    *,
    settings: Settings,
    request: QuestionRequest,
    citations: list[Citation],
    runtime: LLMRuntime | None = None,
    budget: RequestBudget | None = None,
) -> LLMResult:
    if not settings.llm_enabled:
        return LLMResult(answer=None, error_code="not_configured")
    if len(citations) <= LLM_EVIDENCE_BATCH_SIZE:
        return _llm_answer_direct(
            settings=settings,
            request=request,
            citations=citations,
            runtime=runtime,
            budget=budget,
        )
    return _hierarchical_llm_answer(
        settings=settings,
        request=request,
        citations=citations,
        runtime=runtime,
        budget=budget,
    )


def check_deepseek_connection(settings: Settings) -> dict[str, object]:
    """Make a tiny non-thinking request without exposing the configured API key."""

    base = {
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "base_url": settings.llm_base_url,
        "configured": settings.llm_enabled,
    }
    if not settings.llm_enabled:
        return {**base, "status": "not_configured", "latency_ms": None}
    assert settings.llm_api_key is not None
    started = perf_counter()
    try:
        result = httpx.post(
            _chat_completions_url(settings.llm_base_url),
            headers={
                "Authorization": f"Bearer {settings.llm_api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.llm_model,
                "messages": [{"role": "user", "content": "只回答：OK"}],
                "stream": False,
                "max_tokens": 16,
                "thinking": {"type": "disabled"},
            },
            timeout=settings.llm_timeout_seconds,
        )
        result.raise_for_status()
        payload = result.json()
        content = str(payload["choices"][0]["message"]["content"]).strip()
    except httpx.HTTPError as exc:
        return {
            **base,
            "status": _deepseek_error_code(exc),
            "latency_ms": round((perf_counter() - started) * 1000),
        }
    except (KeyError, IndexError, TypeError, ValueError):
        return {
            **base,
            "status": "invalid_response",
            "latency_ms": round((perf_counter() - started) * 1000),
        }
    return {
        **base,
        "status": "ok",
        "latency_ms": round((perf_counter() - started) * 1000),
        "response": content,
    }


@dataclass(frozen=True)
class AnswerContext:
    retrieval: SearchResponse
    citations: list[Citation]
    evidence_status: Literal["supported", "partial", "no_evidence"]
    unsupported_entity: str | None
    reflection_status: Literal[
        "disabled", "skipped", "sufficient", "retried", "fallback"
    ] = "skipped"
    retrieval_rounds: int = 1
    missing_aspects: tuple[str, ...] = ()
    reflection_usage: dict[str, int] | None = None
    reflection_error_code: str | None = None


def _retrieval_limit(request: QuestionRequest, execution: QueryExecution) -> int:
    plan = execution.retrieval_plan
    if plan is None or plan.coverage == "relevance":
        return request.top_k
    if plan.coverage == "balanced_period":
        desired = max(request.top_k, 24)
    else:
        desired = max(request.top_k, len(execution.additional_queries) * 3)
    return min(MAX_COMPLEX_RETRIEVAL_CHUNKS, desired)


def _merge_retrieval_rounds(
    initial: SearchResponse,
    retry: SearchResponse,
    *,
    query: str,
    limit: int,
) -> SearchResponse:
    """Preserve first-pass ranking while appending genuinely new retry evidence."""

    hits: list[SearchHit] = []
    seen: set[str] = set()
    for hit in [*initial.hits, *retry.hits]:
        if hit.chunk_id in seen:
            continue
        seen.add(hit.chunk_id)
        hits.append(hit.model_copy(update={"rank": len(hits) + 1}))
        if len(hits) >= limit:
            break
    return initial.model_copy(
        update={
            "query": query,
            "query_terms": list(dict.fromkeys([*initial.query_terms, *retry.query_terms])),
            "hits": hits,
            "retrieval_mode": f"{initial.retrieval_mode}_refined",
            "degraded_components": sorted(
                {*initial.degraded_components, *retry.degraded_components}
            ),
            "coverage_gaps": list(
                dict.fromkeys([*initial.coverage_gaps, *retry.coverage_gaps])
            ),
        }
    )


def _retrieve_context(
    settings: Settings,
    request: QuestionRequest,
    planning: QueryPlanningResult | None = None,
    runtime: LLMRuntime | None = None,
    budget: RequestBudget | None = None,
) -> AnswerContext:
    execution = query_execution(
        request.question,
        planning.plan if planning is not None else None,
        settings.person_aliases_path,
    )
    retrieval_limit = _retrieval_limit(request, execution)
    retrieval = search_hybrid_index(
        keyword_index_path=settings.keyword_index_path,
        vector_index_path=settings.vector_index_path,
        model_cache_dir=settings.model_cache_dir / "fastembed",
        aliases_path=settings.person_aliases_path,
        query=execution.primary_query,
        top_k=retrieval_limit,
        plan=execution.retrieval_plan,
        additional_queries=list(execution.additional_queries),
    )
    retrieval = retrieval.model_copy(update={"query": request.question})
    reflection = assess_retrieval(
        settings,
        request,
        planning.plan if planning is not None else None,
        retrieval,
        runtime,
        budget,
    )
    reflection_status: Literal[
        "disabled", "skipped", "sufficient", "retried", "fallback"
    ] = reflection.status if reflection.status != "retry" else "retried"
    retrieval_rounds = 1
    reflection_error_code = reflection.error_code
    if reflection.status == "retry":
        try:
            retry_limit = min(12, max(6, len(reflection.followup_queries) * 4))
            retry = search_hybrid_index(
                keyword_index_path=settings.keyword_index_path,
                vector_index_path=settings.vector_index_path,
                model_cache_dir=settings.model_cache_dir / "fastembed",
                aliases_path=settings.person_aliases_path,
                query=reflection.followup_queries[0],
                top_k=retry_limit,
                plan=execution.retrieval_plan,
                additional_queries=list(reflection.followup_queries[1:]),
            )
            retrieval = _merge_retrieval_rounds(
                retrieval,
                retry,
                query=request.question,
                limit=min(
                    MAX_COMPLEX_RETRIEVAL_CHUNKS,
                    max(retrieval_limit, request.top_k + retry_limit),
                ),
            )
            retrieval_rounds = 2
        except RetrievalError:
            reflection_status = "fallback"
            reflection_error_code = "retry_retrieval_failed"
    keyword_backed = [hit for hit in retrieval.hits if hit.keyword_rank is not None]
    unsupported_entity = _unsupported_leading_entity(request.question, retrieval.hits)
    keyword_unavailable = "keyword" in retrieval.degraded_components
    if (not keyword_backed and not keyword_unavailable) or unsupported_entity:
        citations: list[Citation] = []
        evidence_status: Literal["supported", "partial", "no_evidence"] = "no_evidence"
    else:
        citations = _citations(retrieval)
        evidence_status = (
            "supported"
            if retrieval.query_intent != "intersection"
            and not retrieval.degraded_components
            and not retrieval.coverage_gaps
            and any(
                hit.keyword_rank is not None and hit.vector_rank is not None
                for hit in retrieval.hits[:5]
            )
            else "partial"
        )
    return AnswerContext(
        retrieval,
        citations,
        evidence_status,
        unsupported_entity,
        reflection_status,
        retrieval_rounds,
        tuple(reflection.assessment.missing_aspects) if reflection.assessment else (),
        reflection.usage,
        reflection_error_code,
    )


def _finish_answer(
    settings: Settings,
    request: QuestionRequest,
    context: AnswerContext,
    llm_result: LLMResult,
    planning: QueryPlanningResult | None = None,
) -> AnswerResponse:
    retrieval, retrieved_citations = context.retrieval, context.citations
    unsupported_entity = context.unsupported_entity
    generator_mode: Literal["extractive", "llm"] = "llm" if llm_result.answer else "extractive"
    answer = llm_result.answer or _extractive_answer(retrieval.query_intent, retrieved_citations)
    citations = _citations_used_by_answer(answer, retrieved_citations)
    limitations = []
    if not retrieved_citations:
        reason = (
            f"检索片段中没有出现问题人物“{unsupported_entity}”。"
            if unsupported_entity
            else "未找到同时得到关键词检索支持的本地证据。"
        )
        limitations.append(f"{reason}系统没有用语义近似结果强行作答。")
    elif not settings.llm_enabled:
        limitations.append(
            "当前未配置生成模型，返回的是证据摘录式答案；配置兼容接口后可生成综合回答。"
        )
    elif llm_result.answer is None:
        if llm_result.error_code in {"uncited_core_claim", "citation_repair_uncited_core_claim"}:
            limitations.append(
                "生成回答仍有事实语句缺少引用，已改为展示证据摘录；"
                "可展开“哪些语句缺少引用”查看原因。"
            )
        else:
            limitations.append(
                f"DeepSeek 生成未通过（{llm_result.error_code}），已安全降级为证据摘录。"
            )
    elif llm_result.error_code == "removed_uncited_claims":
        limitations.append(
            "生成草稿中的未引用段落已移除，其余内容已通过引用核查；可展开“哪些草稿内容已移除”查看。"
        )
    elif llm_result.error_code == "hierarchical_partial_map_fallback":
        limitations.append(
            "部分证据分组未能生成局部摘要，最终综合时已改用该组原文摘录。"
        )
    if retrieved_citations:
        if retrieval.degraded_components:
            limitations.append(
                "检索已降级："
                + "、".join(retrieval.degraded_components)
                + " 分支暂不可用，当前结果可能不完整。"
            )
        if retrieval.coverage_gaps:
            limitations.append(
                f"覆盖计划中有 {len(retrieval.coverage_gaps)} 个检索分组未找到候选证据，"
                "对应部分已作为资料缺口保留。"
            )
        if len(retrieved_citations) > LLM_EVIDENCE_BATCH_SIZE:
            limitations.append(
                f"本题使用 {len(retrieved_citations)} 条证据候选执行了分组摘要和最终综合，"
                "以避免单次 Top-K 截断跨阶段材料。"
            )
        limitations.append("答案仅代表当前已入库文献的检索结果，不等同于完整历史结论。")
        if retrieval.query_intent == "intersection":
            limitations.append(
                "检索片段是交集研究线索；两人的具体互动须由原文支持，不能仅凭人名共现确认。"
            )
        if retrieval.query_year_range and not retrieval.query_years:
            start_year, end_year = retrieval.query_year_range
            limitations.append(
                f"时间条件按 {start_year}—{end_year} 年范围召回资料；"
                "这不是精确起止日期，具体活动的时期归属须结合原文核对。"
            )
    if planning is not None and planning.status == "fallback":
        limitations.append(
            f"查询理解模型未通过（{planning.error_code}），本次已使用用户原问题检索。"
        )
    if context.reflection_status == "retried":
        aspects = "、".join(context.missing_aspects)
        limitations.append(
            f"首轮证据评估发现仍缺少{aspects or '部分核心方面'}，"
            "已执行一轮定向补充检索。"
        )
    elif context.reflection_status == "fallback":
        limitations.append(
            "证据充分性评估或补充检索未通过，已保留首轮检索结果继续回答。"
        )
    return AnswerResponse(
        question=request.question,
        answer=answer,
        evidence_status=context.evidence_status,
        generator_mode=generator_mode,
        llm_status=(
            "not_applicable"
            if not citations
            else "disabled"
            if not settings.llm_enabled
            else "used"
            if llm_result.answer
            else "fallback"
        ),
        model_name=settings.llm_model if settings.llm_enabled else None,
        llm_usage=llm_result.usage,
        llm_error_code=llm_result.error_code if citations and settings.llm_enabled else None,
        uncited_claims=list(llm_result.uncited_claims),
        query_plan=planning.plan if planning is not None else None,
        query_planner_status=planning.status if planning is not None else "not_applicable",
        query_planner_model=planning.model_name if planning is not None else None,
        query_planner_usage=planning.usage if planning is not None else None,
        query_planner_error_code=planning.error_code if planning is not None else None,
        retrieval_mode=retrieval.retrieval_mode,
        query_intent=retrieval.query_intent,
        retrieval_reflection_status=context.reflection_status,
        retrieval_rounds=context.retrieval_rounds,
        retrieval_missing_aspects=list(context.missing_aspects),
        retrieval_reflection_usage=context.reflection_usage,
        retrieval_reflection_error_code=context.reflection_error_code,
        citations=citations,
        retrieved_evidence_count=len(retrieved_citations),
        limitations=limitations,
    )


def _finish_structured_answer(
    settings: Settings,
    structured: AnswerResponse,
    llm_result: LLMResult,
) -> AnswerResponse:
    limitations = list(structured.limitations)
    if not settings.llm_enabled:
        limitations.append(
            "当前未配置生成模型，返回的是结构化记录摘录；配置兼容接口后可生成综合回答。"
        )
    elif llm_result.answer is None:
        limitations.append(
            f"DeepSeek 生成未通过（{llm_result.error_code}），已安全降级为结构化记录摘录。"
        )
    elif llm_result.error_code == "removed_uncited_claims":
        limitations.append("生成草稿中的未引用段落已移除，其余内容已通过引用核查。")
    elif llm_result.error_code == "hierarchical_partial_map_fallback":
        limitations.append("部分证据分组未生成局部摘要，最终综合时已改用该组原文摘录。")
    if len(structured.citations) > LLM_EVIDENCE_BATCH_SIZE:
        limitations.append(
            f"本题使用 {len(structured.citations)} 条结构化证据执行了分组摘要和最终综合。"
        )
    answer = llm_result.answer or structured.answer
    citations = _citations_used_by_answer(answer, structured.citations)
    return structured.model_copy(
        update={
            "answer": answer,
            "generator_mode": "llm" if llm_result.answer else "extractive",
            "llm_status": (
                "disabled"
                if not settings.llm_enabled
                else "used"
                if llm_result.answer
                else "fallback"
            ),
            "model_name": settings.llm_model if settings.llm_enabled else None,
            "llm_usage": llm_result.usage,
            "llm_error_code": llm_result.error_code if settings.llm_enabled else None,
            "uncited_claims": list(llm_result.uncited_claims),
            "citations": citations,
            "retrieved_evidence_count": len(structured.citations),
            "limitations": limitations,
        }
    )


def _clarification_response(
    request: QuestionRequest, planning: QueryPlanningResult
) -> AnswerResponse:
    assert planning.plan is not None
    assert planning.plan.clarification_question is not None
    return AnswerResponse(
        question=request.question,
        answer=planning.plan.clarification_question,
        evidence_status="no_evidence",
        generator_mode="extractive",
        llm_status="not_applicable",
        query_plan=planning.plan,
        query_planner_status=planning.status,
        query_planner_model=planning.model_name,
        query_planner_usage=planning.usage,
        query_planner_error_code=planning.error_code,
        retrieval_mode="query_clarification",
        query_intent=planning.plan.intent,
        citations=[],
        limitations=["问题存在会改变检索范围的歧义，澄清后再查询可避免丢弃限制条件。"],
    )


def answer_question(
    settings: Settings,
    request: QuestionRequest,
    runtime: LLMRuntime | None = None,
    budget: RequestBudget | None = None,
) -> AnswerResponse:
    budget = budget or RequestBudget.start(settings.request_timeout_seconds)
    full_text = answer_full_text_question(settings, request)
    if full_text is not None:
        return full_text
    structured = answer_structured_question(settings, request)
    if structured is not None:
        if requires_structured_generation(structured):
            llm_result = _llm_answer(
                settings=settings,
                request=request,
                citations=structured.citations,
                runtime=runtime,
                budget=budget,
            )
            return _finish_structured_answer(settings, structured, llm_result)
        return structured
    planning = plan_question(settings, request, runtime, budget)
    if planning.plan is not None and planning.plan.needs_clarification:
        return _clarification_response(request, planning)
    context = _retrieve_context(settings, request, planning, runtime, budget)
    llm_result = (
        _llm_answer(
            settings=settings,
            request=request,
            citations=context.citations,
            runtime=runtime,
            budget=budget,
        )
        if context.citations
        else LLMResult(answer=None, error_code="no_evidence")
    )
    return _finish_answer(settings, request, context, llm_result, planning)
