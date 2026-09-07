from __future__ import annotations

import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, cast

import httpx

from history_agent.answering.models import AnswerResponse, Citation, QuestionRequest
from history_agent.answering.structured import answer_structured_question
from history_agent.answering.validation import remove_uncited_claim_blocks, validate_grounded_answer
from history_agent.config import Settings
from history_agent.retrieval.hybrid import search_hybrid_index
from history_agent.retrieval.models import SearchHit, SearchResponse

WHITESPACE = re.compile(r"\s+")
SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？；])")
LEADING_ENTITY = re.compile(
    r"^(?:请问|我想知道|想知道|帮我查)?(?P<entity>[\u3400-\u4dbf\u4e00-\u9fff·]{2,18})"
    r"(?:在|于)(?=(?:18|19|20)\d{2}年)"
)
ENTITY_SEPARATOR = re.compile(r"[、和与]")
PROMPT_VERSION = "grounded-answer-v9"


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
    if len(text) <= limit:
        return text
    positions = [text.find(term) for term in query_terms if len(term) >= 2]
    positions = [position for position in positions if position >= 0]
    center = min(positions) if positions else 0
    people = set(query_people or [])
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
                query_people=response.query_people
                if response.query_intent == "intersection"
                else [],
            ),
            source_type=hit.source_type,
            verification_status=hit.verification_status,
            extraction_methods=hit.extraction_methods,
        )
        for index, hit in enumerate(response.hits, start=1)
    ]


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
    settings: Settings, request_payload: dict[str, object]
) -> LLMResult:
    assert settings.llm_api_key is not None
    try:
        result = httpx.post(
            _chat_completions_url(settings.llm_base_url),
            headers={
                "Authorization": f"Bearer {settings.llm_api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
            json=request_payload,
            timeout=settings.llm_timeout_seconds,
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
    *, settings: Settings, request: QuestionRequest, citations: list[Citation]
) -> dict[str, object]:
    evidence = "\n\n".join(
        (
            f"[{item.evidence_id}] 《{item.document}》PDF第{item.pdf_page}页"
            f"；章节：{' > '.join(item.section) or '未识别'}\n{item.quote}"
        )
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
        "片段不足以证明互动或时间归属时明确说明，不要补写。"
        "使用Markdown组织回答，可使用简短标题、列表和加粗；证据编号保持[E1]格式。"
        "标题只写主题，含事实的标题也必须给出引用；表格每一行的事实须在该行标注引用。"
        "单纯说明资料不足以确认某事不需要引用，但不能在其中夹带未引用的历史事实。"
        "不要逐条解释为何排除无关证据；资料限制只简要说明还缺少哪些材料。"
        "说明证据时间范围有限时不要逐年罗列证据年份，使用概括表述。"
    )
    history = [item.model_dump() for item in request.history[-6:]]
    messages: list[dict[str, object]] = [
        {"role": "system", "content": system},
        *history,
        {
            "role": "user",
            "content": f"问题：{request.question}\n\n仅可使用的本地证据：\n{evidence}",
        },
    ]
    request_payload: dict[str, object] = {
        "model": settings.llm_model,
        "messages": messages,
        "stream": False,
        "max_tokens": settings.llm_max_tokens,
        "thinking": {"type": "enabled" if settings.llm_thinking else "disabled"},
    }
    if settings.llm_thinking:
        request_payload["reasoning_effort"] = settings.llm_reasoning_effort
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


def _llm_answer(
    *, settings: Settings, request: QuestionRequest, citations: list[Citation]
) -> LLMResult:
    if not settings.llm_enabled:
        return LLMResult(answer=None, error_code="not_configured")
    request_payload = _llm_request_payload(settings=settings, request=request, citations=citations)
    first = _request_deepseek_completion(settings, request_payload)
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
    repair_payload = _repair_request_payload(
        request_payload, first.answer, citations, validation.uncited_claims
    )
    repaired = _request_deepseek_completion(settings, repair_payload)
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


def _retrieve_context(settings: Settings, request: QuestionRequest) -> AnswerContext:
    retrieval = search_hybrid_index(
        keyword_index_path=settings.keyword_index_path,
        vector_index_path=settings.vector_index_path,
        model_cache_dir=settings.model_cache_dir / "fastembed",
        aliases_path=settings.person_aliases_path,
        query=request.question,
        top_k=request.top_k,
    )
    keyword_backed = [hit for hit in retrieval.hits if hit.keyword_rank is not None]
    unsupported_entity = _unsupported_leading_entity(request.question, retrieval.hits)
    if not keyword_backed or unsupported_entity:
        citations: list[Citation] = []
        evidence_status: Literal["supported", "partial", "no_evidence"] = "no_evidence"
    else:
        citations = _citations(retrieval)
        evidence_status = (
            "supported"
            if retrieval.query_intent != "intersection"
            and any(
                hit.keyword_rank is not None and hit.vector_rank is not None
                for hit in retrieval.hits[:5]
            )
            else "partial"
        )
    return AnswerContext(retrieval, citations, evidence_status, unsupported_entity)


def _finish_answer(
    settings: Settings, request: QuestionRequest, context: AnswerContext, llm_result: LLMResult
) -> AnswerResponse:
    retrieval, citations = context.retrieval, context.citations
    unsupported_entity = context.unsupported_entity
    generator_mode: Literal["extractive", "llm"] = "llm" if llm_result.answer else "extractive"
    answer = llm_result.answer or _extractive_answer(retrieval.query_intent, citations)
    limitations = []
    if not citations:
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
            "生成草稿中的未引用段落已移除，其余内容已通过引用核查；"
            "可展开“哪些草稿内容已移除”查看。"
        )
    if citations:
        limitations.append("答案仅代表当前已入库文献的检索结果，不等同于完整历史结论。")
        if retrieval.query_intent == "intersection":
            limitations.append(
                "检索片段是交集研究线索；两人的具体互动须由原文支持，不能仅凭人名共现确认。"
            )
        if retrieval.query_year_range and not retrieval.query_years:
            start_year, end_year = retrieval.query_year_range
            limitations.append(
                f"时期名称按 {start_year}—{end_year} 年范围召回资料；"
                "这不是精确起止日期，具体活动的时期归属须结合原文核对。"
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
        retrieval_mode=retrieval.retrieval_mode,
        query_intent=retrieval.query_intent,
        citations=citations,
        limitations=limitations,
    )


def answer_question(settings: Settings, request: QuestionRequest) -> AnswerResponse:
    structured = answer_structured_question(settings, request)
    if structured is not None:
        return structured
    context = _retrieve_context(settings, request)
    llm_result = (
        _llm_answer(settings=settings, request=request, citations=context.citations)
        if context.citations
        else LLMResult(answer=None, error_code="no_evidence")
    )
    return _finish_answer(settings, request, context, llm_result)
