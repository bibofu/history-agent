from __future__ import annotations

import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import httpx

from history_agent.answering.models import AnswerResponse, Citation, QuestionRequest
from history_agent.config import Settings
from history_agent.retrieval.hybrid import search_hybrid_index
from history_agent.retrieval.models import SearchHit

WHITESPACE = re.compile(r"\s+")
SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？；])")
EVIDENCE_MARKER = re.compile(r"\[(E\d+)\]")
LEADING_ENTITY = re.compile(
    r"^(?:请问|我想知道|想知道|帮我查)?(?P<entity>[\u3400-\u4dbf\u4e00-\u9fff·]{2,8})"
    r"(?:在|于)(?=(?:18|19|20)\d{2}年)"
)


def _compact(text: str) -> str:
    return WHITESPACE.sub(" ", text).strip()


def _quote_for_hit(hit: SearchHit, query_terms: list[str], limit: int = 280) -> str:
    text = _compact(hit.text)
    if len(text) <= limit:
        return text
    positions = [text.find(term) for term in query_terms if len(term) >= 2]
    positions = [position for position in positions if position >= 0]
    center = min(positions) if positions else 0
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
    entity = match.group("entity")
    if any(
        entity in hit.title
        or entity in hit.text
        or any(entity in part for part in hit.section_path)
        for hit in hits
    ):
        return None
    return entity


def _citations(response: Any) -> list[Citation]:
    return [
        Citation(
            evidence_id=f"E{index}",
            document_id=hit.document_id,
            document=hit.title,
            volume=hit.volume,
            pdf_page=hit.pdf_page_start,
            section=hit.section_path,
            quote=_quote_for_hit(hit, response.query_terms),
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
        "intersection": "根据当前本地资料，两位人物的交集可从以下共同事件核查：",
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


def _llm_answer(
    *, settings: Settings, request: QuestionRequest, citations: list[Citation]
) -> LLMResult:
    if not settings.llm_enabled:
        return LLMResult(answer=None, error_code="not_configured")
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
        "出现一个本次证据编号；引用格式只能是[E1]、[E2]这种形式。"
    )
    history = [item.model_dump() for item in request.history[-6:]]
    messages = [
        {"role": "system", "content": system},
        *history,
        {
            "role": "user",
            "content": f"问题：{request.question}\n\n仅可使用的本地证据：\n{evidence}",
        },
    ]
    assert settings.llm_api_key is not None
    request_payload: dict[str, object] = {
        "model": settings.llm_model,
        "messages": messages,
        "stream": False,
        "max_tokens": settings.llm_max_tokens,
        "thinking": {"type": "enabled" if settings.llm_thinking else "disabled"},
    }
    if settings.llm_thinking:
        request_payload["reasoning_effort"] = settings.llm_reasoning_effort
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
    markers = set(EVIDENCE_MARKER.findall(answer))
    allowed_markers = {item.evidence_id for item in citations}
    if not markers:
        return LLMResult(answer=None, error_code="missing_evidence_markers", usage=usage)
    if not markers.issubset(allowed_markers):
        return LLMResult(answer=None, error_code="invalid_evidence_marker", usage=usage)
    return LLMResult(answer=answer, usage=usage)


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


def answer_question(settings: Settings, request: QuestionRequest) -> AnswerResponse:
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
            if any(
                hit.keyword_rank is not None and hit.vector_rank is not None
                for hit in retrieval.hits[:5]
            )
            else "partial"
        )
    llm_result = (
        _llm_answer(settings=settings, request=request, citations=citations)
        if citations
        else LLMResult(answer=None, error_code="no_evidence")
    )
    generator_mode: Literal["extractive", "llm"] = (
        "llm" if llm_result.answer else "extractive"
    )
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
        limitations.append(
            f"DeepSeek 生成未通过（{llm_result.error_code}），已安全降级为证据摘录。"
        )
    if citations:
        limitations.append("答案仅代表当前已入库文献的检索结果，不等同于完整历史结论。")
    return AnswerResponse(
        question=request.question,
        answer=answer,
        evidence_status=evidence_status,
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
        retrieval_mode=retrieval.retrieval_mode,
        query_intent=retrieval.query_intent,
        citations=citations,
        limitations=limitations,
    )
