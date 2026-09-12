"""Incremental answers with provisional text and an authoritative validated final result."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from starlette.concurrency import run_in_threadpool

from history_agent.answering.full_text import answer_full_text_question
from history_agent.answering.llamaindex_llm import (
    DeepSeekLlamaIndexLLM,
    chat_messages,
    response_usage,
)
from history_agent.answering.models import Citation, QuestionRequest
from history_agent.answering.query_understanding import plan_question
from history_agent.answering.retrieval_reflection import should_reflect
from history_agent.answering.runtime import LLMRuntime, RequestBudget
from history_agent.answering.service import (
    LLM_EVIDENCE_BATCH_SIZE,
    LLMResult,
    _aretrieve_context,
    _attach_query_planning,
    _citation_date_repair_payload,
    _clarification_response,
    _deepseek_error_code,
    _finish_answer,
    _finish_structured_answer,
    _has_ambiguous_timeline_years,
    _llm_request_payload,
    _merge_usage,
    _prefer_llm_result,
    _prepare_hierarchical_answer,
    _repair_request_payload,
    _salvage_llm_result,
    _timeline_year_repair_payload,
)
from history_agent.answering.structured import (
    answer_structured_question,
    requires_structured_generation,
)
from history_agent.answering.validation import validate_grounded_answer
from history_agent.config import Settings


@dataclass(frozen=True)
class AnswerStreamEvent:
    event: Literal["status", "delta", "reset", "done", "error"]
    data: dict[str, Any]

    def encode(self) -> str:
        return f"event: {self.event}\ndata: {json.dumps(self.data, ensure_ascii=False)}\n\n"


async def _stream_completion(
    settings: Settings,
    request_payload: dict[str, object],
    runtime: LLMRuntime | None = None,
    budget: RequestBudget | None = None,
) -> AsyncGenerator[str | LLMResult, None]:
    assert settings.llm_api_key is not None
    parts: list[str] = []
    usage: dict[str, int] | None = None
    finish_reason: str | None = None
    finished = False
    try:
        timeout = (
            budget.timeout(settings.llm_timeout_seconds)
            if budget is not None
            else settings.llm_timeout_seconds
        )
        llm = DeepSeekLlamaIndexLLM(
            settings=settings,
            runtime=runtime,
            timeout_seconds=timeout,
        )
        stream = llm.raw_astream_chat(
            chat_messages(request_payload),
            request_payload=request_payload,
        )
        async with aclosing(stream):
            async for response in stream:
                usage = response_usage(response) or usage
                if response.additional_kwargs.get("done"):
                    finished = True
                    break
                content = response.delta
                if content:
                    parts.append(content)
                    yield content
                raw_finish_reason = response.additional_kwargs.get("finish_reason")
                if raw_finish_reason is not None:
                    finish_reason = str(raw_finish_reason)
    except httpx.HTTPError as exc:
        yield LLMResult(answer=None, error_code=_deepseek_error_code(exc), usage=usage)
        return
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        yield LLMResult(answer=None, error_code="invalid_response", usage=usage)
        return
    if finish_reason == "length":
        yield LLMResult(answer=None, error_code="max_tokens_exhausted", usage=usage)
    elif not finished or finish_reason != "stop":
        yield LLMResult(answer=None, error_code="incomplete_stream", usage=usage)
    else:
        answer = "".join(parts).strip()
        yield LLMResult(
            answer=answer or None, error_code=None if answer else "empty_response", usage=usage
        )


async def _stream_llm_answer(
    settings: Settings,
    request: QuestionRequest,
    citations: list[Citation],
    runtime: LLMRuntime | None = None,
    budget: RequestBudget | None = None,
    *,
    request_payload: dict[str, object] | None = None,
    initial_usage: dict[str, int] | None = None,
) -> AsyncGenerator[AnswerStreamEvent | LLMResult, None]:
    payload = request_payload or _llm_request_payload(
        settings=settings, request=request, citations=citations
    )
    usage = initial_usage
    safe_first: LLMResult | None = None
    for attempt in range(2):
        result = LLMResult(answer=None, error_code="incomplete_stream")
        # Closing nested iterators releases the upstream HTTP connection on cancellation.
        async with aclosing(_stream_completion(settings, payload, runtime, budget)) as completion:
            async for item in completion:
                if isinstance(item, str):
                    # Keep the first draft visible while a repair completion runs in
                    # the background. Re-streaming the repair makes the answer clear
                    # itself and type out again before the authoritative final event.
                    if attempt == 0:
                        yield AnswerStreamEvent("delta", {"text": item})
                else:
                    result = item
        usage = _merge_usage(usage, result.usage)
        prefix = "citation_repair_" if attempt else ""
        if result.answer is None:
            preferred = _prefer_llm_result(safe_first, None, usage)
            if preferred is not None:
                yield preferred
                return
            yield LLMResult(answer=None, error_code=f"{prefix}{result.error_code}", usage=usage)
            return
        yield AnswerStreamEvent("status", {"message": "正在核查引用…"})
        validation = validate_grounded_answer(result.answer, citations)
        if validation.valid:
            if _has_ambiguous_timeline_years(request, result.answer):
                if attempt:
                    yield LLMResult(
                        answer=None,
                        error_code="timeline_year_repair_ambiguous_timeline_years",
                        usage=usage,
                    )
                    return
                payload = _timeline_year_repair_payload(payload, result.answer)
                yield AnswerStreamEvent("status", {"message": "正在补全跨年时间点的年份…"})
                continue
            preferred = _prefer_llm_result(safe_first, LLMResult(answer=result.answer), usage)
            assert preferred is not None
            yield preferred
            return
        if validation.error_code == "citation_date_mismatch":
            if attempt:
                yield LLMResult(
                    answer=None,
                    error_code="citation_date_repair_citation_date_mismatch",
                    usage=usage,
                )
                return
            payload = _citation_date_repair_payload(
                payload, result.answer, validation.date_mismatches
            )
            yield AnswerStreamEvent("status", {"message": "正在校正与证据不一致的日期…"})
            continue
        if validation.error_code == "uncited_core_claim":
            salvaged = _salvage_llm_result(
                result.answer, citations, validation.uncited_claims, usage
            )
            if salvaged is not None and _has_ambiguous_timeline_years(
                request, salvaged.answer or ""
            ):
                if attempt:
                    salvaged = None
                else:
                    payload = _timeline_year_repair_payload(
                        payload, salvaged.answer or result.answer
                    )
                    yield AnswerStreamEvent(
                        "status", {"message": "正在补全跨年时间点的年份…"}
                    )
                    continue
            if attempt:
                preferred = _prefer_llm_result(safe_first, salvaged, usage)
                if preferred is not None:
                    yield preferred
                    return
            elif salvaged is not None:
                # The first draft is already safe after removing rejected blocks;
                # do not add a second upstream round trip merely to recover prose.
                yield salvaged
                return
            else:
                safe_first = salvaged
        if attempt or validation.error_code != "uncited_core_claim":
            preferred = _prefer_llm_result(safe_first, None, usage)
            if preferred is not None:
                yield preferred
                return
            yield LLMResult(
                answer=None,
                error_code=f"{prefix}{validation.error_code}",
                usage=usage,
                uncited_claims=validation.uncited_claims,
            )
            return
        payload = _repair_request_payload(
            payload, result.answer, citations, validation.uncited_claims
        )
        yield AnswerStreamEvent("status", {"message": "正在后台补全引用…"})


async def _stream_hierarchical_llm_answer(
    settings: Settings,
    request: QuestionRequest,
    citations: list[Citation],
    runtime: LLMRuntime | None,
    budget: RequestBudget,
) -> AsyncGenerator[AnswerStreamEvent | LLMResult, None]:
    preparation = await run_in_threadpool(
        _prepare_hierarchical_answer,
        settings=settings,
        request=request,
        citations=citations,
        runtime=runtime,
        budget=budget,
    )
    async with aclosing(
        _stream_llm_answer(
            settings,
            request,
            citations,
            runtime,
            budget,
            request_payload=preparation.request_payload,
            initial_usage=preparation.usage,
        )
    ) as generation:
        async for item in generation:
            if (
                isinstance(item, LLMResult)
                and item.answer
                and preparation.map_failed
                and item.error_code is None
            ):
                yield LLMResult(
                    answer=item.answer,
                    error_code="hierarchical_partial_map_fallback",
                    usage=item.usage,
                    uncited_claims=item.uncited_claims,
                )
            else:
                yield item


async def stream_answer_question(
    settings: Settings,
    request: QuestionRequest,
    runtime: LLMRuntime | None = None,
    budget: RequestBudget | None = None,
) -> AsyncGenerator[AnswerStreamEvent, None]:
    budget = budget or RequestBudget.start(settings.request_timeout_seconds)
    yield AnswerStreamEvent("status", {"message": "正在分析问题…"})
    if settings.llm_query_planning and settings.llm_enabled:
        yield AnswerStreamEvent("status", {"message": "正在理解问题并规划检索…"})
    planning = await run_in_threadpool(plan_question, settings, request, runtime, budget)
    if planning.plan is not None and planning.plan.needs_clarification:
        yield AnswerStreamEvent("done", _clarification_response(request, planning).model_dump())
        return
    full_text = await run_in_threadpool(answer_full_text_question, settings, request)
    if full_text is not None:
        yield AnswerStreamEvent("done", _attach_query_planning(full_text, planning).model_dump())
        return
    structured = await run_in_threadpool(
        answer_structured_question, settings, request, planning.plan
    )
    if structured is not None:
        structured = _attach_query_planning(structured, planning)
        if requires_structured_generation(structured):
            result = LLMResult(answer=None, error_code="not_configured")
            if settings.llm_enabled and len(structured.citations) > LLM_EVIDENCE_BATCH_SIZE:
                yield AnswerStreamEvent("status", {"message": "正在分组归纳结构化史料…"})
                async with aclosing(
                    _stream_hierarchical_llm_answer(
                        settings, request, structured.citations, runtime, budget
                    )
                ) as generation:
                    async for item in generation:
                        if isinstance(item, LLMResult):
                            result = item
                        else:
                            yield item
            elif settings.llm_enabled:
                yield AnswerStreamEvent("status", {"message": "正在归纳结构化史料…"})
                async with aclosing(
                    _stream_llm_answer(settings, request, structured.citations, runtime, budget)
                ) as generation:
                    async for item in generation:
                        if isinstance(item, LLMResult):
                            result = item
                        else:
                            yield item
            final = _finish_structured_answer(settings, structured, result)
            yield AnswerStreamEvent("done", final.model_dump())
            return
        yield AnswerStreamEvent("done", structured.model_dump())
        return
    retrieval_status = (
        "正在检索并检查证据覆盖…"
        if should_reflect(settings, request, planning.plan)
        else "正在检索本地史料…"
    )
    yield AnswerStreamEvent("status", {"message": retrieval_status})
    context = await _aretrieve_context(settings, request, planning, runtime, budget)
    if context.retrieval_rounds > 1:
        yield AnswerStreamEvent("status", {"message": "已针对证据缺口完成补充检索…"})
    result = LLMResult(answer=None, error_code="not_configured")
    if not context.citations:
        result = LLMResult(answer=None, error_code="no_evidence")
    elif settings.llm_enabled and len(context.citations) > LLM_EVIDENCE_BATCH_SIZE:
        yield AnswerStreamEvent("status", {"message": "正在分组归纳跨阶段证据…"})
        async with aclosing(
            _stream_hierarchical_llm_answer(settings, request, context.citations, runtime, budget)
        ) as generation:
            async for item in generation:
                if isinstance(item, LLMResult):
                    result = item
                else:
                    yield item
    elif settings.llm_enabled:
        yield AnswerStreamEvent("status", {"message": "正在生成，引用待核查…"})
        async with aclosing(
            _stream_llm_answer(settings, request, context.citations, runtime, budget)
        ) as generation:
            async for item in generation:
                if isinstance(item, LLMResult):
                    result = item
                else:
                    yield item
    final = _finish_answer(settings, request, context, result, planning)
    # Always replace provisional text, including repaired answers and safe fallbacks.
    yield AnswerStreamEvent("done", final.model_dump())
