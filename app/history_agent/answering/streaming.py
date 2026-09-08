"""Incremental answers with provisional text and an authoritative validated final result."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from starlette.concurrency import run_in_threadpool

from history_agent.answering.models import Citation, QuestionRequest
from history_agent.answering.query_understanding import plan_question
from history_agent.answering.runtime import LLMRuntime, RequestBudget
from history_agent.answering.service import (
    LLMResult,
    _chat_completions_url,
    _clarification_response,
    _deepseek_error_code,
    _finish_answer,
    _llm_request_payload,
    _merge_usage,
    _prefer_llm_result,
    _repair_request_payload,
    _retrieve_context,
    _salvage_llm_result,
)
from history_agent.answering.structured import answer_structured_question
from history_agent.answering.validation import validate_grounded_answer
from history_agent.config import Settings


@dataclass(frozen=True)
class AnswerStreamEvent:
    event: Literal["status", "delta", "reset", "done", "error"]
    data: dict[str, Any]

    def encode(self) -> str:
        return f"event: {self.event}\ndata: {json.dumps(self.data, ensure_ascii=False)}\n\n"


async def _sse_data(response: httpx.Response) -> AsyncIterator[str]:
    data: list[str] = []
    async for line in response.aiter_lines():
        if not line:
            if data:
                yield "\n".join(data)
                data.clear()
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    if data:
        yield "\n".join(data)


@asynccontextmanager
async def _ephemeral_stream(
    settings: Settings,
    request_payload: dict[str, object],
    timeout: float,
) -> AsyncIterator[httpx.Response]:
    assert settings.llm_api_key is not None
    async with (
        httpx.AsyncClient(timeout=timeout) as client,
        client.stream(
            "POST",
            _chat_completions_url(settings.llm_base_url),
            headers={
                "Authorization": f"Bearer {settings.llm_api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
            json={**request_payload, "stream": True, "stream_options": {"include_usage": True}},
        ) as response,
    ):
        yield response


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
        stream = (
            runtime.stream(
                "POST",
                _chat_completions_url(settings.llm_base_url),
                timeout=timeout,
                headers={
                    "Authorization": f"Bearer {settings.llm_api_key.get_secret_value()}",
                    "Content-Type": "application/json",
                },
                json={**request_payload, "stream": True, "stream_options": {"include_usage": True}},
            )
            if runtime is not None
            else _ephemeral_stream(settings, request_payload, timeout)
        )
        async with stream as response:
            response.raise_for_status()
            async for data in _sse_data(response):
                if data == "[DONE]":
                    finished = True
                    break
                payload = json.loads(data)
                if "error" in payload:
                    raise ValueError("upstream error")
                raw_usage = payload.get("usage")
                if isinstance(raw_usage, dict):
                    usage = {
                        key: int(raw_usage[key])
                        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                        if key in raw_usage
                    }
                choices = payload["choices"]
                if not choices:  # Also accept compatible providers' usage-only chunk.
                    continue
                choice = choices[0]
                delta = choice.get("delta", {})
                content = delta.get("content")
                if content is not None and not isinstance(content, str):
                    raise ValueError("invalid content delta")
                if content:
                    parts.append(content)
                    yield content
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
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
) -> AsyncGenerator[AnswerStreamEvent | LLMResult, None]:
    payload = _llm_request_payload(settings=settings, request=request, citations=citations)
    usage: dict[str, int] | None = None
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
            preferred = _prefer_llm_result(safe_first, LLMResult(answer=result.answer), usage)
            assert preferred is not None
            yield preferred
            return
        if validation.error_code == "uncited_core_claim":
            salvaged = _salvage_llm_result(
                result.answer, citations, validation.uncited_claims, usage
            )
            if attempt:
                preferred = _prefer_llm_result(safe_first, salvaged, usage)
                if preferred is not None:
                    yield preferred
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


async def stream_answer_question(
    settings: Settings,
    request: QuestionRequest,
    runtime: LLMRuntime | None = None,
    budget: RequestBudget | None = None,
) -> AsyncGenerator[AnswerStreamEvent, None]:
    budget = budget or RequestBudget.start(settings.request_timeout_seconds)
    yield AnswerStreamEvent("status", {"message": "正在分析问题…"})
    structured = await run_in_threadpool(answer_structured_question, settings, request)
    if structured is not None:
        yield AnswerStreamEvent("done", structured.model_dump())
        return
    if settings.llm_query_planning and settings.llm_enabled:
        yield AnswerStreamEvent("status", {"message": "正在理解问题并规划检索…"})
    planning = await run_in_threadpool(plan_question, settings, request, runtime, budget)
    if planning.plan is not None and planning.plan.needs_clarification:
        yield AnswerStreamEvent("done", _clarification_response(request, planning).model_dump())
        return
    yield AnswerStreamEvent("status", {"message": "正在检索本地史料…"})
    context = await run_in_threadpool(_retrieve_context, settings, request, planning)
    result = LLMResult(answer=None, error_code="not_configured")
    if not context.citations:
        result = LLMResult(answer=None, error_code="no_evidence")
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
