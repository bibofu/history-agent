"""LlamaIndex LLM adapter for the project's DeepSeek-compatible endpoint."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import httpx
from llama_index.core.callbacks import CallbackManager, CBEventType, EventPayload
from llama_index.core.llms import (
    LLM,
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from pydantic import Field, PrivateAttr, SecretStr

from history_agent.answering.runtime import LLMRuntime, RequestBudget
from history_agent.config import Settings


def chat_messages(payload: dict[str, object]) -> list[ChatMessage]:
    """Convert an OpenAI-compatible payload into LlamaIndex messages."""

    raw_messages = payload.get("messages", [])
    if not isinstance(raw_messages, list):
        raise TypeError("request messages must be a list")
    messages: list[ChatMessage] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            raise TypeError("request message must be an object")
        messages.append(
            ChatMessage(
                role=MessageRole(str(item["role"])),
                content=str(item.get("content", "")),
            )
        )
    return messages


def response_usage(response: ChatResponse) -> dict[str, int] | None:
    raw = response.additional_kwargs.get("usage")
    if not isinstance(raw, dict):
        return None
    usage = {
        key: int(raw[key])
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if key in raw
    }
    return usage or None


def _sse_usage(payload: dict[str, Any]) -> dict[str, int] | None:
    raw = payload.get("usage")
    if not isinstance(raw, dict):
        return None
    usage = {
        key: int(raw[key])
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if key in raw
    }
    return usage or None


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


class DeepSeekLlamaIndexLLM(LLM):
    """DeepSeek chat model with LlamaIndex callbacks and project request budgets."""

    model_name: str
    base_url: str
    api_key: SecretStr = Field(exclude=True, repr=False)
    max_tokens: int
    timeout_seconds: float
    thinking: bool = False
    reasoning_effort: str = "high"
    _runtime: LLMRuntime | None = PrivateAttr(default=None)
    _budget: RequestBudget | None = PrivateAttr(default=None)

    def __init__(
        self,
        *,
        settings: Settings,
        runtime: LLMRuntime | None = None,
        budget: RequestBudget | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        callback_manager: CallbackManager | None = None,
    ) -> None:
        if settings.llm_api_key is None:
            raise ValueError("DeepSeek API key is not configured")
        model_data: dict[str, Any] = {
            "model_name": model or settings.llm_model,
            "base_url": settings.llm_base_url,
            "api_key": settings.llm_api_key,
            "max_tokens": settings.llm_max_tokens,
            "timeout_seconds": timeout_seconds or settings.llm_timeout_seconds,
            "thinking": settings.llm_thinking,
            "reasoning_effort": settings.llm_reasoning_effort,
            "callback_manager": (
                callback_manager
                or (runtime.callback_manager if runtime is not None else CallbackManager())
            ),
        }
        super().__init__(**model_data)
        self._runtime = runtime
        self._budget = budget

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=64_000,
            num_output=self.max_tokens,
            is_chat_model=True,
            model_name=self.model_name,
        )

    @property
    def _url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

    def _timeout(self) -> float:
        if self._budget is None:
            return self.timeout_seconds
        return self._budget.timeout(self.timeout_seconds)

    def _request_payload(
        self,
        messages: Sequence[ChatMessage],
        kwargs: dict[str, Any],
        *,
        stream: bool,
    ) -> dict[str, object]:
        supplied = kwargs.get("request_payload")
        if supplied is not None and not isinstance(supplied, dict):
            raise TypeError("request_payload must be a dictionary")
        payload: dict[str, object] = dict(supplied or {})
        payload.setdefault("model", self.model_name)
        payload.setdefault("max_tokens", self.max_tokens)
        payload.setdefault("thinking", {"type": "enabled" if self.thinking else "disabled"})
        payload["messages"] = [
            {"role": message.role.value, "content": message.content or ""} for message in messages
        ]
        payload["stream"] = stream
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _sync_response(
        self, messages: Sequence[ChatMessage], kwargs: dict[str, Any]
    ) -> ChatResponse:
        request_payload = self._request_payload(messages, kwargs, stream=False)
        timeout = self._timeout()
        response = (
            self._runtime.post(
                self._url,
                headers=self._headers,
                json=request_payload,
                timeout=timeout,
            )
            if self._runtime is not None
            else httpx.post(
                self._url,
                headers=self._headers,
                json=request_payload,
                timeout=timeout,
            )
        )
        response.raise_for_status()
        payload = response.json()
        choice = payload["choices"][0]
        content = str(choice["message"]["content"])
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=content),
            raw=payload,
            additional_kwargs={
                "usage": payload.get("usage"),
                "finish_reason": choice.get("finish_reason"),
            },
        )

    async def _async_response(
        self, messages: Sequence[ChatMessage], kwargs: dict[str, Any]
    ) -> ChatResponse:
        request_payload = self._request_payload(messages, kwargs, stream=False)
        timeout = self._timeout()
        if self._runtime is not None:
            response = await self._runtime.apost(
                self._url,
                headers=self._headers,
                json=request_payload,
                timeout=timeout,
            )
        else:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    self._url,
                    headers=self._headers,
                    json=request_payload,
                )
        response.raise_for_status()
        payload = response.json()
        choice = payload["choices"][0]
        content = str(choice["message"]["content"])
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=content),
            raw=payload,
            additional_kwargs={
                "usage": payload.get("usage"),
                "finish_reason": choice.get("finish_reason"),
            },
        )

    @asynccontextmanager
    async def _stream_response(
        self, request_payload: dict[str, object], timeout: float
    ) -> AsyncIterator[httpx.Response]:
        if self._runtime is not None:
            async with self._runtime.stream(
                "POST",
                self._url,
                timeout=timeout,
                headers=self._headers,
                json=request_payload,
            ) as response:
                yield response
        else:
            async with (
                httpx.AsyncClient(timeout=timeout) as client,
                client.stream(
                    "POST",
                    self._url,
                    headers=self._headers,
                    json=request_payload,
                ) as response,
            ):
                yield response

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self._sync_response(messages, kwargs)

    @llm_chat_callback()
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return await self._async_response(messages, kwargs)

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        def generate() -> ChatResponseGen:
            response = self._sync_response(messages, kwargs)
            yield response.model_copy(update={"delta": response.message.content})

        return generate()

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        return self.raw_astream_chat(messages, **kwargs)

    def raw_astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        """Return a cancellation-safe stream while preserving callback events.

        LlamaIndex's async callback wrapper does not close the wrapped provider
        generator when its consumer disconnects, so the web path uses this
        lifecycle-aware entry point on the same adapter.
        """

        async def generate() -> ChatResponseAsyncGen:
            request_payload = self._request_payload(messages, kwargs, stream=True)
            parts: list[str] = []
            usage: dict[str, int] | None = None
            callback_payload = {
                EventPayload.MESSAGES.value: list(messages),
                EventPayload.SERIALIZED.value: self.to_payload(),
            }
            with self.callback_manager.event(CBEventType.LLM, payload=callback_payload):
                async with self._stream_response(request_payload, self._timeout()) as response:
                    response.raise_for_status()
                    async for data in _sse_data(response):
                        if data == "[DONE]":
                            yield ChatResponse(
                                message=ChatMessage(
                                    role=MessageRole.ASSISTANT, content="".join(parts)
                                ),
                                delta="",
                                additional_kwargs={"usage": usage, "done": True},
                            )
                            return
                        payload = json.loads(data)
                        if "error" in payload:
                            raise ValueError("upstream error")
                        usage = _sse_usage(payload) or usage
                        choices = payload.get("choices", [])
                        if not choices:
                            continue
                        choice = choices[0]
                        delta = choice.get("delta", {}).get("content") or ""
                        if not isinstance(delta, str):
                            raise ValueError("invalid content delta")
                        parts.append(delta)
                        yield ChatResponse(
                            message=ChatMessage(role=MessageRole.ASSISTANT, content="".join(parts)),
                            raw=payload,
                            delta=delta,
                            additional_kwargs={
                                "usage": usage,
                                "finish_reason": choice.get("finish_reason"),
                            },
                        )

        return generate()

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        del formatted
        response = self.chat([ChatMessage(role=MessageRole.USER, content=prompt)], **kwargs)
        return CompletionResponse(
            text=response.message.content or "",
            raw=response.raw,
            additional_kwargs=response.additional_kwargs,
        )

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        del formatted
        response = await self.achat([ChatMessage(role=MessageRole.USER, content=prompt)], **kwargs)
        return CompletionResponse(
            text=response.message.content or "",
            raw=response.raw,
            additional_kwargs=response.additional_kwargs,
        )

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        del formatted

        def generate() -> CompletionResponseGen:
            for response in self.stream_chat(
                [ChatMessage(role=MessageRole.USER, content=prompt)], **kwargs
            ):
                yield CompletionResponse(
                    text=response.message.content or "",
                    delta=response.delta,
                    raw=response.raw,
                    additional_kwargs=response.additional_kwargs,
                )

        return generate()

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        del formatted

        async def generate() -> CompletionResponseAsyncGen:
            stream = await self.astream_chat(
                [ChatMessage(role=MessageRole.USER, content=prompt)], **kwargs
            )
            async for response in stream:
                yield CompletionResponse(
                    text=response.message.content or "",
                    delta=response.delta,
                    raw=response.raw,
                    additional_kwargs=response.additional_kwargs,
                )

        return generate()

    @classmethod
    def class_name(cls) -> str:
        return "history_agent_deepseek"
