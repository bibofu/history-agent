"""Shared HTTP resources, admission control, and per-request deadlines for LLM calls."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from time import monotonic
from typing import Any

import httpx
from llama_index.core.callbacks import CallbackManager


@dataclass(frozen=True)
class RequestBudget:
    deadline: float

    @classmethod
    def start(cls, total_seconds: float) -> RequestBudget:
        return cls(monotonic() + total_seconds)

    def timeout(self, cap_seconds: float) -> float:
        remaining = self.deadline - monotonic()
        if remaining <= 0:
            raise httpx.TimeoutException("request time budget exhausted")
        return min(cap_seconds, remaining)


class LLMRuntime:
    """Application-scoped HTTP pools guarded by one shared concurrency limit."""

    def __init__(self, max_concurrency: int) -> None:
        limits = httpx.Limits(
            max_connections=max_concurrency,
            max_keepalive_connections=max_concurrency,
        )
        self.sync_client = httpx.Client(
            limits=limits,
            transport=httpx.HTTPTransport(retries=1),
        )
        self.async_client = httpx.AsyncClient(
            limits=limits,
            transport=httpx.AsyncHTTPTransport(retries=1),
        )
        self._capacity = threading.BoundedSemaphore(max_concurrency)
        self.callback_manager = CallbackManager()

    def post(
        self,
        url: str,
        *,
        timeout: float,
        headers: dict[str, str],
        json: dict[str, object],
    ) -> httpx.Response:
        if not self._capacity.acquire(timeout=timeout):
            raise httpx.PoolTimeout("LLM concurrency limit reached")
        try:
            return self.sync_client.post(
                url,
                timeout=timeout,
                headers=headers,
                json=json,
            )
        finally:
            self._capacity.release()

    async def apost(
        self,
        url: str,
        *,
        timeout: float,
        headers: dict[str, str],
        json: dict[str, object],
    ) -> httpx.Response:
        acquired = await asyncio.to_thread(self._capacity.acquire, True, timeout)
        if not acquired:
            raise httpx.PoolTimeout("LLM concurrency limit reached")
        try:
            return await self.async_client.post(
                url,
                timeout=timeout,
                headers=headers,
                json=json,
            )
        finally:
            self._capacity.release()

    @asynccontextmanager
    async def stream(
        self,
        method: str,
        url: str,
        *,
        timeout: float,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> AsyncIterator[httpx.Response]:
        acquired = await asyncio.to_thread(self._capacity.acquire, True, timeout)
        if not acquired:
            raise httpx.PoolTimeout("LLM concurrency limit reached")
        try:
            async with self.async_client.stream(
                method,
                url,
                timeout=timeout,
                headers=headers,
                json=json,
            ) as response:
                yield response
        finally:
            self._capacity.release()

    async def aclose(self) -> None:
        await self.async_client.aclose()
        self.sync_client.close()
