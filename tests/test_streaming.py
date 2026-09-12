from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from history_agent.answering.models import AnswerResponse, QueryPlan, QuestionRequest
from history_agent.answering.query_understanding import QueryPlanningResult
from history_agent.answering.service import (
    AnswerContext,
    HierarchicalPreparation,
    LLMResult,
    answer_question,
)
from history_agent.answering.streaming import _stream_completion, stream_answer_question
from history_agent.config import Settings
from history_agent.errors import RetrievalError
from history_agent.web.app import create_app
from test_answering import _citation
from test_retrieval import _hit, _response


class ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.visited = 0
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            self.visited += 1
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


def _chunk(text: str | None = None, *, finish: str | None = None, usage: int = 0) -> bytes:
    payload = {
        "choices": [{"delta": {"content": text}, "finish_reason": finish}],
        "usage": {"total_tokens": usage} if usage else None,
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\r\n\r\n".encode()


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        llm_api_key="test-only-key",
        llm_query_planning=False,
    )


def _provider(
    monkeypatch: pytest.MonkeyPatch, streams: list[ChunkStream], *, status: int = 200
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(status, stream=streams[len(requests) - 1])

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs),
    )
    return requests


def _context(monkeypatch: pytest.MonkeyPatch) -> None:
    context = AnswerContext(_response([_hit("one", 1, page=688)]), [_citation()], "partial", None)

    async def retrieve(*args: object) -> AnswerContext:
        return context

    monkeypatch.setattr(
        "history_agent.answering.streaming.answer_structured_question", lambda *a: None
    )
    monkeypatch.setattr("history_agent.answering.streaming._aretrieve_context", retrieve)


def _events(settings: Settings | None = None) -> list[Any]:
    async def collect() -> list[Any]:
        return [
            event
            async for event in stream_answer_question(
                settings or _settings(), QuestionRequest(question="测试观点")
            )
        ]

    return asyncio.run(collect())


def test_provider_delivers_delta_before_reading_remainder_and_closes_on_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = ChunkStream([_chunk("第一段"), _chunk("第二段", finish="stop"), b"data: [DONE]\n\n"])
    requests = _provider(monkeypatch, [body])

    async def read_first() -> None:
        stream = _stream_completion(_settings(), {"messages": []})
        assert await anext(stream) == "第一段"
        assert body.visited == 1
        await stream.aclose()
        assert body.closed

    asyncio.run(read_first())
    assert requests[0]["stream"] is True
    assert requests[0]["stream_options"] == {"include_usage": True}


def test_stream_preserves_unicode_and_final_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    wire = _chunk("参加有关") + _chunk("会议。[E1]", finish="stop", usage=20) + b"data: [DONE]\n\n"
    body = ChunkStream([bytes([byte]) for byte in wire])
    _provider(monkeypatch, [body])
    _context(monkeypatch)
    events = _events()
    assert "".join(e.data["text"] for e in events if e.event == "delta") == "参加有关会议。[E1]"
    assert events[-1].event == "done"
    assert events[-1].data["llm_status"] == "used"
    assert events[-1].data["llm_usage"] == {"total_tokens": 20}
    assert events[-1].data["citations"][0]["pdf_page"] == 688
    assert body.closed


def test_stream_returns_valid_salvage_without_background_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = ChunkStream(
        [
            _chunk("参加有关会议。[E1]\n\n随后主持工作。", finish="stop", usage=20),
            b"data: [DONE]\n\n",
        ]
    )
    second = ChunkStream(
        [_chunk("参加有关会议并主持工作。[E1]", finish="stop", usage=10), b"data: [DONE]\n\n"]
    )
    requests = _provider(monkeypatch, [first, second])
    _context(monkeypatch)
    events = _events()
    assert all(e.event != "reset" for e in events)
    assert "".join(e.data["text"] for e in events if e.event == "delta") == (
        "参加有关会议。[E1]\n\n随后主持工作。"
    )
    assert not any(e.event == "status" and e.data["message"] == "正在后台补全引用…" for e in events)
    assert events[-1].data["answer"] == "参加有关会议。[E1]"
    assert events[-1].data["llm_usage"] == {"total_tokens": 20}
    assert len(requests) == 1


@pytest.mark.parametrize(
    "chunks,reason",
    [
        ([_chunk("错误。[E99]", finish="stop"), b"data: [DONE]\n\n"], "invalid_evidence_marker"),
        ([_chunk("参加有关会议。[E1]", finish="stop")], "incomplete_stream"),
        (
            [_chunk("参加有关会议。[E1]", finish="length"), b"data: [DONE]\n\n"],
            "max_tokens_exhausted",
        ),
        ([b"data: invalid-json\n\n"], "invalid_response"),
    ],
)
def test_failed_stream_is_replaced_with_evidence_excerpt(
    monkeypatch: pytest.MonkeyPatch, chunks: list[bytes], reason: str
) -> None:
    _provider(monkeypatch, [ChunkStream(chunks)])
    _context(monkeypatch)
    data = _events()[-1].data
    assert data["llm_status"] == "fallback"
    assert data["generator_mode"] == "extractive"
    assert "E99" not in data["answer"]
    assert any(reason in item for item in data["limitations"])


def test_stream_does_not_send_reasoning_as_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    reasoning = b'data: {"choices":[{"delta":{"reasoning_content":"private reasoning"}}]}\n\n'
    body = ChunkStream(
        [reasoning, _chunk("参加有关会议。[E1]", finish="stop"), b"data: [DONE]\n\n"]
    )
    _provider(monkeypatch, [body])
    _context(monkeypatch)
    events = _events()
    assert "private reasoning" not in "".join(e.encode() for e in events)


def test_no_model_configuration_finishes_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    _context(monkeypatch)
    data = _events(_settings().model_copy(update={"llm_api_key": None}))[-1].data
    assert data["llm_status"] == "disabled"


def test_stream_uses_semantic_plan_before_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = QueryPlan(
        intent="event_overview",
        normalized_question="介绍中国共产党第一次全国代表大会",
        search_queries=["中国共产党第一次全国代表大会 代表 决议"],
    )
    planning = QueryPlanningResult(
        plan,
        "used",
        model_name="deepseek-v4-flash",
        usage={"total_tokens": 50},
    )
    context = AnswerContext(
        _response([]),
        [],
        "no_evidence",
        None,
        reflection_status="retried",
        retrieval_rounds=2,
        missing_aspects=("会议结果",),
    )
    received: list[QueryPlanningResult] = []
    monkeypatch.setattr(
        "history_agent.answering.streaming.answer_structured_question", lambda *a: None
    )
    monkeypatch.setattr("history_agent.answering.streaming.plan_question", lambda *a: planning)

    async def retrieve(*args: Any) -> AnswerContext:
        received.append(args[2])
        return context

    monkeypatch.setattr("history_agent.answering.streaming._aretrieve_context", retrieve)
    settings = _settings().model_copy(update={"llm_query_planning": True})
    events = _events(settings)

    assert received == [planning]
    assert any(event.event == "status" and "规划检索" in event.data["message"] for event in events)
    assert any(event.event == "status" and "补充检索" in event.data["message"] for event in events)
    assert events[-1].data["query_planner_status"] == "used"
    assert events[-1].data["query_plan"]["normalized_question"] == plan.normalized_question


def test_stream_endpoint_sends_final_result_and_safe_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _context(monkeypatch)
    settings = _settings().model_copy(update={"llm_api_key": None})
    client = TestClient(create_app(settings))
    response = client.post("/api/questions/stream", json={"question": "测试观点"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: done\n" in response.text
    assert "event: status\n" in response.text
    assert client.post("/api/questions/stream", json={"question": ""}).status_code == 422

    async def failed(*args: object) -> LLMResult:
        raise RetrievalError("internal path and secret must not leak")

    monkeypatch.setattr("history_agent.answering.streaming._aretrieve_context", failed)
    response = client.post("/api/questions/stream", json={"question": "测试观点"})
    assert "event: error\n" in response.text
    assert "internal path" not in response.text


def test_disabled_planner_uses_rag_without_generic_clarification() -> None:
    client = TestClient(create_app(_settings().model_copy(update={"llm_api_key": None})))
    response = client.post("/api/questions/stream", json={"question": "毛泽东有哪些经历"})
    assert "event: done\n" in response.text
    assert "请明确人物和年份" not in response.text


def test_structured_summary_streams_through_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    structured = AnswerResponse(
        question="周恩来在1956年主要有哪些经历？",
        answer="结构化记录摘录。[E1]",
        evidence_status="partial",
        generator_mode="extractive",
        llm_status="not_applicable",
        retrieval_mode="structured_timeline",
        query_intent="timeline",
        citations=[_citation()],
    )
    body = ChunkStream([_chunk("可归纳为外事和会议工作。[E1]", finish="stop"), b"data: [DONE]\n\n"])
    _provider(monkeypatch, [body])
    monkeypatch.setattr(
        "history_agent.answering.streaming.answer_structured_question",
        lambda *args: structured,
    )

    events = _events()

    assert any(
        event.event == "status" and "归纳结构化史料" in event.data["message"] for event in events
    )
    assert events[-1].data["retrieval_mode"] == "structured_timeline"
    assert events[-1].data["llm_status"] == "used"
    assert events[-1].data["answer"] == "可归纳为外事和会议工作。[E1]"


def test_structured_record_list_also_streams_through_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structured = AnswerResponse(
        question="请列出1956年周恩来的时间线",
        answer="结构化记录摘录。[E1]",
        evidence_status="partial",
        generator_mode="extractive",
        llm_status="not_applicable",
        retrieval_mode="structured_timeline",
        query_intent="timeline",
        citations=[_citation()],
    )
    body = ChunkStream([_chunk("整理后的记录。[E1]", finish="stop"), b"data: [DONE]\n\n"])
    _provider(monkeypatch, [body])
    monkeypatch.setattr(
        "history_agent.answering.streaming.answer_structured_question",
        lambda *args: structured,
    )

    async def collect() -> list[Any]:
        return [
            event
            async for event in stream_answer_question(
                _settings(), QuestionRequest(question="请列出1956年周恩来的时间线")
            )
        ]

    events = asyncio.run(collect())
    assert events[-1].data["llm_status"] == "used"
    assert events[-1].data["answer"] == "整理后的记录。[E1]"


def test_cancelling_answer_closes_nested_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    body = ChunkStream([_chunk("参加有关"), _chunk("会议。[E1]", finish="stop")])
    _provider(monkeypatch, [body])
    _context(monkeypatch)

    async def cancel() -> None:
        answers = stream_answer_question(_settings(), QuestionRequest(question="测试观点"))
        async for event in answers:
            if event.event == "delta":
                break
        await answers.aclose()
        assert body.closed
        assert body.visited == 1

    asyncio.run(cancel())


def test_valid_stream_salvage_does_not_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    invalid = [_chunk("参加有关会议。[E1]\n\n随后主持工作。", finish="stop"), b"data: [DONE]\n\n"]
    requests = _provider(monkeypatch, [ChunkStream(invalid), ChunkStream(invalid)])
    _context(monkeypatch)
    final = _events()[-1].data
    assert len(requests) == 1
    assert final["llm_status"] == "used"
    assert final["llm_error_code"] == "removed_uncited_claims"
    assert final["answer"] == "参加有关会议。[E1]"
    assert final["uncited_claims"] == ["随后主持工作。"]
    assert any("未引用段落已移除" in item for item in final["limitations"])


def test_large_evidence_set_uses_hierarchical_generation_in_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    citations = [
        _citation().model_copy(update={"evidence_id": f"E{index}"}) for index in range(1, 14)
    ]
    context = AnswerContext(_response([]), citations, "partial", None)

    async def retrieve(*args: object) -> AnswerContext:
        return context

    monkeypatch.setattr(
        "history_agent.answering.streaming.answer_structured_question", lambda *args: None
    )
    monkeypatch.setattr("history_agent.answering.streaming._aretrieve_context", retrieve)
    monkeypatch.setattr(
        "history_agent.answering.streaming._prepare_hierarchical_answer",
        lambda **kwargs: HierarchicalPreparation(
            request_payload={"messages": []},
            usage={"total_tokens": 5},
            map_failed=False,
        ),
    )
    body = ChunkStream(
        [
            _chunk("分层", usage=20),
            _chunk("综合答案。[E1][E13]", finish="stop"),
            b"data: [DONE]\n\n",
        ]
    )
    _provider(monkeypatch, [body])

    events = _events()

    assert any(event.event == "status" and "分组归纳" in event.data["message"] for event in events)
    assert [event.data["text"] for event in events if event.event == "delta"] == [
        "分层",
        "综合答案。[E1][E13]",
    ]
    assert events[-1].data["answer"] == "分层综合答案。[E1][E13]"
    assert events[-1].data["llm_usage"] == {"total_tokens": 25}
    assert [item["evidence_id"] for item in events[-1].data["citations"]] == ["E1", "E13"]
    assert events[-1].data["retrieved_evidence_count"] == 13
    assert any("13 条证据" in item for item in events[-1].data["limitations"])


@pytest.mark.parametrize("missing_fact", [False, True])
def test_ceremony_overview_has_same_validation_and_diagnostics_in_both_apis(
    monkeypatch: pytest.MonkeyPatch,
    missing_fact: bool,
) -> None:
    # A local provider fixture for the reported question, not a recorded model answer.
    fact = "1949年10月1日，开国大典在北京天安门广场举行。"
    draft = f"## 开国大典\n\n> {fact}\n> [E1]\n\n"
    draft += "现有资料不足以确认所有参与人员的具体分工。"
    if missing_fact:
        draft += "\n\n随后主持其他会议。"
    citation = _citation(fact)
    context = AnswerContext(_response([_hit("one", 1, page=688)]), [citation], "partial", None)
    for module in ("service", "streaming"):

        async def retrieve(*args: object) -> AnswerContext:
            return context

        monkeypatch.setattr(
            f"history_agent.answering.{module}.answer_structured_question", lambda *a: None
        )
        monkeypatch.setattr(f"history_agent.answering.{module}._aretrieve_context", retrieve)
    sync_calls = []

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        sync_calls.append(kwargs["json"])
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": draft}}], "usage": {"total_tokens": 20}},
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    expected_calls = 1
    requests = _provider(
        monkeypatch,
        [
            ChunkStream([_chunk(draft, finish="stop", usage=20), b"data: [DONE]\n\n"])
            for _ in range(expected_calls)
        ],
    )
    question = QuestionRequest(question="介绍一下开国大典的情况")
    synchronous = answer_question(_settings(), question).model_dump()

    async def collect() -> list[Any]:
        return [event async for event in stream_answer_question(_settings(), question)]

    streamed = asyncio.run(collect())[-1].data
    assert synchronous == streamed
    assert len(sync_calls) == len(requests) == expected_calls
    if missing_fact:
        assert streamed["llm_status"] == "used"
        assert streamed["llm_error_code"] == "removed_uncited_claims"
        assert streamed["uncited_claims"] == ["随后主持其他会议。"]
        assert "随后主持其他会议" not in streamed["answer"]
        assert fact in streamed["answer"]
    else:
        assert streamed["llm_status"] == "used"
        assert streamed["answer"] == draft
        assert streamed["uncited_claims"] == []
        assert streamed["llm_error_code"] is None


def test_rate_limit_becomes_final_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _provider(monkeypatch, [ChunkStream([])], status=429)
    _context(monkeypatch)
    final = _events()[-1].data
    assert final["llm_status"] == "fallback"
    assert any("rate_limited" in item for item in final["limitations"])


def test_read_timeout_replaces_an_unfinished_draft(monkeypatch: pytest.MonkeyPatch) -> None:
    class TimedOutStream(ChunkStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield _chunk("尚未完成的文字")
            raise httpx.ReadTimeout("test timeout")

    body = TimedOutStream([])
    _provider(monkeypatch, [body])
    _context(monkeypatch)
    events = _events()
    assert any(e.event == "delta" for e in events)
    final = events[-1].data
    assert final["llm_status"] == "fallback"
    assert "尚未完成的文字" not in final["answer"]
    assert body.closed
