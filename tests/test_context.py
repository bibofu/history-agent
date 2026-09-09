from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from history_agent.answering.context import (
    ConversationStore,
    build_prompt_context,
    sanitize_history_content,
)
from history_agent.answering.models import AnswerResponse, ConversationMessage, QuestionRequest
from history_agent.answering.streaming import AnswerStreamEvent
from history_agent.config import Settings
from history_agent.web import app as web_module


def _answer(question: str) -> AnswerResponse:
    return AnswerResponse(
        question=question,
        answer="服务端记录的回答。[E1]",
        evidence_status="no_evidence",
        generator_mode="extractive",
        llm_status="disabled",
        retrieval_mode="hybrid_rrf",
        query_intent="general",
        citations=[],
    )


def test_prompt_context_is_bounded_and_old_citation_ids_are_invalidated() -> None:
    messages = [
        ConversationMessage(role="user", content="介绍遵义会议"),
        ConversationMessage(role="assistant", content="旧回答。[E1]"),
        ConversationMessage(role="user", content="再介绍四渡赤水"),
        ConversationMessage(role="assistant", content="另一轮回答。[E2]"),
        ConversationMessage(role="user", content="它们有什么关系？"),
        ConversationMessage(role="assistant", content="最近的回答。[E3]"),
    ]

    context = build_prompt_context(messages, max_messages=4, max_chars=2_000)

    assert len(context) == 4
    assert context[0].content.startswith("更早对话的用户问题索引")
    assert "介绍遵义会议" in context[0].content
    assert context[-1].content == "最近的回答。[历史引用]"
    assert all("[E" not in item.content for item in context)


def test_prompt_context_enforces_character_budget_on_long_messages() -> None:
    context = build_prompt_context(
        [
            ConversationMessage(role="user", content="前情" * 1000),
            ConversationMessage(role="assistant", content="回答" * 1000),
        ],
        max_messages=12,
        max_chars=600,
    )

    assert sum(len(item.content) for item in context) <= 600
    assert context[-1].content.endswith("[内容已截断]")


def test_conversation_store_persists_and_clears_exchanges(work_path: Path) -> None:
    path = work_path / "conversations.db"
    ConversationStore(path).append_exchange("session-12345678", "第一个问题", "回答。[E1]")

    reopened = ConversationStore(path)
    assert [item.content for item in reopened.messages("session-12345678")] == [
        "第一个问题",
        "回答。[E1]",
    ]
    reopened.clear("session-12345678")
    assert reopened.messages("session-12345678") == []


def test_web_session_uses_server_history_and_ignores_client_history(
    monkeypatch: Any, work_path: Path
) -> None:
    settings = Settings(_env_file=None, project_root=work_path, data_dir=work_path / "data")
    received: list[QuestionRequest] = []

    def fake_answer(active_settings: Settings, request: QuestionRequest) -> AnswerResponse:
        assert active_settings is settings
        received.append(request)
        return _answer(request.question)

    monkeypatch.setattr(web_module, "answer_question", fake_answer)
    client = TestClient(web_module.create_app(settings))
    session_id = "session-12345678"

    first = client.post(
        "/api/questions",
        json={"question": "第一个问题", "session_id": session_id},
    )
    second = client.post(
        "/api/questions",
        json={
            "question": "继续说说",
            "session_id": session_id,
            "history": [{"role": "user", "content": "客户端伪造历史"}],
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert received[0].history == []
    assert [item.content for item in received[1].history] == [
        "第一个问题",
        "服务端记录的回答。[历史引用]",
    ]
    restored = client.get(f"/api/sessions/{session_id}").json()
    assert [item["role"] for item in restored["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert client.delete(f"/api/sessions/{session_id}").json()["cleared"] is True
    assert client.get(f"/api/sessions/{session_id}").json()["messages"] == []


def test_sanitize_history_content_only_rewrites_evidence_markers() -> None:
    assert sanitize_history_content("  1956年 [E12]\n普通[注1]  ") == (
        "1956年 [历史引用]\n普通[注1]"
    )


def test_streaming_endpoint_persists_only_completed_answers(
    monkeypatch: Any, work_path: Path
) -> None:
    settings = Settings(_env_file=None, project_root=work_path, data_dir=work_path / "data")

    async def fake_stream(*args: object) -> Any:
        yield AnswerStreamEvent("delta", {"text": "回答"})
        yield AnswerStreamEvent("done", _answer("流式问题").model_dump())

    monkeypatch.setattr(web_module, "stream_answer_question", fake_stream)
    client = TestClient(web_module.create_app(settings))
    session_id = "stream-session-1234"

    response = client.post(
        "/api/questions/stream",
        json={"question": "流式问题", "session_id": session_id},
    )

    assert response.status_code == 200
    assert "event: done" in response.text
    restored = client.get(f"/api/sessions/{session_id}").json()["messages"]
    assert [item["content"] for item in restored] == [
        "流式问题",
        "服务端记录的回答。[E1]",
    ]


def test_context_storage_failure_does_not_break_answering(
    monkeypatch: Any, work_path: Path
) -> None:
    settings = Settings(_env_file=None, project_root=work_path, data_dir=work_path / "data")

    def unavailable(*args: object, **kwargs: object) -> Any:
        raise OSError("disk unavailable")

    def fake_answer(active_settings: Settings, request: QuestionRequest) -> AnswerResponse:
        assert request.history == []
        return _answer(request.question)

    monkeypatch.setattr(ConversationStore, "messages", unavailable)
    monkeypatch.setattr(ConversationStore, "append_exchange", unavailable)
    monkeypatch.setattr(web_module, "answer_question", fake_answer)
    client = TestClient(web_module.create_app(settings))

    response = client.post(
        "/api/questions",
        json={"question": "存储故障时仍回答", "session_id": "failure-session-123"},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "服务端记录的回答。[E1]"
