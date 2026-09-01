from pathlib import Path
from typing import Any

import httpx
from fastapi.testclient import TestClient
from history_agent.answering.models import AnswerResponse, Citation, QuestionRequest
from history_agent.answering.service import (
    _extractive_answer,
    _llm_answer,
    _unsupported_leading_entity,
)
from history_agent.config import Settings
from history_agent.web import app as web_module


def _citation(quote: str = "【1956年1月】参加会议并讨论科学规划。") -> Citation:
    return Citation(
        evidence_id="E1",
        document_id="zhou",
        document="周恩来年谱",
        pdf_page=688,
        section=["1956年"],
        quote=quote,
        source_type="chronology",
        verification_status="verified",
        extraction_methods=["pymupdf"],
    )


def test_extractive_answer_keeps_evidence_marker() -> None:
    answer = _extractive_answer("timeline", [_citation()])

    assert "[E1]" in answer
    assert "1956年" in answer


def test_extractive_answer_skips_bare_ellipsis_sentence() -> None:
    answer = _extractive_answer(
        "viewpoint",
        [_citation("……。要做系统的由历史到现状的调查研究。后续文字。")],
    )

    assert "- ……。[E1]" not in answer
    assert "要做系统" in answer


def test_question_request_rejects_unbounded_history() -> None:
    request = QuestionRequest(question="测试问题")

    assert request.top_k == 8
    assert request.history == []


def test_unknown_leading_person_must_appear_in_evidence() -> None:
    assert (
        _unsupported_leading_entity(
            "爱因斯坦在1925年担任了什么党内职务？", []
        )
        == "爱因斯坦"
    )


def test_api_health_and_question_contract(monkeypatch: Any) -> None:
    settings = Settings(
        project_root=Path.cwd(), data_dir=Path("test-data-that-does-not-exist")
    )

    expected = AnswerResponse(
        question="周恩来在1956年做了什么？",
        answer="参加有关会议。[E1]",
        evidence_status="supported",
        generator_mode="extractive",
        llm_status="disabled",
        retrieval_mode="hybrid_rrf",
        query_intent="timeline",
        citations=[_citation()],
    )

    def fake_answer(active_settings: Settings, request: QuestionRequest) -> AnswerResponse:
        assert active_settings is settings
        assert request.question == expected.question
        return expected

    monkeypatch.setattr(web_module, "answer_question", fake_answer)
    client = TestClient(web_module.create_app(settings))

    health = client.get("/api/health")
    response = client.post("/api/questions", json={"question": expected.question})

    assert health.status_code == 200
    assert health.json()["indexes"] == {"keyword": False, "vector": False}
    assert response.status_code == 200
    assert response.json()["citations"][0]["pdf_page"] == 688


def test_deepseek_v4_request_and_usage(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        captured["url"] = url
        captured.update(kwargs)
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "choices": [{"message": {"content": "参加有关会议。[E1]"}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                },
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    settings = Settings(_env_file=None, llm_api_key="sk-test", llm_thinking=True)
    result = _llm_answer(
        settings=settings,
        request=QuestionRequest(question="周恩来在1956年做了什么？"),
        citations=[_citation()],
    )

    assert result.answer == "参加有关会议。[E1]"
    assert result.usage == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
    }
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    body = captured["json"]
    assert body["model"] == "deepseek-v4-pro"
    assert body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == "high"
    assert "temperature" not in body


def test_deepseek_rejects_unknown_evidence_marker(monkeypatch: Any) -> None:
    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": "错误引用。[E99]"}}]},
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    settings = Settings(_env_file=None, llm_api_key="sk-test")
    result = _llm_answer(
        settings=settings,
        request=QuestionRequest(question="周恩来在1956年做了什么？"),
        citations=[_citation()],
    )

    assert result.answer is None
    assert result.error_code == "invalid_evidence_marker"
