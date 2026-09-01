from pathlib import Path
from typing import Any

import httpx
from fastapi.testclient import TestClient
from history_agent.answering.models import AnswerResponse, Citation, QuestionRequest
from history_agent.answering.service import (
    _extractive_answer,
    _llm_answer,
    _quote_for_hit,
    _unsupported_leading_entity,
)
from history_agent.answering.validation import validate_grounded_answer
from history_agent.config import Settings
from history_agent.retrieval.models import SearchHit
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


def test_quote_window_preserves_nearby_supporting_facts() -> None:
    hit = SearchHit(
        rank=1,
        chunk_id="observation",
        document_id="red-star",
        title="西行漫记",
        filename="red-star.pdf",
        source_type="contemporary_observation",
        verification_status="verified",
        pdf_page_start=118,
        pdf_page_end=118,
        section_path=[],
        text="毛泽东" + "性格质朴。" * 45 + "他是军事和政治战略家。" + "尾声。" * 30,
        year_mentions=[],
        people=["毛泽东"],
        extraction_methods=["ocr"],
        score=1.0,
        matched_terms=[],
    )

    quote = _quote_for_hit(hit, ["毛泽东"])

    assert "军事和政治战略家" in quote
    assert len(quote) <= 422


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


def test_compound_leading_people_are_checked_individually() -> None:
    hit = SearchHit(
        rank=1,
        chunk_id="intersection",
        document_id="history",
        title="测试史料",
        filename="history.pdf",
        source_type="history",
        verification_status="verified",
        pdf_page_start=1,
        pdf_page_end=1,
        section_path=[],
        text="1975年，毛泽东听取邓小平汇报。",
        year_mentions=[1975],
        people=["毛泽东", "邓小平"],
        extraction_methods=["text_layer"],
        score=1.0,
        matched_terms=[],
    )

    assert _unsupported_leading_entity("毛泽东和邓小平在1975年有哪些交集？", [hit]) is None


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


def test_validation_accepts_cited_core_fact_lines() -> None:
    result = validate_grounded_answer(
        "## 主要经历\n- 1956年1月，周恩来参加有关会议。[E1]\n\n资料范围有限。",
        [_citation()],
    )

    assert result.valid is True
    assert result.used_evidence_ids == ("E1",)


def test_validation_rejects_uncited_core_fact_line() -> None:
    result = validate_grounded_answer(
        "- 1956年1月，周恩来参加有关会议。[E1]\n- 随后主持科学规划工作。",
        [_citation()],
    )

    assert result.valid is False
    assert result.error_code == "uncited_core_claim"
    assert result.uncited_claims == ("随后主持科学规划工作。",)


def test_validation_rejects_fabricated_pdf_page() -> None:
    result = validate_grounded_answer(
        "《周恩来年谱》PDF第999页记载周恩来参加会议。[E1]",
        [_citation()],
    )

    assert result.valid is False
    assert result.error_code == "citation_metadata_mismatch"


def test_validation_accepts_matching_document_and_pdf_page() -> None:
    result = validate_grounded_answer(
        "《周恩来年谱》PDF第688页记载周恩来参加会议。[E1]",
        [_citation()],
    )

    assert result.valid is True


def test_validation_rejects_mismatched_document_name() -> None:
    result = validate_grounded_answer(
        "《林彪年谱》PDF第688页记载周恩来参加会议。[E1]",
        [_citation()],
    )

    assert result.valid is False
    assert result.error_code == "citation_metadata_mismatch"


def test_deepseek_falls_back_when_core_fact_has_no_citation(monkeypatch: Any) -> None:
    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                "1956年1月，周恩来参加有关会议。[E1]\n"
                                "随后主持科学规划工作。"
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    settings = Settings(_env_file=None, llm_api_key="sk-test")
    result = _llm_answer(
        settings=settings,
        request=QuestionRequest(question="周恩来在1956年做了什么？"),
        citations=[_citation()],
    )

    assert result.answer is None
    assert result.error_code == "uncited_core_claim"
