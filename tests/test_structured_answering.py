from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from history_agent.answering.models import ConversationMessage, QuestionRequest
from history_agent.answering.service import answer_question
from history_agent.answering.structured import answer_structured_question
from history_agent.config import Settings
from history_agent.db import Database
from history_agent.web.app import create_app
from test_timeline import _prepare_timeline


def _settings(work_path: Path) -> Settings:
    database, _ = _prepare_timeline(work_path)
    return Settings(
        _env_file=None,
        project_root=work_path,
        data_dir=work_path / "data",
        database_path=database.path,
    )


@pytest.mark.parametrize(
    "question,intent,has_evidence",
    [
        ("周恩来在1943年有哪些经历？", "timeline", True),
        ("请列出1943年周恩来的时间线", "timeline", True),
        ("周恩来在1942年至1943年参加过哪些会议", "timeline", True),
        ("周恩来与林彪在1943年有哪些共同事件？", "intersection", True),
        ("周恩来和毛泽东在1943年有哪些交集", "intersection", False),
        ("林彪和周恩来在1942年有哪些交集", "intersection", False),
    ],
    ids=[f"route-{index}" for index in range(6)],
)
def test_structured_api_bypasses_rag_and_llm(
    work_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    question: str,
    intent: str,
    has_evidence: bool,
) -> None:
    settings = _settings(work_path)

    def unexpected(**kwargs: object) -> None:
        pytest.fail("structured route must not invoke RAG or LLM")

    monkeypatch.setattr("history_agent.answering.service.search_hybrid_index", unexpected)
    monkeypatch.setattr("history_agent.answering.service._llm_answer", unexpected)
    response = TestClient(create_app(settings)).post(
        "/api/questions", json={"question": question, "top_k": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["retrieval_mode"] == f"structured_{intent}"
    assert data["llm_status"] == "not_applicable"
    assert bool(data["citations"]) == has_evidence
    assert data["evidence_status"] != "supported"
    if has_evidence:
        assert "[E1]" in data["answer"]
        assert data["citations"][0]["pdf_page"] > 0
        assert "前 1 条" in data["answer"]
    else:
        assert "不代表" in data["answer"]


@pytest.mark.parametrize(
    "question",
    [
        "周恩来和林彪有哪些交集",
        "周恩来与林彪在1900年有哪些交集",
        "周恩来与林彪在1944至1943年有哪些交集",
        "周恩来与林彪在1943年2月有哪些交集",
        "周恩来与林彪在1943年北京有哪些交集",
        "周恩来在1943年没有参加过哪些会议",
        "周恩来和张三在1943年有哪些交集",
        "周恩来和周恩来在1943年有哪些交集",
        "林彪与周恩来在1943年和1945年有哪些交集",
    ],
    ids=[f"constraint-{index}" for index in range(9)],
)
def test_constraints_are_not_silently_dropped(work_path: Path, question: str) -> None:
    result = answer_structured_question(_settings(work_path), QuestionRequest(question=question))
    assert result is not None
    assert result.citations == []
    assert result.evidence_status == "no_evidence"


def test_general_questions_still_use_rag_route(work_path: Path) -> None:
    settings = _settings(work_path)
    assert (
        answer_structured_question(
            settings, QuestionRequest(question="毛泽东关于调查研究有哪些观点")
        )
        is None
    )
    assert (
        answer_structured_question(
            settings, QuestionRequest(question="毛泽东的早年经历如何影响他的调查研究观点")
        )
        is None
    )


def test_structured_missing_database_returns_actionable_message(work_path: Path) -> None:
    settings = Settings(
        _env_file=None, project_root=work_path, database_path=work_path / "missing.db"
    )
    result = answer_question(settings, QuestionRequest(question="毛泽东在1949年有哪些经历"))
    assert "尚未就绪" in result.answer
    assert not settings.database_path.exists()


def test_elliptical_followup_requests_explicit_people(work_path: Path) -> None:
    settings = _settings(work_path)
    result = answer_structured_question(
        settings,
        QuestionRequest(
            question="那1956年呢",
            history=[ConversationMessage(role="user", content="周恩来和林彪在1943年有哪些交集")],
        ),
    )
    assert result is not None
    assert result.retrieval_mode == "structured_intersection"
    assert result.citations == []
    assert "请明确人物" in result.answer


def test_chat_uses_actual_proof_and_full_page_range(work_path: Path) -> None:
    settings = _settings(work_path)
    with Database(settings.database_path).connect() as connection:
        connection.execute(
            "UPDATE evidence_records SET pdf_page_end=31 "
            "WHERE evidence_id='evidence_event_lin_message'"
        )
        connection.execute(
            "UPDATE evidence_records SET quote='周恩来致电毛泽东，详细报告当时的工作情况。' "
            "WHERE evidence_id='evidence_event_zhou_message'"
        )
    result = answer_structured_question(
        settings, QuestionRequest(question="周恩来和林彪在1943年有哪些交集")
    )
    assert result is not None
    assert result.citations[0].document_id == "lin_biao_chronology"
    assert result.citations[0].pdf_page_end == 31
    assert "林彪、周恩来致电毛泽东" in result.citations[0].quote
