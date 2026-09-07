from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from history_agent.answering.models import ConversationMessage, QuestionRequest
from history_agent.answering.service import answer_question
from history_agent.answering.structured import answer_structured_question
from history_agent.config import Settings
from history_agent.db import Database
from history_agent.retrieval.keyword import PERIOD_RANGES, infer_year_range
from history_agent.retrieval.models import SearchHit, SearchResponse
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
        "周恩来与林彪在1900年有哪些交集",
        "周恩来与林彪在1944至1943年有哪些交集",
        "周恩来与林彪在1943年2月有哪些交集",
        "周恩来与林彪在1943年北京有哪些交集",
        "周恩来与林彪在北京有哪些交集",
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
    assert (
        answer_structured_question(settings, QuestionRequest(question="毛泽东和周恩来的交集"))
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


@pytest.mark.parametrize("period", list(PERIOD_RANGES), ids=lambda name: str(PERIOD_RANGES[name]))
def test_known_periods_select_source_synthesis(work_path: Path, period: str) -> None:
    settings = _settings(work_path)
    for question in (
        f"毛泽东和周恩来在{period}期间有哪些交集？",
        f"请列出周恩来在{period}时期的时间线",
    ):
        assert answer_structured_question(settings, QuestionRequest(question=question)) is None


@pytest.mark.parametrize(
    "question",
    [
        "毛泽东和周恩来在长征期间1935年有哪些交集",
        "毛泽东和周恩来在长征期间1月有哪些交集",
        "毛泽东和周恩来在长征期间北京有哪些交集",
        "毛泽东和周恩来在长征之前有哪些交集",
        "毛泽东和周恩来在长征结束后有哪些交集",
        "毛泽东和周恩来不在长征期间有哪些交集",
        "毛泽东和张三在长征期间有哪些交集",
        "毛泽东和周恩来和张三在长征期间有哪些交集",
        "毛泽东和毛泽东在长征期间有哪些交集",
        "毛泽东和周恩来在长征和抗日战争期间有哪些交集",
        "毛泽东和周恩来在未知时期有哪些交集",
    ],
    ids=[f"period-constraint-{index}" for index in range(11)],
)
def test_period_route_preserves_constraints(work_path: Path, question: str) -> None:
    response = answer_structured_question(_settings(work_path), QuestionRequest(question=question))
    assert response is not None
    assert response.citations == []


def test_long_march_question_reaches_grounded_answer_api(
    work_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(work_path).model_copy(update={"llm_api_key": None})
    question = "毛泽东和周恩来在长征期间有哪些交集？"
    searches: list[str] = []

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("named period synthesis must not depend on joint-action lookup results")

    def search(**kwargs: object) -> SearchResponse:
        query = str(kwargs["query"])
        searches.append(query)
        return SearchResponse(
            query=query,
            query_intent="intersection",
            query_terms=["毛泽东", "周恩来", "长征"],
            query_years=[],
            query_year_range=infer_year_range(query, []),
            query_people=["毛泽东", "周恩来"],
            document_filters=[],
            include_out_of_scope=False,
            retrieval_mode="hybrid_rrf",
            hits=[
                SearchHit(
                    rank=1,
                    chunk_id="long_march",
                    document_id="chronology",
                    title="测试年谱",
                    filename="test.pdf",
                    source_type="chronology",
                    verification_status="verified",
                    pdf_page_start=141,
                    pdf_page_end=141,
                    section_path=["1935年"],
                    text="遵义会议后，中央常委分工以毛泽东为周恩来在军事指挥上的帮助者。",
                    year_mentions=[1935],
                    people=["毛泽东", "周恩来"],
                    extraction_methods=["text_layer"],
                    score=0.03,
                    matched_terms=[],
                    keyword_rank=1,
                    vector_rank=1,
                )
            ],
        )

    monkeypatch.setattr("history_agent.answering.structured.get_person_intersections", unexpected)
    monkeypatch.setattr("history_agent.answering.structured.get_person_timeline", unexpected)
    monkeypatch.setattr("history_agent.answering.service.search_hybrid_index", search)
    response = TestClient(create_app(settings)).post("/api/questions", json={"question": question})
    assert response.status_code == 200
    data = response.json()
    assert searches == [question]
    assert data["retrieval_mode"] == "hybrid_rrf"
    assert data["evidence_status"] == "partial"
    assert data["llm_status"] == "disabled"
    assert "遵义会议" in data["answer"]
    assert "[E1]" in data["answer"]
    assert data["citations"][0]["pdf_page"] == 141
    assert any("1934—1936" in item for item in data["limitations"])
    assert "以下共同事件" not in data["answer"]
