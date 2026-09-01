import json
from pathlib import Path
from typing import Any

import pytest
from history_agent.answering.models import AnswerResponse, Citation
from history_agent.config import Settings
from history_agent.evaluation import answers as answer_evaluation
from history_agent.evaluation.answers import evaluate_answers, quote_matches_page
from history_agent.evaluation.retrieval import RetrievalQuestion
from pydantic import ValidationError


def _citation() -> Citation:
    return Citation(
        evidence_id="E1",
        document_id="zhou",
        document="周恩来年谱",
        pdf_page=688,
        section=["1956年"],
        quote="1956年1月，周恩来参加有关会议。",
        source_type="chronology",
        verification_status="verified",
        extraction_methods=["pymupdf"],
    )


def test_quote_matches_page_ignores_layout_punctuation() -> None:
    assert quote_matches_page(
        "……1956年1月，周恩来参加有关会议。",
        "【1956 年 1 月】\n周恩来参加有关会议。",
    )


def test_unanswerable_question_cannot_declare_gold_evidence() -> None:
    with pytest.raises(ValidationError):
        RetrievalQuestion(
            question_id="bad",
            question="不存在的问题",
            answerable=False,
            expected_evidence=[{"document_id": "doc", "pdf_pages": [1]}],
            tags=["refusal"],
        )


def test_answer_evaluation_scores_grounding_and_refusal(
    work_path: Path, monkeypatch: Any
) -> None:
    question_path = work_path / "questions.json"
    question_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "questions": [
                    {
                        "question_id": "answerable",
                        "question": "周恩来在1956年做了什么？",
                        "expected_document_ids": ["zhou"],
                        "expected_evidence": [
                            {"document_id": "zhou", "pdf_pages": [688]}
                        ],
                        "required_fact_terms": [["参加有关会议"]],
                        "tags": ["timeline"],
                    },
                    {
                        "question_id": "refusal",
                        "question": "爱因斯坦在1925年担任了什么党内职务？",
                        "answerable": False,
                        "tags": ["refusal"],
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    settings = Settings(
        _env_file=None,
        project_root=work_path,
        data_dir=Path("data"),
        llm_api_key=None,
    )

    def fake_answer(active_settings: Settings, request: Any) -> AnswerResponse:
        assert active_settings.llm_enabled is False
        if request.question.startswith("爱因斯坦"):
            return AnswerResponse(
                question=request.question,
                answer="现有本地资料中没有检索到足以回答这个问题的证据。",
                evidence_status="no_evidence",
                generator_mode="extractive",
                llm_status="not_applicable",
                retrieval_mode="hybrid_rrf",
                query_intent="timeline",
                citations=[],
            )
        return AnswerResponse(
            question=request.question,
            answer="- 1956年1月，周恩来参加有关会议。[E1]",
            evidence_status="supported",
            generator_mode="extractive",
            llm_status="disabled",
            retrieval_mode="hybrid_rrf",
            query_intent="timeline",
            citations=[_citation()],
        )

    monkeypatch.setattr(answer_evaluation, "answer_question", fake_answer)
    monkeypatch.setattr(
        answer_evaluation,
        "_load_effective_page_texts",
        lambda active_settings: {
            ("zhou", 688): "【1956 年1月】周恩来参加有关会议。"
        },
    )
    payload = evaluate_answers(
        settings=settings,
        question_set_path=question_path,
        top_k=8,
        run_id="test-run",
    )

    assert payload["metrics"] == {
        "gold_page_hit_rate": 1.0,
        "citation_page_accuracy": 1.0,
        "grounding_pass_rate": 1.0,
        "required_fact_coverage": 1.0,
        "refusal_accuracy": 1.0,
        "answerability_accuracy": 1.0,
    }
    assert all(result["success"] for result in payload["results"])
    assert (settings.reports_dir / "answer_eval_latest.json").is_file()
    assert (settings.reports_dir / "mvp_eval_latest.md").is_file()
