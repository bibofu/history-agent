import math
from typing import Any

import pytest
from history_agent.answering.models import AnswerResponse, Citation
from history_agent.evaluation.golden import (
    EvidenceAnchor,
    ForbiddenClaim,
    GoldenCase,
    GoldenDataset,
    GoldenMetadata,
    GoldFact,
    RelevantEvidence,
    RetrievalGold,
    RouteGold,
    _selected_dimensions,
    aggregate_results,
    evaluate_citations,
    evaluate_context_hits,
    evaluate_generation,
    evaluate_retrieval_hits,
    evaluate_routing,
    hit_matches_evidence,
)
from history_agent.retrieval.models import SearchHit
from pydantic import ValidationError


def _hit(rank: int, page: int, *, end: int | None = None, document: str = "doc") -> SearchHit:
    return SearchHit(
        rank=rank,
        chunk_id=f"chunk-{rank}",
        document_id=document,
        title="测试文献",
        filename="test.pdf",
        source_type="test",
        verification_status="reviewed",
        pdf_page_start=page,
        pdf_page_end=end or page,
        section_path=[],
        text="测试文本",
        year_mentions=[],
        people=[],
        extraction_methods=["manual"],
        score=1.0 / rank,
        matched_terms=[],
    )


def _gold_case(*, complete: bool = True) -> GoldenCase:
    return GoldenCase(
        id="case",
        question="测试问题？",
        category="fact",
        difficulty="easy",
        answerability="answerable",
        eval_dimensions=[
            "routing",
            "retrieval",
            "ranking",
            "context",
            "generation",
            "citation",
        ],
        route_gold=RouteGold(preferred_routes=["hybrid_retrieval"]),
        retrieval_gold=RetrievalGold(
            required_evidence=[EvidenceAnchor(document_id="doc", pdf_page=10)],
            relevant_evidence=(
                [
                    RelevantEvidence(document_id="doc", pdf_page=10, relevance=2),
                    RelevantEvidence(document_id="doc", pdf_page=11, relevance=1),
                ]
                if complete
                else []
            ),
            relevance_complete=complete,
        ),
        gold_facts=[
            GoldFact(
                fact_id="required",
                claim="回答覆盖甲事实",
                importance="required",
                evidence=[EvidenceAnchor(document_id="doc", pdf_page=10)],
                deterministic_patterns=["甲事实"],
            )
        ],
        metadata=GoldenMetadata(
            source="test", reviewed=True, reviewed_by="tester", reviewed_at="2026-09-11"
        ),
    )


def _response(
    *,
    answer: str = "甲事实。[E1]",
    status: str = "supported",
    mode: str = "planned_hybrid_rrf",
    citations: list[Citation] | None = None,
) -> AnswerResponse:
    return AnswerResponse(
        question="测试问题？",
        answer=answer,
        evidence_status=status,  # type: ignore[arg-type]
        generator_mode="extractive",
        llm_status="disabled" if citations else "not_applicable",
        retrieval_mode=mode,
        query_intent="general",
        citations=citations or [],
    )


def _citation(page: int = 10) -> Citation:
    return Citation(
        evidence_id="E1",
        document_id="doc",
        document="测试文献",
        pdf_page=page,
        section=[],
        quote="这是一段长度足够并且来自原页面的测试引文。",
        source_type="test",
        verification_status="reviewed",
        extraction_methods=["manual"],
    )


def test_golden_schema_accepts_a_strict_valid_dataset() -> None:
    dataset = GoldenDataset(
        schema_version=1,
        version="test-v1",
        description="test",
        cases=[_gold_case()],
    )

    assert dataset.cases[0].retrieval_gold is not None
    assert dataset.cases[0].retrieval_gold.relevance_complete is True


@pytest.mark.parametrize(
    "updates",
    [
        {"route_gold": None},
        {"retrieval_gold": None},
        {"gold_facts": []},
        {"eval_dimensions": ["retrieval", "retrieval"]},
    ],
)
def test_invalid_golden_case_is_rejected(updates: dict[str, Any]) -> None:
    payload = _gold_case().model_dump()
    payload.update(updates)

    with pytest.raises(ValidationError):
        GoldenCase.model_validate(payload)


def test_unanswerable_case_rejects_gold_facts() -> None:
    payload = _gold_case().model_dump()
    payload.update(
        {
            "category": "refusal",
            "answerability": "unanswerable",
            "eval_dimensions": ["generation"],
            "route_gold": None,
            "retrieval_gold": None,
        }
    )

    with pytest.raises(ValidationError, match="unanswerable"):
        GoldenCase.model_validate(payload)


def test_page_relevance_matches_any_page_in_cross_page_chunk() -> None:
    evidence = EvidenceAnchor(document_id="doc", pdf_page=11)

    assert hit_matches_evidence(_hit(1, 10, end=12), evidence)
    assert not hit_matches_evidence(
        _hit(1, 12, end=13), EvidenceAnchor(document_id="x", pdf_page=12)
    )


def test_retrieval_metrics_are_page_level_and_hand_verifiable() -> None:
    hits = [_hit(1, 99), _hit(2, 10), _hit(3, 11)]

    result = evaluate_retrieval_hits(_gold_case(), hits)

    assert result["hit_at_5"] is True
    assert result["recall_at_5"] == 1.0
    assert result["precision_at_5"] == 0.4
    assert result["mrr"] == 0.5
    expected = ((3 / math.log2(3)) + (1 / math.log2(4))) / (
        3 + (1 / math.log2(3))
    )
    assert result["ndcg_at_5"] == pytest.approx(expected)
    assert result["required_evidence_ranks"][0]["rank"] == 2


def test_precision_and_ndcg_are_null_for_incomplete_relevance() -> None:
    result = evaluate_retrieval_hits(_gold_case(complete=False), [_hit(1, 10)])

    assert result["hit_at_5"] is True
    assert result["recall_at_5"] == 1.0
    assert result["precision_at_5"] is None
    assert result["precision_at_5_evaluable"] is False
    assert result["ndcg_at_10"] is None
    assert result["ndcg_at_10_evaluable"] is False


def test_duplicate_gold_page_does_not_inflate_recall_or_ndcg() -> None:
    hits = [_hit(1, 10), _hit(2, 10), _hit(3, 11)]

    result = evaluate_retrieval_hits(_gold_case(), hits)

    assert result["recall_at_5"] == 1.0
    assert result["ndcg_at_5"] < 1.0


def test_context_quality_reports_noise_and_redundancy() -> None:
    duplicate = _hit(3, 10)
    hits = [_hit(1, 10), _hit(2, 99), duplicate]

    result = evaluate_context_hits(_gold_case(), hits, top_k=3)

    assert result["relevant_evidence_ratio"] == pytest.approx(2 / 3)
    assert result["irrelevant_context_ratio"] == pytest.approx(1 / 3)
    assert result["duplicate_start_page_ratio"] == pytest.approx(1 / 3)


def test_aggregate_uses_only_evaluable_metric_values() -> None:
    results = [
        {"case_id": "a", "retrieval": {"precision_at_5": None}},
        {"case_id": "b", "retrieval": {"precision_at_5": 0.4}},
    ]

    aggregate = aggregate_results(results)

    assert aggregate["retrieval"]["precision_at_5"] == {
        "value": 0.4,
        "evaluable_cases": 1,
    }


def test_eval_dimensions_filtering_skips_unannotated_layer() -> None:
    case = _gold_case().model_copy(update={"eval_dimensions": ["retrieval"]})

    assert _selected_dimensions(case, "retrieval") == {"retrieval"}
    assert _selected_dimensions(case, "generation") == set()
    assert _selected_dimensions(case, "all") == {"retrieval"}


def test_unanswerable_case_scores_refusal_and_false_answer() -> None:
    case = GoldenCase(
        id="refusal",
        question="资料外问题？",
        category="refusal",
        difficulty="easy",
        answerability="unanswerable",
        eval_dimensions=["generation"],
        metadata=GoldenMetadata(
            source="test", reviewed=True, reviewed_by="tester", reviewed_at="2026-09-11"
        ),
    )

    result = evaluate_generation(
        case,
        _response(answer="现有资料无法回答。", status="no_evidence"),
    )

    assert result["refusal_correct"] is True
    assert result["false_answer"] is False
    assert result["answer_correctness"] is True


def test_required_and_optional_facts_have_separate_recall() -> None:
    case = _gold_case().model_copy(
        update={
            "gold_facts": [
                *_gold_case().gold_facts,
                GoldFact(
                    fact_id="optional",
                    claim="回答覆盖乙事实",
                    importance="optional",
                    evidence=[EvidenceAnchor(document_id="doc", pdf_page=11)],
                    deterministic_patterns=["乙事实"],
                ),
            ]
        }
    )

    result = evaluate_generation(case, _response(answer="这里只覆盖甲事实。[E1]"))

    assert result["required_fact_recall"] == 1.0
    assert result["optional_fact_recall"] == 0.0
    assert result["facts"][1]["judge"] == "not_evaluated"


def test_forbidden_claim_is_an_independent_deterministic_signal() -> None:
    case = _gold_case().model_copy(
        update={
            "forbidden_claims": [
                ForbiddenClaim(
                    claim="错误因果",
                    reason="先后不等于因果",
                    deterministic_patterns=["因此导致"],
                )
            ]
        }
    )

    result = evaluate_generation(case, _response(answer="甲事实，因此导致错误结果。[E1]"))

    assert result["forbidden_claim_violation"] is True
    assert result["answer_correctness"] is False


def test_route_can_match_one_of_multiple_preferred_routes() -> None:
    case = _gold_case().model_copy(
        update={
            "route_gold": RouteGold(
                preferred_routes=["structured_timeline", "hybrid_retrieval"]
            )
        }
    )

    result = evaluate_routing(case, _response(mode="planned_hybrid_rrf"))

    assert result["actual_route"] == "hybrid_retrieval"
    assert result["route_match"] is True


def test_citation_metrics_keep_presence_page_and_semantics_separate() -> None:
    citation = _citation()
    response = _response(citations=[citation])
    page_text = "前文。这是一段长度足够并且来自原页面的测试引文。后文。"

    result = evaluate_citations(
        _gold_case(),
        response,
        page_texts={("doc", 10): page_text},
        semantic_judgment={
            "status": "used",
            "verdict": "pass",
            "rationale": "supported",
        },
    )

    assert result["citation_presence"] is True
    assert result["citation_page_validity"] == 1.0
    assert result["citation_quote_consistency"] == 1.0
    assert result["claim_to_citation_coverage"] is True
    assert result["citation_precision"] == 1.0
    assert result["citation_recall"] == 0.5
    assert result["semantic_support"] is True
