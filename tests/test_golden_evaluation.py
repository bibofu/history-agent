import json
import math
from pathlib import Path
from typing import Any

import pytest
from history_agent.answering.models import AnswerResponse, Citation
from history_agent.config import Settings
from history_agent.errors import RetrievalError
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
    _index_attribution,
    _retrieval_branches,
    _selected_dimensions,
    aggregate_results,
    compare_golden_runs,
    evaluate_citations,
    evaluate_context_hits,
    evaluate_generation,
    evaluate_retrieval_hits,
    evaluate_routing,
    hit_matches_evidence,
    load_golden_dataset,
    run_golden_benchmark,
)
from history_agent.processing.chunks import index_artifact_manifest, index_artifact_sha256
from history_agent.retrieval.models import SearchHit, SearchResponse
from pydantic import ValidationError
from typer.testing import CliRunner


def _hit(
    rank: int,
    page: int,
    *,
    end: int | None = None,
    document: str = "doc",
    text: str = "测试文本",
) -> SearchHit:
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
        text=text,
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


def _citation(page: int = 10, *, end: int | None = None) -> Citation:
    return Citation(
        evidence_id="E1",
        document_id="doc",
        document="测试文献",
        pdf_page=page,
        pdf_page_end=end,
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
    assert result["required_evidence_recall_at_5"] == 1.0
    assert result["precision_at_5"] == 0.4
    assert result["mrr"] == 0.5
    expected = ((3 / math.log2(3)) + (1 / math.log2(4))) / (3 + (1 / math.log2(3)))
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
        {
            "case_id": "a",
            "retrieval": {
                "precision_at_5": 0.9,
                "metric_status": {
                    "precision_at_5": {
                        "value": None,
                        "evaluable": False,
                        "reason": "incomplete",
                    }
                },
            },
        },
        {
            "case_id": "b",
            "retrieval": {
                "precision_at_5": 0.4,
                "metric_status": {
                    "precision_at_5": {
                        "value": 0.4,
                        "evaluable": True,
                        "reason": "complete",
                    }
                },
            },
        },
    ]

    aggregate = aggregate_results(results)

    metric = aggregate["retrieval"]["precision_at_5"]
    assert metric["value"] == 0.4
    assert metric["evaluable_cases"] == 1
    assert metric["total_cases"] == 2
    assert metric["unevaluable_case_ids"] == ["a"]


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
    assert result["optional_fact_recall"] is None
    assert result["facts"][1]["judge"] == "not_evaluated"
    assert result["facts"][1]["evaluable"] is False


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
            "route_gold": RouteGold(preferred_routes=["structured_timeline", "hybrid_retrieval"])
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


def _unanswerable_case() -> GoldenCase:
    return GoldenCase(
        id="refusal",
        question="资料外问题？",
        category="refusal",
        difficulty="easy",
        answerability="unanswerable",
        eval_dimensions=["generation", "citation"],
        metadata=GoldenMetadata(
            source="test", reviewed=True, reviewed_by="tester", reviewed_at="2026-09-11"
        ),
    )


def test_unmatched_fact_is_unknown_without_semantic_judge() -> None:
    result = evaluate_generation(_gold_case(), _response(answer="同义改写但未命中模式。"))

    assert result["facts"][0]["covered"] is None
    assert result["facts"][0]["evaluable"] is False
    assert result["required_fact_recall"] is None
    assert result["answer_correctness"] is None


def test_semantic_judge_explicit_false_is_an_evaluable_failure() -> None:
    result = evaluate_generation(
        _gold_case(),
        _response(answer="同义改写但未命中模式。"),
        fact_judgment={
            "status": "used",
            "decisions": [{"fact_id": "required", "covered": False, "reason": "not stated"}],
        },
    )

    assert result["facts"][0]["covered"] is False
    assert result["required_fact_recall"] == 0.0
    assert result["answer_correctness"] is False


@pytest.mark.parametrize("status", ["timeout", "invalid_response", "not_configured"])
def test_failed_fact_judge_does_not_enter_aggregate(status: str) -> None:
    generation = evaluate_generation(
        _gold_case(),
        _response(answer="未命中。"),
        fact_judgment={"status": status, "decisions": []},
    )
    aggregate = aggregate_results([{"case_id": "x", "generation": generation}])

    assert aggregate["generation"]["required_fact_recall"]["value"] is None
    assert aggregate["generation"]["required_fact_recall"]["evaluable_cases"] == 0


def test_uncited_substantive_unanswerable_answer_is_false_answer() -> None:
    result = evaluate_generation(
        _unanswerable_case(),
        _response(answer="他在1925年担任该职务。", status="supported"),
    )

    assert result["answer_disposition"] == "answered_uncited"
    assert result["refusal_correct"] is False
    assert result["false_answer"] is True


def test_refusal_boilerplate_does_not_hide_a_substantive_false_answer() -> None:
    result = evaluate_generation(
        _unanswerable_case(),
        _response(
            answer="现有资料无法回答，但他在1925年担任该职务。",
            status="no_evidence",
        ),
    )

    assert result["answer_disposition"] == "answered_uncited"
    assert result["false_answer"] is True


def test_unanswerable_answer_with_irrelevant_citation_fails() -> None:
    result = evaluate_generation(
        _unanswerable_case(),
        _response(answer="他担任该职务。[E1]", citations=[_citation(99)]),
    )

    assert result["answer_disposition"] == "answered"
    assert result["refusal_correct"] is False
    assert result["false_answer"] is True


def test_unlabeled_forbidden_claim_metric_is_excluded() -> None:
    generation = evaluate_generation(_gold_case(), _response())
    aggregate = aggregate_results([{"case_id": "x", "generation": generation}])

    assert generation["forbidden_claim_violation"] is None
    assert aggregate["generation"]["forbidden_claim_violation"]["evaluable_cases"] == 0


def test_identical_cross_page_chunks_make_ndcg_unevaluable() -> None:
    hits = [_hit(1, 10, end=11), _hit(2, 10, end=11)]
    result = evaluate_retrieval_hits(_gold_case(), hits)

    assert result["annotated_recall_at_5"] == 1.0
    assert result["ndcg_at_5"] is None
    assert result["metric_status"]["ndcg_at_5"]["evaluable"] is False


def test_citation_range_checks_quote_on_non_start_page() -> None:
    citation = _citation(10, end=11)
    result = evaluate_citations(
        _gold_case(),
        _response(citations=[citation]),
        page_texts={
            ("doc", 10): "起始页没有目标引文。",
            ("doc", 11): "这是一段长度足够并且来自原页面的测试引文。",
        },
    )

    assert result["citation_quote_consistency"] == 1.0
    assert result["details"][0]["quote_match_pages"] == [11]


def test_citation_range_is_unknown_when_a_missing_page_could_hold_quote() -> None:
    result = evaluate_citations(
        _gold_case(),
        _response(citations=[_citation(10, end=11)]),
        page_texts={("doc", 10): "没有目标引文。"},
    )

    assert result["citation_quote_consistency"] is None
    assert result["metric_status"]["citation_quote_consistency"]["evaluable"] is False


def test_same_page_different_text_is_not_true_redundancy() -> None:
    result = evaluate_context_hits(
        _gold_case(),
        [_hit(1, 10, text="第一段文本"), _hit(2, 10, text="第二段完全不同")],
        top_k=2,
    )

    assert result["start_page_concentration_ratio"] == 0.5
    assert result["redundancy_ratio"] == 0.0


def test_empty_and_short_hit_lists_are_handled() -> None:
    empty = evaluate_retrieval_hits(_gold_case(), [])
    short = evaluate_retrieval_hits(_gold_case(), [_hit(1, 10)])

    assert empty["hit_at_5"] is False
    assert empty["precision_at_5"] == 0.0
    assert short["precision_at_5"] == 0.2
    assert short["returned_count"] == 1


def test_repository_golden_dataset_loads_without_complete_relevance() -> None:
    dataset = load_golden_dataset(Path("evals/golden/golden_questions.json"))

    assert dataset.cases
    assert all(
        case.retrieval_gold is None or not case.retrieval_gold.relevance_complete
        for case in dataset.cases
    )


def _search_response(hits: list[SearchHit]) -> SearchResponse:
    return SearchResponse(
        query="测试问题？",
        query_intent="general",
        query_terms=[],
        query_years=[],
        query_year_range=[],
        query_people=[],
        document_filters=[],
        include_out_of_scope=False,
        hits=hits,
        retrieval_mode="hybrid_rrf",
        rag_framework="llamaindex",
        chunker_version="test",
    )


def test_full_runner_uses_stub_backends_and_records_actual_metadata(work_path: Path) -> None:
    case = _gold_case().model_copy(
        update={"eval_dimensions": ["routing", "retrieval", "generation"]}
    )
    dataset_path = work_path / "golden.json"
    dataset_path.write_text(
        GoldenDataset(
            schema_version=1, version="test-v1", description="test", cases=[case]
        ).model_dump_json(),
        encoding="utf-8",
    )
    settings = Settings(project_root=work_path, _env_file=None)
    settings.reports_dir.mkdir(parents=True)
    settings.keyword_index_path.parent.mkdir(parents=True, exist_ok=True)
    settings.keyword_index_path.write_bytes(b"keyword-index")
    settings.vector_index_path.mkdir(parents=True)
    (settings.vector_index_path / "segment.bin").write_bytes(b"vector-index")
    chunking = {
        "version": "manifest-v1",
        "target_chars": 650,
        "max_chars": 900,
        "overlap": 0,
    }
    keyword_manifest = {
        "chunking": chunking,
        "chunk_artifact_sha256": "abc",
        "index_artifact_sha256": index_artifact_sha256(settings.keyword_index_path),
        "build_run_id": "keyword-build",
        "git_commit": "commit",
        "keyword_index_version": "keyword-v1",
    }
    vector_manifest = {
        "chunking": chunking,
        "chunk_artifact_sha256": "abc",
        "index_artifact_sha256": index_artifact_sha256(settings.vector_index_path),
        "build_run_id": "vector-build",
        "git_commit": "commit",
        "embedding_model": "embed-v1",
        "vector_index_version": "vector-v1",
    }
    (settings.reports_dir / "keyword_index_latest.json").write_text(
        json.dumps({"manifest": keyword_manifest}), encoding="utf-8"
    )
    (settings.reports_dir / "vector_index_latest.json").write_text(
        json.dumps({"manifest": vector_manifest}), encoding="utf-8"
    )

    payload = run_golden_benchmark(
        settings=settings,
        dataset_path=dataset_path,
        write_reports=False,
        search_backend=lambda **_: _search_response([_hit(1, 10)]),
        answer_backend=lambda *_: _response(),
    )

    assert payload["results"][0]["generation"]["answer_correctness"] is True
    assert payload["run_metadata"]["evaluated_case_ids"] == ["case"]
    assert payload["run_metadata"]["retrieval_config"]["answer_top_k"] == 10
    assert payload["run_metadata"]["chunking"] == chunking
    assert payload["run_metadata"]["embedding_model"] == "embed-v1"
    assert payload["run_metadata"]["chunk_artifact_sha256"] == "abc"
    assert payload["run_metadata"]["index_manifest_consistent"] is True
    assert payload["run_metadata"]["attribution_safe"] is True


def test_index_manifest_uses_actual_chunk_report(work_path: Path) -> None:
    chunks = work_path / "data" / "processed" / "chunks"
    reports = work_path / "data" / "reports"
    chunks.mkdir(parents=True)
    reports.mkdir(parents=True)
    (chunks / "doc.jsonl").write_text('{"text":"one"}\n', encoding="utf-8")
    index_path = work_path / "keyword.db"
    index_path.write_bytes(b"index")
    first = index_artifact_manifest(
        chunks_dir=chunks,
        reports_dir=reports,
        project_root=work_path,
        run_id="old",
        keyword_index_version="k1",
        index_path=index_path,
    )
    assert first["chunking"] is None
    assert first["warnings"] == ["chunk build manifest is missing"]
    chunk_manifest = {
        "chunking": {"version": "v", "target_chars": 1, "max_chars": 2, "overlap": 0},
        "chunk_artifact_sha256": first["chunk_artifact_sha256"],
        "build_run_id": "chunks-1",
    }
    (reports / "chunk_build_latest.json").write_text(
        json.dumps({"manifest": chunk_manifest}), encoding="utf-8"
    )

    manifest = index_artifact_manifest(
        chunks_dir=chunks,
        reports_dir=reports,
        project_root=work_path,
        run_id="keyword-1",
        keyword_index_version="k1",
        index_path=index_path,
    )

    assert manifest["chunking"] == chunk_manifest["chunking"]
    assert manifest["chunk_artifact_sha256"] == chunk_manifest["chunk_artifact_sha256"]
    assert manifest["build_run_id"] == "keyword-1"
    assert manifest["index_artifact_sha256"]


def _comparison_run(
    *, sha: str = "same", case_ids: list[str] | None = None, value: float = 0.5
) -> dict[str, Any]:
    ids = case_ids or ["a"]
    return {
        "schema_version": 1,
        "dataset": {"sha256": sha},
        "requested_dimension": "all",
        "run_metadata": {
            "retrieval_config": {"top_k": 10},
            "attribution_safe": True,
            "evaluator": {
                "version": "unified-golden-evaluator-v2",
                "metric_definition_version": "golden-metrics-v2",
                "page_mapping_version": "page-range-evidence-unit-v2",
                "text_normalization_version": "alnum-casefold-v1",
                "relevance_schema_version": "page-positive-and-graded-v1",
            },
        },
        "results": [{"case_id": item} for item in ids],
        "aggregate": {
            "retrieval": {
                "hit_rate_at_5": {
                    "value": value,
                    "evaluable_cases": len(ids),
                    "evaluable_case_ids": ids,
                    "evaluation_bases": ["deterministic"],
                }
            },
            "operational": {"token_usage": {"answer": {"total_tokens": 3}}},
        },
    }


def test_compare_success_incompatible_and_evaluable_set_warning(work_path: Path) -> None:
    path_a = work_path / "a.json"
    path_b = work_path / "b.json"
    path_a.write_text(json.dumps(_comparison_run()), encoding="utf-8")
    path_b.write_text(json.dumps(_comparison_run(value=0.7)), encoding="utf-8")
    success = compare_golden_runs(path_a, path_b)
    assert success["compatible"] is True
    assert success["metrics"]["retrieval.hit_rate_at_5"]["delta"] == 0.2

    path_b.write_text(json.dumps(_comparison_run(sha="different")), encoding="utf-8")
    assert compare_golden_runs(path_a, path_b)["compatible"] is False

    changed = _comparison_run()
    changed["aggregate"]["retrieval"]["hit_rate_at_5"]["evaluable_case_ids"] = []
    path_b.write_text(json.dumps(changed), encoding="utf-8")
    warning = compare_golden_runs(path_a, path_b)
    assert warning["metrics"]["retrieval.hit_rate_at_5"]["delta"] is None
    assert any("evaluable case set differs" in item for item in warning["warnings"])


def test_golden_cli_help_json_and_error_paths(
    monkeypatch: pytest.MonkeyPatch, work_path: Path
) -> None:
    from history_agent import cli

    runner = CliRunner()
    assert runner.invoke(cli.app, ["eval", "golden", "--help"]).exit_code == 0
    settings = Settings(project_root=work_path, _env_file=None)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    captured: dict[str, Any] = {}

    def successful_run(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"run_id": "x", "aggregate": {"case_count": 0, "retrieval": {}}}

    monkeypatch.setattr(cli, "run_golden_benchmark", successful_run)
    success = runner.invoke(cli.app, ["eval", "golden", "--json", "--case-id", "a"])
    assert success.exit_code == 0
    assert json.loads(success.stdout)["run_id"] == "x"
    assert captured["case_ids"] == {"a"}

    def fail(**_: Any) -> dict[str, Any]:
        raise RetrievalError("missing index")

    monkeypatch.setattr(cli, "run_golden_benchmark", fail)
    failure = runner.invoke(cli.app, ["eval", "golden", "--json"])
    assert failure.exit_code == 2
    assert json.loads(failure.stdout)["error"] == "RetrievalError"


def test_golden_compare_cli_json_and_incompatible_exit(work_path: Path) -> None:
    from history_agent import cli

    path_a = work_path / "a.json"
    path_b = work_path / "b.json"
    path_a.write_text(json.dumps(_comparison_run()), encoding="utf-8")
    path_b.write_text(json.dumps(_comparison_run(value=0.75)), encoding="utf-8")
    runner = CliRunner()

    success = runner.invoke(cli.app, ["eval", "golden-compare", str(path_a), str(path_b), "--json"])
    assert success.exit_code == 0
    assert json.loads(success.stdout)["compatible"] is True

    path_b.write_text(json.dumps(_comparison_run(sha="other")), encoding="utf-8")
    incompatible = runner.invoke(
        cli.app, ["eval", "golden-compare", str(path_a), str(path_b), "--json"]
    )
    assert incompatible.exit_code == 2
    assert json.loads(incompatible.stdout)["compatible"] is False


def _required_fact(fact_id: str, pattern: str, *, importance: str = "required") -> GoldFact:
    return GoldFact(
        fact_id=fact_id,
        claim=f"事实{fact_id}",
        importance=importance,  # type: ignore[arg-type]
        evidence=[EvidenceAnchor(document_id="doc", pdf_page=10)],
        deterministic_patterns=[pattern],
    )


def test_fact_recall_aggregate_is_micro_over_fact_units() -> None:
    one_fact = _gold_case().model_copy(
        update={"id": "one", "gold_facts": [_required_fact("f1", "命中")]}
    )
    three_facts = _gold_case().model_copy(
        update={
            "id": "three",
            "gold_facts": [
                _required_fact("f1", "不会命中1"),
                _required_fact("f2", "不会命中2"),
                _required_fact("f3", "不会命中3"),
            ],
        }
    )
    first = evaluate_generation(one_fact, _response(answer="命中"))
    second = evaluate_generation(
        three_facts,
        _response(answer="均未覆盖"),
        fact_judgment={
            "status": "used",
            "decisions": [
                {"fact_id": fact_id, "covered": False, "reason": "not stated"}
                for fact_id in ("f1", "f2", "f3")
            ],
        },
    )

    aggregate = aggregate_results(
        [
            {"case_id": "one", "generation": first},
            {"case_id": "three", "generation": second},
        ]
    )["generation"]["required_fact_recall"]

    assert aggregate["value"] == 0.25
    assert aggregate["aggregation"] == "micro_over_fact_units"
    assert aggregate["evaluable_unit_ids"] == [
        "one:f1",
        "three:f1",
        "three:f2",
        "three:f3",
    ]


def _fact_metric_run(unit_ids: list[str], value: float) -> dict[str, Any]:
    run = _comparison_run(value=0.5)
    evaluator = run["run_metadata"]["evaluator"]
    evaluator.update(
        {
            "fact_judge_version": "fact-v1",
            "fact_judge_provider": "test-provider",
            "fact_judge_model": "judge-a",
            "citation_judge_version": "citation-v1",
            "citation_judge_provider": "test-provider",
            "citation_judge_model": "judge-a",
        }
    )
    run["aggregate"]["generation"] = {
        "required_fact_recall": {
            "value": value,
            "evaluable_cases": 1,
            "evaluable_case_ids": ["a"],
            "evaluable_unit_ids": unit_ids,
            "evaluation_bases": ["semantic"],
        },
        "answer_correctness": {
            "value": value,
            "evaluable_cases": 1,
            "evaluable_case_ids": ["a"],
            "evaluation_bases": ["deterministic"],
        },
    }
    return run


def test_fact_unit_difference_suppresses_only_fact_delta(work_path: Path) -> None:
    path_a = work_path / "a.json"
    path_b = work_path / "b.json"
    path_a.write_text(json.dumps(_fact_metric_run(["a:f1"], 1.0)), encoding="utf-8")
    path_b.write_text(
        json.dumps(_fact_metric_run(["a:f1", "a:f2", "a:f3"], 2 / 3)),
        encoding="utf-8",
    )

    result = compare_golden_runs(path_a, path_b)

    fact = result["metrics"]["generation.required_fact_recall"]
    assert fact["delta"] is None
    assert "evaluable fact unit set differs" in fact["non_comparable_reasons"]
    correctness = result["metrics"]["generation.answer_correctness"]
    assert correctness["comparable"] is True


def test_judge_version_only_blocks_semantic_metrics(work_path: Path) -> None:
    run_a = _fact_metric_run(["a:f1"], 0.5)
    run_b = _fact_metric_run(["a:f1"], 0.7)
    run_b["run_metadata"]["evaluator"]["fact_judge_version"] = "fact-v2"
    path_a = work_path / "a.json"
    path_b = work_path / "b.json"
    path_a.write_text(json.dumps(run_a), encoding="utf-8")
    path_b.write_text(json.dumps(run_b), encoding="utf-8")

    result = compare_golden_runs(path_a, path_b)

    assert result["metrics"]["generation.required_fact_recall"]["comparable"] is False
    assert result["metrics"]["retrieval.hit_rate_at_5"]["comparable"] is True


def test_generation_model_change_does_not_block_retrieval(work_path: Path) -> None:
    run_a = _comparison_run(value=0.5)
    run_b = _comparison_run(value=0.6)
    run_a["run_metadata"]["generation"] = {"model": "model-a"}
    run_b["run_metadata"]["generation"] = {"model": "model-b"}
    path_a = work_path / "a.json"
    path_b = work_path / "b.json"
    path_a.write_text(json.dumps(run_a), encoding="utf-8")
    path_b.write_text(json.dumps(run_b), encoding="utf-8")

    result = compare_golden_runs(path_a, path_b)

    assert result["metrics"]["retrieval.hit_rate_at_5"]["delta"] == 0.1
    assert "generation.model" in result["metadata_differences"]


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("version", "evaluator version differs"),
        ("metric_definition_version", "metric definition version differs"),
    ],
)
def test_evaluator_definition_change_blocks_metric_delta(
    work_path: Path, field: str, reason: str
) -> None:
    run_a = _comparison_run(value=0.5)
    run_b = _comparison_run(value=0.6)
    run_b["run_metadata"]["evaluator"][field] = "changed"
    path_a = work_path / "a.json"
    path_b = work_path / "b.json"
    path_a.write_text(json.dumps(run_a), encoding="utf-8")
    path_b.write_text(json.dumps(run_b), encoding="utf-8")

    metric = compare_golden_runs(path_a, path_b)["metrics"]["retrieval.hit_rate_at_5"]

    assert metric["delta"] is None
    assert reason in metric["non_comparable_reasons"]


def _manifest(branch: str, chunk_hash: str = "chunks") -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "chunking": {
            "version": "v1",
            "target_chars": 650,
            "max_chars": 900,
            "overlap": 0,
        },
        "chunk_artifact_sha256": chunk_hash,
        "index_artifact_sha256": f"{branch}-index",
        "build_run_id": f"{branch}-run",
        "git_commit": "commit",
    }
    if branch == "keyword":
        manifest["keyword_index_version"] = "keyword-v1"
    else:
        manifest["vector_index_version"] = "vector-v1"
        manifest["embedding_model"] = "embed-v1"
    return manifest


def _index_reports(keyword_hash: str = "chunks", vector_hash: str = "chunks") -> dict[str, Any]:
    return {
        "keyword_index": {
            "manifest": _manifest("keyword", keyword_hash),
            "artifact_verified": True,
        },
        "vector_index": {
            "manifest": _manifest("vector", vector_hash),
            "artifact_verified": True,
        },
    }


def test_index_attribution_consistent_and_inconsistent_manifests() -> None:
    consistent = _index_attribution(_index_reports(), ["keyword", "vector"])
    inconsistent = _index_attribution(_index_reports(vector_hash="other"), ["keyword", "vector"])

    assert consistent["index_manifest_consistent"] is True
    assert consistent["attribution_safe"] is True
    assert consistent["chunk_artifact_sha256"] == "chunks"
    assert inconsistent["index_manifest_consistent"] is False
    assert inconsistent["attribution_safe"] is False
    assert inconsistent["chunk_artifact_sha256"] is None
    assert inconsistent["chunking"] is None


def test_index_attribution_legacy_and_single_branch_degradation() -> None:
    legacy = _index_attribution(
        {"keyword_index": {"manifest": None, "artifact_verified": False}},
        ["keyword"],
    )
    reports = _index_reports(vector_hash="other")
    keyword_only = _index_attribution(reports, ["keyword"])

    assert legacy["attribution_safe"] is False
    assert "no index manifest" in legacy["attribution_unsafe_reasons"][0]
    assert keyword_only["attribution_safe"] is True
    assert keyword_only["embedding_model"] is None
    used, degraded = _retrieval_branches(
        [
            {
                "operational": {
                    "retrieval_mode": "keyword_only",
                    "degraded_components": ["vector"],
                }
            }
        ]
    )
    assert used == ["keyword"]
    assert degraded == ["vector"]


def test_optional_only_generation_case_is_rejected_but_non_generation_is_compatible() -> None:
    payload = _gold_case().model_dump()
    payload["gold_facts"] = [_required_fact("optional", "可选", importance="optional").model_dump()]
    with pytest.raises(ValidationError, match="at least one required"):
        GoldenCase.model_validate(payload)

    payload["eval_dimensions"] = ["retrieval"]
    assert GoldenCase.model_validate(payload).gold_facts[0].importance == "optional"


def test_old_report_missing_evaluator_metadata_is_explicitly_not_comparable(
    work_path: Path,
) -> None:
    old = _comparison_run(value=0.5)
    old["run_metadata"].pop("evaluator")
    old["run_metadata"].pop("attribution_safe")
    current = _comparison_run(value=0.6)
    path_a = work_path / "old.json"
    path_b = work_path / "current.json"
    path_a.write_text(json.dumps(old), encoding="utf-8")
    path_b.write_text(json.dumps(current), encoding="utf-8")

    result = compare_golden_runs(path_a, path_b)

    assert result["compatible"] is True
    assert result["attribution_safe"] is False
    metric = result["metrics"]["retrieval.hit_rate_at_5"]
    assert metric["comparable"] is False
    assert "evaluator version is missing" in metric["non_comparable_reasons"]
    from history_agent import cli

    cli_result = CliRunner().invoke(
        cli.app, ["eval", "golden-compare", str(path_a), str(path_b)]
    )
    assert cli_result.exit_code == 0
    assert "metric not comparable" in cli_result.stdout


def test_compare_cli_reports_attribution_unsafe_without_failing(work_path: Path) -> None:
    from history_agent import cli

    run_a = _comparison_run(value=0.5)
    run_b = _comparison_run(value=0.6)
    run_b["run_metadata"]["attribution_safe"] = False
    run_b["run_metadata"]["attribution_unsafe_reasons"] = ["legacy manifest"]
    path_a = work_path / "a.json"
    path_b = work_path / "b.json"
    path_a.write_text(json.dumps(run_a), encoding="utf-8")
    path_b.write_text(json.dumps(run_b), encoding="utf-8")

    output = CliRunner().invoke(cli.app, ["eval", "golden-compare", str(path_a), str(path_b)])

    assert output.exit_code == 0
    assert "comparison possible but attribution unsafe" in output.stdout
    json_output = CliRunner().invoke(
        cli.app, ["eval", "golden-compare", str(path_a), str(path_b), "--json"]
    )
    assert json_output.exit_code == 0
    assert json.loads(json_output.stdout)["attribution_safe"] is False
