from __future__ import annotations

import json
from pathlib import Path

import pytest
from history_agent.evaluation.comprehensive import (
    ComprehensiveManifest,
    StructuredQuestion,
    audit_comprehensive_question_set,
    evaluate_structured_questions,
)


def test_repository_comprehensive_set_has_105_evidenced_questions() -> None:
    project_root = Path.cwd()

    result = audit_comprehensive_question_set(
        project_root=project_root,
        manifest_path=project_root / "evals" / "comprehensive_evaluation.json",
    )

    assert result["passed"] is True
    assert result["question_count"] == 105
    assert result["evidence_questions"] == 105
    assert result["source_counts"] == {
        "answer": 47,
        "intersection": 40,
        "structured": 18,
    }
    assert result["category_counts"] == {
        "timeline": 23,
        "intersection": 47,
        "viewpoint": 8,
        "event": 11,
        "organization": 9,
        "conflict": 4,
        "refusal": 3,
    }
    assert result["review_metadata"] == {
        "intersection_review_method": "reviewer_confirmed_without_case_rationales",
        "intersection_reason_quality": "waived",
        "structured_review_method": "curated_page_anchored_records",
        "structured_reviewed_by": "project_maintainer",
        "structured_reviewed_at": "2026-09-06",
    }


def test_manifest_requires_a_minimum_for_every_core_category() -> None:
    path = Path("evals/comprehensive_evaluation.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["category_minimums"]["conflict"]

    with pytest.raises(ValueError, match="every core category"):
        ComprehensiveManifest.model_validate(payload)


def test_structured_question_rejects_kind_specific_field_mismatch() -> None:
    with pytest.raises(ValueError, match="timeline questions require"):
        StructuredQuestion.model_validate(
            {
                "question_id": "bad",
                "question": "错误样本",
                "kind": "timeline",
                "category": "timeline",
                "person_id": "mao_zedong",
                "at": "1949",
                "expected_source_event_id": "event",
                "expected_document_id": "document",
                "expected_pdf_page": 1,
            }
        )


def test_structured_evaluation_checks_timeline_and_organization_evidence(
    monkeypatch: pytest.MonkeyPatch, work_path: Path
) -> None:
    question_set = work_path / "structured.json"
    question_set.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "review_method": "test",
                "reviewed_by": "tester",
                "reviewed_at": "2026-09-06",
                "scope": "test",
                "questions": [
                    {
                        "question_id": "timeline",
                        "question": "timeline",
                        "kind": "timeline",
                        "category": "timeline",
                        "person_id": "person",
                        "year": 1949,
                        "expected_source_event_id": "source-event",
                        "expected_document_id": "document",
                        "expected_pdf_page": 12,
                    },
                    {
                        "question_id": "organization",
                        "question": "organization",
                        "kind": "organization",
                        "category": "organization",
                        "person_id": "person",
                        "at": "1949",
                        "expected_relationship_id": "relationship",
                        "expected_source_event_id": "source-event",
                        "expected_document_id": "document",
                        "expected_pdf_page": 12,
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    class Evidence:
        source_event_id = "source-event"
        document_id = "document"
        pdf_page_start = 12
        pdf_page_end = 12

    class Event:
        source_event_ids = ["source-event"]
        evidence = [Evidence()]

    class TimelineResponse:
        events = [Event()]
        has_more = False
        limit = 200

    class Relation:
        relationship_id = "relationship"
        source_event_id = "source-event"
        evidence = [Evidence()]

    class OrganizationResponse:
        relationships = [Relation()]

    monkeypatch.setattr(
        "history_agent.evaluation.comprehensive.get_person_timeline",
        lambda *args, **kwargs: TimelineResponse(),
    )
    monkeypatch.setattr(
        "history_agent.evaluation.comprehensive.get_organization_relationships",
        lambda *args, **kwargs: OrganizationResponse(),
    )

    result = evaluate_structured_questions(
        database=object(),  # type: ignore[arg-type]
        question_set_path=question_set,
    )

    assert result["passed"] is True
    assert result["passed_questions"] == 2
