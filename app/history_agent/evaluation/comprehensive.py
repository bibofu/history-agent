from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from history_agent.db import Database
from history_agent.evaluation.intersections import IntersectionCase
from history_agent.evaluation.retrieval import load_question_set
from history_agent.research.organization import get_organization_relationships
from history_agent.research.timeline import get_person_timeline

CoreCategory = Literal[
    "timeline",
    "intersection",
    "viewpoint",
    "event",
    "organization",
    "conflict",
    "refusal",
]
StructuredKind = Literal["timeline", "organization"]

CORE_CATEGORIES: tuple[CoreCategory, ...] = (
    "timeline",
    "intersection",
    "viewpoint",
    "event",
    "organization",
    "conflict",
    "refusal",
)


class ComprehensiveSources(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_questions: str
    intersection_questions: str
    structured_questions: str


class ComprehensiveManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    version: str
    description: str
    sources: ComprehensiveSources
    minimum_questions: int = Field(ge=1)
    category_minimums: dict[CoreCategory, int]

    @model_validator(mode="after")
    def require_all_categories(self) -> ComprehensiveManifest:
        if set(self.category_minimums) != set(CORE_CATEGORIES):
            raise ValueError("category_minimums must cover every core category")
        if any(minimum < 1 for minimum in self.category_minimums.values()):
            raise ValueError("category minimums must be positive")
        return self


class StructuredQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    question: str
    kind: StructuredKind
    category: CoreCategory
    person_id: str
    year: int | None = Field(default=None, ge=1, le=9999)
    at: str | None = None
    expected_source_event_id: str
    expected_relationship_id: str | None = None
    expected_document_id: str
    expected_pdf_page: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_kind_fields(self) -> StructuredQuestion:
        if self.kind == "timeline":
            if self.category != "timeline" or self.year is None or self.at is not None:
                raise ValueError("timeline questions require category=timeline and year")
            if self.expected_relationship_id is not None:
                raise ValueError("timeline questions cannot declare a relationship ID")
        else:
            if self.category != "organization" or self.at is None or self.year is not None:
                raise ValueError("organization questions require category=organization and at")
            if self.expected_relationship_id is None:
                raise ValueError("organization questions require a relationship ID")
        return self


class StructuredQuestionSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    review_method: str
    reviewed_by: str
    reviewed_at: str
    scope: str
    questions: list[StructuredQuestion] = Field(min_length=1)


def load_structured_question_set(path: Path) -> StructuredQuestionSet:
    return StructuredQuestionSet.model_validate(_load_json(path))


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _primary_answer_category(tags: list[str], *, answerable: bool) -> CoreCategory:
    if not answerable:
        return "refusal"
    for category in CORE_CATEGORIES:
        if category in tags:
            return category
    if "contemporary_observation" in tags:
        return "viewpoint"
    raise ValueError("answer question has no core category")


def audit_comprehensive_question_set(
    *, project_root: Path, manifest_path: Path
) -> dict[str, object]:
    """Validate and inventory the composite M8.1 evaluation question set."""

    manifest = ComprehensiveManifest.model_validate(_load_json(manifest_path))
    answer_path = _project_path(project_root, manifest.sources.answer_questions)
    intersection_path = _project_path(
        project_root, manifest.sources.intersection_questions
    )
    structured_path = _project_path(project_root, manifest.sources.structured_questions)
    answer_set = load_question_set(answer_path)
    intersection_payload = _load_json(intersection_path)
    if not isinstance(intersection_payload, dict):
        raise ValueError("intersection question set must be a JSON object")
    raw_intersections = intersection_payload.get("cases")
    if not isinstance(raw_intersections, list):
        raise ValueError("intersection question set requires cases")
    intersections = [IntersectionCase.model_validate(item) for item in raw_intersections]
    structured = load_structured_question_set(structured_path)

    question_ids: list[str] = []
    category_counts: Counter[str] = Counter()
    evidence_questions = 0
    for answer in answer_set.questions:
        question_ids.append(answer.question_id)
        category = _primary_answer_category(answer.tags, answerable=answer.answerable)
        category_counts[category] += 1
        evidence_questions += int(
            not answer.answerable or bool(answer.expected_evidence)
        )
    for intersection in intersections:
        question_ids.append(f"intersection::{intersection.id}")
        category_counts["intersection"] += 1
        evidence_questions += int(
            bool(
                intersection.source_event_id
                and intersection.document_id
                and intersection.pdf_page
            )
        )
    for structured_question in structured.questions:
        question_ids.append(structured_question.question_id)
        category_counts[structured_question.category] += 1
        evidence_questions += int(
            bool(
                structured_question.expected_source_event_id
                and structured_question.expected_document_id
                and structured_question.expected_pdf_page
            )
        )

    total = len(question_ids)
    duplicate_ids = sorted(
        question_id for question_id, count in Counter(question_ids).items() if count > 1
    )
    missing_categories = [
        category
        for category in CORE_CATEGORIES
        if category_counts[category] < manifest.category_minimums[category]
    ]
    gates = {
        f"至少 {manifest.minimum_questions} 个问题": total
        >= manifest.minimum_questions,
        "七类核心能力达到样本下限": not missing_categories,
        "所有问题均有证据或明确拒答标签": evidence_questions == total,
        "问题 ID 全局唯一": not duplicate_ids,
    }
    return {
        "version": manifest.version,
        "manifest": str(manifest_path),
        "question_count": total,
        "source_counts": {
            "answer": len(answer_set.questions),
            "intersection": len(intersections),
            "structured": len(structured.questions),
        },
        "category_counts": {
            category: category_counts[category] for category in CORE_CATEGORIES
        },
        "category_minimums": manifest.category_minimums,
        "evidence_questions": evidence_questions,
        "duplicate_ids": duplicate_ids,
        "missing_categories": missing_categories,
        "review_metadata": {
            "intersection_review_method": intersection_payload.get("review_method"),
            "intersection_reason_quality": intersection_payload.get("reason_quality"),
            "structured_review_method": structured.review_method,
            "structured_reviewed_by": structured.reviewed_by,
            "structured_reviewed_at": structured.reviewed_at,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def evaluate_structured_questions(
    *, database: Database, question_set_path: Path
) -> dict[str, object]:
    """Check that each curated structured query still returns its anchored evidence."""

    question_set = load_structured_question_set(question_set_path)
    results: list[dict[str, object]] = []
    for item in question_set.questions:
        matched = False
        if item.kind == "timeline":
            offset = 0
            while True:
                timeline_response = get_person_timeline(
                    database,
                    person_id=item.person_id,
                    start_year=item.year,
                    end_year=item.year,
                    limit=200,
                    offset=offset,
                )
                matched = any(
                    item.expected_source_event_id in event.source_event_ids
                    and any(
                        evidence.document_id == item.expected_document_id
                        and evidence.pdf_page_start
                        <= item.expected_pdf_page
                        <= evidence.pdf_page_end
                        for evidence in event.evidence
                        if evidence.source_event_id == item.expected_source_event_id
                    )
                    for event in timeline_response.events
                )
                if matched or not timeline_response.has_more:
                    break
                offset += timeline_response.limit
        else:
            assert item.at is not None
            organization_response = get_organization_relationships(
                database,
                person_id=item.person_id,
                at=item.at,
                limit=200,
            )
            matched = any(
                relation.relationship_id == item.expected_relationship_id
                and relation.source_event_id == item.expected_source_event_id
                and any(
                    evidence.document_id == item.expected_document_id
                    and evidence.pdf_page_start
                    <= item.expected_pdf_page
                    <= evidence.pdf_page_end
                    for evidence in relation.evidence
                )
                for relation in organization_response.relationships
            )
        results.append(
            {
                "question_id": item.question_id,
                "kind": item.kind,
                "success": matched,
                "expected_source_event_id": item.expected_source_event_id,
                "expected_document_id": item.expected_document_id,
                "expected_pdf_page": item.expected_pdf_page,
            }
        )
    passed = sum(int(bool(item["success"])) for item in results)
    return {
        "questions": len(results),
        "passed_questions": passed,
        "pass_rate": round(passed / len(results), 6),
        "passed": passed == len(results),
        "review_method": question_set.review_method,
        "results": results,
    }
