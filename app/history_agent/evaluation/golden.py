"""Unified page-grounded Golden Dataset and layered RAG evaluators.

The retrieval evaluator intentionally sends the case question directly to the
retrieval backend.  This isolates index/chunk/fusion changes from the production
query planner.  Routing and generation dimensions separately run the production
answer pipeline, so a benchmark report can show where an improvement did (or did
not) propagate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from history_agent import __version__
from history_agent.answering.models import AnswerResponse, QuestionRequest
from history_agent.answering.service import PROMPT_VERSION, answer_question
from history_agent.answering.validation import validate_grounded_answer
from history_agent.config import Settings, get_settings
from history_agent.evaluation.answers import _load_effective_page_texts, quote_matches_page
from history_agent.evaluation.semantic import (
    FACT_COVERAGE_JUDGE_VERSION,
    SEMANTIC_CITATION_JUDGE_VERSION,
    judge_citation_semantics,
    judge_fact_coverage,
)
from history_agent.processing.chunks import index_artifact_sha256
from history_agent.retrieval.hybrid import RRF_K, search_hybrid_index
from history_agent.retrieval.models import SearchHit, SearchResponse

GoldenCategory = Literal[
    "timeline",
    "intersection",
    "event",
    "viewpoint",
    "fact",
    "multi_hop",
    "conflict",
    "organization",
    "refusal",
    "adversarial",
]
Difficulty = Literal["easy", "medium", "hard"]
Answerability = Literal["answerable", "unanswerable"]
EvalDimension = Literal[
    "routing",
    "retrieval",
    "ranking",
    "context",
    "generation",
    "citation",
]
RequestedDimension = Literal[
    "all",
    "routing",
    "retrieval",
    "ranking",
    "context",
    "generation",
    "citation",
]
FactImportance = Literal["required", "optional"]

ALL_DIMENSIONS: tuple[EvalDimension, ...] = (
    "routing",
    "retrieval",
    "ranking",
    "context",
    "generation",
    "citation",
)
DEFAULT_K_VALUES = (5, 10)
GOLDEN_EVALUATOR_VERSION = "unified-golden-evaluator-v2"
METRIC_DEFINITION_VERSION = "golden-metrics-v2"
PAGE_MAPPING_VERSION = "page-range-evidence-unit-v2"
TEXT_NORMALIZATION_VERSION = "alnum-casefold-v1"
RELEVANCE_SCHEMA_VERSION = "page-positive-and-graded-v1"


class EvidenceAnchor(BaseModel):
    """Stable gold identity; deliberately independent of chunk IDs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(min_length=1)
    pdf_page: int = Field(ge=1)

    @property
    def key(self) -> tuple[str, int]:
        return self.document_id, self.pdf_page


class RelevantEvidence(EvidenceAnchor):
    relevance: Literal[1, 2]


class RetrievalGold(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required_evidence: list[EvidenceAnchor] = Field(default_factory=list)
    relevant_evidence: list[RelevantEvidence] = Field(default_factory=list)
    relevance_complete: bool = False

    @model_validator(mode="after")
    def validate_relevance(self) -> RetrievalGold:
        required = [item.key for item in self.required_evidence]
        relevant = [item.key for item in self.relevant_evidence]
        if len(required) != len(set(required)):
            raise ValueError("required_evidence contains duplicate pages")
        if len(relevant) != len(set(relevant)):
            raise ValueError("relevant_evidence contains duplicate pages")
        if self.relevance_complete and not relevant:
            raise ValueError("complete relevance annotation cannot be empty")
        if relevant and not set(required).issubset(relevant):
            raise ValueError("relevant_evidence must include every required evidence page")
        grades = {item.key: item.relevance for item in self.relevant_evidence}
        if any(grades.get(key) != 2 for key in required if key in grades):
            raise ValueError("required evidence must have relevance=2")
        return self


class RouteGold(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preferred_routes: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_routes(self) -> RouteGold:
        normalized = [item.strip() for item in self.preferred_routes]
        if any(not item for item in normalized):
            raise ValueError("preferred routes cannot be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("preferred routes must be unique")
        self.preferred_routes = normalized
        return self


class GoldFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fact_id: str = Field(min_length=1)
    claim: str = Field(min_length=2)
    importance: FactImportance
    evidence: list[EvidenceAnchor] = Field(min_length=1)
    deterministic_patterns: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_patterns(self) -> GoldFact:
        if any(not item.strip() for item in self.deterministic_patterns):
            raise ValueError("deterministic fact patterns cannot be blank")
        keys = [item.key for item in self.evidence]
        if len(keys) != len(set(keys)):
            raise ValueError("gold fact evidence contains duplicate pages")
        return self


class ForbiddenClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=2)
    reason: str = Field(min_length=2)
    deterministic_patterns: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_patterns(self) -> ForbiddenClaim:
        if any(not item.strip() for item in self.deterministic_patterns):
            raise ValueError("deterministic forbidden-claim patterns cannot be blank")
        return self


class GoldenMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1)
    reviewed: bool
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    legacy_case_id: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def require_reviewer_for_reviewed_data(self) -> GoldenMetadata:
        if self.reviewed and not (self.reviewed_by and self.reviewed_at):
            raise ValueError("reviewed metadata requires reviewed_by and reviewed_at")
        return self


class GoldenCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    question: str = Field(min_length=2)
    category: GoldenCategory
    difficulty: Difficulty
    answerability: Answerability
    eval_dimensions: list[EvalDimension] = Field(min_length=1)
    route_gold: RouteGold | None = None
    retrieval_gold: RetrievalGold | None = None
    gold_facts: list[GoldFact] = Field(default_factory=list)
    forbidden_claims: list[ForbiddenClaim] = Field(default_factory=list)
    reference_answer: str | None = None
    semantic_criteria: list[str] = Field(default_factory=list)
    metadata: GoldenMetadata

    @model_validator(mode="after")
    def validate_case_contract(self) -> GoldenCase:
        if len(self.eval_dimensions) != len(set(self.eval_dimensions)):
            raise ValueError("eval_dimensions must be unique")
        if "routing" in self.eval_dimensions and self.route_gold is None:
            raise ValueError("routing cases require route_gold")
        evidence_dimensions = {"retrieval", "ranking", "context", "citation"}
        needs_evidence = bool(evidence_dimensions.intersection(self.eval_dimensions))
        if self.answerability == "unanswerable":
            if self.gold_facts:
                raise ValueError("unanswerable cases cannot declare gold facts")
            if self.retrieval_gold and (
                self.retrieval_gold.required_evidence or self.retrieval_gold.relevant_evidence
            ):
                raise ValueError("unanswerable cases cannot declare gold evidence")
        else:
            if needs_evidence and (
                self.retrieval_gold is None
                or not (
                    self.retrieval_gold.required_evidence or self.retrieval_gold.relevant_evidence
                )
            ):
                raise ValueError("answerable evidence dimensions require page-level gold")
            if "generation" in self.eval_dimensions and not self.gold_facts:
                raise ValueError("answerable generation cases require gold_facts")
            if "generation" in self.eval_dimensions and not any(
                fact.importance == "required" for fact in self.gold_facts
            ):
                raise ValueError(
                    "answerable generation cases require at least one required gold fact"
                )
        fact_ids = [item.fact_id for item in self.gold_facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("gold fact IDs must be unique within a case")
        if self.retrieval_gold is not None:
            gold_pages = {item.key for item in self.retrieval_gold.required_evidence} | {
                item.key for item in self.retrieval_gold.relevant_evidence
            }
            fact_pages = {anchor.key for fact in self.gold_facts for anchor in fact.evidence}
            if not fact_pages.issubset(gold_pages):
                raise ValueError("gold fact evidence must be declared in retrieval_gold")
        return self


class GoldenDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    version: str = Field(min_length=1)
    description: str = Field(min_length=1)
    cases: list[GoldenCase] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_case_ids(self) -> GoldenDataset:
        ids = [item.id for item in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("golden case IDs must be unique")
        return self


def load_golden_dataset(path: Path) -> GoldenDataset:
    return GoldenDataset.model_validate_json(path.read_text(encoding="utf-8"))


def _hit_page_keys(hit: SearchHit) -> set[tuple[str, int]]:
    """Map a chunk to every physical page in its inclusive page range."""

    return {(hit.document_id, page) for page in range(hit.pdf_page_start, hit.pdf_page_end + 1)}


def hit_matches_evidence(hit: SearchHit, evidence: EvidenceAnchor) -> bool:
    return evidence.key in _hit_page_keys(hit)


def _known_relevance(case: GoldenCase) -> dict[tuple[str, int], int]:
    if case.retrieval_gold is None:
        return {}
    known = {item.key: 2 for item in case.retrieval_gold.required_evidence}
    for item in case.retrieval_gold.relevant_evidence:
        known[item.key] = max(known.get(item.key, 0), item.relevance)
    return known


def _matched_gold_keys(
    hits: Iterable[SearchHit], gold: dict[tuple[str, int], int]
) -> set[tuple[str, int]]:
    returned = set().union(*(_hit_page_keys(hit) for hit in hits)) if gold else set()
    return returned.intersection(gold)


def _evidence_unit_key(hit: SearchHit) -> tuple[str, int, int, str]:
    fingerprint = hashlib.sha256(_normalize_text(hit.text).encode("utf-8")).hexdigest()
    return hit.document_id, hit.pdf_page_start, hit.pdf_page_end, fingerprint


def _ranked_relevance(
    hits: list[SearchHit], gold: dict[tuple[str, int], int], k: int
) -> tuple[list[int], bool]:
    """Assign one gain per stable evidence unit and consume all pages it covers."""

    unused = set(gold)
    seen_units: set[tuple[str, int, int, str]] = set()
    grades: list[int] = []
    ambiguous = False
    for hit in hits[:k]:
        unit = _evidence_unit_key(hit)
        if unit in seen_units:
            grades.append(0)
            continue
        seen_units.add(unit)
        matches = _hit_page_keys(hit).intersection(unused)
        ambiguous = ambiguous or len(matches) > 1
        grade = max((gold[key] for key in matches), default=0)
        grades.append(grade)
        unused.difference_update(matches)
    grades.extend([0] * (k - len(grades)))
    return grades, ambiguous


def _dcg(grades: list[int]) -> float:
    return float(sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, 1)))


def _record_metric(
    result: dict[str, Any],
    key: str,
    value: bool | int | float | None,
    *,
    evaluable: bool,
    reason: str,
    evaluation_bases: list[str] | None = None,
    evaluable_unit_ids: list[str] | None = None,
    unevaluable_unit_ids: list[str] | None = None,
) -> None:
    result[key] = value if evaluable else None
    status = result.setdefault("metric_status", {})
    status[key] = {
        "value": value if evaluable else None,
        "evaluable": evaluable,
        "reason": reason,
        "evaluation_bases": evaluation_bases or ["deterministic"],
        "evaluable_unit_ids": evaluable_unit_ids or [],
        "unevaluable_unit_ids": unevaluable_unit_ids or [],
    }


def evaluate_retrieval_hits(
    case: GoldenCase,
    hits: list[SearchHit],
    *,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> dict[str, Any]:
    """Evaluate page-level retrieval/ranking without invoking answer generation."""

    gold = _known_relevance(case)
    complete = bool(case.retrieval_gold and case.retrieval_gold.relevance_complete)
    result: dict[str, Any] = {
        "known_relevant_pages": len(gold),
        "relevance_annotation_complete": complete,
        "returned_count": len(hits),
        "metric_status": {},
    }
    for k in k_values:
        prefix = hits[:k]
        matched = _matched_gold_keys(prefix, gold)
        required_keys = {
            item.key
            for item in (case.retrieval_gold.required_evidence if case.retrieval_gold else [])
        }
        returned_keys = set().union(*(_hit_page_keys(hit) for hit in prefix)) if prefix else set()
        relevant_hits = sum(bool(_hit_page_keys(hit).intersection(gold)) for hit in prefix)
        _record_metric(
            result,
            f"hit_at_{k}",
            bool(matched),
            evaluable=bool(gold),
            reason="evaluated against annotated gold pages" if gold else "no annotated gold pages",
        )
        known_recall = len(matched) / len(gold) if gold else None
        _record_metric(
            result,
            f"annotated_recall_at_{k}",
            known_recall,
            evaluable=bool(gold),
            reason=(
                "recall over annotated gold pages; annotations may be incomplete"
                if gold
                else "no annotated gold pages"
            ),
        )
        result[f"recall_at_{k}"] = result[f"annotated_recall_at_{k}"]
        result["metric_status"][f"recall_at_{k}"] = result["metric_status"][
            f"annotated_recall_at_{k}"
        ]
        _record_metric(
            result,
            f"required_evidence_recall_at_{k}",
            len(returned_keys.intersection(required_keys)) / len(required_keys)
            if required_keys
            else None,
            evaluable=bool(required_keys),
            reason=(
                "recall over annotated required evidence pages"
                if required_keys
                else "no required evidence pages"
            ),
        )
        _record_metric(
            result,
            f"precision_at_{k}",
            relevant_hits / k,
            evaluable=complete,
            reason=(
                "evaluated against complete relevance annotations"
                if complete
                else "relevance annotations are incomplete"
            ),
        )
        result[f"precision_at_{k}_evaluable"] = complete
        if complete:
            grades, ambiguous = _ranked_relevance(hits, gold, k)
            ideal = sorted(gold.values(), reverse=True)[:k]
            ideal.extend([0] * (k - len(ideal)))
            ideal_dcg = _dcg(ideal)
            value = _dcg(grades) / ideal_dcg if ideal_dcg else None
        else:
            ambiguous = False
            value = None
        ndcg_evaluable = complete and not ambiguous and value is not None
        _record_metric(
            result,
            f"ndcg_at_{k}",
            value,
            evaluable=ndcg_evaluable,
            reason=(
                "cross-page evidence maps to multiple gold pages; graded gain is ambiguous"
                if ambiguous
                else "evaluated against complete relevance annotations"
                if ndcg_evaluable
                else "relevance annotations are incomplete or have no ideal gain"
            ),
        )
        result[f"ndcg_at_{k}_evaluable"] = ndcg_evaluable
    first_rank = next(
        (rank for rank, hit in enumerate(hits, 1) if _hit_page_keys(hit).intersection(gold)),
        None,
    )
    _record_metric(
        result,
        "mrr",
        1.0 / first_rank if first_rank is not None else 0.0,
        evaluable=bool(gold),
        reason="first rank of an annotated gold page" if gold else "no annotated gold pages",
    )
    required = case.retrieval_gold.required_evidence if case.retrieval_gold else []
    result["required_evidence_ranks"] = [
        {
            "document_id": item.document_id,
            "pdf_page": item.pdf_page,
            "rank": next(
                (rank for rank, hit in enumerate(hits, 1) if hit_matches_evidence(hit, item)),
                None,
            ),
        }
        for item in required
    ]
    return result


def evaluate_context_hits(case: GoldenCase, hits: list[SearchHit], *, top_k: int) -> dict[str, Any]:
    """Score the final retrieval candidate list used as component context."""

    selected = hits[:top_k]
    gold = _known_relevance(case)
    known_relevant = sum(bool(_hit_page_keys(hit).intersection(gold)) for hit in selected)
    complete = bool(case.retrieval_gold and case.retrieval_gold.relevance_complete)
    seen_starts: set[tuple[str, int]] = set()
    seen_units: set[tuple[str, int, int, str]] = set()
    duplicate_starts = 0
    duplicate_units = 0
    for hit in selected:
        key = (hit.document_id, hit.pdf_page_start)
        duplicate_starts += int(key in seen_starts)
        seen_starts.add(key)
        unit = _evidence_unit_key(hit)
        duplicate_units += int(unit in seen_units)
        seen_units.add(unit)
    total = len(selected)
    known_ratio = known_relevant / total if total else None
    result: dict[str, Any] = {
        "context_size": total,
        "relevance_ratio_evaluable": complete,
        "metric_status": {},
    }
    _record_metric(
        result,
        "known_relevant_ratio_lower_bound",
        known_ratio,
        evaluable=total > 0,
        reason="lower bound from annotated pages" if total else "empty context",
    )
    _record_metric(
        result,
        "relevant_evidence_ratio",
        known_ratio,
        evaluable=complete and total > 0,
        reason="evaluated against complete relevance annotations"
        if complete
        else "relevance annotations are incomplete",
    )
    _record_metric(
        result,
        "irrelevant_context_ratio",
        1.0 - known_ratio if known_ratio is not None else None,
        evaluable=complete and total > 0,
        reason="evaluated against complete relevance annotations"
        if complete
        else "relevance annotations are incomplete",
    )
    concentration = duplicate_starts / total if total else None
    _record_metric(
        result,
        "start_page_concentration_ratio",
        concentration,
        evaluable=total > 0,
        reason="repeated document/start-page locations" if total else "empty context",
    )
    result["duplicate_start_page_ratio"] = concentration
    result["metric_status"]["duplicate_start_page_ratio"] = result["metric_status"][
        "start_page_concentration_ratio"
    ]
    _record_metric(
        result,
        "redundancy_ratio",
        duplicate_units / total if total else None,
        evaluable=total > 0,
        reason="duplicate page-range and normalized-text fingerprints"
        if total
        else "empty context",
    )
    return result


def _normalize_text(value: str) -> str:
    return "".join(character.casefold() for character in value if character.isalnum())


def _pattern_match(answer: str, patterns: list[str], fallback: str) -> bool:
    normalized = _normalize_text(answer)
    candidates = patterns or [fallback]
    return any(_normalize_text(item) in normalized for item in candidates)


REFUSAL_PATTERNS = (
    re.compile(
        r"(?:现有|当前|本地).{0,12}(?:资料|证据).{0,16}(?:不足|没有|无法|不能).{0,12}(?:回答|判断|确认|支持)"
    ),
    re.compile(
        r"(?:无法|不能).{0,10}(?:依据|从).{0,12}(?:现有|当前|本地)?(?:资料|证据).{0,12}(?:回答|判断|确认)"
    ),
    re.compile(r"(?:资料|证据).{0,12}(?:不足|缺失).{0,12}(?:无法|不能).{0,12}(?:回答|判断|确认)"),
)
CLARIFICATION_PATTERN = re.compile(
    r"(?:请|需要).{0,10}(?:明确|补充|说明).{0,16}(?:人物|年份|时间|范围|条件|问题)"
)
SUBSTANTIVE_ASSERTION_PATTERN = re.compile(
    r"(?:18|19|20)\d{2}年|(?:是|为|担任|发生|位于|当选|参加|参与|提出|成立|导致|造成)"
)
NEGATED_ASSERTION_PATTERN = re.compile(r"无法|不能|不足|没有证据|未能|不可确认|难以判断")


def classify_answer_disposition(response: AnswerResponse) -> tuple[str, str]:
    """Classify outcome without treating evidence flags alone as a refusal."""

    text = response.answer.strip()
    if response.retrieval_mode == "query_clarification" or (
        not response.citations and CLARIFICATION_PATTERN.search(text)
    ):
        return "clarification", "production route requested clarification"
    if response.citations:
        return "answered", "answer includes one or more citations"
    refusal_semantics = any(pattern.search(text) for pattern in REFUSAL_PATTERNS)
    clauses = re.split(r"[。！？；\n]|但是|然而|不过|但", text)
    substantive = any(
        SUBSTANTIVE_ASSERTION_PATTERN.search(clause)
        and not NEGATED_ASSERTION_PATTERN.search(clause)
        for clause in clauses
    )
    if response.evidence_status == "no_evidence" and refusal_semantics and not substantive:
        return "refused", "explicit evidence-scoped refusal with no citations"
    if text:
        return "answered_uncited", "substantive output has no citations"
    return "unknown", "empty answer cannot be classified"


def evaluate_generation(
    case: GoldenCase,
    response: AnswerResponse,
    *,
    fact_judgment: dict[str, Any] | None = None,
    semantic_judgment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fact_status = (fact_judgment or {}).get("status", "not_run")
    decisions = {
        item["fact_id"]: item
        for item in (fact_judgment or {}).get("decisions", [])
        if fact_status == "used"
    }
    fact_results: list[dict[str, Any]] = []
    for fact in case.gold_facts:
        deterministic = _pattern_match(response.answer, fact.deterministic_patterns, fact.claim)
        semantic = decisions.get(fact.fact_id)
        if deterministic:
            covered, judge, reason = True, "deterministic", "matched an annotated pattern"
        elif semantic is not None:
            covered = bool(semantic["covered"])
            judge = "semantic"
            reason = str(semantic["reason"])
        else:
            covered = None
            judge = "not_evaluated"
            reason = "no deterministic match; semantic fallback was not available"
        fact_results.append(
            {
                "fact_id": fact.fact_id,
                "unit_id": f"{case.id}:{fact.fact_id}",
                "importance": fact.importance,
                "covered": covered,
                "evaluable": covered is not None,
                "judge": judge,
                "reason": reason,
            }
        )
    required = [item for item in fact_results if item["importance"] == "required"]
    optional = [item for item in fact_results if item["importance"] == "optional"]
    forbidden = [
        {
            "claim": item.claim,
            "reason": item.reason,
            "violated": _pattern_match(response.answer, item.deterministic_patterns, item.claim),
        }
        for item in case.forbidden_claims
    ]
    semantic_used = (semantic_judgment or {}).get("status") == "used"
    unsupported_claims = (
        list((semantic_judgment or {}).get("unsupported_claims", [])) if semantic_used else []
    )
    disposition, disposition_reason = classify_answer_disposition(response)
    refusal_evaluable = case.answerability == "unanswerable" and disposition != "unknown"
    refusal_correct = disposition == "refused" if refusal_evaluable else None
    required_evaluable = [item for item in required if item["evaluable"]]
    optional_evaluable = [item for item in optional if item["evaluable"]]
    required_bases = sorted({str(item["judge"]) for item in required_evaluable})
    optional_bases = sorted({str(item["judge"]) for item in optional_evaluable})
    required_recall = (
        sum(bool(item["covered"]) for item in required_evaluable) / len(required_evaluable)
        if required_evaluable
        else None
    )
    optional_recall = (
        sum(bool(item["covered"]) for item in optional_evaluable) / len(optional_evaluable)
        if optional_evaluable
        else None
    )
    semantic_failed = semantic_used and (semantic_judgment or {}).get("verdict") == "fail"
    forbidden_violation = any(bool(item["violated"]) for item in forbidden) or semantic_failed
    if case.answerability == "unanswerable":
        correct_evaluable = refusal_evaluable
        correct = (
            bool(refusal_correct) and not forbidden_violation and not semantic_failed
            if correct_evaluable
            else None
        )
    else:
        correct_evaluable = len(required_evaluable) == len(required)
        correct = (
            required_recall == 1.0 and not forbidden_violation and not semantic_failed
            if correct_evaluable
            else None
        )
    result: dict[str, Any] = {
        "facts": fact_results,
        "fact_counts": {
            "required_total": len(required),
            "required_evaluable": len(required_evaluable),
            "optional_total": len(optional),
            "optional_evaluable": len(optional_evaluable),
        },
        "fact_judge_status": fact_status,
        "forbidden_claims": forbidden,
        "unsupported_claims": unsupported_claims,
        "answer_disposition": disposition,
        "answer_disposition_reason": disposition_reason,
        "metric_status": {},
        "correctness_basis": (
            "deterministic+semantic" if semantic_used or decisions else "deterministic"
        ),
    }
    _record_metric(
        result,
        "required_fact_recall",
        required_recall,
        evaluable=bool(required_evaluable),
        reason="recall over evaluable required facts"
        if required_evaluable
        else "no required facts were evaluable",
        evaluation_bases=required_bases,
        evaluable_unit_ids=[str(item["unit_id"]) for item in required_evaluable],
        unevaluable_unit_ids=[str(item["unit_id"]) for item in required if not item["evaluable"]],
    )
    _record_metric(
        result,
        "optional_fact_recall",
        optional_recall,
        evaluable=bool(optional_evaluable),
        reason="recall over evaluable optional facts"
        if optional_evaluable
        else "no optional facts were evaluable",
        evaluation_bases=optional_bases,
        evaluable_unit_ids=[str(item["unit_id"]) for item in optional_evaluable],
        unevaluable_unit_ids=[str(item["unit_id"]) for item in optional if not item["evaluable"]],
    )
    _record_metric(
        result,
        "forbidden_claim_violation",
        forbidden_violation,
        evaluable=bool(forbidden) or semantic_used,
        reason=(
            "checked annotated forbidden-claim patterns and/or semantic unsupported claims"
            if forbidden or semantic_used
            else "case has no forbidden-claim labels and semantic judge was unavailable"
        ),
        evaluation_bases=(
            ["deterministic", "semantic_citation"]
            if forbidden and semantic_used
            else ["semantic_citation"]
            if semantic_used
            else ["deterministic"]
        ),
    )
    _record_metric(
        result,
        "unsupported_claim_detected",
        bool(unsupported_claims),
        evaluable=semantic_used,
        reason="semantic citation judge completed"
        if semantic_used
        else "semantic citation judge was not available",
        evaluation_bases=["semantic_citation"],
    )
    _record_metric(
        result,
        "refusal_correct",
        refusal_correct,
        evaluable=refusal_evaluable,
        reason=disposition_reason
        if refusal_evaluable
        else "not an unanswerable case or disposition unknown",
    )
    false_answer = disposition in {"answered", "answered_uncited"} if refusal_evaluable else None
    _record_metric(
        result,
        "false_answer",
        false_answer,
        evaluable=refusal_evaluable,
        reason=disposition_reason
        if refusal_evaluable
        else "not an unanswerable case or disposition unknown",
    )
    _record_metric(
        result,
        "answer_correctness",
        correct,
        evaluable=correct_evaluable,
        reason="all required correctness signals are evaluable"
        if correct_evaluable
        else "at least one required fact or refusal disposition is unevaluable",
        evaluation_bases=sorted(
            {
                *(str(item["judge"]) for item in required_evaluable),
                *(["semantic_citation"] if semantic_used else []),
                "deterministic",
            }
        ),
    )
    return result


def evaluate_citations(
    case: GoldenCase,
    response: AnswerResponse,
    *,
    page_texts: dict[tuple[str, int], str],
    semantic_judgment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gold = _known_relevance(case)
    complete = bool(case.retrieval_gold and case.retrieval_gold.relevance_complete)
    cited_gold: set[tuple[str, int]] = set()
    relevant_citations = 0
    details: list[dict[str, Any]] = []
    for citation in response.citations:
        end = citation.pdf_page_end or citation.pdf_page
        pages = {(citation.document_id, page) for page in range(citation.pdf_page, end + 1)}
        matches = pages.intersection(gold)
        cited_gold.update(matches)
        relevant_citations += int(bool(matches))
        available = {
            page: page_texts.get((citation.document_id, page))
            for page in range(citation.pdf_page, end + 1)
        }
        match_pages = [
            page
            for page, text in available.items()
            if text is not None and quote_matches_page(citation.quote, text)
        ]
        available_text = "\n".join(text for text in available.values() if text is not None)
        range_match = bool(match_pages) or bool(
            available_text and quote_matches_page(citation.quote, available_text)
        )
        all_pages_exist = all(text is not None for text in available.values())
        quote_evaluable = range_match or all_pages_exist
        quote_match: bool | None = range_match if quote_evaluable else None
        details.append(
            {
                "evidence_id": citation.evidence_id,
                "document_id": citation.document_id,
                "pdf_page": citation.pdf_page,
                "pdf_page_end": citation.pdf_page_end,
                "page_exists": all_pages_exist,
                "page_range_evaluable": True,
                "quote_matches_page": quote_match,
                "quote_matches_range": quote_match,
                "quote_check_evaluable": quote_evaluable,
                "quote_match_pages": match_pages,
                "matches_gold_page": bool(matches),
            }
        )
    count = len(details)
    validation = (
        validate_grounded_answer(response.answer, response.citations)
        if response.citations
        else None
    )
    result: dict[str, Any] = {
        "citation_presence": bool(response.citations),
        "uncited_claims": list(validation.uncited_claims) if validation else [],
        "citation_precision_evaluable": complete and bool(count),
        "semantic_judge_status": (semantic_judgment or {}).get("status", "not_run"),
        "semantic_judge_reason": (semantic_judgment or {}).get("rationale"),
        "details": details,
        "metric_status": {},
    }
    _record_metric(
        result,
        "citation_presence_correct",
        bool(response.citations) == (case.answerability == "answerable"),
        evaluable=True,
        reason="compared citation presence with answerability",
    )
    _record_metric(
        result,
        "citation_page_validity",
        sum(bool(item["page_exists"]) for item in details) / count if count else None,
        evaluable=count > 0,
        reason="all pages in each citation range are available" if count else "no citations",
    )
    quote_details = [item for item in details if item["quote_check_evaluable"]]
    _record_metric(
        result,
        "citation_quote_consistency",
        sum(bool(item["quote_matches_page"]) for item in quote_details) / len(quote_details)
        if quote_details
        else None,
        evaluable=len(quote_details) == count and count > 0,
        reason="quote checked across every page in each citation range"
        if len(quote_details) == count and count
        else "one or more citation ranges could not be fully verified",
    )
    coverage_value = bool(validation.valid) if validation is not None else False
    _record_metric(
        result,
        "claim_to_citation_coverage",
        coverage_value,
        evaluable=case.answerability == "answerable",
        reason="deterministic claim-to-citation validation"
        if case.answerability == "answerable"
        else "not applicable to an unanswerable case",
    )
    _record_metric(
        result,
        "citation_precision",
        relevant_citations / count if count else None,
        evaluable=complete and count > 0,
        reason="evaluated against complete relevance annotations"
        if complete and count
        else "relevance annotations are incomplete or there are no citations",
    )
    citation_recall = len(cited_gold) / len(gold) if gold else None
    _record_metric(
        result,
        "annotated_citation_recall",
        citation_recall,
        evaluable=bool(gold),
        reason="recall over annotated gold pages" if gold else "no annotated gold pages",
    )
    result["citation_recall"] = result["annotated_citation_recall"]
    result["metric_status"]["citation_recall"] = result["metric_status"][
        "annotated_citation_recall"
    ]
    semantic_used = (semantic_judgment or {}).get("status") == "used"
    _record_metric(
        result,
        "semantic_support",
        (semantic_judgment or {}).get("verdict") == "pass",
        evaluable=semantic_used,
        reason="semantic citation judge completed"
        if semantic_used
        else "semantic citation judge was not available",
        evaluation_bases=["semantic_citation"],
    )
    return result


def normalize_route(retrieval_mode: str) -> str:
    if retrieval_mode.startswith("structured_"):
        return retrieval_mode
    if retrieval_mode == "full_text_section":
        return "full_text_section"
    if retrieval_mode == "query_clarification":
        return "query_clarification"
    if any(marker in retrieval_mode for marker in ("hybrid", "keyword", "vector")):
        return "hybrid_retrieval"
    return retrieval_mode


def evaluate_routing(case: GoldenCase, response: AnswerResponse) -> dict[str, Any]:
    actual = normalize_route(response.retrieval_mode)
    preferred = case.route_gold.preferred_routes if case.route_gold else []
    result: dict[str, Any] = {
        "actual_route": actual,
        "raw_retrieval_mode": response.retrieval_mode,
        "preferred_routes": preferred,
        "metric_status": {},
    }
    _record_metric(
        result,
        "route_match",
        actual in preferred,
        evaluable=bool(preferred),
        reason="compared with annotated preferred routes"
        if preferred
        else "no preferred route labels",
    )
    return result


def _metric(
    values: list[float],
    *,
    total_cases: int,
    evaluable_case_ids: list[str],
    unevaluable_case_ids: list[str],
    evaluation_bases: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "value": round(mean(values), 6) if values else None,
        "evaluable_cases": len(values),
        "total_cases": total_cases,
        "unevaluable_cases": total_cases - len(values),
        "evaluable_case_ids": evaluable_case_ids,
        "unevaluable_case_ids": unevaluable_case_ids,
        "evaluable_unit_ids": evaluable_case_ids,
        "unevaluable_unit_ids": unevaluable_case_ids,
        "evaluation_bases": evaluation_bases or ["deterministic"],
    }


def _section_values(
    results: list[dict[str, Any]], section: str, key: str
) -> tuple[list[float], int, list[str], list[str]]:
    values: list[float] = []
    evaluable_ids: list[str] = []
    unevaluable_ids: list[str] = []
    total = 0
    for result in results:
        payload = result.get(section)
        if not isinstance(payload, dict):
            continue
        total += 1
        value = payload.get(key)
        status = payload.get("metric_status", {}).get(key, {})
        evaluable = isinstance(status, dict) and status.get("evaluable") is True
        case_id = str(result.get("case_id", ""))
        if evaluable and isinstance(value, (bool, int, float)):
            values.append(float(value))
            evaluable_ids.append(case_id)
        else:
            unevaluable_ids.append(case_id)
    return values, total, evaluable_ids, unevaluable_ids


def _aggregate_metric(results: list[dict[str, Any]], section: str, key: str) -> dict[str, Any]:
    values, total, evaluable_ids, unevaluable_ids = _section_values(results, section, key)
    bases = sorted(
        {
            str(basis)
            for result in results
            for payload in [result.get(section)]
            if isinstance(payload, dict)
            for status in [payload.get("metric_status", {}).get(key)]
            if isinstance(status, dict) and status.get("evaluable") is True
            for basis in status.get("evaluation_bases", ["deterministic"])
        }
    )
    return _metric(
        values,
        total_cases=total,
        evaluable_case_ids=evaluable_ids,
        unevaluable_case_ids=unevaluable_ids,
        evaluation_bases=bases,
    )


def _aggregate_fact_recall(
    results: list[dict[str, Any]], importance: FactImportance
) -> dict[str, Any]:
    evaluable_units: list[str] = []
    unevaluable_units: list[str] = []
    evaluable_cases: set[str] = set()
    unevaluable_cases: set[str] = set()
    bases: set[str] = set()
    covered = 0
    total_cases = 0
    for result in results:
        generation = result.get("generation")
        if not isinstance(generation, dict):
            continue
        total_cases += 1
        case_id = str(result.get("case_id", ""))
        case_has_evaluable = False
        for fact in generation.get("facts", []):
            if not isinstance(fact, dict) or fact.get("importance") != importance:
                continue
            unit_id = str(fact.get("unit_id") or f"{case_id}:{fact.get('fact_id', '')}")
            if fact.get("evaluable") is True and isinstance(fact.get("covered"), bool):
                evaluable_units.append(unit_id)
                covered += int(fact["covered"])
                case_has_evaluable = True
                bases.add(str(fact.get("judge") or "deterministic"))
            else:
                unevaluable_units.append(unit_id)
        if case_has_evaluable:
            evaluable_cases.add(case_id)
        else:
            unevaluable_cases.add(case_id)
    return {
        "value": round(covered / len(evaluable_units), 6) if evaluable_units else None,
        "evaluable_cases": len(evaluable_cases),
        "total_cases": total_cases,
        "unevaluable_cases": len(unevaluable_cases),
        "evaluable_case_ids": sorted(evaluable_cases),
        "unevaluable_case_ids": sorted(unevaluable_cases),
        "evaluable_units": len(evaluable_units),
        "total_units": len(evaluable_units) + len(unevaluable_units),
        "unevaluable_units": len(unevaluable_units),
        "evaluable_unit_ids": evaluable_units,
        "unevaluable_unit_ids": unevaluable_units,
        "evaluation_bases": sorted(bases),
        "aggregation": "micro_over_fact_units",
    }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    dimensions = {
        name: sum(isinstance(item.get(name), dict) for item in results) for name in ALL_DIMENSIONS
    }
    aggregate: dict[str, Any] = {
        "case_count": len(results),
        "dimension_case_counts": dimensions,
        "retrieval": {},
        "ranking": {},
        "context": {},
        "generation": {},
        "citation": {},
        "routing": {},
    }
    for k in DEFAULT_K_VALUES:
        aggregate["retrieval"][f"hit_rate_at_{k}"] = _aggregate_metric(
            results, "retrieval", f"hit_at_{k}"
        )
        aggregate["retrieval"][f"annotated_recall_at_{k}"] = _aggregate_metric(
            results, "retrieval", f"annotated_recall_at_{k}"
        )
        aggregate["retrieval"][f"recall_at_{k}"] = aggregate["retrieval"][
            f"annotated_recall_at_{k}"
        ]
        aggregate["retrieval"][f"required_evidence_recall_at_{k}"] = _aggregate_metric(
            results, "retrieval", f"required_evidence_recall_at_{k}"
        )
        aggregate["retrieval"][f"precision_at_{k}"] = _aggregate_metric(
            results, "retrieval", f"precision_at_{k}"
        )
        aggregate["retrieval"][f"ndcg_at_{k}"] = _aggregate_metric(
            results, "retrieval", f"ndcg_at_{k}"
        )
        aggregate["ranking"][f"ndcg_at_{k}"] = _aggregate_metric(results, "ranking", f"ndcg_at_{k}")
    aggregate["retrieval"]["mrr"] = _aggregate_metric(results, "retrieval", "mrr")
    aggregate["ranking"]["mrr"] = _aggregate_metric(results, "ranking", "mrr")
    for key in (
        "known_relevant_ratio_lower_bound",
        "relevant_evidence_ratio",
        "irrelevant_context_ratio",
        "duplicate_start_page_ratio",
        "start_page_concentration_ratio",
        "redundancy_ratio",
    ):
        aggregate["context"][key] = _aggregate_metric(results, "context", key)
    aggregate["generation"]["required_fact_recall"] = _aggregate_fact_recall(results, "required")
    aggregate["generation"]["optional_fact_recall"] = _aggregate_fact_recall(results, "optional")
    for key in (
        "answer_correctness",
        "forbidden_claim_violation",
        "unsupported_claim_detected",
        "refusal_correct",
        "false_answer",
    ):
        aggregate["generation"][key] = _aggregate_metric(results, "generation", key)
    for key in (
        "citation_presence_correct",
        "citation_page_validity",
        "citation_quote_consistency",
        "claim_to_citation_coverage",
        "citation_precision",
        "citation_recall",
        "semantic_support",
    ):
        aggregate["citation"][key] = _aggregate_metric(results, "citation", key)
    aggregate["citation"]["annotated_citation_recall"] = _aggregate_metric(
        results, "citation", "annotated_citation_recall"
    )
    aggregate["routing"]["route_accuracy"] = _aggregate_metric(results, "routing", "route_match")
    aggregate["routing"]["mismatch_case_ids"] = [
        item["case_id"]
        for item in results
        if isinstance(item.get("routing"), dict) and not item["routing"]["route_match"]
    ]
    latency_keys = ("retrieval_latency_ms", "answer_latency_ms", "total_latency_ms")
    aggregate["operational"] = {
        key: _aggregate_metric(results, "operational", key) for key in latency_keys
    }
    usage_totals: dict[str, dict[str, int]] = {}
    for result in results:
        operational = result.get("operational")
        if not isinstance(operational, dict):
            continue
        usage = operational.get("token_usage")
        if not isinstance(usage, dict):
            continue
        for stage, values in usage.items():
            if not isinstance(values, dict):
                continue
            target = usage_totals.setdefault(stage, {})
            for key, value in values.items():
                if isinstance(value, int):
                    target[key] = target.get(key, 0) + value
    aggregate["operational"]["token_usage"] = usage_totals
    return aggregate


def _git_metadata(project_root: Path) -> dict[str, Any]:
    base = ["git", "-c", f"safe.directory={project_root.as_posix()}"]

    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                [*base, *args],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
    }


def _load_index_metadata(settings: Settings) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    warnings: list[str] = []
    for name in ("keyword_index_latest.json", "vector_index_latest.json"):
        path = settings.reports_dir / name
        if not path.is_file():
            warnings.append(f"missing index report: {name}")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            warnings.append(f"unreadable index report: {name}")
            continue
        report_name = name.removesuffix("_latest.json")
        manifest = payload.get("manifest")
        reports[report_name] = {
            "summary": {
                key: payload.get(key)
                for key in ("run_id", "index_version", "model_name", "chunks")
                if payload.get(key) is not None
            },
            "manifest": manifest if isinstance(manifest, dict) else None,
        }
        if not isinstance(manifest, dict):
            warnings.append(f"{report_name} is legacy and has no artifact manifest")
        else:
            index_path = (
                settings.keyword_index_path
                if report_name == "keyword_index"
                else settings.vector_index_path
            )
            expected_artifact = manifest.get("index_artifact_sha256")
            try:
                actual_artifact = index_artifact_sha256(index_path)
            except OSError:
                actual_artifact = None
            verified = bool(expected_artifact) and expected_artifact == actual_artifact
            reports[report_name]["artifact_verified"] = verified
            reports[report_name]["actual_index_artifact_sha256"] = actual_artifact
            if not expected_artifact:
                warnings.append(f"{report_name} manifest cannot identify its index artifact")
            elif not verified:
                warnings.append(f"{report_name} manifest does not match the queried index artifact")
    manifests = [
        item["manifest"] for item in reports.values() if isinstance(item.get("manifest"), dict)
    ]
    chunk_hashes = {item.get("chunk_artifact_sha256") for item in manifests}
    if len(chunk_hashes) > 1:
        warnings.append("keyword and vector indexes were built from different chunk artifacts")
    for item in manifests:
        warnings.extend(str(value) for value in item.get("warnings", []))
    return {"reports": reports, "warnings": list(dict.fromkeys(warnings))}


def _retrieval_branches(results: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    used: set[str] = set()
    degraded: set[str] = set()

    def branches_for_mode(mode: str) -> set[str]:
        if "keyword_only" in mode:
            return {"keyword"}
        if "vector_only" in mode:
            return {"vector"}
        if "hybrid" in mode:
            return {"keyword", "vector"}
        if "keyword" in mode:
            return {"keyword"}
        if "vector" in mode:
            return {"vector"}
        return set()

    for result in results:
        operational = result.get("operational")
        if isinstance(operational, dict):
            local_degraded = {str(item) for item in operational.get("degraded_components", [])}
            degraded.update(local_degraded)
            mode = operational.get("retrieval_mode")
            if isinstance(mode, str):
                used.update(branches_for_mode(mode).difference(local_degraded))
        answer = result.get("answer")
        if isinstance(answer, dict) and isinstance(answer.get("retrieval_mode"), str):
            answer_mode = str(answer["retrieval_mode"])
            used.update(branches_for_mode(answer_mode))
            if "keyword_only" in answer_mode:
                degraded.add("vector")
            elif "vector_only" in answer_mode:
                degraded.add("keyword")
    return sorted(used), sorted(degraded)


def _required_manifest_fields(branch: str) -> tuple[str, ...]:
    common = (
        "chunk_artifact_sha256",
        "index_artifact_sha256",
        "build_run_id",
        "git_commit",
    )
    if branch == "vector":
        return (*common, "embedding_model", "vector_index_version")
    return (*common, "keyword_index_version")


def _index_attribution(reports: dict[str, Any], actual_branches: list[str]) -> dict[str, Any]:
    reasons: list[str] = []
    selected_by_branch: dict[str, dict[str, Any]] = {}
    for branch in actual_branches:
        report = reports.get(f"{branch}_index")
        manifest = report.get("manifest") if isinstance(report, dict) else None
        if not isinstance(manifest, dict):
            reasons.append(f"actual {branch} branch has no index manifest")
            continue
        assert isinstance(report, dict)
        selected_by_branch[branch] = manifest
        if report.get("artifact_verified") is not True:
            reasons.append(f"actual {branch} index artifact is not verified")
        missing = [key for key in _required_manifest_fields(branch) if not manifest.get(key)]
        chunking = manifest.get("chunking")
        if not isinstance(chunking, dict) or any(
            chunking.get(key) is None for key in ("version", "target_chars", "max_chars", "overlap")
        ):
            missing.append("chunking")
        if missing:
            reasons.append(f"actual {branch} manifest is missing: {', '.join(missing)}")
    if not actual_branches:
        reasons.append("no retrieval branch usage was recorded")
    selected = list(selected_by_branch.values())
    hashes = {manifest.get("chunk_artifact_sha256") for manifest in selected}
    chunkings = {
        json.dumps(manifest.get("chunking"), sort_keys=True, ensure_ascii=False)
        for manifest in selected
    }
    if len(hashes) > 1:
        reasons.append("actual retrieval branches use different chunk artifacts")
    if len(chunkings) > 1:
        reasons.append("actual retrieval branches use different chunking manifests")
    consistent = (
        len(selected) == len(actual_branches)
        and bool(selected)
        and len(hashes) == 1
        and len(chunkings) == 1
    )
    unified_hash = next(iter(hashes)) if consistent else None
    unified_chunking = selected[0].get("chunking") if consistent else None
    vector_manifest = selected_by_branch.get("vector")
    return {
        "index_manifest_consistent": consistent,
        "attribution_safe": not reasons,
        "attribution_unsafe_reasons": reasons,
        "chunk_artifact_sha256": unified_hash,
        "chunking": unified_chunking,
        "embedding_model": (
            vector_manifest.get("embedding_model") if isinstance(vector_manifest, dict) else None
        ),
    }


def _run_metadata(
    settings: Settings,
    *,
    run_name: str | None,
    top_k: int,
    with_llm: bool,
    semantic_judge: bool,
    dataset_sha256: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    indexes = _load_index_metadata(settings)
    reports = indexes["reports"]
    actual_branches, degraded_branches = _retrieval_branches(results)
    attribution = _index_attribution(reports, actual_branches)
    answers = [item["answer"] for item in results if isinstance(item.get("answer"), dict)]
    llm_used = any(item.get("generator_mode") == "llm" for item in answers)
    planner_used = any(item.get("query_planner_status") == "used" for item in answers)
    reflection_used = any(
        item.get("retrieval_reflection_status") in {"sufficient", "retried"} for item in answers
    )
    semantic_used = any(
        (
            isinstance(item.get("citation"), dict)
            and item["citation"].get("semantic_judge_status") == "used"
        )
        or (
            isinstance(item.get("generation"), dict)
            and item["generation"].get("fact_judge_status") == "used"
        )
        for item in results
    )
    fact_judge_used = any(
        isinstance(item.get("generation"), dict)
        and any(
            isinstance(fact, dict) and fact.get("judge") == "semantic"
            for fact in item["generation"].get("facts", [])
        )
        for item in results
    )
    citation_judge_used = any(
        isinstance(item.get("citation"), dict)
        and item["citation"].get("semantic_judge_status") == "used"
        for item in results
    )
    return {
        "run_name": run_name,
        "timestamp": datetime.now(UTC).isoformat(),
        "git": _git_metadata(settings.project_root),
        "project_version": __version__,
        "prompt_version": PROMPT_VERSION,
        "with_llm": with_llm,
        "semantic_judge_requested": semantic_judge,
        "evaluator": {
            "version": GOLDEN_EVALUATOR_VERSION,
            "metric_definition_version": METRIC_DEFINITION_VERSION,
            "page_mapping_version": PAGE_MAPPING_VERSION,
            "text_normalization_version": TEXT_NORMALIZATION_VERSION,
            "relevance_schema_version": RELEVANCE_SCHEMA_VERSION,
            "fact_judge_version": FACT_COVERAGE_JUDGE_VERSION,
            "citation_judge_version": SEMANTIC_CITATION_JUDGE_VERSION,
            "judge_provider": settings.llm_provider if semantic_used else None,
            "fact_judge_provider": settings.llm_provider if fact_judge_used else None,
            "citation_judge_provider": (settings.llm_provider if citation_judge_used else None),
            "fact_judge_model": settings.llm_model if fact_judge_used else None,
            "citation_judge_model": settings.llm_model if citation_judge_used else None,
        },
        "rag_framework": "llamaindex",
        "dataset_sha256": dataset_sha256,
        "evaluated_case_ids": [item["case_id"] for item in results],
        "embedding_model": attribution["embedding_model"],
        "chunk_artifact_sha256": attribution["chunk_artifact_sha256"],
        "chunking": attribution["chunking"],
        "index_manifest_consistent": attribution["index_manifest_consistent"],
        "attribution_safe": attribution["attribution_safe"],
        "attribution_unsafe_reasons": attribution["attribution_unsafe_reasons"],
        "actual_retrieval_branches": actual_branches,
        "degraded_retrieval_branches": degraded_branches,
        "retrieval_config": {
            "backend": "hybrid_rrf",
            "rrf_k": RRF_K,
            "top_k": top_k,
            "component_top_k": top_k,
            "answer_top_k": min(top_k, 12),
            "component_query": "raw_case_question",
        },
        "reranker": None,
        "generation": {
            "with_llm_requested": with_llm,
            "llm_used": llm_used,
            "model": next(
                (item.get("model_name") for item in answers if item.get("generator_mode") == "llm"),
                None,
            ),
        },
        "query_planner": {
            "enabled": with_llm and settings.llm_query_planning,
            "used": planner_used,
            "model": next(
                (
                    item.get("query_planner_model")
                    for item in answers
                    if item.get("query_planner_status") == "used"
                ),
                None,
            ),
        },
        "retrieval_reflection": {
            "enabled": with_llm and settings.llm_retrieval_reflection,
            "used": reflection_used,
            "model": settings.llm_query_planner_model if reflection_used else None,
        },
        "semantic_judge": {
            "requested": semantic_judge,
            "used": semantic_used,
            "model": settings.llm_model if semantic_used else None,
            "citation_version": SEMANTIC_CITATION_JUDGE_VERSION,
            "fact_version": FACT_COVERAGE_JUDGE_VERSION,
        },
        "execution": {
            "degraded_components": degraded_branches,
            "fallback_used": any(
                isinstance(item.get("answer"), dict)
                and (
                    item["answer"].get("llm_status") == "fallback"
                    or item["answer"].get("query_planner_status") == "fallback"
                    or item["answer"].get("retrieval_reflection_status") == "fallback"
                )
                for item in results
            ),
        },
        "index_versions": reports,
        "warnings": indexes["warnings"],
    }


def _selected_dimensions(case: GoldenCase, requested: RequestedDimension) -> set[EvalDimension]:
    available = set(case.eval_dimensions)
    if requested == "all":
        return available
    return {requested} if requested in available else set()


def _filter_cases(
    dataset: GoldenDataset,
    *,
    dimension: RequestedDimension,
    case_ids: set[str] | None,
    categories: set[str] | None,
    limit: int | None,
) -> list[GoldenCase]:
    selected = [
        case
        for case in dataset.cases
        if (not case_ids or case.id in case_ids)
        and (not categories or case.category in categories)
        and bool(_selected_dimensions(case, dimension))
    ]
    return selected[:limit] if limit is not None else selected


def run_golden_benchmark(
    *,
    settings: Settings,
    dataset_path: Path,
    dimension: RequestedDimension = "all",
    case_ids: set[str] | None = None,
    categories: set[str] | None = None,
    limit: int | None = None,
    top_k: int = 10,
    run_name: str | None = None,
    with_llm: bool = False,
    semantic_judge: bool = False,
    write_reports: bool = True,
    search_backend: Callable[..., SearchResponse] = search_hybrid_index,
    answer_backend: Callable[..., AnswerResponse] = answer_question,
) -> dict[str, Any]:
    if top_k < max(DEFAULT_K_VALUES):
        raise ValueError("top_k must be at least 10 so @5 and @10 stay comparable")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    dataset = load_golden_dataset(dataset_path)
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    cases = _filter_cases(
        dataset,
        dimension=dimension,
        case_ids=case_ids,
        categories=categories,
        limit=limit,
    )
    if not cases:
        raise ValueError("no golden cases match the requested filters and dimension")
    unknown_ids = (case_ids or set()).difference(case.id for case in dataset.cases)
    if unknown_ids:
        raise ValueError("unknown golden case IDs: " + ", ".join(sorted(unknown_ids)))
    page_texts = (
        _load_effective_page_texts(settings)
        if any("citation" in _selected_dimensions(case, dimension) for case in cases)
        else {}
    )
    answer_settings = settings if with_llm else settings.model_copy(update={"llm_api_key": None})
    results: list[dict[str, Any]] = []
    for case in cases:
        selected = _selected_dimensions(case, dimension)
        started = perf_counter()
        retrieval_response: SearchResponse | None = None
        retrieval_ms = 0
        if selected.intersection({"retrieval", "ranking", "context"}):
            retrieval_started = perf_counter()
            retrieval_response = search_backend(
                keyword_index_path=settings.keyword_index_path,
                vector_index_path=settings.vector_index_path,
                model_cache_dir=settings.model_cache_dir / "fastembed",
                aliases_path=settings.person_aliases_path,
                query=case.question,
                top_k=top_k,
            )
            retrieval_ms = round((perf_counter() - retrieval_started) * 1000)
        response: AnswerResponse | None = None
        answer_ms = 0
        if selected.intersection({"routing", "generation", "citation"}):
            answer_started = perf_counter()
            response = answer_backend(
                answer_settings,
                QuestionRequest(question=case.question, top_k=min(top_k, 12)),
            )
            answer_ms = round((perf_counter() - answer_started) * 1000)
        semantic_result: dict[str, Any] | None = None
        if semantic_judge and response is not None and response.citations:
            criteria = [*case.semantic_criteria]
            criteria.extend(item.reason for item in case.forbidden_claims)
            semantic_result = judge_citation_semantics(
                settings,
                question=case.question,
                answer=response.answer,
                citations=response.citations,
                semantic_criteria=criteria,
            )
        fact_result: dict[str, Any] | None = None
        if semantic_judge and response is not None and "generation" in selected and case.gold_facts:
            unmatched = [
                {"fact_id": fact.fact_id, "claim": fact.claim}
                for fact in case.gold_facts
                if not _pattern_match(response.answer, fact.deterministic_patterns, fact.claim)
            ]
            fact_result = judge_fact_coverage(
                settings,
                question=case.question,
                answer=response.answer,
                facts=unmatched,
            )
        retrieval_metrics = (
            evaluate_retrieval_hits(case, retrieval_response.hits)
            if retrieval_response is not None
            else None
        )
        ranking_metrics: dict[str, Any] | None = None
        if retrieval_metrics is not None:
            ranking_keys = {
                key
                for key in retrieval_metrics
                if key == "mrr" or key.startswith("ndcg_") or key == "required_evidence_ranks"
            }
            ranking_metrics = {key: retrieval_metrics[key] for key in ranking_keys}
            ranking_metrics["metric_status"] = {
                key: value
                for key, value in retrieval_metrics["metric_status"].items()
                if key == "mrr" or key.startswith("ndcg_")
            }
        token_usage: dict[str, Any] = {}
        if response is not None:
            for key, usage in (
                ("answer_generation", response.llm_usage),
                ("query_planner", response.query_planner_usage),
                ("retrieval_reflection", response.retrieval_reflection_usage),
            ):
                if usage:
                    token_usage[key] = usage
        if semantic_result and semantic_result.get("usage"):
            token_usage["semantic_citation_judge"] = semantic_result["usage"]
        if fact_result and fact_result.get("usage"):
            token_usage["fact_coverage_judge"] = fact_result["usage"]
        case_result: dict[str, Any] = {
            "case_id": case.id,
            "question": case.question,
            "category": case.category,
            "difficulty": case.difficulty,
            "answerability": case.answerability,
            "evaluated_dimensions": sorted(selected),
            "retrieval": retrieval_metrics if "retrieval" in selected else None,
            "ranking": (ranking_metrics if "ranking" in selected else None),
            "context": (
                evaluate_context_hits(case, retrieval_response.hits, top_k=top_k)
                if "context" in selected and retrieval_response is not None
                else None
            ),
            "generation": (
                evaluate_generation(
                    case,
                    response,
                    fact_judgment=fact_result,
                    semantic_judgment=semantic_result,
                )
                if "generation" in selected and response is not None
                else None
            ),
            "citation": (
                evaluate_citations(
                    case,
                    response,
                    page_texts=page_texts,
                    semantic_judgment=semantic_result,
                )
                if "citation" in selected and response is not None
                else None
            ),
            "routing": (
                evaluate_routing(case, response)
                if "routing" in selected and response is not None
                else None
            ),
            "answer": (
                {
                    "text": response.answer,
                    "evidence_status": response.evidence_status,
                    "generator_mode": response.generator_mode,
                    "llm_status": response.llm_status,
                    "model_name": response.model_name,
                    "query_planner_status": response.query_planner_status,
                    "query_planner_model": response.query_planner_model,
                    "retrieval_reflection_status": response.retrieval_reflection_status,
                    "retrieval_mode": response.retrieval_mode,
                    "retrieved_evidence_count": response.retrieved_evidence_count,
                    "citation_count": len(response.citations),
                }
                if response is not None
                else None
            ),
            "retrieved": (
                [
                    {
                        "rank": hit.rank,
                        "document_id": hit.document_id,
                        "pdf_page_start": hit.pdf_page_start,
                        "pdf_page_end": hit.pdf_page_end,
                        "score": hit.score,
                        "keyword_rank": hit.keyword_rank,
                        "vector_rank": hit.vector_rank,
                    }
                    for hit in retrieval_response.hits
                ]
                if retrieval_response is not None
                else None
            ),
            "operational": {
                "retrieval_latency_ms": (retrieval_ms if retrieval_response is not None else None),
                "answer_latency_ms": answer_ms if response is not None else None,
                "total_latency_ms": round((perf_counter() - started) * 1000),
                "token_usage": token_usage,
                "degraded_components": (
                    retrieval_response.degraded_components if retrieval_response is not None else []
                ),
                "retrieval_mode": (
                    retrieval_response.retrieval_mode if retrieval_response is not None else None
                ),
                "metric_status": {},
            },
        }
        operational = case_result["operational"]
        assert isinstance(operational, dict)
        for latency_key in (
            "retrieval_latency_ms",
            "answer_latency_ms",
            "total_latency_ms",
        ):
            latency_value = operational[latency_key]
            _record_metric(
                operational,
                latency_key,
                latency_value,
                evaluable=latency_value is not None,
                reason="stage executed" if latency_value is not None else "stage not requested",
            )
        results.append(case_result)
    run_id = uuid4().hex
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "dataset": {
            "path": str(dataset_path),
            "version": dataset.version,
            "sha256": dataset_sha256,
        },
        "requested_dimension": dimension,
        "filters": {
            "case_ids": sorted(case_ids or []),
            "categories": sorted(categories or []),
            "limit": limit,
        },
        "run_metadata": _run_metadata(
            settings,
            run_name=run_name,
            top_k=top_k,
            with_llm=with_llm,
            semantic_judge=semantic_judge,
            dataset_sha256=dataset_sha256,
            results=results,
        ),
        "aggregate": aggregate_results(results),
        "results": results,
    }
    if write_reports:
        settings.reports_dir.mkdir(parents=True, exist_ok=True)
        rendered = json.dumps(payload, ensure_ascii=False, indent=2)
        (settings.reports_dir / f"golden_benchmark_{run_id}.json").write_text(
            rendered, encoding="utf-8"
        )
        (settings.reports_dir / "golden_benchmark_latest.json").write_text(
            rendered, encoding="utf-8"
        )
    return payload


def _flatten_values(value: Any, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_values(child, path))
    else:
        flattened[prefix] = value
    return flattened


def _aggregate_metric_nodes(value: Any, prefix: str = "") -> dict[str, dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}
    if not isinstance(value, dict):
        return nodes
    if {"value", "evaluable_cases"}.issubset(value):
        nodes[prefix] = value
        return nodes
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        nodes.update(_aggregate_metric_nodes(child, path))
    return nodes


def _metric_compatibility_reasons(
    key: str,
    metric_a: dict[str, Any],
    metric_b: dict[str, Any],
    evaluator_a: dict[str, Any],
    evaluator_b: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []

    def require_equal(field: str, label: str) -> None:
        value_a = evaluator_a.get(field)
        value_b = evaluator_b.get(field)
        if value_a is None or value_b is None:
            reasons.append(f"{label} is missing")
        elif value_a != value_b:
            reasons.append(f"{label} differs")

    require_equal("version", "evaluator version")
    require_equal("metric_definition_version", "metric definition version")
    page_mapped = key.startswith(("retrieval.", "ranking.", "context.")) or key in {
        "citation.citation_page_validity",
        "citation.citation_quote_consistency",
        "citation.citation_precision",
        "citation.citation_recall",
        "citation.annotated_citation_recall",
    }
    relevance_based = key.startswith(("retrieval.", "ranking.")) or key in {
        "context.known_relevant_ratio_lower_bound",
        "context.relevant_evidence_ratio",
        "context.irrelevant_context_ratio",
        "citation.citation_precision",
        "citation.citation_recall",
        "citation.annotated_citation_recall",
    }
    normalization_based = key in {
        "context.redundancy_ratio",
        "citation.citation_quote_consistency",
        "citation.claim_to_citation_coverage",
    }
    if page_mapped:
        require_equal("page_mapping_version", "page mapping version")
    if relevance_based:
        require_equal("relevance_schema_version", "relevance schema version")
    if normalization_based:
        require_equal("text_normalization_version", "text normalization version")

    bases_a = set(metric_a.get("evaluation_bases", []))
    bases_b = set(metric_b.get("evaluation_bases", []))
    if bases_a != bases_b:
        reasons.append("actual evaluation basis differs")
    bases = bases_a | bases_b
    if "semantic" in bases:
        require_equal("fact_judge_version", "fact judge version")
        require_equal("fact_judge_provider", "fact judge provider")
        require_equal("fact_judge_model", "fact judge model")
    if "semantic_citation" in bases:
        require_equal("citation_judge_version", "citation judge version")
        require_equal("citation_judge_provider", "citation judge provider")
        require_equal("citation_judge_model", "citation judge model")
    return reasons


def compare_golden_runs(run_a_path: Path, run_b_path: Path) -> dict[str, Any]:
    """Compare reports at global, metric-definition, denominator, and attribution levels."""

    run_a = json.loads(run_a_path.read_text(encoding="utf-8"))
    run_b = json.loads(run_b_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []
    schema_a = run_a.get("schema_version")
    schema_b = run_b.get("schema_version")
    schema_match = schema_a is not None and schema_a == schema_b
    if not schema_match:
        errors.append("Golden result schema version differs or is missing")
    sha_a = run_a.get("dataset", {}).get("sha256")
    sha_b = run_b.get("dataset", {}).get("sha256")
    sha_match = bool(sha_a) and sha_a == sha_b
    if not sha_match:
        errors.append("dataset SHA-256 differs or is missing")
    ids_a = [item.get("case_id") for item in run_a.get("results", [])]
    ids_b = [item.get("case_id") for item in run_b.get("results", [])]
    case_ids_match = set(ids_a) == set(ids_b)
    if not case_ids_match:
        errors.append("evaluated case IDs differ")
    dimension_a = run_a.get("requested_dimension")
    dimension_b = run_b.get("requested_dimension")
    dimension_match = dimension_a is not None and dimension_a == dimension_b
    if not dimension_match:
        errors.append("requested dimensions differ or are missing")

    run_metadata_a = run_a.get("run_metadata", {})
    run_metadata_b = run_b.get("run_metadata", {})
    metadata_a = _flatten_values(run_metadata_a)
    metadata_b = _flatten_values(run_metadata_b)
    ignored_metadata = {"timestamp", "run_name", "git.dirty"}
    metadata_differences = {
        key: {"run_a": metadata_a.get(key), "run_b": metadata_b.get(key)}
        for key in sorted(set(metadata_a) | set(metadata_b))
        if key not in ignored_metadata and metadata_a.get(key) != metadata_b.get(key)
    }
    evaluator_a = run_metadata_a.get("evaluator", {}) if isinstance(run_metadata_a, dict) else {}
    evaluator_b = run_metadata_b.get("evaluator", {}) if isinstance(run_metadata_b, dict) else {}
    attribution_a = (
        run_metadata_a.get("attribution_safe") is True
        if isinstance(run_metadata_a, dict)
        else False
    )
    attribution_b = (
        run_metadata_b.get("attribution_safe") is True
        if isinstance(run_metadata_b, dict)
        else False
    )
    attribution_safe = attribution_a and attribution_b
    if not attribution_safe:
        warnings.append(
            "comparison is possible, but configuration attribution is unsafe for one or both runs"
        )

    nodes_a = _aggregate_metric_nodes(run_a.get("aggregate", {}))
    nodes_b = _aggregate_metric_nodes(run_b.get("aggregate", {}))
    metrics: dict[str, Any] = {}
    for key in sorted(set(nodes_a) | set(nodes_b)):
        a = nodes_a.get(key)
        b = nodes_b.get(key)
        reasons = list(errors)
        if a is None or b is None:
            reasons.append("metric is present in only one run")
            a = a or {}
            b = b or {}
        fact_metric = key in {
            "generation.required_fact_recall",
            "generation.optional_fact_recall",
        }
        unit_ids_a = a.get("evaluable_unit_ids" if fact_metric else "evaluable_case_ids", [])
        unit_ids_b = b.get("evaluable_unit_ids" if fact_metric else "evaluable_case_ids", [])
        if fact_metric and ("evaluable_unit_ids" not in a or "evaluable_unit_ids" not in b):
            reasons.append("fact-level evaluable unit IDs are missing")
        elif set(unit_ids_a) != set(unit_ids_b):
            unit_label = "fact unit" if fact_metric else "case"
            reasons.append(f"evaluable {unit_label} set differs")
            warnings.append(f"evaluable {unit_label} set differs for {key}; delta suppressed")
        if a and b:
            reasons.extend(_metric_compatibility_reasons(key, a, b, evaluator_a, evaluator_b))
        value_a = a.get("value")
        value_b = b.get("value")
        numeric = isinstance(value_a, (int, float)) and isinstance(value_b, (int, float))
        if not numeric:
            reasons.append("one or both metric values are not numeric")
        reasons = list(dict.fromkeys(reasons))
        metric_comparable = not reasons
        delta = (
            round(value_b - value_a, 6)
            if metric_comparable
            and isinstance(value_a, (int, float))
            and isinstance(value_b, (int, float))
            else None
        )
        metrics[key] = {
            "run_a": value_a,
            "run_b": value_b,
            "delta": delta,
            "evaluable_cases_a": a.get("evaluable_cases"),
            "evaluable_cases_b": b.get("evaluable_cases"),
            "evaluable_unit_ids_a": unit_ids_a,
            "evaluable_unit_ids_b": unit_ids_b,
            "comparable": metric_comparable,
            "non_comparable_reasons": reasons,
        }

    usage_a = _flatten_values(
        run_a.get("aggregate", {}).get("operational", {}).get("token_usage", {})
    )
    usage_b = _flatten_values(
        run_b.get("aggregate", {}).get("operational", {}).get("token_usage", {})
    )
    token_usage = {
        key: {
            "run_a": usage_a.get(key),
            "run_b": usage_b.get(key),
            "delta": (
                usage_b[key] - usage_a[key]
                if not errors
                and isinstance(usage_a.get(key), int)
                and isinstance(usage_b.get(key), int)
                else None
            ),
            "comparable": (
                not errors
                and isinstance(usage_a.get(key), int)
                and isinstance(usage_b.get(key), int)
            ),
            "non_comparable_reasons": list(errors)
            or (
                []
                if isinstance(usage_a.get(key), int) and isinstance(usage_b.get(key), int)
                else ["one or both token usage values are missing"]
            ),
        }
        for key in sorted(set(usage_a) | set(usage_b))
    }
    global_checks = {
        "schema_version_match": schema_match,
        "dataset_sha256_match": sha_match,
        "case_ids_match": case_ids_match,
        "requested_dimension_match": dimension_match,
    }
    return {
        "schema_version": 1,
        "run_a": str(run_a_path),
        "run_b": str(run_b_path),
        "compatible": not errors,
        "attribution_safe": attribution_safe,
        "attribution": {
            "run_a": attribution_a,
            "run_b": attribution_b,
            "run_a_reasons": (
                run_metadata_a.get("attribution_unsafe_reasons", [])
                if isinstance(run_metadata_a, dict)
                else ["run metadata is missing"]
            ),
            "run_b_reasons": (
                run_metadata_b.get("attribution_unsafe_reasons", [])
                if isinstance(run_metadata_b, dict)
                else ["run metadata is missing"]
            ),
        },
        "global_checks": global_checks,
        "checks": global_checks,
        "errors": errors,
        "warnings": list(dict.fromkeys(warnings)),
        "metadata_differences": metadata_differences,
        "metrics": metrics,
        "token_usage": token_usage,
    }


def _module_main() -> None:
    parser = argparse.ArgumentParser(description="Run the Unified RAG Golden Benchmark.")
    parser.add_argument("--dataset", default="evals/golden/golden_questions.json")
    parser.add_argument("--dimension", choices=("all", *ALL_DIMENSIONS), default="all")
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--category", action="append", dest="categories")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--run-name")
    parser.add_argument("--with-llm", action="store_true")
    parser.add_argument("--semantic-judge", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    payload = run_golden_benchmark(
        settings=settings,
        dataset_path=settings.project_root / args.dataset,
        dimension=args.dimension,
        case_ids=set(args.case_ids or []),
        categories=set(args.categories or []),
        limit=args.limit,
        top_k=args.top_k,
        run_name=args.run_name,
        with_llm=args.with_llm,
        semantic_judge=args.semantic_judge,
    )
    print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))
    print(settings.reports_dir / "golden_benchmark_latest.json")


if __name__ == "__main__":
    _module_main()
