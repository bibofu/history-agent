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
import inspect
import json
import math
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
    judge_citation_semantics,
    judge_fact_coverage,
)
from history_agent.processing.chunks import CHUNKER_VERSION, split_text
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
                self.retrieval_gold.required_evidence
                or self.retrieval_gold.relevant_evidence
            ):
                raise ValueError("unanswerable cases cannot declare gold evidence")
        else:
            if needs_evidence and (
                self.retrieval_gold is None
                or not (
                    self.retrieval_gold.required_evidence
                    or self.retrieval_gold.relevant_evidence
                )
            ):
                raise ValueError("answerable evidence dimensions require page-level gold")
            if "generation" in self.eval_dimensions and not self.gold_facts:
                raise ValueError("answerable generation cases require gold_facts")
        fact_ids = [item.fact_id for item in self.gold_facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("gold fact IDs must be unique within a case")
        if self.retrieval_gold is not None:
            gold_pages = {
                item.key for item in self.retrieval_gold.required_evidence
            } | {item.key for item in self.retrieval_gold.relevant_evidence}
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

    return {
        (hit.document_id, page)
        for page in range(hit.pdf_page_start, hit.pdf_page_end + 1)
    }


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


def _ranked_relevance(
    hits: list[SearchHit], gold: dict[tuple[str, int], int], k: int
) -> list[int]:
    """Assign each gold page once so duplicate chunks cannot inflate nDCG."""

    unused = set(gold)
    grades: list[int] = []
    for hit in hits[:k]:
        matches = _hit_page_keys(hit).intersection(unused)
        grade = max((gold[key] for key in matches), default=0)
        grades.append(grade)
        if matches:
            best = min(key for key in matches if gold[key] == grade)
            unused.remove(best)
    grades.extend([0] * (k - len(grades)))
    return grades


def _dcg(grades: list[int]) -> float:
    return float(
        sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, 1))
    )


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
    }
    for k in k_values:
        prefix = hits[:k]
        matched = _matched_gold_keys(prefix, gold)
        relevant_hits = sum(bool(_hit_page_keys(hit).intersection(gold)) for hit in prefix)
        result[f"hit_at_{k}"] = bool(matched) if gold else None
        result[f"recall_at_{k}"] = len(matched) / len(gold) if gold else None
        result[f"precision_at_{k}"] = relevant_hits / k if complete else None
        result[f"precision_at_{k}_evaluable"] = complete
        if complete:
            grades = _ranked_relevance(hits, gold, k)
            ideal = sorted(gold.values(), reverse=True)[:k]
            ideal.extend([0] * (k - len(ideal)))
            ideal_dcg = _dcg(ideal)
            result[f"ndcg_at_{k}"] = _dcg(grades) / ideal_dcg if ideal_dcg else None
        else:
            result[f"ndcg_at_{k}"] = None
        result[f"ndcg_at_{k}_evaluable"] = complete
    first_rank = next(
        (
            rank
            for rank, hit in enumerate(hits, 1)
            if _hit_page_keys(hit).intersection(gold)
        ),
        None,
    )
    result["mrr"] = 1.0 / first_rank if first_rank is not None else 0.0 if gold else None
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


def evaluate_context_hits(
    case: GoldenCase, hits: list[SearchHit], *, top_k: int
) -> dict[str, Any]:
    """Score the final retrieval candidate list used as component context."""

    selected = hits[:top_k]
    gold = _known_relevance(case)
    known_relevant = sum(
        bool(_hit_page_keys(hit).intersection(gold)) for hit in selected
    )
    complete = bool(case.retrieval_gold and case.retrieval_gold.relevance_complete)
    seen_starts: set[tuple[str, int]] = set()
    duplicate_starts = 0
    for hit in selected:
        key = (hit.document_id, hit.pdf_page_start)
        duplicate_starts += int(key in seen_starts)
        seen_starts.add(key)
    total = len(selected)
    known_ratio = known_relevant / total if total else None
    return {
        "context_size": total,
        "known_relevant_ratio_lower_bound": known_ratio,
        "relevant_evidence_ratio": known_ratio if complete else None,
        "irrelevant_context_ratio": (
            (1.0 - known_ratio) if complete and known_ratio is not None else None
        ),
        "relevance_ratio_evaluable": complete,
        "duplicate_start_page_ratio": duplicate_starts / total if total else None,
    }


def _normalize_text(value: str) -> str:
    return "".join(character.casefold() for character in value if character.isalnum())


def _pattern_match(answer: str, patterns: list[str], fallback: str) -> bool:
    normalized = _normalize_text(answer)
    candidates = patterns or [fallback]
    return any(_normalize_text(item) in normalized for item in candidates)


def evaluate_generation(
    case: GoldenCase,
    response: AnswerResponse,
    *,
    fact_judgment: dict[str, Any] | None = None,
    semantic_judgment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    decisions = {
        item["fact_id"]: item
        for item in (fact_judgment or {}).get("decisions", [])
    }
    fact_results: list[dict[str, Any]] = []
    for fact in case.gold_facts:
        deterministic = _pattern_match(
            response.answer, fact.deterministic_patterns, fact.claim
        )
        semantic = decisions.get(fact.fact_id)
        if deterministic:
            covered, judge, reason = True, "deterministic", "matched an annotated pattern"
        elif semantic is not None:
            covered = bool(semantic["covered"])
            judge = "semantic"
            reason = str(semantic["reason"])
        else:
            covered = False
            judge = "not_evaluated"
            reason = "no deterministic match; semantic fallback was not available"
        fact_results.append(
            {
                "fact_id": fact.fact_id,
                "importance": fact.importance,
                "covered": covered,
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
            "violated": _pattern_match(
                response.answer, item.deterministic_patterns, item.claim
            ),
        }
        for item in case.forbidden_claims
    ]
    unsupported_claims = list((semantic_judgment or {}).get("unsupported_claims", []))
    unsupported_claims.extend(item["claim"] for item in forbidden if item["violated"])
    unsupported_claims = list(dict.fromkeys(unsupported_claims))
    semantic_used = (semantic_judgment or {}).get("verdict") is not None
    unsupported_evaluable = semantic_used or bool(forbidden)
    has_answer = response.evidence_status != "no_evidence" and bool(response.citations)
    refusal_correct = not has_answer if case.answerability == "unanswerable" else None
    required_recall = (
        sum(bool(item["covered"]) for item in required) / len(required)
        if required
        else None
    )
    optional_recall = (
        sum(bool(item["covered"]) for item in optional) / len(optional)
        if optional
        else None
    )
    semantic_failed = (semantic_judgment or {}).get("verdict") == "fail"
    forbidden_violation = any(bool(item["violated"]) for item in forbidden)
    if case.answerability == "unanswerable":
        correct = bool(refusal_correct) and not forbidden_violation and not semantic_failed
    else:
        correct = required_recall == 1.0 and not forbidden_violation and not semantic_failed
    return {
        "required_fact_recall": required_recall,
        "optional_fact_recall": optional_recall,
        "facts": fact_results,
        "fact_judge_status": (fact_judgment or {}).get("status", "not_run"),
        "forbidden_claim_violation": forbidden_violation,
        "forbidden_claims": forbidden,
        "unsupported_claims": unsupported_claims,
        "unsupported_claim_detected": (
            bool(unsupported_claims) if unsupported_evaluable else None
        ),
        "refusal_correct": refusal_correct,
        "false_answer": not bool(refusal_correct)
        if case.answerability == "unanswerable"
        else None,
        "answer_correctness": correct,
        "correctness_basis": (
            "deterministic+semantic" if semantic_used or decisions else "deterministic"
        ),
    }


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
        pages = {
            (citation.document_id, page)
            for page in range(citation.pdf_page, end + 1)
        }
        matches = pages.intersection(gold)
        cited_gold.update(matches)
        relevant_citations += int(bool(matches))
        page_text = page_texts.get((citation.document_id, citation.pdf_page))
        details.append(
            {
                "evidence_id": citation.evidence_id,
                "document_id": citation.document_id,
                "pdf_page": citation.pdf_page,
                "pdf_page_end": citation.pdf_page_end,
                "page_exists": page_text is not None,
                "quote_matches_page": bool(
                    page_text and quote_matches_page(citation.quote, page_text)
                ),
                "matches_gold_page": bool(matches),
            }
        )
    count = len(details)
    validation = (
        validate_grounded_answer(response.answer, response.citations)
        if response.citations
        else None
    )
    return {
        "citation_presence": bool(response.citations),
        "citation_presence_correct": bool(response.citations)
        == (case.answerability == "answerable"),
        "citation_page_validity": (
            sum(bool(item["page_exists"]) for item in details) / count if count else None
        ),
        "citation_quote_consistency": (
            sum(bool(item["quote_matches_page"]) for item in details) / count
            if count
            else None
        ),
        "claim_to_citation_coverage": (
            bool(validation.valid)
            if validation is not None
            else False
            if case.answerability == "answerable"
            else None
        ),
        "uncited_claims": list(validation.uncited_claims) if validation else [],
        "citation_precision": relevant_citations / count if complete and count else None,
        "citation_precision_evaluable": complete and bool(count),
        "citation_recall": len(cited_gold) / len(gold) if gold else None,
        "semantic_support": (
            (semantic_judgment or {}).get("verdict") == "pass"
            if (semantic_judgment or {}).get("verdict") is not None
            else None
        ),
        "semantic_judge_status": (semantic_judgment or {}).get("status", "not_run"),
        "semantic_judge_reason": (semantic_judgment or {}).get("rationale"),
        "details": details,
    }


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
    return {
        "actual_route": actual,
        "raw_retrieval_mode": response.retrieval_mode,
        "preferred_routes": preferred,
        "route_match": actual in preferred,
    }


def _metric(values: list[float]) -> dict[str, Any]:
    return {
        "value": round(mean(values), 6) if values else None,
        "evaluable_cases": len(values),
    }


def _section_values(
    results: list[dict[str, Any]], section: str, key: str
) -> list[float]:
    values: list[float] = []
    for result in results:
        payload = result.get(section)
        if not isinstance(payload, dict):
            continue
        value = payload.get(key)
        if isinstance(value, (bool, int, float)):
            values.append(float(value))
    return values


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    dimensions = {
        name: sum(isinstance(item.get(name), dict) for item in results)
        for name in ALL_DIMENSIONS
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
        aggregate["retrieval"][f"hit_rate_at_{k}"] = _metric(
            _section_values(results, "retrieval", f"hit_at_{k}")
        )
        aggregate["retrieval"][f"recall_at_{k}"] = _metric(
            _section_values(results, "retrieval", f"recall_at_{k}")
        )
        aggregate["retrieval"][f"precision_at_{k}"] = _metric(
            _section_values(results, "retrieval", f"precision_at_{k}")
        )
        aggregate["retrieval"][f"ndcg_at_{k}"] = _metric(
            _section_values(results, "retrieval", f"ndcg_at_{k}")
        )
        aggregate["ranking"][f"ndcg_at_{k}"] = _metric(
            _section_values(results, "ranking", f"ndcg_at_{k}")
        )
    aggregate["retrieval"]["mrr"] = _metric(
        _section_values(results, "retrieval", "mrr")
    )
    aggregate["ranking"]["mrr"] = _metric(_section_values(results, "ranking", "mrr"))
    for key in (
        "known_relevant_ratio_lower_bound",
        "relevant_evidence_ratio",
        "irrelevant_context_ratio",
        "duplicate_start_page_ratio",
    ):
        aggregate["context"][key] = _metric(_section_values(results, "context", key))
    for key in (
        "required_fact_recall",
        "optional_fact_recall",
        "answer_correctness",
        "forbidden_claim_violation",
        "unsupported_claim_detected",
        "refusal_correct",
        "false_answer",
    ):
        aggregate["generation"][key] = _metric(
            _section_values(results, "generation", key)
        )
    for key in (
        "citation_presence_correct",
        "citation_page_validity",
        "citation_quote_consistency",
        "claim_to_citation_coverage",
        "citation_precision",
        "citation_recall",
        "semantic_support",
    ):
        aggregate["citation"][key] = _metric(_section_values(results, "citation", key))
    aggregate["routing"]["route_accuracy"] = _metric(
        _section_values(results, "routing", "route_match")
    )
    aggregate["routing"]["mismatch_case_ids"] = [
        item["case_id"]
        for item in results
        if isinstance(item.get("routing"), dict) and not item["routing"]["route_match"]
    ]
    latency_keys = ("retrieval_latency_ms", "answer_latency_ms", "total_latency_ms")
    aggregate["operational"] = {
        key: _metric(_section_values(results, "operational", key))
        for key in latency_keys
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
    result: dict[str, Any] = {}
    for name in ("keyword_index_latest.json", "vector_index_latest.json"):
        path = settings.reports_dir / name
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        result[name.removesuffix("_latest.json")] = {
            key: payload.get(key)
            for key in ("run_id", "index_version", "model_name", "chunks")
            if payload.get(key) is not None
        }
    return result


def _run_metadata(
    settings: Settings, *, run_name: str | None, top_k: int, with_llm: bool
) -> dict[str, Any]:
    signature = inspect.signature(split_text)
    indexes = _load_index_metadata(settings)
    vector = indexes.get("vector_index", {})
    return {
        "run_name": run_name,
        "timestamp": datetime.now(UTC).isoformat(),
        "git": _git_metadata(settings.project_root),
        "project_version": __version__,
        "prompt_version": PROMPT_VERSION,
        "rag_framework": "llamaindex",
        "embedding_model": vector.get("model_name"),
        "chunking": {
            "version": CHUNKER_VERSION,
            "target_chars": signature.parameters["target_chars"].default,
            "max_chars": signature.parameters["max_chars"].default,
            "overlap": 0,
        },
        "retrieval_config": {
            "backend": "hybrid_rrf",
            "rrf_k": RRF_K,
            "top_k": top_k,
            "component_query": "raw_case_question",
        },
        "reranker": None,
        "llm": settings.llm_model if with_llm else None,
        "index_versions": indexes,
    }


def _selected_dimensions(
    case: GoldenCase, requested: RequestedDimension
) -> set[EvalDimension]:
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
    answer_settings = (
        settings if with_llm else settings.model_copy(update={"llm_api_key": None})
    )
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
        if (
            semantic_judge
            and response is not None
            and "generation" in selected
            and case.gold_facts
        ):
            unmatched = [
                {"fact_id": fact.fact_id, "claim": fact.claim}
                for fact in case.gold_facts
                if not _pattern_match(
                    response.answer, fact.deterministic_patterns, fact.claim
                )
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
            "ranking": (
                {
                    key: value
                    for key, value in (retrieval_metrics or {}).items()
                    if key == "mrr"
                    or key.startswith("ndcg_")
                    or key == "required_evidence_ranks"
                }
                if "ranking" in selected
                else None
            ),
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
                "retrieval_latency_ms": (
                    retrieval_ms if retrieval_response is not None else None
                ),
                "answer_latency_ms": answer_ms if response is not None else None,
                "total_latency_ms": round((perf_counter() - started) * 1000),
                "token_usage": token_usage,
            },
        }
        results.append(case_result)
    run_id = uuid4().hex
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "dataset": {
            "path": str(dataset_path),
            "version": dataset.version,
            "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        },
        "requested_dimension": dimension,
        "filters": {
            "case_ids": sorted(case_ids or []),
            "categories": sorted(categories or []),
            "limit": limit,
        },
        "run_metadata": _run_metadata(
            settings, run_name=run_name, top_k=top_k, with_llm=with_llm
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


def _module_main() -> None:
    parser = argparse.ArgumentParser(description="Run the Unified RAG Golden Benchmark.")
    parser.add_argument(
        "--dataset", default="evals/golden/golden_questions.json"
    )
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
