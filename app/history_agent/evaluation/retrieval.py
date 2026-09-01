from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from history_agent.config import Settings
from history_agent.retrieval.hybrid import search_hybrid_index


class GoldEvidence(BaseModel):
    document_id: str
    pdf_pages: list[int] = Field(min_length=1)


class RetrievalQuestion(BaseModel):
    question_id: str
    question: str
    answerable: bool = True
    expected_document_ids: list[str] = Field(default_factory=list)
    expected_years: list[int] = Field(default_factory=list)
    expected_evidence: list[GoldEvidence] = Field(default_factory=list)
    required_fact_terms: list[list[str]] = Field(default_factory=list)
    forbidden_answer_terms: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_expected_answer(self) -> RetrievalQuestion:
        if self.answerable and not self.expected_document_ids:
            raise ValueError("answerable questions require expected_document_ids")
        if not self.answerable and self.expected_evidence:
            raise ValueError("unanswerable questions cannot declare expected_evidence")
        if any(not alternatives for alternatives in self.required_fact_terms):
            raise ValueError("required_fact_terms groups cannot be empty")
        return self


class RetrievalQuestionSet(BaseModel):
    schema_version: int
    questions: list[RetrievalQuestion] = Field(min_length=1)


def load_question_set(path: Path) -> RetrievalQuestionSet:
    return RetrievalQuestionSet.model_validate_json(path.read_text(encoding="utf-8"))


def evaluate_retrieval(
    *,
    settings: Settings,
    question_set_path: Path,
    top_k: int,
    run_id: str,
) -> dict[str, object]:
    question_set = load_question_set(question_set_path)
    results: list[dict[str, object]] = []
    reciprocal_ranks: list[float] = []
    tag_totals: dict[str, int] = defaultdict(int)
    tag_hits: dict[str, int] = defaultdict(int)
    for item in question_set.questions:
        if not item.answerable:
            results.append(
                {
                    "question_id": item.question_id,
                    "question": item.question,
                    "answerable": False,
                    "success": None,
                    "target_document_rank": None,
                    "year_hit": None,
                    "expected_document_ids": [],
                    "expected_years": item.expected_years,
                    "returned": [],
                }
            )
            continue
        response = search_hybrid_index(
            keyword_index_path=settings.keyword_index_path,
            vector_index_path=settings.vector_index_path,
            model_cache_dir=settings.model_cache_dir / "fastembed",
            aliases_path=settings.person_aliases_path,
            query=item.question,
            top_k=top_k,
        )
        document_rank = next(
            (
                hit.rank
                for hit in response.hits
                if hit.document_id in item.expected_document_ids
            ),
            None,
        )
        year_hit = not item.expected_years or any(
            set(item.expected_years).intersection(hit.year_mentions)
            for hit in response.hits
        )
        success = document_rank is not None and year_hit
        reciprocal_ranks.append(1.0 / document_rank if document_rank else 0.0)
        for tag in item.tags:
            tag_totals[tag] += 1
            tag_hits[tag] += int(success)
        results.append(
            {
                "question_id": item.question_id,
                "question": item.question,
                "answerable": True,
                "success": success,
                "target_document_rank": document_rank,
                "year_hit": year_hit,
                "expected_document_ids": item.expected_document_ids,
                "expected_years": item.expected_years,
                "returned": [
                    {
                        "rank": hit.rank,
                        "document_id": hit.document_id,
                        "pdf_page": hit.pdf_page_start,
                        "years": hit.year_mentions,
                        "score": hit.score,
                    }
                    for hit in response.hits
                ],
            }
        )
    evaluated_results = [result for result in results if result["answerable"]]
    hits = sum(int(bool(result["success"])) for result in evaluated_results)
    total = len(evaluated_results)
    payload: dict[str, object] = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "question_set": str(question_set_path),
        "retrieval_mode": "hybrid_rrf",
        "top_k": top_k,
        "questions": len(results),
        "evaluated_questions": total,
        "hits": hits,
        "recall_at_k": round(hits / total, 6),
        "mean_reciprocal_rank": round(sum(reciprocal_ranks) / total, 6),
        "tag_recall": {
            tag: round(tag_hits[tag] / count, 6)
            for tag, count in sorted(tag_totals.items())
        },
        "results": results,
    }
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    (settings.reports_dir / f"retrieval_eval_{run_id}.json").write_text(
        rendered, encoding="utf-8"
    )
    (settings.reports_dir / "retrieval_eval_latest.json").write_text(
        rendered, encoding="utf-8"
    )
    return payload
