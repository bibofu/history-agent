from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from history_agent.config import Settings
from history_agent.retrieval.hybrid import search_hybrid_index


class RetrievalQuestion(BaseModel):
    question_id: str
    question: str
    expected_document_ids: list[str] = Field(min_length=1)
    expected_years: list[int] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


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
    hits = sum(int(bool(result["success"])) for result in results)
    total = len(results)
    payload: dict[str, object] = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "question_set": str(question_set_path),
        "retrieval_mode": "hybrid_rrf",
        "top_k": top_k,
        "questions": total,
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
