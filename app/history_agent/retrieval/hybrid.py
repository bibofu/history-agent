from __future__ import annotations

from pathlib import Path

from history_agent.retrieval.keyword import search_keyword_index
from history_agent.retrieval.models import SearchHit, SearchResponse
from history_agent.retrieval.vector import search_vector_index

RRF_K = 60
INTENT_SOURCE_BONUS = 0.004


def _source_bonus(
    intent: str, source_type: str, title: str, query_people: list[str]
) -> float:
    if intent == "timeline":
        if query_people and any(person in title for person in query_people):
            return INTENT_SOURCE_BONUS
        if not query_people and source_type == "chronology":
            return INTENT_SOURCE_BONUS
    if intent == "viewpoint" and source_type == "selected_works":
        return INTENT_SOURCE_BONUS
    return 0.0


def fuse_search_responses(
    keyword: SearchResponse,
    vector: SearchResponse,
    *,
    top_k: int,
) -> SearchResponse:
    """Fuse keyword and semantic rankings with reciprocal-rank fusion."""

    by_chunk: dict[str, SearchHit] = {}
    scores: dict[str, float] = {}
    for hit in keyword.hits:
        merged = hit.model_copy(deep=True)
        merged.keyword_rank = hit.rank
        merged.keyword_score = hit.score
        merged.vector_rank = None
        merged.vector_score = None
        merged.rrf_score = None
        by_chunk[hit.chunk_id] = merged
        scores[hit.chunk_id] = 1.0 / (RRF_K + hit.rank)
    for hit in vector.hits:
        existing = by_chunk.get(hit.chunk_id)
        if existing is None:
            existing = hit.model_copy(deep=True)
            existing.keyword_rank = None
            existing.keyword_score = None
            by_chunk[hit.chunk_id] = existing
            scores[hit.chunk_id] = 0.0
        existing.vector_rank = hit.rank
        existing.vector_score = hit.score
        scores[hit.chunk_id] += 1.0 / (RRF_K + hit.rank)

    for chunk_id, hit in by_chunk.items():
        scores[chunk_id] += _source_bonus(
            keyword.query_intent,
            hit.source_type,
            hit.title,
            keyword.query_people,
        )
        hit.rrf_score = round(scores[chunk_id], 8)
        hit.score = hit.rrf_score

    ranked = sorted(
        by_chunk.values(),
        key=lambda hit: (
            -scores[hit.chunk_id],
            hit.pdf_page_start,
            hit.chunk_id,
        ),
    )
    # A page may yield several adjacent chunks. One result per physical page gives
    # the answer layer a broader, less repetitive evidence set.
    selected: list[SearchHit] = []
    seen_pages: set[tuple[str, int]] = set()
    for hit in ranked:
        page_key = (hit.document_id, hit.pdf_page_start)
        if page_key in seen_pages:
            continue
        seen_pages.add(page_key)
        hit.rank = len(selected) + 1
        selected.append(hit)
        if len(selected) == top_k:
            break

    return SearchResponse(
        query=keyword.query,
        query_intent=keyword.query_intent,
        query_terms=keyword.query_terms,
        query_years=keyword.query_years,
        query_year_range=keyword.query_year_range,
        query_people=keyword.query_people,
        document_filters=keyword.document_filters,
        include_out_of_scope=keyword.include_out_of_scope,
        hits=selected,
        retrieval_mode="hybrid_rrf",
    )


def search_hybrid_index(
    *,
    keyword_index_path: Path,
    vector_index_path: Path,
    model_cache_dir: Path,
    aliases_path: Path,
    query: str,
    top_k: int = 10,
    document_ids: list[str] | None = None,
    include_out_of_scope: bool = False,
) -> SearchResponse:
    candidate_k = min(100, max(30, top_k * 4))
    keyword = search_keyword_index(
        index_path=keyword_index_path,
        query=query,
        aliases_path=aliases_path,
        top_k=candidate_k,
        document_ids=document_ids,
        include_out_of_scope=include_out_of_scope,
    )
    vector = search_vector_index(
        index_path=vector_index_path,
        model_cache_dir=model_cache_dir,
        aliases_path=aliases_path,
        query=query,
        top_k=candidate_k,
        document_ids=document_ids,
        include_out_of_scope=include_out_of_scope,
    )
    return fuse_search_responses(keyword, vector, top_k=top_k)
