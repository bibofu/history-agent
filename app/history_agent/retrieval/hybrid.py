from __future__ import annotations

import re
from pathlib import Path

from history_agent.retrieval.keyword import search_keyword_index
from history_agent.retrieval.models import SearchHit, SearchResponse
from history_agent.retrieval.vector import search_vector_index

RRF_K = 60
INTENT_SOURCE_BONUS = 0.004
TEMPORAL_BUCKET_COUNT = 3
MIN_TEMPORAL_COVERAGE_YEARS = 5
TEMPORAL_FOCUS_MARKERS = ("初期", "前期", "早期", "中期", "后期", "晚期", "末期")
OBSERVATION_QUERY_MARKERS = ("怎样记述", "如何记述", "怎样描述", "如何描述")
OBSERVATION_EXPANSION = "外貌 性格 生活 印象"
CPC_CONGRESS_SHORT_NAME = re.compile(
    r"中共(?:第)?(?P<ordinal>[一二三四五六七八九十]{1,3})大"
)


def expand_query(query: str) -> str:
    """Add restrained search hints for source observations and named congresses."""

    expansions: list[str] = []
    if any(marker in query for marker in OBSERVATION_QUERY_MARKERS):
        expansions.append(OBSERVATION_EXPANSION)
    congress_ordinals = dict.fromkeys(
        match.group("ordinal") for match in CPC_CONGRESS_SHORT_NAME.finditer(query)
    )
    for ordinal in congress_ordinals:
        expansions.append(f"中国共产党第{ordinal}次全国代表大会")
    return f"{query} {' '.join(expansions)}" if expansions else query


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
    if intent == "observation" and source_type == "contemporary_observation":
        return INTENT_SOURCE_BONUS
    return 0.0


def _primary_year(hit: SearchHit, start_year: int, end_year: int) -> int | None:
    years = sorted(year for year in hit.year_mentions if start_year <= year <= end_year)
    if not years:
        return None
    positions = [
        (hit.text.find(f"{year}年"), year)
        for year in years
        if hit.text.find(f"{year}年") >= 0
    ]
    return min(positions)[1] if positions else years[0]


def _select_with_temporal_coverage(
    ranked: list[SearchHit],
    *,
    query: str,
    query_year_range: list[int],
    top_k: int,
) -> list[SearchHit]:
    if (
        len(query_year_range) != 2
        or query_year_range[1] - query_year_range[0] + 1 < MIN_TEMPORAL_COVERAGE_YEARS
        or top_k < TEMPORAL_BUCKET_COUNT
        or any(marker in query for marker in TEMPORAL_FOCUS_MARKERS)
    ):
        return ranked[:top_k]

    start_year, end_year = query_year_range
    span = end_year - start_year + 1
    buckets: list[list[SearchHit]] = [[] for _ in range(TEMPORAL_BUCKET_COUNT)]
    for hit in ranked:
        year = _primary_year(hit, start_year, end_year)
        if year is None:
            continue
        bucket_index = min(
            TEMPORAL_BUCKET_COUNT - 1,
            (year - start_year) * TEMPORAL_BUCKET_COUNT // span,
        )
        buckets[bucket_index].append(hit)

    quota = max(1, top_k // TEMPORAL_BUCKET_COUNT)
    selected_ids: set[str] = set()
    for bucket in buckets:
        for hit in bucket[:quota]:
            selected_ids.add(hit.chunk_id)
    for hit in ranked:
        if len(selected_ids) >= top_k:
            break
        selected_ids.add(hit.chunk_id)
    return [hit for hit in ranked if hit.chunk_id in selected_ids][:top_k]


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
    unique_pages: list[SearchHit] = []
    seen_pages: set[tuple[str, int]] = set()
    for hit in ranked:
        page_key = (hit.document_id, hit.pdf_page_start)
        if page_key in seen_pages:
            continue
        seen_pages.add(page_key)
        unique_pages.append(hit)

    selected = _select_with_temporal_coverage(
        unique_pages,
        query=keyword.query,
        query_year_range=keyword.query_year_range,
        top_k=top_k,
    )
    for rank, hit in enumerate(selected, start=1):
        hit.rank = rank

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
    search_query = expand_query(query)
    keyword = search_keyword_index(
        index_path=keyword_index_path,
        query=search_query,
        aliases_path=aliases_path,
        top_k=candidate_k,
        document_ids=document_ids,
        include_out_of_scope=include_out_of_scope,
    )
    vector = search_vector_index(
        index_path=vector_index_path,
        model_cache_dir=model_cache_dir,
        aliases_path=aliases_path,
        query=search_query,
        top_k=candidate_k,
        document_ids=document_ids,
        include_out_of_scope=include_out_of_scope,
    )
    response = fuse_search_responses(keyword, vector, top_k=top_k)
    return response.model_copy(update={"query": query})
