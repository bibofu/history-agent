from __future__ import annotations

import logging
import re
from pathlib import Path

from history_agent.errors import RetrievalError
from history_agent.retrieval.keyword import YEAR, search_keyword_index
from history_agent.retrieval.models import RetrievalPlan, SearchHit, SearchResponse
from history_agent.retrieval.vector import search_vector_index

RRF_K = 60
INTENT_SOURCE_BONUS = 0.004
TEMPORAL_BUCKET_COUNT = 3
MIN_TEMPORAL_COVERAGE_YEARS = 5
TEMPORAL_FOCUS_MARKERS = ("初期", "前期", "早期", "中期", "后期", "晚期", "末期")
OBSERVATION_QUERY_MARKERS = ("怎样记述", "如何记述", "怎样描述", "如何描述")
OBSERVATION_EXPANSION = "外貌 性格 生活 印象"
CPC_CONGRESS_ORDINALS = (
    "一",
    "二",
    "三",
    "四",
    "五",
    "六",
    "七",
    "八",
    "九",
    "十",
    "十一",
    "十二",
    "十三",
    "十四",
    "十五",
    "十六",
    "十七",
    "十八",
    "十九",
    "二十",
)
CPC_CONGRESS_NUMBER = {
    ordinal: number for number, ordinal in enumerate(CPC_CONGRESS_ORDINALS, start=1)
}
CPC_CONGRESS_SHORT_NAME = re.compile(r"中共(?:第)?(?P<ordinal>[一二三四五六七八九十]{1,3})大")
CPC_CONGRESS_RANGE = re.compile(
    r"中共(?:第)?(?P<start>[一二三四五六七八九十]{1,3})大\s*"
    r"(?:至|到|—|–|-|~|～)\s*"
    r"中共(?:第)?(?P<end>[一二三四五六七八九十]{1,3})大"
)
logger = logging.getLogger(__name__)


def cpc_congress_ordinals(query: str) -> list[str]:
    ordinals: list[str] = []
    for match in CPC_CONGRESS_RANGE.finditer(query):
        start = CPC_CONGRESS_NUMBER.get(match.group("start"))
        end = CPC_CONGRESS_NUMBER.get(match.group("end"))
        if start is None or end is None:
            continue
        step = 1 if start <= end else -1
        ordinals.extend(
            CPC_CONGRESS_ORDINALS[number - 1] for number in range(start, end + step, step)
        )
    ordinals.extend(match.group("ordinal") for match in CPC_CONGRESS_SHORT_NAME.finditer(query))
    return list(dict.fromkeys(ordinals))


def expand_query(query: str) -> str:
    """Add restrained search hints for source observations and named congresses."""

    expansions: list[str] = []
    if any(marker in query for marker in OBSERVATION_QUERY_MARKERS):
        expansions.append(OBSERVATION_EXPANSION)
    for ordinal in cpc_congress_ordinals(query):
        expansions.append(f"中国共产党第{ordinal}次全国代表大会")
    return f"{query} {' '.join(expansions)}" if expansions else query


def _source_bonus(intent: str, source_type: str, title: str, query_people: list[str]) -> float:
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
    section_years = [
        int(match.group(1))
        for section in hit.section_path
        for match in YEAR.finditer(section)
        if start_year <= int(match.group(1)) <= end_year
    ]
    if section_years:
        return section_years[0]
    positions = [
        (hit.text.find(f"{year}年"), year) for year in years if hit.text.find(f"{year}年") >= 0
    ]
    return min(positions)[1] if positions else years[0]


def _select_with_temporal_coverage(
    ranked: list[SearchHit],
    *,
    query: str,
    query_year_range: list[int],
    top_k: int,
    coverage: str = "auto",
) -> list[SearchHit]:
    if coverage in {"relevance", "per_item"}:
        return ranked[:top_k]
    forced = coverage in {"per_year", "balanced_period"}
    if (
        len(query_year_range) != 2
        or (
            not forced
            and query_year_range[1] - query_year_range[0] + 1 < MIN_TEMPORAL_COVERAGE_YEARS
        )
        or top_k < TEMPORAL_BUCKET_COUNT
        or (not forced and any(marker in query for marker in TEMPORAL_FOCUS_MARKERS))
    ):
        return ranked[:top_k]

    start_year, end_year = query_year_range
    span = end_year - start_year + 1
    if coverage == "per_year" or span <= top_k:
        covered_ids: set[str] = set()
        for target_year in range(start_year, end_year + 1):
            match = next(
                (hit for hit in ranked if _primary_year(hit, start_year, end_year) == target_year),
                None,
            )
            if match is not None:
                covered_ids.add(match.chunk_id)
        for hit in ranked:
            if len(covered_ids) >= top_k:
                break
            covered_ids.add(hit.chunk_id)
        return [hit for hit in ranked if hit.chunk_id in covered_ids][:top_k]

    buckets: list[list[SearchHit]] = [[] for _ in range(TEMPORAL_BUCKET_COUNT)]
    for hit in ranked:
        primary_year = _primary_year(hit, start_year, end_year)
        if primary_year is None:
            continue
        bucket_index = min(
            TEMPORAL_BUCKET_COUNT - 1,
            (primary_year - start_year) * TEMPORAL_BUCKET_COUNT // span,
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


def _best_congress_hits(ranked: list[SearchHit], query: str) -> list[SearchHit]:
    ordinals = cpc_congress_ordinals(query)
    selected: list[SearchHit] = []
    for ordinal in ordinals:
        markers = (
            f"党的第{ordinal}次全国代表大会",
            f"中国共产党第{ordinal}次全国代表大会",
            f"第{ordinal}次全国代表大会",
        )
        matches = [
            (index, hit)
            for index, hit in enumerate(ranked)
            if any(marker in " ".join(hit.section_path) for marker in markers)
            or any(marker in hit.text for marker in markers)
        ]
        if not matches:
            continue
        _, match = max(
            matches,
            key=lambda item: (
                any(marker in " ".join(item[1].section_path) for marker in markers),
                any(marker in item[1].text for marker in markers),
                "召开" in item[1].text or "举行" in item[1].text,
                item[1].source_type == "official_history",
                -item[0],
            ),
        )
        selected.append(match)
    return selected


def _select_with_congress_coverage(
    ranked: list[SearchHit], *, query: str, top_k: int
) -> list[SearchHit]:
    ordinals = cpc_congress_ordinals(query)
    if len(ordinals) < 2 or len(ordinals) > top_k:
        return ranked
    covered_ids = {hit.chunk_id for hit in _best_congress_hits(ranked, query)}
    for hit in ranked:
        if len(covered_ids) >= top_k:
            break
        covered_ids.add(hit.chunk_id)
    return [hit for hit in ranked if hit.chunk_id in covered_ids]


def _prepend_congress_keyword_hits(
    primary: SearchResponse, congress_hits: list[SearchHit]
) -> SearchResponse:
    combined: list[SearchHit] = []
    seen_ids: set[str] = set()
    for hit in [*congress_hits, *primary.hits]:
        if hit.chunk_id in seen_ids:
            continue
        seen_ids.add(hit.chunk_id)
        combined.append(hit.model_copy(update={"rank": len(combined) + 1}))
    return primary.model_copy(update={"hits": combined})


def fuse_search_responses(
    keyword: SearchResponse,
    vector: SearchResponse,
    *,
    top_k: int,
    coverage: str = "auto",
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
    preferred_ids = {hit.chunk_id for hit in _best_congress_hits(ranked, keyword.query)}
    unique_by_page: dict[tuple[str, int], SearchHit] = {}
    for hit in ranked:
        page_key = (hit.document_id, hit.pdf_page_start)
        existing = unique_by_page.get(page_key)
        if existing is None or (
            hit.chunk_id in preferred_ids and existing.chunk_id not in preferred_ids
        ):
            unique_by_page[page_key] = hit
    unique_pages = sorted(unique_by_page.values(), key=lambda hit: ranked.index(hit))

    congress_covered = _select_with_congress_coverage(
        unique_pages, query=keyword.query, top_k=top_k
    )
    selected = _select_with_temporal_coverage(
        congress_covered,
        query=keyword.query,
        query_year_range=keyword.query_year_range,
        top_k=top_k,
        coverage=coverage,
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


def _single_branch_response(
    response: SearchResponse,
    *,
    branch: str,
    failed: str,
    top_k: int,
    coverage: str,
) -> SearchResponse:
    hits: list[SearchHit] = []
    for hit in response.hits:
        updates: dict[str, object] = {
            "keyword_rank": hit.rank if branch == "keyword" else None,
            "vector_rank": hit.rank if branch == "vector" else None,
        }
        hits.append(hit.model_copy(update=updates))
    selected = _select_with_temporal_coverage(
        hits,
        query=response.query,
        query_year_range=response.query_year_range,
        top_k=top_k,
        coverage=coverage,
    )
    for rank, hit in enumerate(selected, start=1):
        hit.rank = rank
    return response.model_copy(
        update={
            "hits": selected,
            "retrieval_mode": f"{branch}_only",
            "degraded_components": [failed],
        }
    )


def _search_one(
    *,
    keyword_index_path: Path,
    vector_index_path: Path,
    model_cache_dir: Path,
    aliases_path: Path,
    query: str,
    top_k: int,
    document_ids: list[str] | None,
    include_out_of_scope: bool,
    plan: RetrievalPlan | None,
) -> SearchResponse:
    congress_count = len(cpc_congress_ordinals(query))
    candidate_k = min(100, max(30, top_k * 4, congress_count * 12))
    search_query = expand_query(query)
    keyword: SearchResponse | None = None
    vector: SearchResponse | None = None
    keyword_error: Exception | None = None
    vector_error: Exception | None = None
    try:
        keyword = search_keyword_index(
            index_path=keyword_index_path,
            query=search_query,
            aliases_path=aliases_path,
            top_k=candidate_k,
            document_ids=document_ids,
            include_out_of_scope=include_out_of_scope,
            plan=plan,
        )
        congress_hits: list[SearchHit] = []
        if congress_count >= 2:
            for ordinal in cpc_congress_ordinals(query):
                formal_name = f"中国共产党第{ordinal}次全国代表大会"
                targeted = search_keyword_index(
                    index_path=keyword_index_path,
                    query=formal_name,
                    aliases_path=aliases_path,
                    top_k=24,
                    document_ids=document_ids,
                    include_out_of_scope=include_out_of_scope,
                    section_path_contains=f"第{ordinal}次全国代表大会",
                    plan=plan,
                )
                best_hits = _best_congress_hits(targeted.hits, query)
                if best_hits:
                    congress_hits.append(best_hits[0])
            keyword = _prepend_congress_keyword_hits(keyword, congress_hits)
    except Exception as exc:
        keyword_error = exc
        logger.warning(
            "keyword retrieval degraded", extra={"context": {"error": type(exc).__name__}}
        )
    try:
        vector = search_vector_index(
            index_path=vector_index_path,
            model_cache_dir=model_cache_dir,
            aliases_path=aliases_path,
            query=search_query,
            top_k=candidate_k,
            document_ids=document_ids,
            include_out_of_scope=include_out_of_scope,
            plan=plan,
        )
    except Exception as exc:
        vector_error = exc
        logger.warning(
            "vector retrieval degraded", extra={"context": {"error": type(exc).__name__}}
        )
    coverage = plan.coverage if plan is not None else "auto"
    if keyword is not None and vector is not None:
        return fuse_search_responses(keyword, vector, top_k=top_k, coverage=coverage)
    if keyword is not None:
        return _single_branch_response(
            keyword, branch="keyword", failed="vector", top_k=top_k, coverage=coverage
        )
    if vector is not None:
        return _single_branch_response(
            vector, branch="vector", failed="keyword", top_k=top_k, coverage=coverage
        )
    cause = vector_error or keyword_error
    raise RetrievalError("Keyword and vector retrieval are both unavailable.") from cause


def _fuse_query_variants(
    responses: list[SearchResponse],
    *,
    query: str,
    top_k: int,
    plan: RetrievalPlan,
) -> SearchResponse:
    by_chunk: dict[str, SearchHit] = {}
    scores: dict[str, float] = {}
    for response in responses:
        for hit in response.hits:
            current = by_chunk.get(hit.chunk_id)
            if current is None:
                current = hit.model_copy(deep=True)
                by_chunk[hit.chunk_id] = current
                scores[hit.chunk_id] = 0.0
            else:
                ranks = [rank for rank in (current.keyword_rank, hit.keyword_rank) if rank]
                current.keyword_rank = min(ranks) if ranks else None
                ranks = [rank for rank in (current.vector_rank, hit.vector_rank) if rank]
                current.vector_rank = min(ranks) if ranks else None
            scores[hit.chunk_id] += 1.0 / (RRF_K + hit.rank)
    ranked = sorted(
        by_chunk.values(),
        key=lambda hit: (-scores[hit.chunk_id], hit.pdf_page_start, hit.chunk_id),
    )
    unique_pages: list[SearchHit] = []
    pages: set[tuple[str, int]] = set()
    for hit in ranked:
        page = (hit.document_id, hit.pdf_page_start)
        if page not in pages:
            pages.add(page)
            unique_pages.append(hit)
    if plan.coverage == "per_item":
        reserved: list[SearchHit] = []
        reserved_ids: set[str] = set()
        for response in responses[1:]:
            match = next((hit for hit in response.hits if hit.chunk_id not in reserved_ids), None)
            if match is not None:
                reserved.append(by_chunk[match.chunk_id])
                reserved_ids.add(match.chunk_id)
        selected = reserved[:top_k]
        selected.extend(hit for hit in unique_pages if hit.chunk_id not in reserved_ids)
        selected = selected[:top_k]
    else:
        selected = _select_with_temporal_coverage(
            unique_pages,
            query=query,
            query_year_range=plan.query_year_range,
            top_k=top_k,
            coverage=plan.coverage,
        )
    for rank, hit in enumerate(selected, start=1):
        hit.rank = rank
        hit.rrf_score = round(scores[hit.chunk_id], 8)
        hit.score = hit.rrf_score
    degraded = sorted(
        {component for response in responses for component in response.degraded_components}
    )
    return SearchResponse(
        query=query,
        query_intent=plan.query_intent,
        query_terms=list(dict.fromkeys(term for item in responses for term in item.query_terms)),
        query_years=plan.query_years,
        query_year_range=plan.query_year_range,
        query_people=plan.query_people,
        document_filters=responses[0].document_filters,
        include_out_of_scope=responses[0].include_out_of_scope,
        hits=selected,
        retrieval_mode="planned_hybrid_rrf",
        degraded_components=degraded,
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
    plan: RetrievalPlan | None = None,
    additional_queries: list[str] | None = None,
) -> SearchResponse:
    queries = list(dict.fromkeys([query, *(additional_queries or [])]))
    responses = [
        _search_one(
            keyword_index_path=keyword_index_path,
            vector_index_path=vector_index_path,
            model_cache_dir=model_cache_dir,
            aliases_path=aliases_path,
            query=item,
            top_k=top_k,
            document_ids=document_ids,
            include_out_of_scope=include_out_of_scope,
            plan=plan,
        )
        for item in queries
    ]
    if plan is not None and len(responses) > 1:
        return _fuse_query_variants(responses, query=query, top_k=top_k, plan=plan)
    return responses[0].model_copy(update={"query": query})
