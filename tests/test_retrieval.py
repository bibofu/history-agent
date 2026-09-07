from history_agent.retrieval.hybrid import _primary_year, expand_query, fuse_search_responses
from history_agent.retrieval.keyword import (
    hard_filter_people,
    has_explicit_year_range,
    infer_query_intent,
    infer_year_range,
    tokenize_query,
)
from history_agent.retrieval.models import SearchHit, SearchResponse


def _hit(
    chunk_id: str,
    rank: int,
    *,
    page: int,
    source_type: str = "history",
    years: list[int] | None = None,
) -> SearchHit:
    return SearchHit(
        rank=rank,
        chunk_id=chunk_id,
        document_id="doc",
        title="测试文献",
        filename="test.pdf",
        source_type=source_type,
        verification_status="verified",
        pdf_page_start=page,
        pdf_page_end=page,
        section_path=[],
        text=f"evidence {chunk_id}",
        year_mentions=years or [1956],
        people=["周恩来"],
        extraction_methods=["text_layer"],
        score=float(10 - rank),
        matched_terms=[],
    )


def _response(hits: list[SearchHit], *, intent: str = "general") -> SearchResponse:
    return SearchResponse(
        query="测试问题",
        query_intent=intent,
        query_terms=["测试"],
        query_years=[],
        query_year_range=[],
        query_people=[],
        document_filters=[],
        include_out_of_scope=False,
        hits=hits,
    )


def test_query_routing_extracts_intent_and_period() -> None:
    assert infer_query_intent("毛泽东和周恩来在长征期间的交集") == "intersection"
    assert infer_query_intent("周恩来在1956年主要有哪些经历") == "timeline"
    assert infer_query_intent("周恩来在长征期间有哪些经历") == "timeline"
    assert infer_query_intent("毛泽东关于调查研究的观点") == "viewpoint"
    assert infer_query_intent("毛泽东对抗日战争的谋划") == "viewpoint"
    assert infer_query_intent("毛泽东在矛盾论中怎样分析主要矛盾") == "viewpoint"
    assert infer_query_intent("斯诺在西行漫记中怎样记述毛泽东") == "observation"
    assert infer_year_range("长征期间", []) == [1934, 1936]
    assert has_explicit_year_range("梳理毛泽东在1921-1926年期间的活动")
    assert has_explicit_year_range("1921年至1926年")
    assert has_explicit_year_range("1921～1926")
    assert not has_explicit_year_range("比较毛泽东在1921年和1926年的活动")
    assert "毛泽" in tokenize_query("毛泽东关于调查研究的观点")
    assert hard_filter_people(["毛泽东"], "viewpoint") == []
    assert hard_filter_people(["埃德加·斯诺", "毛泽东"], "observation") == []
    assert hard_filter_people(["毛泽东", "周恩来"], "intersection") == [
        "毛泽东",
        "周恩来",
    ]


def test_observation_query_expansion_is_restrained() -> None:
    expanded = expand_query("斯诺在西行漫记中怎样记述毛泽东？")

    assert expanded.endswith("外貌 性格 生活 印象")
    assert expand_query("周恩来在1956年有哪些经历？") == "周恩来在1956年有哪些经历？"


def test_cpc_congress_short_name_expands_to_formal_name() -> None:
    assert expand_query("中共一大的情况") == (
        "中共一大的情况 中国共产党第一次全国代表大会"
    )
    assert expand_query("比较中共一大和中共二大") == (
        "比较中共一大和中共二大 "
        "中国共产党第一次全国代表大会 中国共产党第二次全国代表大会"
    )


def test_generic_question_words_do_not_pollute_keyword_query() -> None:
    assert tokenize_query("中共一大的情况") == ["中共", "共一", "一大"]


def test_rrf_rewards_results_found_by_both_retrievers() -> None:
    keyword = _response([_hit("a", 1, page=1), _hit("shared", 2, page=2)])
    vector = _response([_hit("shared", 1, page=2), _hit("b", 2, page=3)])

    result = fuse_search_responses(keyword, vector, top_k=3)

    assert result.retrieval_mode == "hybrid_rrf"
    assert result.hits[0].chunk_id == "shared"
    assert result.hits[0].keyword_rank == 2
    assert result.hits[0].vector_rank == 1


def test_rrf_returns_only_one_chunk_per_pdf_page() -> None:
    keyword = _response([_hit("a", 1, page=1), _hit("b", 2, page=1)])
    vector = _response([_hit("c", 1, page=2)])

    result = fuse_search_responses(keyword, vector, top_k=3)

    assert [(hit.chunk_id, hit.pdf_page_start) for hit in result.hits] == [
        ("a", 1),
        ("c", 2),
    ]


def test_rrf_adds_intent_source_bonus() -> None:
    keyword = _response(
        [
            _hit("history", 1, page=1, source_type="history"),
            _hit("chronology", 2, page=2, source_type="chronology"),
        ],
        intent="timeline",
    )
    vector = _response([])

    result = fuse_search_responses(keyword, vector, top_k=2)

    assert result.hits[0].chunk_id == "chronology"


def test_wide_period_query_balances_early_middle_and_late_evidence() -> None:
    hits = [
        *[_hit(f"early-{index}", index, page=index, years=[1937]) for index in range(1, 7)],
        *[
            _hit(f"middle-{index}", index + 6, page=index + 6, years=[1941])
            for index in range(1, 3)
        ],
        *[
            _hit(f"late-{index}", index + 8, page=index + 8, years=[1945])
            for index in range(1, 3)
        ],
    ]
    keyword = _response(hits).model_copy(
        update={"query": "抗日战争的谋划", "query_year_range": [1937, 1945]}
    )

    result = fuse_search_responses(keyword, _response([]), top_k=6)

    assert [hit.chunk_id for hit in result.hits] == [
        "early-1",
        "early-2",
        "middle-1",
        "middle-2",
        "late-1",
        "late-2",
    ]


def test_short_year_range_keeps_evidence_from_each_year() -> None:
    hits = [
        *[_hit(f"edge-{index}", index, page=index, years=[1921]) for index in range(1, 7)],
        _hit("1922", 7, page=7, years=[1922]),
        _hit("1923", 8, page=8, years=[1923]),
        _hit("1924", 9, page=9, years=[1924]),
        _hit("1925", 10, page=10, years=[1925]),
        _hit("1926", 11, page=11, years=[1926]),
    ]
    keyword = _response(hits).model_copy(
        update={"query": "1921-1926年期间的活动", "query_year_range": [1921, 1926]}
    )

    result = fuse_search_responses(keyword, _response([]), top_k=6)

    assert {hit.year_mentions[0] for hit in result.hits} == set(range(1921, 1927))


def test_primary_year_prefers_chronology_heading_over_incidental_body_year() -> None:
    hit = _hit("chronology", 1, page=1, years=[1925, 1926])
    hit.section_path = ["1925年"]
    hit.text = "年底讨论了1926年的工作计划。"

    assert _primary_year(hit, 1921, 1926) == 1925


def test_focused_subperiod_query_keeps_relevance_order() -> None:
    hits = [
        *[_hit(f"early-{index}", index, page=index, years=[1937]) for index in range(1, 4)],
        _hit("late", 4, page=4, years=[1945]),
    ]
    keyword = _response(hits).model_copy(
        update={"query": "抗日战争初期的谋划", "query_year_range": [1937, 1945]}
    )

    result = fuse_search_responses(keyword, _response([]), top_k=3)

    assert [hit.chunk_id for hit in result.hits] == ["early-1", "early-2", "early-3"]
