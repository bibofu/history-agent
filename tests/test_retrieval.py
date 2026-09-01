from history_agent.retrieval.hybrid import expand_query, fuse_search_responses
from history_agent.retrieval.keyword import (
    hard_filter_people,
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
        year_mentions=[1956],
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
    assert infer_query_intent("毛泽东关于调查研究的观点") == "viewpoint"
    assert infer_query_intent("毛泽东在矛盾论中怎样分析主要矛盾") == "viewpoint"
    assert infer_query_intent("斯诺在西行漫记中怎样记述毛泽东") == "observation"
    assert infer_year_range("长征期间", []) == [1934, 1936]
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
