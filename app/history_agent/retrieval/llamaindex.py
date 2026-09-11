"""LlamaIndex adapters for the project's domain-specific hybrid retriever.

LlamaIndex owns the public retrieval contract and node lifecycle.  The existing
keyword/vector implementation remains a deliberately custom backend because it
contains corpus-specific ranking, temporal coverage, and safety rules.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from llama_index.core import QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode

from history_agent.retrieval.models import SearchHit, SearchResponse

SearchBackend = Callable[..., SearchResponse]


def _node_score(hit: SearchHit) -> float:
    return hit.rrf_score if hit.rrf_score is not None else hit.score


def search_hit_to_node(hit: SearchHit) -> NodeWithScore:
    """Convert a domain result into LlamaIndex's canonical evidence node."""

    payload = hit.model_dump(mode="json")
    node = TextNode(
        id_=hit.chunk_id,
        text=hit.text,
        metadata={
            "document_id": hit.document_id,
            "title": hit.title,
            "source_type": hit.source_type,
            "pdf_page_start": hit.pdf_page_start,
            "pdf_page_end": hit.pdf_page_end,
            "section_path": hit.section_path,
            "year_mentions": hit.year_mentions,
            "people": hit.people,
            "history_agent_hit": payload,
        },
        excluded_embed_metadata_keys=["history_agent_hit"],
        excluded_llm_metadata_keys=["history_agent_hit"],
    )
    return NodeWithScore(node=node, score=_node_score(hit))


def node_to_search_hit(node: NodeWithScore, rank: int) -> SearchHit:
    """Restore a validated domain result after LlamaIndex post-processing."""

    raw = node.node.metadata.get("history_agent_hit")
    if not isinstance(raw, dict):
        raise ValueError("LlamaIndex node is missing history_agent_hit metadata")
    return SearchHit.model_validate(raw).model_copy(
        update={"rank": rank, "score": node.score if node.score is not None else raw["score"]}
    )


class HistoryHybridRetriever(BaseRetriever):
    """LlamaIndex retriever backed by the project's keyword/vector fusion."""

    def __init__(
        self,
        *,
        search_backend: SearchBackend,
        search_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self._search_backend = search_backend
        self._search_kwargs = search_kwargs
        self.last_response: SearchResponse | None = None

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        kwargs = {
            **self._search_kwargs,
            "query": query_bundle.query_str,
            "additional_queries": list(query_bundle.custom_embedding_strs or []),
        }
        self.last_response = self._search_backend(**kwargs)
        return [search_hit_to_node(hit) for hit in self.last_response.hits]


class HistoryEvidencePostprocessor(BaseNodePostprocessor):
    """Deduplicate and cap evidence inside the LlamaIndex node pipeline."""

    max_nodes: int

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        del query_bundle
        selected: list[NodeWithScore] = []
        seen: set[str] = set()
        for node in nodes:
            if node.node.node_id in seen:
                continue
            # Reject malformed adapter output before it can enter synthesis.
            node_to_search_hit(node, len(selected) + 1)
            seen.add(node.node.node_id)
            selected.append(node)
            if len(selected) >= self.max_nodes:
                break
        return selected


def retrieve_with_llamaindex(
    *,
    search_backend: SearchBackend,
    query: str,
    additional_queries: list[str],
    top_k: int,
    search_kwargs: dict[str, Any],
) -> SearchResponse:
    """Execute one retrieval round through LlamaIndex abstractions."""

    retriever = HistoryHybridRetriever(
        search_backend=search_backend,
        search_kwargs={**search_kwargs, "top_k": top_k},
    )
    bundle = QueryBundle(query_str=query, custom_embedding_strs=additional_queries)
    nodes = retriever.retrieve(bundle)
    nodes = HistoryEvidencePostprocessor(max_nodes=top_k).postprocess_nodes(
        nodes, query_bundle=bundle
    )
    if retriever.last_response is None:
        raise RuntimeError("LlamaIndex retriever completed without a search response")
    hits = [node_to_search_hit(node, rank) for rank, node in enumerate(nodes, start=1)]
    return retriever.last_response.model_copy(
        update={"hits": hits, "rag_framework": "llamaindex"}
    )
