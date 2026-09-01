from __future__ import annotations

from pydantic import BaseModel, Field


class KeywordIndexSummary(BaseModel):
    run_id: str
    started_at: str
    finished_at: str
    index_version: str
    documents: int = Field(ge=0)
    chunks: int = Field(ge=0)
    index_path: str
    size_bytes: int = Field(ge=0)


class VectorIndexSummary(BaseModel):
    run_id: str
    started_at: str
    finished_at: str
    index_version: str
    model_name: str
    vector_size: int = Field(ge=1)
    documents: int = Field(ge=0)
    chunks: int = Field(ge=0)
    collection_name: str
    index_path: str
    size_bytes: int = Field(ge=0)


class SearchHit(BaseModel):
    rank: int = Field(ge=1)
    chunk_id: str
    document_id: str
    title: str
    filename: str
    volume: str | None = None
    source_type: str
    verification_status: str
    pdf_page_start: int
    pdf_page_end: int
    section_path: list[str]
    text: str
    year_mentions: list[int]
    people: list[str]
    extraction_methods: list[str]
    score: float
    matched_terms: list[str]
    keyword_rank: int | None = Field(default=None, ge=1)
    vector_rank: int | None = Field(default=None, ge=1)
    keyword_score: float | None = None
    vector_score: float | None = None
    rrf_score: float | None = None


class SearchResponse(BaseModel):
    query: str
    query_intent: str
    query_terms: list[str]
    query_years: list[int]
    query_year_range: list[int]
    query_people: list[str]
    document_filters: list[str]
    include_out_of_scope: bool
    hits: list[SearchHit]
    retrieval_mode: str = "keyword"
