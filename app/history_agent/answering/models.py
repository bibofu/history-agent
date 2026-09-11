from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=10_000)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    top_k: int = Field(default=12, ge=1, le=12)
    history: list[ConversationMessage] = Field(default_factory=list, max_length=12)
    session_id: str | None = Field(
        default=None,
        min_length=16,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
    )


class QueryEntity(BaseModel):
    type: Literal["person", "organization", "event", "place", "document", "other"]
    text: str = Field(min_length=1, max_length=80)
    canonical: str = Field(min_length=1, max_length=120)


ShortQueryText = Annotated[str, Field(min_length=1, max_length=180)]
ConstraintText = Annotated[str, Field(min_length=1, max_length=120)]


class QueryPlan(BaseModel):
    """Validated semantic interpretation used to build a retrieval query."""

    intent: Literal[
        "general",
        "event_overview",
        "timeline",
        "intersection",
        "viewpoint",
        "observation",
        "comparison",
        "causal_analysis",
    ] = "general"
    normalized_question: str = Field(min_length=2, max_length=500)
    search_queries: list[ShortQueryText] = Field(default_factory=list, max_length=8)
    entities: list[QueryEntity] = Field(default_factory=list, max_length=12)
    start_year: int | None = Field(default=None, ge=1800, le=2100)
    end_year: int | None = Field(default=None, ge=1800, le=2100)
    coverage: Literal["relevance", "per_year", "per_item", "balanced_period"] = "relevance"
    constraints: list[ConstraintText] = Field(default_factory=list, max_length=12)
    needs_clarification: bool = False
    clarification_question: str | None = Field(default=None, max_length=300)


class Citation(BaseModel):
    evidence_id: str = Field(pattern=r"^E[1-9]\d*$")
    document_id: str
    document: str
    volume: str | None = None
    pdf_page: int = Field(ge=1)
    pdf_page_end: int | None = Field(default=None, ge=1)
    section: list[str]
    quote: str
    source_type: str
    verification_status: str
    extraction_methods: list[str]


class AnswerResponse(BaseModel):
    question: str
    answer: str
    evidence_status: Literal["supported", "partial", "no_evidence"]
    generator_mode: Literal["extractive", "llm"]
    llm_status: Literal["used", "disabled", "fallback", "not_applicable"]
    model_name: str | None = None
    llm_usage: dict[str, int] | None = None
    llm_error_code: str | None = None
    uncited_claims: list[str] = Field(default_factory=list)
    query_plan: QueryPlan | None = None
    query_planner_status: Literal["used", "disabled", "fallback", "not_applicable"] = (
        "not_applicable"
    )
    query_planner_model: str | None = None
    query_planner_usage: dict[str, int] | None = None
    query_planner_error_code: str | None = None
    retrieval_mode: str
    query_intent: str
    retrieval_reflection_status: Literal[
        "not_applicable", "disabled", "skipped", "sufficient", "retried", "fallback"
    ] = "not_applicable"
    retrieval_rounds: int = Field(default=1, ge=1, le=3)
    retrieval_missing_aspects: list[str] = Field(default_factory=list, max_length=4)
    retrieval_reflection_usage: dict[str, int] | None = None
    retrieval_reflection_error_code: str | None = None
    citations: list[Citation]
    retrieved_evidence_count: int = Field(default=0, ge=0)
    limitations: list[str] = Field(default_factory=list)
    rag_framework: Literal["llamaindex"] = "llamaindex"
