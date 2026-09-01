from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=10_000)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    top_k: int = Field(default=8, ge=1, le=12)
    history: list[ConversationMessage] = Field(default_factory=list, max_length=12)


class Citation(BaseModel):
    evidence_id: str
    document_id: str
    document: str
    volume: str | None = None
    pdf_page: int
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
    retrieval_mode: str
    query_intent: str
    citations: list[Citation]
    limitations: list[str] = Field(default_factory=list)
