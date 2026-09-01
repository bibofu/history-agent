from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class StructureEntry(BaseModel):
    entry_id: str
    document_id: str
    level: int = Field(ge=1)
    title: str
    pdf_page_start: int = Field(ge=1)
    pdf_page_end: int = Field(ge=1)
    source: Literal["pdf_outline", "year_heading", "document_root"]


class ChunkRecord(BaseModel):
    chunk_id: str
    document_id: str
    file_sha256: str
    title: str
    filename: str
    creators: list[str]
    source_type: str
    edition: str | None = None
    volume: str | None = None
    verification_status: str
    chunk_index: int = Field(ge=0)
    page_chunk_index: int = Field(ge=0)
    pdf_page_start: int = Field(ge=1)
    pdf_page_end: int = Field(ge=1)
    printed_page_start: str | None = None
    printed_page_end: str | None = None
    section_path: list[str] = Field(default_factory=list)
    text: str
    search_text: str
    character_count: int = Field(ge=1)
    year_mentions: list[int] = Field(default_factory=list)
    date_mentions: list[str] = Field(default_factory=list)
    scope_status: Literal["in_scope", "out_of_scope", "mixed", "unknown"]
    people: list[str] = Field(default_factory=list)
    extraction_methods: list[str] = Field(default_factory=list)
    content_hash: str
    cleaner_version: str
    chunker_version: str
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None


class DocumentChunkResult(BaseModel):
    document_id: str
    filename: str
    effective_pages: int
    skipped_pages: int
    structure_entries: int
    chunks: int
    characters: int
    scope_counts: dict[str, int]
    output_path: str


class ChunkBuildSummary(BaseModel):
    run_id: str
    started_at: str
    finished_at: str
    documents: list[DocumentChunkResult]

    @property
    def totals(self) -> dict[str, int]:
        return {
            "documents": len(self.documents),
            "effective_pages": sum(document.effective_pages for document in self.documents),
            "skipped_pages": sum(document.skipped_pages for document in self.documents),
            "structure_entries": sum(
                document.structure_entries for document in self.documents
            ),
            "chunks": sum(document.chunks for document in self.documents),
            "characters": sum(document.characters for document in self.documents),
        }
