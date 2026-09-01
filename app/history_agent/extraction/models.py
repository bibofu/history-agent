from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

PageStatus = Literal["extracted", "ocr_required", "empty", "failed"]
ExtractionMethod = Literal[
    "pypdf_text_layer",
    "pymupdf_text_layer",
    "pdfplumber_text_layer",
    "ocr",
    "none",
]


class PageRecord(BaseModel):
    document_id: str
    file_sha256: str
    pdf_page: int = Field(ge=1)
    printed_page: str | None = None
    status: PageStatus
    extraction_method: ExtractionMethod
    extractor_version: str
    raw_text: str
    normalized_text: str
    text_sha256: str
    character_count: int = Field(ge=0)
    cjk_character_count: int = Field(ge=0)
    replacement_character_count: int = Field(ge=0)
    image_object_count: int = Field(default=0, ge=0)
    quality_flags: list[str] = Field(default_factory=list)
    ocr_confidence: float | None = None
    error: str | None = None
    run_id: str
    processed_at: str


class SampleDocumentResult(BaseModel):
    document_id: str
    filename: str
    pages_requested: list[int]
    output_path: str
    status_counts: dict[str, int]


class SampleExtractionSummary(BaseModel):
    run_id: str
    started_at: str
    finished_at: str
    documents: list[SampleDocumentResult]


class DocumentExtractionResult(BaseModel):
    document_id: str
    filename: str
    total_pages: int
    processed_pages: int
    reused_pages: int
    status_counts: dict[str, int]
    total_characters: int
    output_path: str
    elapsed_seconds: float = Field(ge=0)


class FullExtractionSummary(BaseModel):
    run_id: str
    parser: str
    started_at: str
    finished_at: str
    documents: list[DocumentExtractionResult]

    @property
    def totals(self) -> dict[str, int]:
        totals: dict[str, int] = {"documents": len(self.documents), "pages": 0, "characters": 0}
        for document in self.documents:
            totals["pages"] += document.total_pages
            totals["characters"] += document.total_characters
            for status, count in document.status_counts.items():
                key = f"status_{status}"
                totals[key] = totals.get(key, 0) + count
        return totals


class OcrDocumentResult(BaseModel):
    document_id: str
    filename: str
    candidate_pages: int
    processed_pages: int
    reused_pages: int
    remaining_pages: int
    status_counts: dict[str, int]
    total_characters: int
    mean_confidence: float | None = None
    output_path: str
    elapsed_seconds: float = Field(ge=0)


class OcrExtractionSummary(BaseModel):
    run_id: str
    engine_version: str
    started_at: str
    finished_at: str
    documents: list[OcrDocumentResult]

    @property
    def totals(self) -> dict[str, int]:
        return {
            "documents": len(self.documents),
            "candidates": sum(document.candidate_pages for document in self.documents),
            "processed": sum(document.processed_pages for document in self.documents),
            "reused": sum(document.reused_pages for document in self.documents),
            "remaining": sum(document.remaining_pages for document in self.documents),
            "characters": sum(document.total_characters for document in self.documents),
            "failed": sum(
                document.status_counts.get("failed", 0) for document in self.documents
            ),
        }
