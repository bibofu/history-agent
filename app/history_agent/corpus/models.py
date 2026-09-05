from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

FileStatus = Literal["new", "changed", "unchanged", "renamed", "missing", "unregistered"]


class CatalogDocument(BaseModel):
    document_id: str = Field(pattern=r"^[a-z][a-z0-9_]+$")
    filename: str
    title: str
    creators: list[str] = Field(default_factory=list)
    source_type: str
    edition: str | None = None
    volume: str | None = None
    source_series: list[str] = Field(default_factory=list)
    verification_status: str
    enabled: bool = True
    ocr_strategy: Literal[
        "none",
        "partial_if_needed",
        "partial_required",
        "full_required",
    ] = "partial_if_needed"
    expected_page_count: int | None = Field(default=None, ge=1)
    ocr_pages: list[Annotated[int, Field(ge=1)]] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def validate_ocr_pages(self) -> CatalogDocument:
        if self.expected_page_count is not None and any(
            page > self.expected_page_count for page in self.ocr_pages
        ):
            raise ValueError("ocr_pages must be within expected_page_count")
        self.ocr_pages = sorted(set(self.ocr_pages))
        return self

    @field_validator("filename")
    @classmethod
    def filename_must_be_a_basename(cls, value: str) -> str:
        if "/" in value or "\\" in value or value in {".", ".."}:
            raise ValueError("filename must not contain a directory")
        if not value.lower().endswith(".pdf"):
            raise ValueError("filename must end with .pdf")
        return value


class CorpusCatalog(BaseModel):
    schema_version: int = 1
    documents: list[CatalogDocument]

    @model_validator(mode="after")
    def identifiers_and_filenames_must_be_unique(self) -> CorpusCatalog:
        ids = [document.document_id for document in self.documents]
        filenames = [document.filename.casefold() for document in self.documents]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate document_id in catalog")
        if len(filenames) != len(set(filenames)):
            raise ValueError("duplicate filename in catalog")
        return self


class FileObservation(BaseModel):
    document_id: str | None = None
    filename: str
    status: FileStatus
    sha256: str | None = None
    size_bytes: int | None = None
    page_count: int | None = None
    expected_page_count: int | None = None
    page_count_matches: bool | None = None
    relative_path: str | None = None
    message: str | None = None


class CorpusScanSummary(BaseModel):
    run_id: str
    started_at: str
    finished_at: str
    status: Literal["succeeded", "failed"]
    observations: list[FileObservation]

    @property
    def counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for observation in self.observations:
            result[observation.status] = result.get(observation.status, 0) + 1
        return result
