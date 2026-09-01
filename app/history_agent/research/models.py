from __future__ import annotations

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator

StableId = str
ReviewStatus = Literal["unreviewed", "needs_review", "confirmed", "rejected"]
ExtractionMethod = Literal["rule", "llm", "rule_llm", "manual", "import"]
DatePrecision = Literal["day", "month", "year", "unknown"]
DateCertainty = Literal["exact", "approximate", "inferred", "unknown"]

STABLE_ID = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
YEAR_VALUE = re.compile(r"^(?:18|19|20)\d{2}$")
MONTH_VALUE = re.compile(r"^(?:18|19|20)\d{2}-(?:0[1-9]|1[0-2])$")
HIGH_RISK_RELATION_TYPES = {"direct_subordinate"}
EVENT_REQUIRED_RELATION_TYPES = {"co_attended"}


class PersonAlias(BaseModel):
    name: str = Field(min_length=1)
    alias_type: Literal[
        "name", "courtesy_name", "title", "pen_name", "transliteration", "other"
    ] = "name"
    notes: str | None = None


class PersonRecord(BaseModel):
    person_id: str = Field(pattern=STABLE_ID.pattern)
    canonical_name: str = Field(min_length=1)
    aliases: list[PersonAlias] = Field(default_factory=list)
    description: str | None = None

    @model_validator(mode="after")
    def aliases_must_be_unique(self) -> PersonRecord:
        names = [alias.name.casefold() for alias in self.aliases]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate aliases for {self.person_id}")
        if self.canonical_name.casefold() in names:
            raise ValueError("canonical_name must not be repeated as an alias")
        return self


class PersonAmbiguity(BaseModel):
    ambiguity_id: str = Field(pattern=STABLE_ID.pattern)
    mention: str = Field(min_length=1)
    candidate_person_ids: list[str] = Field(min_length=2)
    status: Literal["unresolved", "resolved"] = "unresolved"
    resolved_person_id: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def resolution_must_be_a_candidate(self) -> PersonAmbiguity:
        if len(self.candidate_person_ids) != len(set(self.candidate_person_ids)):
            raise ValueError("ambiguity candidates must be unique")
        if self.status == "resolved" and self.resolved_person_id not in self.candidate_person_ids:
            raise ValueError("resolved_person_id must be one of the ambiguity candidates")
        if self.status == "unresolved" and self.resolved_person_id is not None:
            raise ValueError("unresolved ambiguity cannot have resolved_person_id")
        return self


class PersonCatalog(BaseModel):
    schema_version: int = 2
    people: list[PersonRecord]
    ambiguities: list[PersonAmbiguity] = Field(default_factory=list)

    @model_validator(mode="after")
    def catalog_references_must_be_valid(self) -> PersonCatalog:
        person_ids = [person.person_id for person in self.people]
        canonical_names = [person.canonical_name.casefold() for person in self.people]
        if len(person_ids) != len(set(person_ids)):
            raise ValueError("duplicate person_id")
        if len(canonical_names) != len(set(canonical_names)):
            raise ValueError("duplicate canonical_name")
        known = set(person_ids)
        for ambiguity in self.ambiguities:
            if not set(ambiguity.candidate_person_ids).issubset(known):
                raise ValueError(f"unknown ambiguity candidate in {ambiguity.ambiguity_id}")
        return self

    def alias_map(self) -> dict[str, list[str]]:
        return {
            person.canonical_name: [alias.name for alias in person.aliases]
            for person in self.people
        }


class RelationTypeDefinition(BaseModel):
    relation_type: str = Field(pattern=STABLE_ID.pattern)
    label: str = Field(min_length=1)
    object_kind: Literal["person", "organization", "person_or_organization"]
    extraction_policy: Literal["allowed", "candidate_only", "manual_only"]
    requires_human_review: bool = False
    requires_event: bool = False
    description: str = Field(min_length=1)


class RelationTypeCatalog(BaseModel):
    schema_version: int = 1
    relation_types: list[RelationTypeDefinition]

    @model_validator(mode="after")
    def relation_types_must_be_unique(self) -> RelationTypeCatalog:
        values = [item.relation_type for item in self.relation_types]
        if len(values) != len(set(values)):
            raise ValueError("duplicate relation_type")
        return self


class TemporalPoint(BaseModel):
    value: str | None = None
    precision: DatePrecision = "unknown"
    certainty: DateCertainty = "unknown"
    original_text: str | None = None

    @model_validator(mode="after")
    def value_matches_precision(self) -> TemporalPoint:
        if self.precision == "unknown":
            if self.value is not None:
                raise ValueError("unknown precision requires a null value")
            return self
        if self.value is None:
            raise ValueError("known precision requires a value")
        if self.precision == "year" and not YEAR_VALUE.fullmatch(self.value):
            raise ValueError("year precision requires YYYY")
        if self.precision == "month" and not MONTH_VALUE.fullmatch(self.value):
            raise ValueError("month precision requires YYYY-MM")
        if self.precision == "day":
            try:
                date.fromisoformat(self.value)
            except ValueError as exc:
                raise ValueError("day precision requires a valid YYYY-MM-DD") from exc
        return self


class EvidenceReference(BaseModel):
    evidence_id: str = Field(pattern=STABLE_ID.pattern)
    document_id: str = Field(min_length=1)
    chunk_id: str | None = None
    pdf_page_start: int = Field(ge=1)
    pdf_page_end: int = Field(ge=1)
    quote: str = Field(min_length=12, max_length=1200)
    extraction_methods: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def page_range_must_be_ordered(self) -> EvidenceReference:
        if self.pdf_page_end < self.pdf_page_start:
            raise ValueError("pdf_page_end must be on or after pdf_page_start")
        return self


class EventParticipant(BaseModel):
    person_id: str = Field(pattern=STABLE_ID.pattern)
    role: str | None = None
    mention_text: str = Field(min_length=1)
    mention_source: Literal["explicit", "chronology_subject", "inferred"] = "explicit"


class HistoricalEvent(BaseModel):
    event_id: str = Field(pattern=STABLE_ID.pattern)
    name: str = Field(min_length=1)
    event_type: str = Field(pattern=STABLE_ID.pattern)
    start: TemporalPoint = Field(default_factory=TemporalPoint)
    end: TemporalPoint | None = None
    location_text: str | None = None
    organization_names: list[str] = Field(default_factory=list)
    description: str = Field(min_length=1)
    participants: list[EventParticipant] = Field(min_length=1)
    evidence: list[EvidenceReference] = Field(min_length=1)
    extraction_method: ExtractionMethod
    extraction_confidence: float = Field(ge=0, le=1)
    review_status: ReviewStatus = "unreviewed"
    extractor_version: str | None = None


class HistoricalRelationship(BaseModel):
    relationship_id: str = Field(pattern=STABLE_ID.pattern)
    relation_type: str = Field(pattern=STABLE_ID.pattern)
    subject_person_id: str = Field(pattern=STABLE_ID.pattern)
    subject_mention_text: str = Field(min_length=1)
    object_person_id: str | None = Field(default=None, pattern=STABLE_ID.pattern)
    object_mention_text: str | None = None
    organization_name: str | None = None
    role_title: str | None = None
    start: TemporalPoint = Field(default_factory=TemporalPoint)
    end: TemporalPoint | None = None
    event_id: str | None = Field(default=None, pattern=STABLE_ID.pattern)
    description: str | None = None
    evidence: list[EvidenceReference] = Field(min_length=1)
    extraction_method: ExtractionMethod
    extraction_confidence: float = Field(ge=0, le=1)
    review_status: ReviewStatus = "unreviewed"
    reviewed_by: str | None = None
    extractor_version: str | None = None

    @model_validator(mode="after")
    def relationship_context_is_auditable(self) -> HistoricalRelationship:
        if self.object_person_id is None and not self.organization_name:
            raise ValueError("relationship requires an object person or organization")
        if self.object_person_id is not None and not self.object_mention_text:
            raise ValueError("object person relationship requires its original mention text")
        if self.relation_type == "held_position" and not self.role_title:
            raise ValueError("held_position requires a role_title")
        if self.relation_type in EVENT_REQUIRED_RELATION_TYPES and self.event_id is None:
            raise ValueError(f"{self.relation_type} requires a shared event_id")
        if (
            self.relation_type in HIGH_RISK_RELATION_TYPES
            and self.review_status == "confirmed"
            and not self.reviewed_by
        ):
            raise ValueError("confirmed high-risk relationship requires a human reviewer")
        if (
            self.relation_type in HIGH_RISK_RELATION_TYPES
            and self.review_status == "confirmed"
            and (not self.organization_name or self.start.precision == "unknown")
        ):
            raise ValueError(
                "confirmed high-risk relationship requires organization and time context"
            )
        return self


class PersonCandidate(BaseModel):
    person_id: str
    canonical_name: str
    matched_form: str
    status: Literal["active", "merged"]
    merged_into_person_id: str | None = None


class PersonResolution(BaseModel):
    query: str
    normalized_query: str
    status: Literal["resolved", "ambiguous", "not_found"]
    candidates: list[PersonCandidate] = Field(default_factory=list)
