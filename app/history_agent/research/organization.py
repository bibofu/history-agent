from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Sequence
from datetime import date
from typing import Literal, cast

from pydantic import BaseModel, Field

from history_agent.db import Database
from history_agent.errors import ResearchDataError
from history_agent.research.models import (
    DateCertainty,
    DatePrecision,
    EvidenceReference,
    HistoricalRelationship,
    TemporalPoint,
)
from history_agent.research.store import ResearchStore

ORGANIZATION_RELATION_EXTRACTOR_VERSION = "organization-rules-v1"
QUERYABLE_RELATION_TYPES = ("held_position", "member_of", "led", "direct_subordinate")
QUERYABLE_REVIEW_STATUSES = ("confirmed", "unreviewed", "needs_review")

SUBJECT_APPOINTMENT_VERB = re.compile(r"(?:被任命为|被委任为|被选为|当选为|担任|兼任|任(?!命))")
CLAUSE_END = re.compile(r"[，,。；;]|并(?:出席|参加|主持|会见|访问|前往|赴)")
ROLE_SUFFIXES = (
    "第一书记",
    "代理书记",
    "副秘书长",
    "秘书长",
    "副总司令",
    "总司令",
    "副司令员",
    "司令员",
    "副政治委员",
    "政治委员",
    "副委员长",
    "委员长",
    "第一副主席",
    "第二副主席",
    "第三副主席",
    "副主席",
    "主席",
    "第一副总理",
    "第二副总理",
    "副总理",
    "总理",
    "副部长",
    "部长",
    "副主任",
    "主任",
    "副书记",
    "书记",
    "总参谋长",
    "副参谋长",
    "参谋长",
    "委员",
    "政委",
)
ORGANIZATION_MARKERS = (
    "中共中央",
    "中央",
    "国务院",
    "政务院",
    "委员会",
    "军委",
    "军区",
    "野战军",
    "兵团",
    "政府",
    "书记处",
    "政治局",
    "外交部",
    "国防部",
    "组织部",
)


class OrganizationRelationEvidence(BaseModel):
    evidence_id: str
    document_id: str
    document_title: str
    volume: str | None = None
    pdf_page_start: int = Field(ge=1)
    pdf_page_end: int = Field(ge=1)
    quote: str
    extraction_methods: list[str] = Field(default_factory=list)


class OrganizationRelationItem(BaseModel):
    relationship_id: str
    relation_type: str
    relation_label: str
    subject_person_id: str
    canonical_name: str
    organization_name: str
    role_title: str | None = None
    start: TemporalPoint
    end: TemporalPoint | None = None
    source_event_id: str | None = None
    description: str | None = None
    extraction_method: str
    confidence: float = Field(ge=0, le=1)
    review_status: str
    verification_level: Literal["confirmed", "automatic_candidate", "pending_review"]
    evidence: list[OrganizationRelationEvidence] = Field(min_length=1)


class OrganizationRelationResponse(BaseModel):
    person_id: str
    canonical_name: str
    at: str | None = None
    relation_types: list[str]
    review_statuses: list[str]
    total: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1)
    has_more: bool
    relationships: list[OrganizationRelationItem]
    limitation: str


class OrganizationExtractionSummary(BaseModel):
    extractor_version: str = ORGANIZATION_RELATION_EXTRACTOR_VERSION
    events_scanned: int
    candidates: int
    created: int
    updated: int
    skipped: int
    rejected_without_organization: int
    samples: list[dict[str, object]] = Field(default_factory=list)
    dry_run: bool = False


def _stable_id(*values: str) -> str:
    digest = hashlib.sha256("\x1f".join(values).encode("utf-8")).hexdigest()[:20]
    return f"relationship_{digest}"


def _position_parts(value: str) -> tuple[str, str] | None:
    cleaned = re.sub(r"\s+", "", value).strip(" ：:、，和及的")
    if "和" in cleaned:
        return None
    for role in ROLE_SUFFIXES:
        if not cleaned.endswith(role):
            continue
        organization = cleaned[: -len(role)].strip(" 的")
        if len(organization) < 2:
            return None
        if role == "部长" and not organization.endswith("部"):
            organization += "部"
        if organization == "中央":
            return None
        if not any(marker in organization for marker in ORGANIZATION_MARKERS):
            return None
        return organization, role
    return None


def _appointment_phrases(text: str, mention: str) -> list[str]:
    normalized = re.sub(r"\s+", "", text)
    phrases: list[str] = []
    for found in re.finditer(re.escape(mention), normalized):
        context = normalized[max(0, found.start() - 16) : found.start()]
        if re.search(r"(?:要求|建议|提议|主张)(?:由)?$", context):
            continue
        tail = normalized[found.end() : found.end() + 100]
        verb = SUBJECT_APPOINTMENT_VERB.match(tail)
        if verb is None:
            continue
        value = tail[verb.end() :]
        end = CLAUSE_END.search(value)
        phrase = value[: end.start()] if end else value
        phrases.extend(re.split(r"(?:并)?兼(?:任)?|、", phrase))
    if phrases:
        return phrases
    prefix_pattern = re.compile(
        rf"(?:任命|委任|选举|决定由){re.escape(mention)}(?:为|担任|兼任)"
    )
    prefix = prefix_pattern.search(normalized)
    if prefix is not None:
        value = normalized[prefix.end() : prefix.end() + 100]
        end = CLAUSE_END.search(value)
        phrase = value[: end.start()] if end else value
        return re.split(r"(?:并)?兼(?:任)?|、", phrase)
    verb = SUBJECT_APPOINTMENT_VERB.match(normalized)
    if verb is None:
        return []
    value = normalized[verb.end() :]
    end = CLAUSE_END.search(value)
    return re.split(
        r"(?:并)?兼(?:任)?|、", value[: end.start()] if end else value
    )


def _load_event_rows(database: Database) -> list[sqlite3.Row]:
    with database.connect() as connection:
        return connection.execute(
            """
            SELECT e.*, ep.person_id, ep.mention_text, ep.mention_source,
                   er.evidence_id, er.document_id, er.chunk_id,
                   er.pdf_page_start, er.pdf_page_end, er.quote,
                   er.extraction_methods_json
            FROM historical_events e
            JOIN event_participants ep ON ep.event_id = e.event_id
            JOIN event_evidence ee ON ee.event_id = e.event_id
            JOIN evidence_records er ON er.evidence_id = ee.evidence_id
            WHERE e.review_status != 'rejected'
              AND (e.event_type = 'appointment' OR e.description GLOB '*任*')
            ORDER BY e.event_id, ep.person_id, er.evidence_id
            """
        ).fetchall()


def extract_organization_relationships(
    database: Database, *, dry_run: bool = False
) -> OrganizationExtractionSummary:
    """Extract conservative held-position candidates from dated source events."""

    database.initialize()
    rows = _load_event_rows(database)
    events_scanned = len({str(row["event_id"]) for row in rows})
    candidates: dict[str, HistoricalRelationship] = {}
    rejected = 0
    for row in rows:
        mention = str(row["mention_text"])
        if row["mention_source"] == "inferred":
            continue
        for phrase in _appointment_phrases(str(row["description"]), mention):
            parts = _position_parts(phrase)
            if parts is None:
                rejected += 1
                continue
            organization, role = parts
            relationship_id = _stable_id(
                str(row["event_id"]), str(row["person_id"]), organization, role
            )
            evidence = EvidenceReference(
                evidence_id=str(row["evidence_id"]),
                document_id=str(row["document_id"]),
                chunk_id=row["chunk_id"],
                pdf_page_start=int(row["pdf_page_start"]),
                pdf_page_end=int(row["pdf_page_end"]),
                quote=str(row["quote"]),
                extraction_methods=json.loads(str(row["extraction_methods_json"])),
            )
            existing = candidates.get(relationship_id)
            if existing is not None:
                if evidence.evidence_id not in {item.evidence_id for item in existing.evidence}:
                    existing.evidence.append(evidence)
                continue
            candidates[relationship_id] = HistoricalRelationship(
                relationship_id=relationship_id,
                relation_type="held_position",
                subject_person_id=str(row["person_id"]),
                subject_mention_text=mention,
                organization_name=organization,
                role_title=role,
                start=TemporalPoint(
                    value=row["start_value"],
                    precision=cast(DatePrecision, str(row["start_precision"])),
                    certainty=cast(DateCertainty, str(row["start_certainty"])),
                    original_text=row["start_original_text"],
                ),
                event_id=str(row["event_id"]),
                description=str(row["description"]),
                evidence=[evidence],
                extraction_method="rule",
                extraction_confidence=0.9,
                review_status="unreviewed",
                extractor_version=ORGANIZATION_RELATION_EXTRACTOR_VERSION,
            )
    created = updated = skipped = 0
    if not dry_run:
        created, updated, skipped = ResearchStore(database).sync_generated_relationships(
            list(candidates.values())
        )
    samples: list[dict[str, object]] = [
        {
            "relationship_id": item.relationship_id,
            "person_id": item.subject_person_id,
            "date": item.start.value,
            "organization": item.organization_name,
            "role": item.role_title,
            "description": item.description,
            "document_id": item.evidence[0].document_id,
            "pdf_page": item.evidence[0].pdf_page_start,
        }
        for item in list(candidates.values())[:20]
    ]
    return OrganizationExtractionSummary(
        events_scanned=events_scanned,
        candidates=len(candidates),
        created=created,
        updated=updated,
        skipped=skipped,
        rejected_without_organization=rejected,
        samples=samples,
        dry_run=dry_run,
    )


def _parse_query_period(value: str) -> tuple[date, date]:
    try:
        if re.fullmatch(r"\d{4}", value):
            year = int(value)
            return date(year, 1, 1), date(year, 12, 31)
        if re.fullmatch(r"\d{4}-\d{2}", value):
            year, month = (int(part) for part in value.split("-"))
            start = date(year, month, 1)
            next_month = date(year + (month == 12), month % 12 + 1, 1)
            return start, date.fromordinal(next_month.toordinal() - 1)
        point = date.fromisoformat(value)
        return point, point
    except ValueError as exc:
        raise ResearchDataError("at must be YYYY, YYYY-MM, or YYYY-MM-DD") from exc


def _temporal_bounds(value: str, precision: str, *, end: bool) -> date:
    if precision == "day":
        return date.fromisoformat(value)
    if precision == "month":
        lower, upper = _parse_query_period(value)
        return upper if end else lower
    lower, upper = _parse_query_period(value[:4])
    return upper if end else lower


def _overlaps_at(row: sqlite3.Row, at_start: date, at_end: date) -> bool:
    if row["start_value"] is None or row["start_precision"] == "unknown":
        return False
    relation_start = _temporal_bounds(
        str(row["start_value"]), str(row["start_precision"]), end=False
    )
    relation_end = _temporal_bounds(
        str(row["start_value"]), str(row["start_precision"]), end=True
    )
    if row["end_value"] is not None and row["end_precision"] != "unknown":
        relation_end = _temporal_bounds(
            str(row["end_value"]), str(row["end_precision"]), end=True
        )
    return relation_start <= at_end and relation_end >= at_start


def _placeholders(values: Sequence[str]) -> str:
    return ", ".join("?" for _ in values)


def get_organization_relationships(
    database: Database,
    *,
    person_id: str,
    at: str | None = None,
    relation_types: Sequence[str] | None = None,
    review_statuses: Sequence[str] | None = None,
    limit: int = 50,
    offset: int = 0,
) -> OrganizationRelationResponse:
    database.initialize()
    selected_types = tuple(relation_types or QUERYABLE_RELATION_TYPES)
    selected_statuses = tuple(review_statuses or QUERYABLE_REVIEW_STATUSES)
    if not selected_types or not selected_statuses:
        raise ResearchDataError("relationship filters cannot be empty")
    unknown_types = sorted(set(selected_types) - set(QUERYABLE_RELATION_TYPES))
    unknown_statuses = sorted(set(selected_statuses) - set(QUERYABLE_REVIEW_STATUSES))
    if unknown_types:
        raise ResearchDataError(
            "unsupported organization relation type: " + ", ".join(unknown_types)
        )
    if unknown_statuses:
        raise ResearchDataError(
            "unsupported relationship review status: " + ", ".join(unknown_statuses)
        )
    if not 1 <= limit <= 200 or offset < 0:
        raise ResearchDataError("relationship pagination is invalid")
    at_period = _parse_query_period(at) if at is not None else None
    with database.connect() as connection:
        person = connection.execute(
            "SELECT canonical_name FROM persons WHERE person_id = ? AND status = 'active'",
            (person_id,),
        ).fetchone()
        if person is None:
            raise ResearchDataError(f"unknown person_id: {person_id}")
        rows = connection.execute(
            f"""
            SELECT r.*, rt.label AS relation_label, p.canonical_name,
                   er.evidence_id, er.document_id, d.title AS document_title,
                   d.volume, er.pdf_page_start, er.pdf_page_end, er.quote,
                   er.extraction_methods_json
            FROM person_relationships r
            JOIN relation_types rt ON rt.relation_type = r.relation_type
            JOIN persons p ON p.person_id = r.subject_person_id
            JOIN relationship_evidence re ON re.relationship_id = r.relationship_id
            JOIN evidence_records er ON er.evidence_id = re.evidence_id
            JOIN documents d ON d.document_id = er.document_id
            WHERE r.subject_person_id = ?
              AND r.relation_type IN ({_placeholders(selected_types)})
              AND r.review_status IN ({_placeholders(selected_statuses)})
              AND r.organization_name IS NOT NULL
              AND (
                  r.review_status != 'confirmed'
                  OR (r.start_value IS NOT NULL AND r.start_precision != 'unknown')
              )
            ORDER BY r.start_value, r.relationship_id, er.evidence_id
            """,
            (person_id, *selected_types, *selected_statuses),
        ).fetchall()
    grouped: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        if at_period is not None and not _overlaps_at(row, *at_period):
            continue
        grouped.setdefault(str(row["relationship_id"]), []).append(row)
    items: list[OrganizationRelationItem] = []
    for relation_rows in grouped.values():
        row = relation_rows[0]
        status = str(row["review_status"])
        verification: Literal["confirmed", "automatic_candidate", "pending_review"] = (
            "confirmed" if status == "confirmed" else
            "pending_review" if status == "needs_review" else "automatic_candidate"
        )
        end = None
        if row["end_precision"] is not None:
            end = TemporalPoint(
                value=row["end_value"], precision=row["end_precision"],
                certainty=row["end_certainty"], original_text=row["end_original_text"],
            )
        items.append(OrganizationRelationItem(
            relationship_id=str(row["relationship_id"]),
            relation_type=str(row["relation_type"]),
            relation_label=str(row["relation_label"]),
            subject_person_id=str(row["subject_person_id"]),
            canonical_name=str(row["canonical_name"]),
            organization_name=str(row["organization_name"]),
            role_title=row["role_title"],
            start=TemporalPoint(
                value=row["start_value"], precision=row["start_precision"],
                certainty=row["start_certainty"], original_text=row["start_original_text"],
            ),
            end=end,
            source_event_id=row["event_id"],
            description=row["description"],
            extraction_method=str(row["extraction_method"]),
            confidence=float(row["extraction_confidence"]),
            review_status=status,
            verification_level=verification,
            evidence=[OrganizationRelationEvidence(
                evidence_id=str(evidence["evidence_id"]),
                document_id=str(evidence["document_id"]),
                document_title=str(evidence["document_title"]),
                volume=evidence["volume"],
                pdf_page_start=int(evidence["pdf_page_start"]),
                pdf_page_end=int(evidence["pdf_page_end"]),
                quote=str(evidence["quote"]),
                extraction_methods=json.loads(str(evidence["extraction_methods_json"])),
            ) for evidence in relation_rows],
        ))
    total = len(items)
    page = items[offset : offset + limit]
    return OrganizationRelationResponse(
        person_id=person_id,
        canonical_name=str(person["canonical_name"]),
        at=at,
        relation_types=list(selected_types),
        review_statuses=list(selected_statuses),
        total=total,
        offset=offset,
        limit=limit,
        has_more=offset + len(page) < total,
        relationships=page,
        limitation=(
            "未人工确认的记录仅为有日期、机构和页码证据的自动候选；结束时间为空时，"
            "时间点过滤只匹配史料明确记载任职的日期范围，不推断其后持续在任。"
            "直接下属关系只允许人工录入和确认。"
        ),
    )
