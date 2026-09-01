from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Literal, cast

from pydantic import BaseModel, Field

from history_agent.db import Database
from history_agent.errors import ResearchDataError
from history_agent.research.models import ReviewStatus, TemporalPoint

TimelineRecordKind = Literal["canonical", "source"]
TimelineVerificationLevel = Literal["confirmed", "automatic", "pending_review"]
TimelineReviewStatus = Literal["unreviewed", "needs_review", "confirmed"]
DEFAULT_REVIEW_STATUSES: tuple[TimelineReviewStatus, ...] = (
    "confirmed",
    "unreviewed",
    "needs_review",
)


class TimelineParticipant(BaseModel):
    person_id: str
    canonical_name: str
    role: str | None = None
    mention_text: str
    mention_source: Literal["explicit", "chronology_subject", "inferred"]


class TimelineEvidence(BaseModel):
    evidence_id: str
    source_event_id: str
    document_id: str
    document_title: str
    volume: str | None = None
    pdf_page_start: int = Field(ge=1)
    pdf_page_end: int = Field(ge=1)
    quote: str
    source_type: str
    verification_status: str
    extraction_methods: list[str]


class TimelineEvent(BaseModel):
    event_id: str
    record_kind: TimelineRecordKind
    source_event_ids: list[str] = Field(min_length=1)
    name: str
    event_type: str
    start: TemporalPoint
    end: TemporalPoint | None = None
    location_text: str | None = None
    organization_names: list[str] = Field(default_factory=list)
    description: str
    participants: list[TimelineParticipant] = Field(min_length=1)
    evidence: list[TimelineEvidence] = Field(min_length=1)
    extraction_method: str
    confidence: float = Field(ge=0, le=1)
    review_status: ReviewStatus
    verification_level: TimelineVerificationLevel
    candidate_kind: Literal["high_confidence", "uncertain"] | None = None
    field_variants: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)


class PersonTimelineResponse(BaseModel):
    person_id: str
    canonical_name: str
    start_year: int | None = None
    end_year: int | None = None
    event_types: list[str] = Field(default_factory=list)
    review_statuses: list[TimelineReviewStatus]
    total: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1)
    has_more: bool
    events: list[TimelineEvent]


def _placeholders(values: Sequence[str]) -> str:
    return ", ".join("?" for _ in values)


def _temporal_payload(row: sqlite3.Row, prefix: str) -> dict[str, Any] | None:
    precision = row[f"{prefix}_precision"]
    if precision is None:
        return None
    return {
        "value": row[f"{prefix}_value"],
        "precision": precision,
        "certainty": row[f"{prefix}_certainty"],
        "original_text": row[f"{prefix}_original_text"],
    }


def _verification_level(review_status: str) -> TimelineVerificationLevel:
    if review_status == "confirmed":
        return "confirmed"
    if review_status == "needs_review":
        return "pending_review"
    return "automatic"


def _validate_filters(
    *,
    start_year: int | None,
    end_year: int | None,
    event_types: Sequence[str],
    review_statuses: Sequence[str],
    limit: int,
    offset: int,
) -> None:
    if start_year is not None and not 1 <= start_year <= 9999:
        raise ResearchDataError("start_year must be between 1 and 9999")
    if end_year is not None and not 1 <= end_year <= 9999:
        raise ResearchDataError("end_year must be between 1 and 9999")
    if start_year is not None and end_year is not None and start_year > end_year:
        raise ResearchDataError("start_year must be on or before end_year")
    if not 1 <= limit <= 200:
        raise ResearchDataError("timeline limit must be between 1 and 200")
    if offset < 0:
        raise ResearchDataError("timeline offset cannot be negative")
    if any(not item.strip() for item in event_types):
        raise ResearchDataError("event_type cannot be empty")
    allowed_statuses = set(DEFAULT_REVIEW_STATUSES)
    unknown_statuses = sorted(set(review_statuses) - allowed_statuses)
    if unknown_statuses:
        raise ResearchDataError(
            "unsupported timeline review status: " + ", ".join(unknown_statuses)
        )


def _event_filters(
    alias: str,
    *,
    start_year: int | None,
    end_year: int | None,
    event_types: Sequence[str],
    review_statuses: Sequence[str],
) -> tuple[str, list[object]]:
    conditions = [
        (
            f"{alias}.is_active = 1"
            if alias in {"c", "active_c"}
            else f"{alias}.review_status != 'rejected'"
        ),
        f"{alias}.review_status IN ({_placeholders(review_statuses)})",
    ]
    parameters: list[object] = list(review_statuses)
    if start_year is not None:
        conditions.append(
            f"CAST(substr(COALESCE({alias}.end_value, {alias}.start_value), 1, 4) "
            "AS INTEGER) >= ?"
        )
        parameters.append(start_year)
    if end_year is not None:
        conditions.append(
            f"CAST(substr({alias}.start_value, 1, 4) AS INTEGER) <= ?"
        )
        parameters.append(end_year)
    if event_types:
        conditions.append(f"{alias}.event_type IN ({_placeholders(event_types)})")
        parameters.extend(event_types)
    return " AND ".join(conditions), parameters


def _load_event_rows(
    connection: sqlite3.Connection,
    *,
    person_id: str,
    start_year: int | None,
    end_year: int | None,
    event_types: Sequence[str],
    review_statuses: Sequence[str],
) -> list[sqlite3.Row]:
    canonical_filters, canonical_parameters = _event_filters(
        "c",
        start_year=start_year,
        end_year=end_year,
        event_types=event_types,
        review_statuses=review_statuses,
    )
    source_filters, source_parameters = _event_filters(
        "e",
        start_year=start_year,
        end_year=end_year,
        event_types=event_types,
        review_statuses=review_statuses,
    )
    active_canonical_filters, active_canonical_parameters = _event_filters(
        "active_c",
        start_year=start_year,
        end_year=end_year,
        event_types=event_types,
        review_statuses=review_statuses,
    )
    rows = connection.execute(
        f"""
        SELECT
            c.canonical_event_id AS record_id, 'canonical' AS record_kind,
            c.name, c.event_type, c.start_value, c.start_precision,
            c.start_certainty, c.start_original_text, c.end_value,
            c.end_precision, c.end_certainty, c.end_original_text,
            c.location_text, c.organization_names_json, c.description,
            'merge' AS extraction_method, c.merge_confidence AS confidence,
            c.review_status, c.candidate_kind, c.field_variants_json
        FROM canonical_events c
        WHERE {canonical_filters}
          AND EXISTS (
              SELECT 1 FROM canonical_event_participants cp
              WHERE cp.canonical_event_id = c.canonical_event_id
                AND cp.person_id = ?
          )
        UNION ALL
        SELECT
            e.event_id AS record_id, 'source' AS record_kind,
            e.name, e.event_type, e.start_value, e.start_precision,
            e.start_certainty, e.start_original_text, e.end_value,
            e.end_precision, e.end_certainty, e.end_original_text,
            e.location_text, e.organization_names_json, e.description,
            e.extraction_method, e.extraction_confidence AS confidence,
            e.review_status, NULL AS candidate_kind, '{{}}' AS field_variants_json
        FROM historical_events e
        WHERE {source_filters}
          AND EXISTS (
              SELECT 1 FROM event_participants ep
              WHERE ep.event_id = e.event_id AND ep.person_id = ?
          )
          AND NOT EXISTS (
              SELECT 1
              FROM canonical_event_members cm
              JOIN canonical_events active_c
                ON active_c.canonical_event_id = cm.canonical_event_id
              WHERE cm.source_event_id = e.event_id
                AND {active_canonical_filters}
          )
        """,
        (
            *canonical_parameters,
            person_id,
            *source_parameters,
            person_id,
            *active_canonical_parameters,
        ),
    ).fetchall()
    return sorted(
        rows,
        key=lambda row: (
            row["start_value"] is None,
            str(row["start_value"] or ""),
            str(row["end_value"] or ""),
            str(row["record_id"]),
        ),
    )


def _selected_ids(rows: list[sqlite3.Row], kind: TimelineRecordKind) -> list[str]:
    return [str(row["record_id"]) for row in rows if row["record_kind"] == kind]


def _load_source_event_ids(
    connection: sqlite3.Connection,
    canonical_ids: list[str],
    source_ids: list[str],
) -> dict[str, list[str]]:
    result = {event_id: [event_id] for event_id in source_ids}
    if canonical_ids:
        rows = connection.execute(
            "SELECT canonical_event_id, source_event_id FROM canonical_event_members "
            f"WHERE canonical_event_id IN ({_placeholders(canonical_ids)}) "
            "ORDER BY canonical_event_id, source_event_id",
            canonical_ids,
        ).fetchall()
        for row in rows:
            result.setdefault(str(row["canonical_event_id"]), []).append(
                str(row["source_event_id"])
            )
    return result


def _load_participants(
    connection: sqlite3.Connection,
    canonical_ids: list[str],
    source_ids: list[str],
) -> dict[str, list[TimelineParticipant]]:
    result: dict[str, list[TimelineParticipant]] = defaultdict(list)
    queries = (
        (
            "canonical",
            canonical_ids,
            "canonical_event_participants",
            "canonical_event_id",
        ),
        ("source", source_ids, "event_participants", "event_id"),
    )
    for _, record_ids, table, id_column in queries:
        if not record_ids:
            continue
        rows = connection.execute(
            f"""
            SELECT p.{id_column} AS record_id, p.person_id, people.canonical_name,
                   p.role, p.mention_text, p.mention_source
            FROM {table} p
            JOIN persons people ON people.person_id = p.person_id
            WHERE p.{id_column} IN ({_placeholders(record_ids)})
            ORDER BY p.{id_column}, p.person_id, p.mention_text
            """,
            record_ids,
        ).fetchall()
        for row in rows:
            result[str(row["record_id"])].append(
                TimelineParticipant.model_validate(dict(row))
            )
    return result


def _load_evidence(
    connection: sqlite3.Connection,
    canonical_ids: list[str],
    source_ids: list[str],
) -> dict[str, list[TimelineEvidence]]:
    result: dict[str, list[TimelineEvidence]] = defaultdict(list)
    if canonical_ids:
        rows = connection.execute(
            f"""
            SELECT ce.canonical_event_id AS record_id, ce.source_event_id,
                   er.*, d.title AS document_title, d.volume, d.source_type,
                   d.verification_status
            FROM canonical_event_evidence ce
            JOIN evidence_records er ON er.evidence_id = ce.evidence_id
            JOIN documents d ON d.document_id = er.document_id
            WHERE ce.canonical_event_id IN ({_placeholders(canonical_ids)})
            ORDER BY ce.canonical_event_id, er.document_id, er.pdf_page_start,
                     er.evidence_id
            """,
            canonical_ids,
        ).fetchall()
        _append_evidence(result, rows)
    if source_ids:
        rows = connection.execute(
            f"""
            SELECT ee.event_id AS record_id, ee.event_id AS source_event_id,
                   er.*, d.title AS document_title, d.volume, d.source_type,
                   d.verification_status
            FROM event_evidence ee
            JOIN evidence_records er ON er.evidence_id = ee.evidence_id
            JOIN documents d ON d.document_id = er.document_id
            WHERE ee.event_id IN ({_placeholders(source_ids)})
            ORDER BY ee.event_id, er.document_id, er.pdf_page_start, er.evidence_id
            """,
            source_ids,
        ).fetchall()
        _append_evidence(result, rows)
    return result


def _append_evidence(
    result: dict[str, list[TimelineEvidence]], rows: list[sqlite3.Row]
) -> None:
    for row in rows:
        payload = dict(row)
        payload["extraction_methods"] = json.loads(
            str(payload.pop("extraction_methods_json"))
        )
        payload.pop("created_at", None)
        payload.pop("chunk_id", None)
        result[str(row["record_id"])].append(TimelineEvidence.model_validate(payload))


def get_person_timeline(
    database: Database,
    *,
    person_id: str,
    start_year: int | None = None,
    end_year: int | None = None,
    event_types: Sequence[str] | None = None,
    review_statuses: Sequence[str] | None = None,
    limit: int = 50,
    offset: int = 0,
) -> PersonTimelineResponse:
    event_type_filter = sorted(set(event_types or []))
    status_filter = cast(
        list[TimelineReviewStatus],
        list(dict.fromkeys(review_statuses or DEFAULT_REVIEW_STATUSES)),
    )
    _validate_filters(
        start_year=start_year,
        end_year=end_year,
        event_types=event_type_filter,
        review_statuses=status_filter,
        limit=limit,
        offset=offset,
    )
    database.initialize()
    with database.connect() as connection:
        person = connection.execute(
            "SELECT person_id, canonical_name, status, merged_into_person_id "
            "FROM persons WHERE person_id = ?",
            (person_id,),
        ).fetchone()
        if person is None:
            raise ResearchDataError(f"unknown person_id: {person_id}")
        if person["status"] != "active":
            target = person["merged_into_person_id"] or "unknown"
            raise ResearchDataError(f"person is merged; use person_id: {target}")
        all_rows = _load_event_rows(
            connection,
            person_id=person_id,
            start_year=start_year,
            end_year=end_year,
            event_types=event_type_filter,
            review_statuses=status_filter,
        )
        selected = all_rows[offset : offset + limit]
        canonical_ids = _selected_ids(selected, "canonical")
        source_ids = _selected_ids(selected, "source")
        source_event_ids = _load_source_event_ids(
            connection, canonical_ids, source_ids
        )
        participants = _load_participants(connection, canonical_ids, source_ids)
        evidence = _load_evidence(connection, canonical_ids, source_ids)
    events: list[TimelineEvent] = []
    for row in selected:
        record_id = str(row["record_id"])
        events.append(
            TimelineEvent.model_validate(
                {
                    "event_id": record_id,
                    "record_kind": row["record_kind"],
                    "source_event_ids": source_event_ids[record_id],
                    "name": row["name"],
                    "event_type": row["event_type"],
                    "start": _temporal_payload(row, "start"),
                    "end": _temporal_payload(row, "end"),
                    "location_text": row["location_text"],
                    "organization_names": json.loads(
                        str(row["organization_names_json"])
                    ),
                    "description": row["description"],
                    "participants": participants[record_id],
                    "evidence": evidence[record_id],
                    "extraction_method": row["extraction_method"],
                    "confidence": row["confidence"],
                    "review_status": row["review_status"],
                    "verification_level": _verification_level(
                        str(row["review_status"])
                    ),
                    "candidate_kind": row["candidate_kind"],
                    "field_variants": json.loads(str(row["field_variants_json"])),
                }
            )
        )
    return PersonTimelineResponse(
        person_id=str(person["person_id"]),
        canonical_name=str(person["canonical_name"]),
        start_year=start_year,
        end_year=end_year,
        event_types=event_type_filter,
        review_statuses=status_filter,
        total=len(all_rows),
        offset=offset,
        limit=limit,
        has_more=offset + len(events) < len(all_rows),
        events=events,
    )
