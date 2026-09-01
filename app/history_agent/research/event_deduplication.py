from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from history_agent.db import Database
from history_agent.errors import ResearchDataError
from history_agent.research.models import (
    EventParticipant,
    EvidenceReference,
    HistoricalEvent,
)
from history_agent.research.people import utc_now

MERGE_VERSION = "event-dedup-rules-v1"
MERGE_METHOD = "cross-source-date-text-people"
DOCUMENT_SUBJECTS = {
    "zhou_enlai_chronology_1949_1976": "zhou_enlai",
    "lin_biao_chronology": "lin_biao",
}
GENERIC_EVENT_TYPE = "activity"
NON_CONTENT = re.compile(r"[^0-9A-Za-z\u3400-\u4dbf\u4e00-\u9fff]+")

DateRelation = Literal["exact_day", "same_value", "same_month"]
CandidateKind = Literal["high_confidence", "uncertain"]


class EventMergePair(BaseModel):
    left_event_id: str
    right_event_id: str
    date_relation: DateRelation
    score: float = Field(ge=0, le=1)
    text_containment: float = Field(ge=0, le=1)
    text_jaccard: float = Field(ge=0, le=1)
    shared_person_ids: list[str] = Field(default_factory=list)
    mutual_chronology_subjects: bool = False
    organization_match: bool = False
    location_match: bool = False
    event_type_match: bool = False


class EventMergeSummary(BaseModel):
    run_id: str
    merge_version: str
    source_events: int
    compared_pairs: int
    candidate_pairs: int
    candidate_groups: int
    high_confidence_groups: int
    uncertain_groups: int
    created: int = 0
    updated: int = 0
    skipped: int = 0
    protected: int = 0
    deactivated: int = 0
    dry_run: bool = False
    samples: list[dict[str, Any]] = Field(default_factory=list)


@dataclass(frozen=True)
class SourceEvent:
    event: HistoricalEvent
    document_ids: tuple[str, ...]
    text_grams: frozenset[str]


@dataclass(frozen=True)
class CanonicalCandidate:
    event: HistoricalEvent
    members: tuple[SourceEvent, ...]
    pair_features: tuple[EventMergePair, ...]
    candidate_kind: CandidateKind
    field_variants: dict[str, list[dict[str, Any]]]
    input_hash: str


class _DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:32]
    return f"{prefix}_{digest}"


def _normalized_text(value: str) -> str:
    return NON_CONTENT.sub("", value).casefold()


def _bigrams(value: str) -> frozenset[str]:
    normalized = _normalized_text(value)
    if len(normalized) < 2:
        return frozenset({normalized}) if normalized else frozenset()
    return frozenset(normalized[index : index + 2] for index in range(len(normalized) - 1))


def _text_similarity(
    left: frozenset[str], right: frozenset[str]
) -> tuple[float, float]:
    if not left or not right:
        return 0.0, 0.0
    intersection = len(left & right)
    containment = intersection / min(len(left), len(right))
    jaccard = intersection / len(left | right)
    return containment, jaccard


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


def _load_source_events(connection: sqlite3.Connection) -> list[SourceEvent]:
    event_rows = connection.execute(
        """
        SELECT * FROM historical_events
        WHERE review_status != 'rejected' AND start_value IS NOT NULL
        ORDER BY event_id
        """
    ).fetchall()
    participants: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in connection.execute(
        "SELECT * FROM event_participants ORDER BY event_id, person_id, mention_text"
    ).fetchall():
        participants[str(row["event_id"])].append(
            {
                "person_id": row["person_id"],
                "role": row["role"],
                "mention_text": row["mention_text"],
                "mention_source": row["mention_source"],
            }
        )
    evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    documents: dict[str, set[str]] = defaultdict(set)
    for row in connection.execute(
        """
        SELECT ee.event_id, er.*
        FROM event_evidence ee
        JOIN evidence_records er USING (evidence_id)
        ORDER BY ee.event_id, er.pdf_page_start, er.evidence_id
        """
    ).fetchall():
        event_id = str(row["event_id"])
        documents[event_id].add(str(row["document_id"]))
        evidence[event_id].append(
            {
                "evidence_id": row["evidence_id"],
                "document_id": row["document_id"],
                "chunk_id": row["chunk_id"],
                "pdf_page_start": row["pdf_page_start"],
                "pdf_page_end": row["pdf_page_end"],
                "quote": row["quote"],
                "extraction_methods": json.loads(str(row["extraction_methods_json"])),
            }
        )
    result: list[SourceEvent] = []
    for row in event_rows:
        event_id = str(row["event_id"])
        if not evidence[event_id] or not documents[event_id]:
            continue
        event = HistoricalEvent.model_validate(
            {
                "event_id": event_id,
                "name": row["name"],
                "event_type": row["event_type"],
                "start": _temporal_payload(row, "start"),
                "end": _temporal_payload(row, "end"),
                "location_text": row["location_text"],
                "organization_names": json.loads(str(row["organization_names_json"])),
                "description": row["description"],
                "participants": participants[event_id],
                "evidence": evidence[event_id],
                "extraction_method": row["extraction_method"],
                "extraction_confidence": row["extraction_confidence"],
                "review_status": row["review_status"],
                "extractor_version": row["extractor_version"],
            }
        )
        result.append(
            SourceEvent(
                event=event,
                document_ids=tuple(sorted(documents[event_id])),
                text_grams=_bigrams(event.description),
            )
        )
    return result


def _date_relation(left: HistoricalEvent, right: HistoricalEvent) -> DateRelation | None:
    if left.start.value is None or right.start.value is None:
        return None
    if (
        left.start.precision == "day"
        and right.start.precision == "day"
        and left.start.value == right.start.value
    ):
        return "exact_day"
    if left.start.value == right.start.value:
        return "same_value"
    if left.start.value[:7] == right.start.value[:7]:
        return "same_month"
    return None


def _pair_features(left: SourceEvent, right: SourceEvent) -> EventMergePair | None:
    if set(left.document_ids) & set(right.document_ids):
        return None
    relation = _date_relation(left.event, right.event)
    if relation is None:
        return None
    containment, jaccard = _text_similarity(left.text_grams, right.text_grams)
    left_people = {item.person_id for item in left.event.participants}
    right_people = {item.person_id for item in right.event.participants}
    shared_people = sorted(left_people & right_people)
    chronology_subjects = {
        DOCUMENT_SUBJECTS[document_id]
        for document_id in (*left.document_ids, *right.document_ids)
        if document_id in DOCUMENT_SUBJECTS
    }
    mutual_subjects = (
        len(chronology_subjects) >= 2
        and chronology_subjects <= left_people
        and chronology_subjects <= right_people
    )
    left_orgs = set(left.event.organization_names)
    right_orgs = set(right.event.organization_names)
    organization_match = bool(left_orgs & right_orgs)
    location_match = bool(
        left.event.location_text
        and left.event.location_text == right.event.location_text
    )
    event_type_match = (
        left.event.event_type == right.event.event_type
        and left.event.event_type != GENERIC_EVENT_TYPE
    )
    date_score = {"exact_day": 0.34, "same_value": 0.24, "same_month": 0.18}[
        relation
    ]
    score = date_score
    score += 0.22 * containment
    score += 0.10 * jaccard
    score += 0.14 * (min(len(shared_people), 2) / 2)
    score += 0.10 * float(mutual_subjects)
    score += 0.04 * float(organization_match)
    score += 0.03 * float(location_match)
    score += 0.03 * float(event_type_match)
    return EventMergePair(
        left_event_id=left.event.event_id,
        right_event_id=right.event.event_id,
        date_relation=relation,
        score=min(1.0, round(score, 6)),
        text_containment=round(containment, 6),
        text_jaccard=round(jaccard, 6),
        shared_person_ids=shared_people,
        mutual_chronology_subjects=mutual_subjects,
        organization_match=organization_match,
        location_match=location_match,
        event_type_match=event_type_match,
    )


def _is_candidate(pair: EventMergePair, minimum_score: float) -> bool:
    if pair.date_relation == "exact_day":
        return pair.score >= minimum_score or (
            pair.text_containment >= 0.82 and pair.text_jaccard >= 0.55
        )
    return (
        pair.score >= max(0.60, minimum_score - 0.10)
        and pair.text_containment >= 0.80
        and pair.text_jaccard >= 0.45
        and bool(pair.shared_person_ids)
        and (pair.mutual_chronology_subjects or pair.text_containment >= 0.90)
    )


def _is_high_confidence(
    pair: EventMergePair, members: tuple[SourceEvent, ...], automatic_score: float
) -> bool:
    return (
        len(members) == 2
        and pair.date_relation == "exact_day"
        and pair.score >= automatic_score
        and pair.text_containment >= 0.75
        and (
            pair.mutual_chronology_subjects
            or (pair.text_containment >= 0.90 and bool(pair.shared_person_ids))
        )
        and all(item.event.start.precision == "day" for item in members)
        and all(
            item.event.start.certainty in {"exact", "inferred"} for item in members
        )
        and any(item.event.start.certainty == "exact" for item in members)
    )


def _representative(members: tuple[SourceEvent, ...]) -> SourceEvent:
    review_rank = {"confirmed": 3, "unreviewed": 2, "needs_review": 1, "rejected": 0}
    return max(
        members,
        key=lambda item: (
            review_rank[item.event.review_status],
            item.event.extraction_confidence,
            item.event.start.certainty == "exact",
            item.event.event_type != GENERIC_EVENT_TYPE,
            -len(item.event.description),
            item.event.event_id,
        ),
    )


def _canonical_participants(
    members: tuple[SourceEvent, ...]
) -> list[EventParticipant]:
    selected: dict[tuple[str, str], EventParticipant] = {}
    source_rank = {"explicit": 3, "chronology_subject": 2, "inferred": 1}
    for member in members:
        for participant in member.event.participants:
            key = (participant.person_id, participant.mention_text)
            existing = selected.get(key)
            if existing is None:
                selected[key] = participant
                continue
            existing_rank = (
                existing.role == "年谱主体",
                source_rank[existing.mention_source],
                existing.role is not None,
            )
            candidate_rank = (
                participant.role == "年谱主体",
                source_rank[participant.mention_source],
                participant.role is not None,
            )
            if candidate_rank > existing_rank:
                selected[key] = participant
    return [selected[key] for key in sorted(selected)]


def _field_variants(members: tuple[SourceEvent, ...]) -> dict[str, list[dict[str, Any]]]:
    variants: dict[str, list[dict[str, Any]]] = {}
    values_by_field: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    for member in members:
        event = member.event
        event_values = {
            "name": event.name,
            "event_type": event.event_type,
            "start": event.start.model_dump(),
            "end": event.end.model_dump() if event.end else None,
            "location_text": event.location_text,
            "organization_names": sorted(event.organization_names),
            "participant_person_ids": sorted(
                {item.person_id for item in event.participants}
            ),
            "description_sha256": hashlib.sha256(
                event.description.encode("utf-8")
            ).hexdigest(),
        }
        for field_name, value in event_values.items():
            values_by_field[field_name].append((event.event_id, value))
    for field_name, field_entries in values_by_field.items():
        grouped: dict[str, dict[str, Any]] = {}
        for event_id, value in field_entries:
            key = json.dumps(value, ensure_ascii=False, sort_keys=True)
            grouped.setdefault(key, {"value": value, "source_event_ids": []})[
                "source_event_ids"
            ].append(event_id)
        if len(grouped) > 1:
            variants[field_name] = list(grouped.values())
    return variants


def _canonical_event_type(members: tuple[SourceEvent, ...], representative: SourceEvent) -> str:
    event_types = {item.event.event_type for item in members}
    specific = event_types - {GENERIC_EVENT_TYPE}
    if len(event_types) == 1:
        return next(iter(event_types))
    if len(specific) == 1:
        return next(iter(specific))
    return representative.event.event_type


def _candidate_from_group(
    members: tuple[SourceEvent, ...],
    pair_features: tuple[EventMergePair, ...],
    automatic_score: float,
) -> CanonicalCandidate:
    member_ids = sorted(item.event.event_id for item in members)
    canonical_id = _stable_id("cev", *member_ids)
    representative = _representative(members)
    high_confidence = len(pair_features) == 1 and _is_high_confidence(
        pair_features[0], members, automatic_score
    )
    candidate_kind: CandidateKind = (
        "high_confidence" if high_confidence else "uncertain"
    )
    confidence = round(
        sum(item.score for item in pair_features) / len(pair_features), 6
    )
    evidence_by_id: dict[str, EvidenceReference] = {}
    organizations: list[str] = []
    for member in members:
        for evidence in member.event.evidence:
            evidence_by_id[evidence.evidence_id] = evidence
        organizations.extend(member.event.organization_names)
    variants = _field_variants(members)
    event = HistoricalEvent(
        event_id=canonical_id,
        name=representative.event.name,
        event_type=_canonical_event_type(members, representative),
        start=representative.event.start,
        end=representative.event.end,
        location_text=representative.event.location_text,
        organization_names=list(dict.fromkeys(organizations)),
        description=representative.event.description,
        participants=_canonical_participants(members),
        evidence=list(evidence_by_id.values()),
        extraction_method="merge",
        extraction_confidence=confidence,
        review_status="unreviewed" if high_confidence else "needs_review",
        extractor_version=MERGE_VERSION,
    )
    hash_payload = {
        "members": [
            {
                "event": member.event.model_dump(),
                "document_ids": member.document_ids,
            }
            for member in members
        ],
        "features": [item.model_dump() for item in pair_features],
        "candidate_kind": candidate_kind,
        "merge_version": MERGE_VERSION,
    }
    input_hash = hashlib.sha256(
        json.dumps(hash_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return CanonicalCandidate(
        event=event,
        members=members,
        pair_features=pair_features,
        candidate_kind=candidate_kind,
        field_variants=variants,
        input_hash=input_hash,
    )


def discover_event_merge_candidates(
    database: Database,
    *,
    minimum_score: float = 0.70,
    automatic_score: float = 0.80,
) -> tuple[list[CanonicalCandidate], int, int]:
    if not 0.5 <= minimum_score <= 0.95:
        raise ResearchDataError("minimum merge score must be between 0.5 and 0.95")
    if not minimum_score <= automatic_score <= 0.99:
        raise ResearchDataError("automatic score must be at least the minimum score")
    database.initialize()
    with database.connect() as connection:
        source_events = _load_source_events(connection)
    events_by_month: dict[str, list[SourceEvent]] = defaultdict(list)
    events_by_id = {item.event.event_id: item for item in source_events}
    for source in source_events:
        assert source.event.start.value is not None
        events_by_month[source.event.start.value[:7]].append(source)
    compared = 0
    pairs: list[EventMergePair] = []
    for month_events in events_by_month.values():
        for left_index, left in enumerate(month_events):
            for right in month_events[left_index + 1 :]:
                if set(left.document_ids) & set(right.document_ids):
                    continue
                compared += 1
                pair = _pair_features(left, right)
                if pair is not None and _is_candidate(pair, minimum_score):
                    pairs.append(pair)
    disjoint = _DisjointSet()
    for pair in pairs:
        disjoint.union(pair.left_event_id, pair.right_event_id)
    member_ids: dict[str, set[str]] = defaultdict(set)
    group_pairs: dict[str, list[EventMergePair]] = defaultdict(list)
    for pair in pairs:
        root = disjoint.find(pair.left_event_id)
        member_ids[root].update((pair.left_event_id, pair.right_event_id))
        group_pairs[root].append(pair)
    candidates: list[CanonicalCandidate] = []
    for root in sorted(member_ids):
        members = tuple(events_by_id[event_id] for event_id in sorted(member_ids[root]))
        pair_features = tuple(
            sorted(
                group_pairs[root],
                key=lambda item: (item.left_event_id, item.right_event_id),
            )
        )
        candidates.append(
            _candidate_from_group(members, pair_features, automatic_score)
        )
    return candidates, len(source_events), compared


def _canonical_row_values(candidate: CanonicalCandidate) -> tuple[object, ...]:
    event = candidate.event
    end = event.end
    representative = _representative(candidate.members)
    return (
        representative.event.event_id,
        event.name,
        event.event_type,
        event.start.value,
        event.start.precision,
        event.start.certainty,
        event.start.original_text,
        end.value if end else None,
        end.precision if end else None,
        end.certainty if end else None,
        end.original_text if end else None,
        event.location_text,
        json.dumps(event.organization_names, ensure_ascii=False),
        event.description,
        MERGE_METHOD,
        event.extraction_confidence,
        candidate.candidate_kind,
        event.review_status,
        json.dumps(candidate.field_variants, ensure_ascii=False, sort_keys=True),
        json.dumps(
            [item.model_dump() for item in candidate.pair_features],
            ensure_ascii=False,
            sort_keys=True,
        ),
        candidate.input_hash,
        MERGE_VERSION,
    )


def _write_candidate_links(
    connection: sqlite3.Connection, candidate: CanonicalCandidate, now: str
) -> None:
    canonical_id = candidate.event.event_id
    representative_id = _representative(candidate.members).event.event_id
    connection.executemany(
        """
        INSERT INTO canonical_event_members (
            canonical_event_id, source_event_id, source_document_ids_json,
            source_snapshot_json, is_representative
        ) VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                canonical_id,
                member.event.event_id,
                json.dumps(member.document_ids, ensure_ascii=False),
                member.event.model_dump_json(),
                int(member.event.event_id == representative_id),
            )
            for member in candidate.members
        ],
    )
    connection.executemany(
        """
        INSERT INTO canonical_event_participants (
            canonical_event_id, person_id, role, mention_text, mention_source
        ) VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                canonical_id,
                item.person_id,
                item.role,
                item.mention_text,
                item.mention_source,
            )
            for item in candidate.event.participants
        ],
    )
    evidence_links = [
        (canonical_id, evidence.evidence_id, member.event.event_id)
        for member in candidate.members
        for evidence in member.event.evidence
    ]
    connection.executemany(
        """
        INSERT INTO canonical_event_evidence (
            canonical_event_id, evidence_id, source_event_id
        ) VALUES (?, ?, ?)
        """,
        evidence_links,
    )
    if candidate.event.review_status == "needs_review":
        _ensure_pending_review_queue(connection, candidate, now)


def _ensure_pending_review_queue(
    connection: sqlite3.Connection,
    candidate: CanonicalCandidate,
    now: str,
    *,
    reactivate_existing: bool = False,
) -> None:
    canonical_id = candidate.event.event_id
    if reactivate_existing:
        existing_queue = connection.execute(
            "SELECT queue_id FROM event_merge_review_queue "
            "WHERE canonical_event_id = ? ORDER BY created_at DESC LIMIT 1",
            (canonical_id,),
        ).fetchone()
        if existing_queue is not None:
            connection.execute(
                "UPDATE event_merge_review_queue SET status = 'pending', updated_at = ? "
                "WHERE queue_id = ?",
                (now, existing_queue["queue_id"]),
            )
            return
    reasons = ["uncertain_event_merge"]
    if len(candidate.members) > 2:
        reasons.append("multi_event_component")
    if any(item.date_relation != "exact_day" for item in candidate.pair_features):
        reasons.append("date_variant")
    if candidate.field_variants:
        reasons.append("field_variants")
    queue_id = _stable_id("merge_review", canonical_id, candidate.input_hash)
    priority = min(90, 55 + round((1 - candidate.event.extraction_confidence) * 50))
    connection.execute(
        """
        INSERT INTO event_merge_review_queue (
            queue_id, canonical_event_id, reason_codes_json, priority,
            status, created_at, updated_at
        ) VALUES (?, ?, ?, ?, 'pending', ?, ?)
        """,
        (
            queue_id,
            canonical_id,
            json.dumps(reasons, ensure_ascii=False),
            priority,
            now,
            now,
        ),
    )


def _sync_candidates(
    database: Database, candidates: list[CanonicalCandidate]
) -> tuple[int, int, int, int, int]:
    now = utc_now()
    created = updated = skipped = protected = deactivated = 0
    current_ids = {item.event.event_id for item in candidates}
    with database.connect() as connection:
        for candidate in candidates:
            canonical_id = candidate.event.event_id
            existing = connection.execute(
                "SELECT * FROM canonical_events WHERE canonical_event_id = ?",
                (canonical_id,),
            ).fetchone()
            if existing is not None and existing["review_status"] in {
                "confirmed",
                "rejected",
            }:
                protected += 1
                continue
            if existing is not None and existing["input_hash"] == candidate.input_hash:
                if not bool(existing["is_active"]) and existing["review_status"] not in {
                    "confirmed",
                    "rejected",
                }:
                    connection.execute(
                        "UPDATE canonical_events SET is_active = 1, updated_at = ? "
                        "WHERE canonical_event_id = ?",
                        (now, canonical_id),
                    )
                    if existing["review_status"] == "needs_review":
                        _ensure_pending_review_queue(
                            connection,
                            candidate,
                            now,
                            reactivate_existing=True,
                        )
                    updated += 1
                else:
                    skipped += 1
                continue
            values = _canonical_row_values(candidate)
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO canonical_events (
                        canonical_event_id, representative_event_id, name, event_type,
                        start_value, start_precision, start_certainty, start_original_text,
                        end_value, end_precision, end_certainty, end_original_text,
                        location_text, organization_names_json, description, merge_method,
                        merge_confidence, candidate_kind, review_status, field_variants_json,
                        merge_features_json, input_hash, merge_version, is_active,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                              ?, ?, ?, 1, ?, ?)
                    """,
                    (canonical_id, *values, now, now),
                )
                created += 1
            else:
                connection.execute(
                    "DELETE FROM event_merge_review_queue "
                    "WHERE canonical_event_id = ? AND status = 'pending'",
                    (canonical_id,),
                )
                connection.execute(
                    "DELETE FROM canonical_event_members WHERE canonical_event_id = ?",
                    (canonical_id,),
                )
                connection.execute(
                    "DELETE FROM canonical_event_participants WHERE canonical_event_id = ?",
                    (canonical_id,),
                )
                connection.execute(
                    "DELETE FROM canonical_event_evidence WHERE canonical_event_id = ?",
                    (canonical_id,),
                )
                connection.execute(
                    """
                    UPDATE canonical_events SET
                        representative_event_id = ?, name = ?, event_type = ?,
                        start_value = ?, start_precision = ?, start_certainty = ?,
                        start_original_text = ?, end_value = ?, end_precision = ?,
                        end_certainty = ?, end_original_text = ?, location_text = ?,
                        organization_names_json = ?, description = ?, merge_method = ?,
                        merge_confidence = ?, candidate_kind = ?, review_status = ?,
                        reviewed_by = NULL, reviewed_at = NULL, field_variants_json = ?,
                        merge_features_json = ?, input_hash = ?, merge_version = ?,
                        is_active = 1, updated_at = ?
                    WHERE canonical_event_id = ?
                    """,
                    (*values, now, canonical_id),
                )
                updated += 1
            _write_candidate_links(connection, candidate, now)
        stale_rows = connection.execute(
            """
            SELECT canonical_event_id FROM canonical_events
            WHERE merge_version = ? AND is_active = 1
              AND review_status IN ('unreviewed', 'needs_review')
            """,
            (MERGE_VERSION,),
        ).fetchall()
        for row in stale_rows:
            canonical_id = str(row["canonical_event_id"])
            if canonical_id in current_ids:
                continue
            connection.execute(
                "UPDATE canonical_events SET is_active = 0, updated_at = ? "
                "WHERE canonical_event_id = ?",
                (now, canonical_id),
            )
            connection.execute(
                "UPDATE event_merge_review_queue SET status = 'dismissed', updated_at = ? "
                "WHERE canonical_event_id = ? AND status = 'pending'",
                (now, canonical_id),
            )
            deactivated += 1
    return created, updated, skipped, protected, deactivated


def _samples(candidates: list[CanonicalCandidate], limit: int = 12) -> list[dict[str, Any]]:
    selected = sorted(
        candidates,
        key=lambda item: (-item.event.extraction_confidence, item.event.event_id),
    )[:limit]
    return [
        {
            "canonical_event_id": item.event.event_id,
            "candidate_kind": item.candidate_kind,
            "merge_confidence": item.event.extraction_confidence,
            "start": item.event.start.model_dump(),
            "name": item.event.name,
            "member_event_ids": [member.event.event_id for member in item.members],
            "document_ids": sorted(
                {
                    document_id
                    for member in item.members
                    for document_id in member.document_ids
                }
            ),
            "evidence": [
                {
                    "document_id": evidence.document_id,
                    "pdf_page_start": evidence.pdf_page_start,
                    "pdf_page_end": evidence.pdf_page_end,
                }
                for evidence in item.event.evidence
            ],
            "variant_fields": sorted(item.field_variants),
        }
        for item in selected
    ]


def merge_duplicate_events(
    *,
    database: Database,
    reports_dir: Path,
    run_id: str,
    minimum_score: float = 0.70,
    automatic_score: float = 0.80,
    dry_run: bool = False,
) -> EventMergeSummary:
    candidates, source_count, compared = discover_event_merge_candidates(
        database,
        minimum_score=minimum_score,
        automatic_score=automatic_score,
    )
    high = sum(item.candidate_kind == "high_confidence" for item in candidates)
    uncertain = len(candidates) - high
    summary = EventMergeSummary(
        run_id=run_id,
        merge_version=MERGE_VERSION,
        source_events=source_count,
        compared_pairs=compared,
        candidate_pairs=sum(len(item.pair_features) for item in candidates),
        candidate_groups=len(candidates),
        high_confidence_groups=high,
        uncertain_groups=uncertain,
        dry_run=dry_run,
        samples=_samples(candidates),
    )
    if not dry_run:
        (
            summary.created,
            summary.updated,
            summary.skipped,
            summary.protected,
            summary.deactivated,
        ) = _sync_candidates(database, candidates)
    reports_dir.mkdir(parents=True, exist_ok=True)
    rendered = summary.model_dump_json(indent=2)
    (reports_dir / f"event_merge_{run_id}.json").write_text(rendered, encoding="utf-8")
    (reports_dir / "event_merge_latest.json").write_text(rendered, encoding="utf-8")
    return summary


def list_event_merge_review_queue(
    database: Database, *, limit: int = 20
) -> list[dict[str, Any]]:
    if not 1 <= limit <= 200:
        raise ResearchDataError("merge review queue limit must be between 1 and 200")
    database.initialize()
    with database.connect() as connection:
        rows = connection.execute(
            """
            SELECT q.*, c.name, c.start_value, c.merge_confidence,
                   c.candidate_kind, c.field_variants_json,
                   COUNT(m.source_event_id) AS member_count
            FROM event_merge_review_queue q
            JOIN canonical_events c USING (canonical_event_id)
            JOIN canonical_event_members m USING (canonical_event_id)
            WHERE q.status = 'pending' AND c.is_active = 1
            GROUP BY q.queue_id
            ORDER BY q.priority DESC, c.merge_confidence DESC, q.created_at
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {
            **dict(row),
            "reason_codes": json.loads(str(row["reason_codes_json"])),
            "variant_fields": sorted(
                json.loads(str(row["field_variants_json"])).keys()
            ),
        }
        for row in rows
    ]


def review_event_merge(
    database: Database,
    *,
    canonical_event_id: str,
    decision: Literal["confirmed", "rejected", "reopened"],
    reviewed_by: str,
    note: str | None = None,
) -> str:
    if not reviewed_by.strip():
        raise ResearchDataError("reviewed_by cannot be empty")
    database.initialize()
    now = utc_now()
    review_id = f"canonical_review_{uuid4().hex}"
    with database.connect() as connection:
        row = connection.execute(
            "SELECT * FROM canonical_events WHERE canonical_event_id = ?",
            (canonical_event_id,),
        ).fetchone()
        if row is None:
            raise ResearchDataError(f"unknown canonical_event_id: {canonical_event_id}")
        previous = str(row["review_status"])
        target = "needs_review" if decision == "reopened" else decision
        if previous == target:
            raise ResearchDataError(f"canonical event is already {target}")
        connection.execute(
            """
            INSERT INTO canonical_event_reviews (
                review_id, canonical_event_id, previous_status, decision,
                reviewed_by, review_note, reviewed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                review_id,
                canonical_event_id,
                previous,
                decision,
                reviewed_by.strip(),
                note,
                now,
            ),
        )
        connection.execute(
            """
            UPDATE canonical_events SET
                review_status = ?, reviewed_by = ?, reviewed_at = ?,
                is_active = ?, updated_at = ?
            WHERE canonical_event_id = ?
            """,
            (
                target,
                None if decision == "reopened" else reviewed_by.strip(),
                None if decision == "reopened" else now,
                int(decision != "rejected"),
                now,
                canonical_event_id,
            ),
        )
        if decision in {"confirmed", "rejected"}:
            queue_status = "resolved" if decision == "confirmed" else "dismissed"
            connection.execute(
                "UPDATE event_merge_review_queue SET status = ?, updated_at = ? "
                "WHERE canonical_event_id = ? AND status = 'pending'",
                (queue_status, now, canonical_event_id),
            )
        else:
            existing_queue = connection.execute(
                "SELECT queue_id FROM event_merge_review_queue "
                "WHERE canonical_event_id = ? ORDER BY created_at DESC LIMIT 1",
                (canonical_event_id,),
            ).fetchone()
            if existing_queue is not None:
                connection.execute(
                    "UPDATE event_merge_review_queue SET status = 'pending', "
                    "updated_at = ? WHERE queue_id = ?",
                    (now, existing_queue["queue_id"]),
                )
            else:
                queue_id = _stable_id("merge_review", canonical_event_id, review_id)
                connection.execute(
                    """
                    INSERT INTO event_merge_review_queue (
                        queue_id, canonical_event_id, reason_codes_json,
                        priority, status, created_at, updated_at
                    ) VALUES (?, ?, '["reopened_for_review"]', 70, 'pending', ?, ?)
                    """,
                    (queue_id, canonical_event_id, now, now),
                )
    return review_id
