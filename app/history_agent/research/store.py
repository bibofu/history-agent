from __future__ import annotations

import json
import sqlite3

from history_agent.db import Database
from history_agent.errors import ResearchDataError
from history_agent.research.models import (
    EvidenceReference,
    HistoricalEvent,
    HistoricalRelationship,
    TemporalPoint,
)
from history_agent.research.people import utc_now


def _temporal_values(point: TemporalPoint | None) -> tuple[str | None, ...]:
    if point is None:
        return (None, None, None, None)
    return (point.value, point.precision, point.certainty, point.original_text)


def _event_values(event: HistoricalEvent) -> tuple[object, ...]:
    return (
        event.name,
        event.event_type,
        *_temporal_values(event.start),
        *_temporal_values(event.end),
        event.location_text,
        json.dumps(event.organization_names, ensure_ascii=False),
        event.description,
        event.extraction_method,
        event.extraction_confidence,
        event.review_status,
        event.extractor_version,
    )


class ResearchStore:
    def __init__(self, database: Database):
        self.database = database

    def save_event(self, event: HistoricalEvent) -> None:
        self.database.initialize()
        now = utc_now()
        with self.database.connect() as connection:
            self._require_active_people(
                connection, [participant.person_id for participant in event.participants]
            )
            try:
                self._insert_event(connection, event, now)
            except sqlite3.IntegrityError as exc:
                raise ResearchDataError(f"Cannot save event {event.event_id}: {exc}") from exc

    def sync_generated_events(
        self, events: list[HistoricalEvent]
    ) -> tuple[int, int, int]:
        """Idempotently sync rule output while protecting reviewed or foreign records."""

        self.database.initialize()
        if not events:
            return 0, 0, 0
        now = utc_now()
        created = 0
        updated = 0
        skipped = 0
        with self.database.connect() as connection:
            self._require_active_people(
                connection,
                [
                    participant.person_id
                    for event in events
                    for participant in event.participants
                ],
            )
            try:
                for event in events:
                    existing = connection.execute(
                        "SELECT * FROM historical_events WHERE event_id = ?",
                        (event.event_id,),
                    ).fetchone()
                    if existing is None:
                        self._insert_event(connection, event, now)
                        created += 1
                        continue
                    if self._event_matches(connection, existing, event):
                        skipped += 1
                        continue
                    is_replaceable = (
                        existing["extraction_method"] == "rule"
                        and existing["extractor_version"] == event.extractor_version
                        and existing["review_status"] in {"unreviewed", "needs_review"}
                    )
                    if not is_replaceable:
                        skipped += 1
                        continue
                    self._update_generated_event(connection, event, now)
                    updated += 1
            except sqlite3.IntegrityError as exc:
                raise ResearchDataError(f"Cannot save event batch: {exc}") from exc
        return created, updated, skipped

    def save_relationship(self, relationship: HistoricalRelationship) -> None:
        self.database.initialize()
        now = utc_now()
        with self.database.connect() as connection:
            person_ids = [relationship.subject_person_id]
            if relationship.object_person_id:
                person_ids.append(relationship.object_person_id)
            self._require_active_people(connection, person_ids)
            relation_type = connection.execute(
                "SELECT * FROM relation_types WHERE relation_type = ?",
                (relationship.relation_type,),
            ).fetchone()
            if relation_type is None:
                raise ResearchDataError(
                    f"unknown relation_type: {relationship.relation_type}"
                )
            self._validate_relationship_policy(relation_type, relationship)
            try:
                connection.execute(
                    """
                    INSERT INTO person_relationships (
                        relationship_id, relation_type, subject_person_id,
                        subject_mention_text, object_person_id, object_mention_text,
                        organization_name, role_title,
                        start_value, start_precision, start_certainty, start_original_text,
                        end_value, end_precision, end_certainty, end_original_text,
                        event_id, description, extraction_method, extraction_confidence,
                        review_status, reviewed_by, extractor_version, created_at, updated_at
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        relationship.relationship_id,
                        relationship.relation_type,
                        relationship.subject_person_id,
                        relationship.subject_mention_text,
                        relationship.object_person_id,
                        relationship.object_mention_text,
                        relationship.organization_name,
                        relationship.role_title,
                        *_temporal_values(relationship.start),
                        *_temporal_values(relationship.end),
                        relationship.event_id,
                        relationship.description,
                        relationship.extraction_method,
                        relationship.extraction_confidence,
                        relationship.review_status,
                        relationship.reviewed_by,
                        relationship.extractor_version,
                        now,
                        now,
                    ),
                )
                for evidence in relationship.evidence:
                    self._save_evidence(connection, evidence, now)
                    connection.execute(
                        """
                        INSERT INTO relationship_evidence (relationship_id, evidence_id)
                        VALUES (?, ?)
                        """,
                        (relationship.relationship_id, evidence.evidence_id),
                    )
            except sqlite3.IntegrityError as exc:
                raise ResearchDataError(
                    f"Cannot save relationship {relationship.relationship_id}: {exc}"
                ) from exc

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection, event: HistoricalEvent, now: str
    ) -> None:
        connection.execute(
            """
            INSERT INTO historical_events (
                event_id, name, event_type,
                start_value, start_precision, start_certainty, start_original_text,
                end_value, end_precision, end_certainty, end_original_text,
                location_text, organization_names_json, description,
                extraction_method, extraction_confidence, review_status,
                extractor_version, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                *_event_values(event),
                now,
                now,
            ),
        )
        ResearchStore._write_event_links(connection, event, now)

    @staticmethod
    def _write_event_links(
        connection: sqlite3.Connection, event: HistoricalEvent, now: str
    ) -> None:
        connection.executemany(
            """
            INSERT INTO event_participants (
                event_id, person_id, role, mention_text, mention_source
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    event.event_id,
                    participant.person_id,
                    participant.role,
                    participant.mention_text,
                    participant.mention_source,
                )
                for participant in event.participants
            ],
        )
        for evidence in event.evidence:
            ResearchStore._save_evidence(connection, evidence, now)
            connection.execute(
                "INSERT INTO event_evidence (event_id, evidence_id) VALUES (?, ?)",
                (event.event_id, evidence.evidence_id),
            )

    @staticmethod
    def _event_matches(
        connection: sqlite3.Connection,
        existing: sqlite3.Row,
        event: HistoricalEvent,
    ) -> bool:
        columns = (
            "name",
            "event_type",
            "start_value",
            "start_precision",
            "start_certainty",
            "start_original_text",
            "end_value",
            "end_precision",
            "end_certainty",
            "end_original_text",
            "location_text",
            "organization_names_json",
            "description",
            "extraction_method",
            "extraction_confidence",
            "review_status",
            "extractor_version",
        )
        if tuple(existing[column] for column in columns) != _event_values(event):
            return False
        actual_participants = {
            (row["person_id"], row["role"], row["mention_text"], row["mention_source"])
            for row in connection.execute(
                "SELECT * FROM event_participants WHERE event_id = ?",
                (event.event_id,),
            ).fetchall()
        }
        expected_participants = {
            (
                participant.person_id,
                participant.role,
                participant.mention_text,
                participant.mention_source,
            )
            for participant in event.participants
        }
        if actual_participants != expected_participants:
            return False
        actual_evidence = {
            row["evidence_id"]
            for row in connection.execute(
                "SELECT evidence_id FROM event_evidence WHERE event_id = ?",
                (event.event_id,),
            ).fetchall()
        }
        return actual_evidence == {item.evidence_id for item in event.evidence}

    @staticmethod
    def _update_generated_event(
        connection: sqlite3.Connection, event: HistoricalEvent, now: str
    ) -> None:
        connection.execute(
            """
            UPDATE historical_events SET
                name = ?, event_type = ?,
                start_value = ?, start_precision = ?, start_certainty = ?,
                start_original_text = ?, end_value = ?, end_precision = ?,
                end_certainty = ?, end_original_text = ?, location_text = ?,
                organization_names_json = ?, description = ?, extraction_method = ?,
                extraction_confidence = ?, review_status = ?, extractor_version = ?,
                updated_at = ?
            WHERE event_id = ?
            """,
            (*_event_values(event), now, event.event_id),
        )
        connection.execute(
            "DELETE FROM event_participants WHERE event_id = ?", (event.event_id,)
        )
        connection.execute(
            "DELETE FROM event_evidence WHERE event_id = ?", (event.event_id,)
        )
        ResearchStore._write_event_links(connection, event, now)

    @staticmethod
    def _require_active_people(
        connection: sqlite3.Connection, person_ids: list[str]
    ) -> None:
        for person_id in set(person_ids):
            row = connection.execute(
                "SELECT status FROM persons WHERE person_id = ?", (person_id,)
            ).fetchone()
            if row is None:
                raise ResearchDataError(f"unknown person_id: {person_id}")
            if row["status"] != "active":
                raise ResearchDataError(f"person is not active: {person_id}")

    @staticmethod
    def _validate_relationship_policy(
        relation_type: sqlite3.Row, relationship: HistoricalRelationship
    ) -> None:
        object_kind = str(relation_type["object_kind"])
        if object_kind == "person" and relationship.object_person_id is None:
            raise ResearchDataError("relation type requires an object person")
        if object_kind == "organization" and not relationship.organization_name:
            raise ResearchDataError("relation type requires an organization")
        if bool(relation_type["requires_event"]) and relationship.event_id is None:
            raise ResearchDataError("relation type requires a shared event")
        if (
            bool(relation_type["requires_human_review"])
            and relationship.review_status == "confirmed"
            and not relationship.reviewed_by
        ):
            raise ResearchDataError("confirmed relation requires a human reviewer")
        if (
            str(relation_type["extraction_policy"]) == "manual_only"
            and relationship.extraction_method != "manual"
        ):
            raise ResearchDataError("manual-only relation cannot be automatically extracted")

    @staticmethod
    def _save_evidence(
        connection: sqlite3.Connection, evidence: EvidenceReference, now: str
    ) -> None:
        existing = connection.execute(
            "SELECT * FROM evidence_records WHERE evidence_id = ?",
            (evidence.evidence_id,),
        ).fetchone()
        expected = (
            evidence.document_id,
            evidence.chunk_id,
            evidence.pdf_page_start,
            evidence.pdf_page_end,
            evidence.quote,
            json.dumps(evidence.extraction_methods, ensure_ascii=False),
        )
        if existing is not None:
            actual = (
                existing["document_id"],
                existing["chunk_id"],
                existing["pdf_page_start"],
                existing["pdf_page_end"],
                existing["quote"],
                existing["extraction_methods_json"],
            )
            if actual != expected:
                raise ResearchDataError(
                    f"evidence_id already refers to different content: {evidence.evidence_id}"
                )
            return
        connection.execute(
            """
            INSERT INTO evidence_records (
                evidence_id, document_id, chunk_id, pdf_page_start, pdf_page_end,
                quote, extraction_methods_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (evidence.evidence_id, *expected, now),
        )
