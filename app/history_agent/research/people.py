from __future__ import annotations

import json
import re
import sqlite3
from datetime import UTC, datetime
from typing import Literal, cast
from uuid import uuid4

from history_agent.db import Database
from history_agent.errors import ResearchDataError
from history_agent.research.models import (
    PersonCandidate,
    PersonCatalog,
    PersonResolution,
    RelationTypeCatalog,
)

WHITESPACE = re.compile(r"\s+")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def normalize_person_name(value: str) -> str:
    return WHITESPACE.sub("", value).casefold()


def sync_person_catalog(database: Database, catalog: PersonCatalog) -> dict[str, int]:
    database.initialize()
    now = utc_now()
    alias_count = 0
    with database.connect() as connection:
        for person in catalog.people:
            connection.execute(
                """
                INSERT INTO persons (
                    person_id, canonical_name, normalized_name, description,
                    status, merged_into_person_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'active', NULL, ?, ?)
                ON CONFLICT(person_id) DO UPDATE SET
                    canonical_name = excluded.canonical_name,
                    normalized_name = excluded.normalized_name,
                    description = excluded.description,
                    updated_at = excluded.updated_at
                """,
                (
                    person.person_id,
                    person.canonical_name,
                    normalize_person_name(person.canonical_name),
                    person.description,
                    now,
                    now,
                ),
            )
            connection.execute(
                "UPDATE person_aliases SET is_active = 0, updated_at = ? "
                "WHERE person_id = ? AND source = 'catalog'",
                (now, person.person_id),
            )
            for alias in person.aliases:
                connection.execute(
                    """
                    INSERT INTO person_aliases (
                        person_id, alias_text, normalized_alias, alias_type,
                        notes, source, is_active, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'catalog', 1, ?, ?)
                    ON CONFLICT(person_id, normalized_alias) DO UPDATE SET
                        alias_text = excluded.alias_text,
                        alias_type = excluded.alias_type,
                        notes = excluded.notes,
                        source = 'catalog',
                        is_active = 1,
                        updated_at = excluded.updated_at
                    """,
                    (
                        person.person_id,
                        alias.name,
                        normalize_person_name(alias.name),
                        alias.alias_type,
                        alias.notes,
                        now,
                        now,
                    ),
                )
                alias_count += 1

        for ambiguity in catalog.ambiguities:
            connection.execute(
                """
                INSERT INTO person_ambiguities (
                    ambiguity_id, mention_text, normalized_mention,
                    candidate_person_ids_json, status, resolved_person_id,
                    notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ambiguity_id) DO UPDATE SET
                    mention_text = excluded.mention_text,
                    normalized_mention = excluded.normalized_mention,
                    candidate_person_ids_json = excluded.candidate_person_ids_json,
                    status = excluded.status,
                    resolved_person_id = excluded.resolved_person_id,
                    notes = excluded.notes,
                    updated_at = excluded.updated_at
                """,
                (
                    ambiguity.ambiguity_id,
                    ambiguity.mention,
                    normalize_person_name(ambiguity.mention),
                    json.dumps(ambiguity.candidate_person_ids, ensure_ascii=False),
                    ambiguity.status,
                    ambiguity.resolved_person_id,
                    ambiguity.notes,
                    now,
                    now,
                ),
            )
    return {
        "people": len(catalog.people),
        "aliases": alias_count,
        "ambiguities": len(catalog.ambiguities),
    }


def sync_relation_types(
    database: Database, catalog: RelationTypeCatalog
) -> dict[str, int]:
    database.initialize()
    now = utc_now()
    with database.connect() as connection:
        for item in catalog.relation_types:
            connection.execute(
                """
                INSERT INTO relation_types (
                    relation_type, label, object_kind, extraction_policy,
                    requires_human_review, requires_event, description, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(relation_type) DO UPDATE SET
                    label = excluded.label,
                    object_kind = excluded.object_kind,
                    extraction_policy = excluded.extraction_policy,
                    requires_human_review = excluded.requires_human_review,
                    requires_event = excluded.requires_event,
                    description = excluded.description,
                    updated_at = excluded.updated_at
                """,
                (
                    item.relation_type,
                    item.label,
                    item.object_kind,
                    item.extraction_policy,
                    int(item.requires_human_review),
                    int(item.requires_event),
                    item.description,
                    now,
                ),
            )
    return {"relation_types": len(catalog.relation_types)}


def resolve_person(database: Database, name: str) -> PersonResolution:
    database.initialize()
    normalized = normalize_person_name(name)
    if not normalized:
        return PersonResolution(
            query=name, normalized_query=normalized, status="not_found", candidates=[]
        )
    with database.connect() as connection:
        ambiguity = connection.execute(
            "SELECT * FROM person_ambiguities WHERE normalized_mention = ?",
            (normalized,),
        ).fetchone()
        if ambiguity is not None and ambiguity["status"] == "unresolved":
            candidate_ids = json.loads(str(ambiguity["candidate_person_ids_json"]))
            candidates = _candidate_rows(connection, candidate_ids, name)
            return PersonResolution(
                query=name,
                normalized_query=normalized,
                status="ambiguous",
                candidates=candidates,
            )

        rows = connection.execute(
            """
            SELECT p.*, p.canonical_name AS matched_form
            FROM persons p
            WHERE p.normalized_name = ?
            UNION
            SELECT p.*, a.alias_text AS matched_form
            FROM person_aliases a
            JOIN persons p ON p.person_id = a.person_id
            WHERE a.normalized_alias = ? AND a.is_active = 1
            ORDER BY person_id
            """,
            (normalized, normalized),
        ).fetchall()
    candidates = [
        PersonCandidate(
            person_id=str(row["person_id"]),
            canonical_name=str(row["canonical_name"]),
            matched_form=str(row["matched_form"]),
            status=cast(Literal["active", "merged"], str(row["status"])),
            merged_into_person_id=row["merged_into_person_id"],
        )
        for row in rows
    ]
    status: Literal["resolved", "ambiguous", "not_found"] = (
        "not_found" if not candidates else "resolved" if len(candidates) == 1 else "ambiguous"
    )
    return PersonResolution(
        query=name,
        normalized_query=normalized,
        status=status,
        candidates=candidates,
    )


def _candidate_rows(
    connection: sqlite3.Connection, person_ids: list[str], matched_form: str
) -> list[PersonCandidate]:
    if not person_ids:
        return []
    placeholders = ", ".join("?" for _ in person_ids)
    rows = connection.execute(
        f"SELECT * FROM persons WHERE person_id IN ({placeholders}) ORDER BY person_id",
        person_ids,
    ).fetchall()
    return [
        PersonCandidate(
            person_id=str(row["person_id"]),
            canonical_name=str(row["canonical_name"]),
            matched_form=matched_form,
            status=cast(Literal["active", "merged"], str(row["status"])),
            merged_into_person_id=row["merged_into_person_id"],
        )
        for row in rows
    ]


def propose_person_merge(
    database: Database,
    *,
    source_person_id: str,
    target_person_id: str,
    reason: str,
    proposed_by: str,
) -> str:
    if source_person_id == target_person_id:
        raise ResearchDataError("source and target person must differ")
    proposal_id = f"merge_{uuid4().hex}"
    database.initialize()
    with database.connect() as connection:
        _require_active_person(connection, source_person_id)
        _require_active_person(connection, target_person_id)
        connection.execute(
            """
            INSERT INTO person_merge_proposals (
                proposal_id, source_person_id, target_person_id, reason,
                status, proposed_by, proposed_at
            ) VALUES (?, ?, ?, ?, 'pending', ?, ?)
            """,
            (
                proposal_id,
                source_person_id,
                target_person_id,
                reason.strip(),
                proposed_by.strip(),
                utc_now(),
            ),
        )
    return proposal_id


def review_person_merge(
    database: Database,
    *,
    proposal_id: str,
    decision: str,
    reviewed_by: str,
    review_note: str | None = None,
) -> None:
    if decision not in {"accepted", "rejected"}:
        raise ResearchDataError("merge decision must be accepted or rejected")
    database.initialize()
    now = utc_now()
    with database.connect() as connection:
        proposal = connection.execute(
            "SELECT * FROM person_merge_proposals WHERE proposal_id = ?",
            (proposal_id,),
        ).fetchone()
        if proposal is None:
            raise ResearchDataError(f"unknown merge proposal: {proposal_id}")
        if proposal["status"] != "pending":
            raise ResearchDataError("merge proposal has already been reviewed")
        if decision == "accepted":
            _require_active_person(connection, str(proposal["source_person_id"]))
            _require_active_person(connection, str(proposal["target_person_id"]))
            connection.execute(
                """
                UPDATE persons
                SET status = 'merged', merged_into_person_id = ?, updated_at = ?
                WHERE person_id = ?
                """,
                (
                    proposal["target_person_id"],
                    now,
                    proposal["source_person_id"],
                ),
            )
        connection.execute(
            """
            UPDATE person_merge_proposals
            SET status = ?, reviewed_by = ?, reviewed_at = ?, review_note = ?
            WHERE proposal_id = ?
            """,
            (decision, reviewed_by.strip(), now, review_note, proposal_id),
        )


def _require_active_person(connection: sqlite3.Connection, person_id: str) -> None:
    row = connection.execute(
        "SELECT status FROM persons WHERE person_id = ?", (person_id,)
    ).fetchone()
    if row is None:
        raise ResearchDataError(f"unknown person_id: {person_id}")
    if row["status"] != "active":
        raise ResearchDataError(f"person is not active: {person_id}")
