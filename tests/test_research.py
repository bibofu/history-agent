from pathlib import Path

import pytest
from history_agent.db import Database
from history_agent.errors import ResearchDataError
from history_agent.research.catalog import load_person_catalog, load_relation_type_catalog
from history_agent.research.models import (
    EventParticipant,
    EvidenceReference,
    HistoricalEvent,
    HistoricalRelationship,
    PersonCatalog,
    RelationTypeCatalog,
    TemporalPoint,
)
from history_agent.research.people import (
    propose_person_merge,
    resolve_person,
    review_person_merge,
    sync_person_catalog,
    sync_relation_types,
)
from history_agent.research.store import ResearchStore
from pydantic import ValidationError


def _people_catalog() -> PersonCatalog:
    return PersonCatalog.model_validate(
        {
            "schema_version": 2,
            "people": [
                {
                    "person_id": "person_alpha",
                    "canonical_name": "甲某",
                    "aliases": [{"name": "同名"}],
                },
                {
                    "person_id": "person_beta",
                    "canonical_name": "乙某",
                    "aliases": [{"name": "同名"}],
                },
            ],
            "ambiguities": [
                {
                    "ambiguity_id": "ambiguity_same_name",
                    "mention": "同名",
                    "candidate_person_ids": ["person_alpha", "person_beta"],
                }
            ],
        }
    )


def _relation_catalog() -> RelationTypeCatalog:
    return RelationTypeCatalog.model_validate(
        {
            "relation_types": [
                {
                    "relation_type": "co_attended",
                    "label": "共同参会",
                    "object_kind": "person",
                    "extraction_policy": "allowed",
                    "requires_event": True,
                    "description": "必须绑定共同事件。",
                },
                {
                    "relation_type": "direct_subordinate",
                    "label": "直接下属",
                    "object_kind": "person",
                    "extraction_policy": "manual_only",
                    "requires_human_review": True,
                    "description": "只能人工确认。",
                },
            ]
        }
    )


def _evidence() -> EvidenceReference:
    return EvidenceReference(
        evidence_id="evidence_one",
        document_id="fixture_document",
        chunk_id="chunk-one",
        pdf_page_start=12,
        pdf_page_end=12,
        quote="甲某和乙某在同一次会议中分别发言并讨论工作。",
        extraction_methods=["text_layer"],
    )


def _insert_document(database: Database) -> None:
    database.initialize()
    with database.connect() as connection:
        connection.execute(
            """
            INSERT INTO documents (
                document_id, title, creators_json, source_type, edition, volume,
                source_series_json, verification_status, enabled, ocr_strategy,
                expected_page_count, notes, created_at, updated_at
            ) VALUES (
                'fixture_document', '测试史料', '[]', 'test', NULL, NULL,
                '[]', 'verified', 1, 'none', 20, NULL, 'now', 'now'
            )
            """
        )


def test_project_person_and_relation_catalogs_are_valid() -> None:
    root = Path.cwd()
    people = load_person_catalog(root / "config" / "person_aliases.json")
    relations = load_relation_type_catalog(root / "config" / "relation_types.json")

    assert people.alias_map()["毛泽东"] == ["毛主席", "润之"]
    assert len({person.person_id for person in people.people}) == len(people.people)
    assert {item.relation_type for item in relations.relation_types} >= {
        "held_position",
        "direct_subordinate",
        "co_attended",
    }


def test_database_migrates_existing_relationship_table(work_path: Path) -> None:
    database = Database(work_path / "history_agent.db")
    with database.connect() as connection:
        connection.execute(
            """
            CREATE TABLE person_relationships (
                relationship_id TEXT PRIMARY KEY,
                relation_type TEXT,
                subject_person_id TEXT,
                object_person_id TEXT
            )
            """
        )

    database.initialize()

    with database.connect() as connection:
        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(person_relationships)"
            ).fetchall()
        }
        version = connection.execute("PRAGMA user_version").fetchone()[0]
    assert {"subject_mention_text", "object_mention_text"}.issubset(columns)
    assert version == 3


def test_ambiguous_alias_is_not_forced_and_merge_is_audited(work_path: Path) -> None:
    database = Database(work_path / "history_agent.db")
    sync_person_catalog(database, _people_catalog())

    resolution = resolve_person(database, "同 名")
    assert resolution.status == "ambiguous"
    assert {candidate.person_id for candidate in resolution.candidates} == {
        "person_alpha",
        "person_beta",
    }
    assert resolve_person(database, "甲某").status == "resolved"

    proposal_id = propose_person_merge(
        database,
        source_person_id="person_beta",
        target_person_id="person_alpha",
        reason="人工确认两条记录为同一人物",
        proposed_by="tester",
    )
    with database.connect() as connection:
        status = connection.execute(
            "SELECT status FROM persons WHERE person_id = 'person_beta'"
        ).fetchone()["status"]
    assert status == "active"

    review_person_merge(
        database,
        proposal_id=proposal_id,
        decision="accepted",
        reviewed_by="reviewer",
        review_note="证据已复核",
    )
    with database.connect() as connection:
        person = connection.execute(
            "SELECT * FROM persons WHERE person_id = 'person_beta'"
        ).fetchone()
        proposal = connection.execute(
            "SELECT * FROM person_merge_proposals WHERE proposal_id = ?", (proposal_id,)
        ).fetchone()
    assert person["status"] == "merged"
    assert person["merged_into_person_id"] == "person_alpha"
    assert proposal["status"] == "accepted"
    assert proposal["reviewed_by"] == "reviewer"


def test_temporal_and_high_risk_relationship_validation() -> None:
    assert TemporalPoint(value="1956", precision="year", certainty="exact").value == "1956"
    with pytest.raises(ValidationError, match="YYYY-MM"):
        TemporalPoint(value="1956-13", precision="month", certainty="exact")
    with pytest.raises(ValidationError, match="shared event_id"):
        HistoricalRelationship(
            relationship_id="relation_one",
            relation_type="co_attended",
            subject_person_id="person_alpha",
            subject_mention_text="甲某",
            object_person_id="person_beta",
            object_mention_text="乙某",
            evidence=[_evidence()],
            extraction_method="rule",
            extraction_confidence=0.8,
        )
    with pytest.raises(ValidationError, match="human reviewer"):
        HistoricalRelationship(
            relationship_id="relation_two",
            relation_type="direct_subordinate",
            subject_person_id="person_alpha",
            subject_mention_text="甲某",
            object_person_id="person_beta",
            object_mention_text="乙某",
            evidence=[_evidence()],
            extraction_method="manual",
            extraction_confidence=1.0,
            review_status="confirmed",
        )


def test_event_and_relationship_require_shared_evidence(work_path: Path) -> None:
    database = Database(work_path / "history_agent.db")
    sync_person_catalog(database, _people_catalog())
    sync_relation_types(database, _relation_catalog())
    _insert_document(database)
    store = ResearchStore(database)
    event = HistoricalEvent(
        event_id="event_one",
        name="测试会议",
        event_type="meeting",
        start=TemporalPoint(
            value="1956-01", precision="month", certainty="exact", original_text="1956年1月"
        ),
        description="甲某和乙某参加同一次会议。",
        participants=[
            EventParticipant(person_id="person_alpha", role="发言", mention_text="甲某"),
            EventParticipant(person_id="person_beta", role="参会", mention_text="乙某"),
        ],
        evidence=[_evidence()],
        extraction_method="rule",
        extraction_confidence=0.9,
    )
    store.save_event(event)
    relationship = HistoricalRelationship(
        relationship_id="relationship_one",
        relation_type="co_attended",
        subject_person_id="person_alpha",
        subject_mention_text="甲某",
        object_person_id="person_beta",
        object_mention_text="乙某",
        event_id=event.event_id,
        start=event.start,
        evidence=[_evidence()],
        extraction_method="rule",
        extraction_confidence=0.9,
    )
    store.save_relationship(relationship)

    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM historical_events").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM event_participants").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM event_evidence").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM relationship_evidence").fetchone()[0] == 1

    invalid = relationship.model_copy(
        update={
            "relationship_id": "relationship_two",
            "relation_type": "direct_subordinate",
            "event_id": None,
            "review_status": "needs_review",
        }
    )
    with pytest.raises(ResearchDataError, match="manual-only"):
        store.save_relationship(invalid)


def test_database_rejects_evidence_for_unknown_document(work_path: Path) -> None:
    database = Database(work_path / "history_agent.db")
    sync_person_catalog(database, _people_catalog())
    event = HistoricalEvent(
        event_id="event_unknown_source",
        name="无来源事件",
        event_type="meeting",
        description="用于验证外键约束。",
        participants=[EventParticipant(person_id="person_alpha", mention_text="甲某")],
        evidence=[_evidence()],
        extraction_method="rule",
        extraction_confidence=0.5,
    )

    with pytest.raises(ResearchDataError, match="FOREIGN KEY"):
        ResearchStore(database).save_event(event)

    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM historical_events").fetchone()[0] == 0
