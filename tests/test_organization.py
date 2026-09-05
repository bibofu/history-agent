from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from history_agent.config import Settings
from history_agent.db import Database
from history_agent.research.models import (
    EventParticipant,
    EvidenceReference,
    HistoricalEvent,
    PersonCatalog,
    RelationTypeCatalog,
    TemporalPoint,
)
from history_agent.research.organization import (
    extract_organization_relationships,
    get_organization_relationships,
)
from history_agent.research.people import sync_person_catalog, sync_relation_types
from history_agent.research.store import ResearchStore
from history_agent.web.app import create_app


def _database(work_path: Path) -> Database:
    database = Database(work_path / "history_agent.db")
    sync_person_catalog(
        database,
        PersonCatalog.model_validate(
            {"people": [{"person_id": "zhou_enlai", "canonical_name": "周恩来"}]}
        ),
    )
    sync_relation_types(
        database,
        RelationTypeCatalog.model_validate_json(
            (Path.cwd() / "config" / "relation_types.json").read_text(encoding="utf-8")
        ),
    )
    with database.connect() as connection:
        connection.execute(
            """
            INSERT INTO documents (
                document_id, title, creators_json, source_type, edition, volume,
                source_series_json, verification_status, enabled, ocr_strategy,
                expected_page_count, notes, created_at, updated_at
            ) VALUES ('zhou_chronology', '周恩来年谱', '[]', 'chronology', NULL, '上卷',
                      '[]', 'verified', 1, 'none', 100, NULL, 'now', 'now')
            """
        )
    return database


def _save_appointment(database: Database) -> None:
    ResearchStore(database).save_event(
        HistoricalEvent(
            event_id="event_appointment",
            name="周恩来担任政务院总理",
            event_type="appointment",
            start=TemporalPoint(
                value="1949-10-01",
                precision="day",
                certainty="exact",
                original_text="一九四九年十月一日",
            ),
            description="周恩来被任命为政务院总理，并出席中央人民政府委员会会议。",
            participants=[
                EventParticipant(
                    person_id="zhou_enlai", mention_text="周恩来", role="年谱主体"
                )
            ],
            evidence=[
                EvidenceReference(
                    evidence_id="evidence_appointment",
                    document_id="zhou_chronology",
                    pdf_page_start=12,
                    pdf_page_end=12,
                    quote="周恩来被任命为政务院总理，并出席中央人民政府委员会会议。",
                    extraction_methods=["text_layer"],
                )
            ],
            extraction_method="rule",
            extraction_confidence=0.9,
        )
    )


def test_extract_and_query_organization_relationships(work_path: Path) -> None:
    database = _database(work_path)
    _save_appointment(database)

    first = extract_organization_relationships(database)
    second = extract_organization_relationships(database)

    assert first.events_scanned == 1
    assert first.created == 1
    assert second.skipped == 1
    result = get_organization_relationships(
        database, person_id="zhou_enlai", at="1949"
    )
    assert result.total == 1
    item = result.relationships[0]
    assert item.organization_name == "政务院"
    assert item.role_title == "总理"
    assert item.verification_level == "automatic_candidate"
    assert item.evidence[0].pdf_page_start == 12
    assert get_organization_relationships(
        database, person_id="zhou_enlai", at="1948"
    ).total == 0
    assert get_organization_relationships(
        database, person_id="zhou_enlai", at="1950"
    ).total == 0


def test_relationship_api_filters_research_range(work_path: Path) -> None:
    database = _database(work_path)
    _save_appointment(database)
    extract_organization_relationships(database)
    settings = Settings(
        _env_file=None,
        project_root=work_path,
        data_dir=work_path / "data",
        database_path=database.path,
    )
    client = TestClient(create_app(settings))

    response = client.get("/api/people/zhou_enlai/relationships", params={"at": "1949"})
    assert response.status_code == 200
    assert response.json()["relationships"][0]["organization_name"] == "政务院"
    assert client.get(
        "/api/people/zhou_enlai/relationships", params={"at": "1980"}
    ).status_code == 422
