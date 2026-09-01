from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from history_agent.config import Settings
from history_agent.db import Database
from history_agent.errors import ResearchDataError
from history_agent.research.event_deduplication import (
    merge_duplicate_events,
    review_event_merge,
)
from history_agent.research.models import (
    EventParticipant,
    EvidenceReference,
    HistoricalEvent,
    PersonCatalog,
)
from history_agent.research.people import sync_person_catalog
from history_agent.research.store import ResearchStore
from history_agent.research.timeline import get_person_timeline
from history_agent.web.app import create_app


def _prepare_database(work_path: Path) -> Database:
    database = Database(work_path / "history_agent.db")
    sync_person_catalog(
        database,
        PersonCatalog.model_validate(
            {
                "people": [
                    {"person_id": "zhou_enlai", "canonical_name": "周恩来"},
                    {"person_id": "lin_biao", "canonical_name": "林彪"},
                    {"person_id": "mao_zedong", "canonical_name": "毛泽东"},
                ]
            }
        ),
    )
    with database.connect() as connection:
        for document_id, title in (
            ("zhou_enlai_chronology_1949_1976", "周恩来年谱"),
            ("lin_biao_chronology", "林彪年谱"),
        ):
            connection.execute(
                """
                INSERT INTO documents (
                    document_id, title, creators_json, source_type, edition, volume,
                    source_series_json, verification_status, enabled, ocr_strategy,
                    expected_page_count, notes, created_at, updated_at
                ) VALUES (?, ?, '[]', 'chronology', NULL, '第一卷', '[]', 'verified',
                          1, 'none', 100, NULL, 'now', 'now')
                """,
                (document_id, title),
            )
    return database


def _save_event(
    database: Database,
    *,
    event_id: str,
    document_id: str,
    date_value: str,
    description: str,
    event_type: str,
    review_status: str,
    page: int,
    include_lin: bool,
) -> None:
    participants = [
        EventParticipant(
            person_id="zhou_enlai",
            role="年谱主体",
            mention_text="周恩来",
            mention_source="chronology_subject",
        ),
        EventParticipant(person_id="mao_zedong", mention_text="毛泽东"),
    ]
    if include_lin:
        participants.append(
            EventParticipant(person_id="lin_biao", mention_text="林彪")
        )
    event = HistoricalEvent.model_validate(
        {
            "event_id": event_id,
            "name": f"周恩来：{description}",
            "event_type": event_type,
            "start": {
                "value": date_value,
                "precision": "day",
                "certainty": (
                    "inferred" if document_id == "lin_biao_chronology" else "exact"
                ),
                "original_text": date_value,
            },
            "description": description,
            "participants": participants,
            "evidence": [
                EvidenceReference(
                    evidence_id=f"evidence_{event_id}",
                    document_id=document_id,
                    pdf_page_start=page,
                    pdf_page_end=page,
                    quote=description,
                    extraction_methods=["text_layer"],
                )
            ],
            "extraction_method": "rule",
            "extraction_confidence": 0.9,
            "review_status": review_status,
            "extractor_version": "test-rules-v1",
        }
    )
    ResearchStore(database).save_event(event)


def _prepare_timeline(work_path: Path) -> tuple[Database, str]:
    database = _prepare_database(work_path)
    _save_event(
        database,
        event_id="event_zhou_message",
        document_id="zhou_enlai_chronology_1949_1976",
        date_value="1943-01-21",
        description="周恩来、林彪致电毛泽东，说明谈判条件并提出两种解决办法。",
        event_type="correspondence",
        review_status="confirmed",
        page=20,
        include_lin=True,
    )
    _save_event(
        database,
        event_id="event_lin_message",
        document_id="lin_biao_chronology",
        date_value="1943-01-21",
        description="林彪、周恩来致电毛泽东，说明谈判条件，提出两种解决办法。",
        event_type="activity",
        review_status="unreviewed",
        page=30,
        include_lin=True,
    )
    _save_event(
        database,
        event_id="event_standalone_meeting",
        document_id="zhou_enlai_chronology_1949_1976",
        date_value="1943-02-01",
        description="周恩来出席会议并报告近期工作安排。",
        event_type="meeting",
        review_status="confirmed",
        page=21,
        include_lin=False,
    )
    summary = merge_duplicate_events(
        database=database,
        reports_dir=work_path / "reports",
        run_id="timeline-merge",
        minimum_score=0.65,
        automatic_score=0.75,
    )
    assert summary.candidate_groups == 1
    with database.connect() as connection:
        canonical_id = str(
            connection.execute(
                "SELECT canonical_event_id FROM canonical_events"
            ).fetchone()[0]
        )
    return database, canonical_id


def test_timeline_combines_canonical_and_unmerged_events(work_path: Path) -> None:
    database, canonical_id = _prepare_timeline(work_path)

    timeline = get_person_timeline(
        database,
        person_id="zhou_enlai",
        start_year=1943,
        end_year=1943,
        limit=1,
    )

    assert timeline.total == 2
    assert timeline.has_more is True
    assert timeline.events[0].event_id == canonical_id
    assert timeline.events[0].record_kind == "canonical"
    assert timeline.events[0].verification_level == "automatic"
    assert timeline.events[0].source_event_ids == [
        "event_lin_message",
        "event_zhou_message",
    ]
    assert len(timeline.events[0].evidence) == 2
    assert {item.document_title for item in timeline.events[0].evidence} == {
        "周恩来年谱",
        "林彪年谱",
    }

    meetings = get_person_timeline(
        database,
        person_id="zhou_enlai",
        start_year=1943,
        end_year=1943,
        event_types=["meeting"],
        review_statuses=["confirmed"],
    )
    assert meetings.total == 1
    assert meetings.events[0].event_id == "event_standalone_meeting"
    assert meetings.events[0].record_kind == "source"
    assert meetings.events[0].verification_level == "confirmed"

    type_variant = get_person_timeline(
        database,
        person_id="zhou_enlai",
        start_year=1943,
        end_year=1943,
        event_types=["activity"],
    )
    assert type_variant.total == 1
    assert type_variant.events[0].event_id == "event_lin_message"
    assert type_variant.events[0].record_kind == "source"

    confirmed_only = get_person_timeline(
        database,
        person_id="zhou_enlai",
        start_year=1943,
        end_year=1943,
        review_statuses=["confirmed"],
    )
    assert {item.event_id for item in confirmed_only.events} == {
        "event_zhou_message",
        "event_standalone_meeting",
    }


def test_rejected_canonical_merge_restores_source_events(work_path: Path) -> None:
    database, canonical_id = _prepare_timeline(work_path)
    review_event_merge(
        database,
        canonical_event_id=canonical_id,
        decision="rejected",
        reviewed_by="tester",
        note="人工判断不是同一事件",
    )

    timeline = get_person_timeline(
        database,
        person_id="zhou_enlai",
        start_year=1943,
        end_year=1943,
    )

    assert timeline.total == 3
    assert {item.record_kind for item in timeline.events} == {"source"}
    assert canonical_id not in {item.event_id for item in timeline.events}


def test_timeline_rejects_invalid_person_and_status(work_path: Path) -> None:
    database, _ = _prepare_timeline(work_path)

    with pytest.raises(ResearchDataError, match="unknown person_id"):
        get_person_timeline(database, person_id="unknown_person")
    with pytest.raises(ResearchDataError, match="unsupported timeline review status"):
        get_person_timeline(
            database,
            person_id="zhou_enlai",
            review_statuses=["rejected"],
        )


def test_timeline_api_contract_and_research_range(work_path: Path) -> None:
    database, canonical_id = _prepare_timeline(work_path)
    settings = Settings(
        _env_file=None,
        project_root=work_path,
        data_dir=work_path / "data",
        database_path=database.path,
    )
    client = TestClient(create_app(settings))

    response = client.get(
        "/api/people/zhou_enlai/timeline",
        params={"start_year": 1943, "end_year": 1943, "limit": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["person_id"] == "zhou_enlai"
    assert payload["total"] == 2
    assert payload["events"][0]["event_id"] == canonical_id
    assert payload["events"][0]["evidence"][0]["pdf_page_start"] >= 1
    assert client.get(
        "/api/people/zhou_enlai/timeline", params={"start_year": 1900}
    ).status_code == 422
    assert client.get("/api/people/unknown_person/timeline").status_code == 404
