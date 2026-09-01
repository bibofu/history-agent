from __future__ import annotations

import json
from pathlib import Path

from history_agent.db import Database
from history_agent.research.event_deduplication import (
    discover_event_merge_candidates,
    list_event_merge_review_queue,
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
                ) VALUES (?, ?, '[]', 'chronology', NULL, NULL, '[]', 'verified',
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
    description: str,
    certainty: str,
    page: int,
) -> None:
    subject_id = (
        "zhou_enlai"
        if document_id == "zhou_enlai_chronology_1949_1976"
        else "lin_biao"
    )
    other_id = "lin_biao" if subject_id == "zhou_enlai" else "zhou_enlai"
    subject_name = "周恩来" if subject_id == "zhou_enlai" else "林彪"
    other_name = "林彪" if other_id == "lin_biao" else "周恩来"
    event = HistoricalEvent.model_validate(
        {
            "event_id": event_id,
            "name": f"{subject_name}：{description}",
            "event_type": "correspondence",
            "start": {
                "value": "1943-01-21",
                "precision": "day",
                "certainty": certainty,
                "original_text": (
                    "1943年1月21日" if certainty == "exact" else "1月21日"
                ),
            },
            "description": description,
            "participants": [
                EventParticipant(
                    person_id=subject_id,
                    role="年谱主体",
                    mention_text=subject_name,
                    mention_source="explicit",
                ),
                EventParticipant(person_id=other_id, mention_text=other_name),
                EventParticipant(person_id="mao_zedong", mention_text="毛泽东"),
            ],
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
            "review_status": "unreviewed",
            "extractor_version": "chronology-rules-v1",
        }
    )
    ResearchStore(database).save_event(event)


def _save_duplicate_pair(database: Database) -> None:
    _save_event(
        database,
        event_id="event_zhou_message",
        document_id="zhou_enlai_chronology_1949_1976",
        description="周恩来、林彪致电毛泽东，说明谈判条件并提出两种解决办法。",
        certainty="exact",
        page=20,
    )
    _save_event(
        database,
        event_id="event_lin_message",
        document_id="lin_biao_chronology",
        description="林彪、周恩来致电毛泽东，说明谈判条件，提出两种解决办法。",
        certainty="inferred",
        page=30,
    )


def test_high_confidence_merge_preserves_sources_and_is_idempotent(
    work_path: Path,
) -> None:
    database = _prepare_database(work_path)
    _save_duplicate_pair(database)

    summary = merge_duplicate_events(
        database=database,
        reports_dir=work_path / "reports",
        run_id="merge-one",
        minimum_score=0.65,
        automatic_score=0.75,
    )

    assert summary.candidate_groups == 1
    assert summary.high_confidence_groups == 1
    assert summary.created == 1
    with database.connect() as connection:
        canonical = connection.execute("SELECT * FROM canonical_events").fetchone()
        member_count = connection.execute(
            "SELECT COUNT(*) FROM canonical_event_members"
        ).fetchone()[0]
        evidence_count = connection.execute(
            "SELECT COUNT(*) FROM canonical_event_evidence"
        ).fetchone()[0]
        source_count = connection.execute(
            "SELECT COUNT(*) FROM historical_events"
        ).fetchone()[0]
    assert canonical["candidate_kind"] == "high_confidence"
    assert canonical["review_status"] == "unreviewed"
    assert member_count == 2
    assert evidence_count == 2
    assert source_count == 2
    assert list_event_merge_review_queue(database) == []

    rerun = merge_duplicate_events(
        database=database,
        reports_dir=work_path / "reports",
        run_id="merge-two",
        minimum_score=0.65,
        automatic_score=0.75,
    )
    assert rerun.created == 0
    assert rerun.skipped == 1

    canonical_id = str(canonical["canonical_event_id"])
    review_event_merge(
        database,
        canonical_event_id=canonical_id,
        decision="confirmed",
        reviewed_by="tester",
    )
    protected_rerun = merge_duplicate_events(
        database=database,
        reports_dir=work_path / "reports",
        run_id="merge-three",
        minimum_score=0.65,
        automatic_score=0.75,
    )
    assert protected_rerun.protected == 1
    assert protected_rerun.skipped == 0
    review_event_merge(
        database,
        canonical_event_id=canonical_id,
        decision="reopened",
        reviewed_by="tester",
    )
    assert len(list_event_merge_review_queue(database)) == 1


def test_uncertain_merge_keeps_variants_and_review_is_reversible(
    work_path: Path,
) -> None:
    database = _prepare_database(work_path)
    _save_duplicate_pair(database)
    summary = merge_duplicate_events(
        database=database,
        reports_dir=work_path / "reports",
        run_id="merge-review",
        minimum_score=0.65,
        automatic_score=0.99,
    )
    assert summary.uncertain_groups == 1
    queue = list_event_merge_review_queue(database)
    assert len(queue) == 1
    canonical_id = str(queue[0]["canonical_event_id"])
    assert {"name", "start", "description_sha256"}.issubset(
        queue[0]["variant_fields"]
    )

    review_event_merge(
        database,
        canonical_event_id=canonical_id,
        decision="confirmed",
        reviewed_by="tester",
        note="两份年谱记载同一电报",
    )
    assert list_event_merge_review_queue(database) == []
    review_event_merge(
        database,
        canonical_event_id=canonical_id,
        decision="reopened",
        reviewed_by="tester",
        note="重新核对日期表述",
    )

    with database.connect() as connection:
        canonical = connection.execute(
            "SELECT * FROM canonical_events WHERE canonical_event_id = ?",
            (canonical_id,),
        ).fetchone()
        reviews = connection.execute(
            "SELECT decision FROM canonical_event_reviews ORDER BY reviewed_at"
        ).fetchall()
        snapshots = connection.execute(
            "SELECT source_snapshot_json FROM canonical_event_members"
        ).fetchall()
        originals = connection.execute(
            "SELECT COUNT(*) FROM historical_events"
        ).fetchone()[0]
    assert canonical["review_status"] == "needs_review"
    assert [row["decision"] for row in reviews] == ["confirmed", "reopened"]
    assert len(list_event_merge_review_queue(database)) == 1
    assert originals == 2
    assert all(json.loads(row["source_snapshot_json"])["evidence"] for row in snapshots)


def test_same_day_unrelated_events_are_not_merged(work_path: Path) -> None:
    database = _prepare_database(work_path)
    _save_event(
        database,
        event_id="event_unrelated_zhou",
        document_id="zhou_enlai_chronology_1949_1976",
        description="周恩来接见代表团并讨论经济建设问题。",
        certainty="exact",
        page=40,
    )
    _save_event(
        database,
        event_id="event_unrelated_lin",
        document_id="lin_biao_chronology",
        description="林彪致电前线部队调整军事部署。",
        certainty="exact",
        page=50,
    )

    candidates, source_count, compared = discover_event_merge_candidates(database)

    assert source_count == 2
    assert compared == 1
    assert candidates == []
