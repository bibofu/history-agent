from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
from history_agent.config import Settings
from history_agent.db import Database
from history_agent.research.model_extraction import (
    EXTRACTOR_VERSION,
    PROMPT_VERSION,
    enrich_events_with_model,
    list_review_queue,
    select_event_candidates,
)
from history_agent.research.models import (
    EventParticipant,
    EvidenceReference,
    HistoricalEvent,
    PersonCatalog,
    ReviewStatus,
    TemporalPoint,
)
from history_agent.research.people import sync_person_catalog
from history_agent.research.store import ResearchStore


def _catalog() -> PersonCatalog:
    return PersonCatalog.model_validate(
        {
            "people": [
                {
                    "person_id": "zhou_enlai",
                    "canonical_name": "周恩来",
                    "aliases": [{"name": "周总理"}],
                },
                {
                    "person_id": "mao_zedong",
                    "canonical_name": "毛泽东",
                    "aliases": [],
                },
            ]
        }
    )


def _settings(work_path: Path) -> Settings:
    return Settings(
        project_root=work_path,
        database_path=work_path / "history_agent.db",
        data_dir=work_path / "data",
        llm_api_key="test-key",
        llm_model="deepseek-v4-pro",
    )


def _prepare_event(
    database: Database, *, review_status: ReviewStatus = "needs_review"
) -> None:
    catalog = _catalog()
    sync_person_catalog(database, catalog)
    with database.connect() as connection:
        connection.execute(
            """
            INSERT INTO documents (
                document_id, title, creators_json, source_type, edition, volume,
                source_series_json, verification_status, enabled, ocr_strategy,
                expected_page_count, notes, created_at, updated_at
            ) VALUES (
                'zhou_chronology', '周恩来年谱', '[]', 'chronology', NULL, NULL,
                '[]', 'verified', 1, 'none', 20, NULL, 'now', 'now'
            )
            """
        )
    event = HistoricalEvent(
        event_id="event_model_one",
        name="周恩来：出席会议",
        event_type="activity",
        start=TemporalPoint(
            value="1956-01-15",
            precision="day",
            certainty="exact",
            original_text="1月15日",
        ),
        description="1月15日，在北京出席中央政治局会议，毛泽东主持会议。",
        participants=[
            EventParticipant(
                person_id="zhou_enlai",
                role="年谱主体",
                mention_text="周恩来",
                mention_source="chronology_subject",
            )
        ],
        evidence=[
            EvidenceReference(
                evidence_id="evidence_model_one",
                document_id="zhou_chronology",
                pdf_page_start=12,
                pdf_page_end=12,
                quote="1月15日，在北京出席中央政治局会议，毛泽东主持会议。",
                extraction_methods=["text_layer"],
            )
        ],
        extraction_method="rule",
        extraction_confidence=0.62,
        review_status=review_status,
        extractor_version="chronology-rules-v1",
    )
    ResearchStore(database).save_event(event)


class _Response:
    def __init__(self, payload: dict[str, object], status_code: int = 200):
        self.payload = payload
        self.status_code = status_code
        self.request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            response = httpx.Response(self.status_code, request=self.request)
            raise httpx.HTTPStatusError("request failed", request=self.request, response=response)

    def json(self) -> dict[str, object]:
        return self.payload


def _api_payload(content: dict[str, object]) -> dict[str, object]:
    return {
        "choices": [
            {
                "message": {"content": json.dumps(content, ensure_ascii=False)},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140},
    }


def _valid_content() -> dict[str, object]:
    return {
        "event_type": "meeting",
        "action_text": "出席中央政治局会议",
        "location_text": "北京",
        "organization_names": ["中央政治局"],
        "participants": [
            {"mention_text": "毛泽东", "role_text": "主持"},
        ],
        "confidence": 0.88,
        "needs_review": False,
        "review_reasons": [],
    }


def test_model_enrichment_is_grounded_audited_and_idempotent(
    work_path: Path, monkeypatch: Any
) -> None:
    settings = _settings(work_path)
    database = Database(settings.database_path)
    _prepare_event(database)
    captured: dict[str, object] = {}

    def fake_post(*args: Any, **kwargs: Any) -> _Response:
        captured.update(kwargs)
        return _Response(_api_payload(_valid_content()))

    monkeypatch.setattr("history_agent.research.model_extraction.httpx.post", fake_post)
    summary = enrich_events_with_model(
        database=database,
        settings=settings,
        catalog=_catalog(),
        reports_dir=settings.reports_dir,
        run_id="run-model-one",
        limit=5,
    )

    assert summary.succeeded == 1
    assert summary.invalid == 0
    assert summary.total_tokens == 140
    request_body = captured["json"]
    assert isinstance(request_body, dict)
    assert request_body["response_format"] == {"type": "json_object"}
    assert request_body["thinking"] == {"type": "disabled"}
    with database.connect() as connection:
        event = connection.execute(
            "SELECT * FROM historical_events WHERE event_id = 'event_model_one'"
        ).fetchone()
        attempt = connection.execute(
            "SELECT * FROM event_extraction_attempts"
        ).fetchone()
        evidence_count = connection.execute(
            "SELECT COUNT(*) FROM event_evidence WHERE event_id = 'event_model_one'"
        ).fetchone()[0]
        people = connection.execute(
            "SELECT person_id, mention_source FROM event_participants "
            "WHERE event_id = 'event_model_one' ORDER BY person_id"
        ).fetchall()
    assert event["start_value"] == "1956-01-15"
    assert event["description"] == "1月15日，在北京出席中央政治局会议，毛泽东主持会议。"
    assert event["event_type"] == "meeting"
    assert event["location_text"] == "北京"
    assert event["extraction_method"] == "rule_llm"
    assert event["extractor_version"] == EXTRACTOR_VERSION
    assert event["review_status"] == "needs_review"
    assert attempt["model_name"] == "deepseek-v4-pro"
    assert attempt["prompt_version"] == PROMPT_VERSION
    assert attempt["status"] == "succeeded"
    assert attempt["before_event_json"]
    assert attempt["merged_event_json"]
    assert evidence_count == 1
    assert [(row["person_id"], row["mention_source"]) for row in people] == [
        ("mao_zedong", "explicit"),
        ("zhou_enlai", "chronology_subject"),
    ]
    queue = list_review_queue(database)
    assert len(queue) == 1
    assert queue[0]["event_id"] == "event_model_one"

    candidates, skipped = select_event_candidates(
        database, catalog=_catalog(), model_name=settings.llm_model, limit=5
    )
    assert candidates == []
    assert skipped == 1


def test_ungrounded_model_output_is_rejected_without_changing_event(
    work_path: Path, monkeypatch: Any
) -> None:
    settings = _settings(work_path)
    database = Database(settings.database_path)
    _prepare_event(database)
    content = _valid_content()
    content["location_text"] = "上海"
    monkeypatch.setattr(
        "history_agent.research.model_extraction.httpx.post",
        lambda *args, **kwargs: _Response(_api_payload(content)),
    )

    summary = enrich_events_with_model(
        database=database,
        settings=settings,
        catalog=_catalog(),
        reports_dir=settings.reports_dir,
        run_id="run-model-invalid",
    )

    assert summary.invalid == 1
    with database.connect() as connection:
        event = connection.execute(
            "SELECT * FROM historical_events WHERE event_id = 'event_model_one'"
        ).fetchone()
        attempt = connection.execute(
            "SELECT * FROM event_extraction_attempts"
        ).fetchone()
    assert event["location_text"] is None
    assert event["extraction_method"] == "rule"
    assert attempt["status"] == "invalid"
    assert attempt["error_code"] == "ungrounded_location_text"
    assert attempt["validated_json"]
    assert attempt["merged_event_json"] is None
    assert list_review_queue(database)[0]["priority"] == 90


def test_invalid_json_and_api_failure_are_audited(work_path: Path, monkeypatch: Any) -> None:
    settings = _settings(work_path)
    database = Database(settings.database_path)
    _prepare_event(database)
    monkeypatch.setattr(
        "history_agent.research.model_extraction.httpx.post",
        lambda *args, **kwargs: _Response(
            {
                "choices": [{"message": {"content": "not-json"}, "finish_reason": "stop"}],
                "usage": {},
            }
        ),
    )
    invalid = enrich_events_with_model(
        database=database,
        settings=settings,
        catalog=_catalog(),
        reports_dir=settings.reports_dir,
        run_id="run-invalid-json",
    )
    assert invalid.invalid == 1

    def fail_post(*args: Any, **kwargs: Any) -> _Response:
        raise httpx.TimeoutException("timeout")

    monkeypatch.setattr("history_agent.research.model_extraction.httpx.post", fail_post)
    failed = enrich_events_with_model(
        database=database,
        settings=settings,
        catalog=_catalog(),
        reports_dir=settings.reports_dir,
        run_id="run-timeout",
        retry_failed=True,
    )
    assert failed.failed == 1
    with database.connect() as connection:
        rows = connection.execute(
            "SELECT status, error_code FROM event_extraction_attempts ORDER BY created_at"
        ).fetchall()
    assert [(row["status"], row["error_code"]) for row in rows] == [
        ("invalid", "schema_validation_failed"),
        ("failed", "timeout"),
    ]


def test_dry_run_does_not_call_model_or_write_attempt(work_path: Path, monkeypatch: Any) -> None:
    settings = _settings(work_path)
    database = Database(settings.database_path)
    _prepare_event(database)

    def unexpected_post(*args: Any, **kwargs: Any) -> _Response:
        raise AssertionError("DeepSeek should not be called")

    monkeypatch.setattr(
        "history_agent.research.model_extraction.httpx.post", unexpected_post
    )
    summary = enrich_events_with_model(
        database=database,
        settings=settings,
        catalog=_catalog(),
        reports_dir=settings.reports_dir,
        run_id="run-dry",
        dry_run=True,
    )
    assert summary.selected == 1
    assert summary.attempt_ids == []
    with database.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM event_extraction_attempts"
        ).fetchone()[0] == 0


def test_confirmed_event_is_never_selected_even_when_explicit(work_path: Path) -> None:
    settings = _settings(work_path)
    database = Database(settings.database_path)
    _prepare_event(database, review_status="confirmed")

    candidates, skipped = select_event_candidates(
        database,
        catalog=_catalog(),
        model_name=settings.llm_model,
        limit=5,
        event_ids=["event_model_one"],
    )

    assert candidates == []
    assert skipped == 0


def test_unknown_person_mention_is_rejected(work_path: Path, monkeypatch: Any) -> None:
    settings = _settings(work_path)
    database = Database(settings.database_path)
    _prepare_event(database)
    content = _valid_content()
    content["participants"] = [
        {"mention_text": "中央政治局", "role_text": "出席"}
    ]
    monkeypatch.setattr(
        "history_agent.research.model_extraction.httpx.post",
        lambda *args, **kwargs: _Response(_api_payload(content)),
    )

    summary = enrich_events_with_model(
        database=database,
        settings=settings,
        catalog=_catalog(),
        reports_dir=settings.reports_dir,
        run_id="run-unknown-person",
    )

    assert summary.invalid == 1
    with database.connect() as connection:
        attempt = connection.execute(
            "SELECT * FROM event_extraction_attempts"
        ).fetchone()
    assert attempt["error_code"] == "unknown_participant"
    assert attempt["validated_json"]
