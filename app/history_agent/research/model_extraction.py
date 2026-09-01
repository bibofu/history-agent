from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from history_agent.config import Settings
from history_agent.db import Database
from history_agent.errors import ResearchDataError
from history_agent.research.models import (
    EventParticipant,
    HistoricalEvent,
    PersonCatalog,
)
from history_agent.research.people import normalize_person_name, utc_now

PROMPT_VERSION = "event-extraction-json-v1"
EXTRACTOR_VERSION = "chronology-rules-v1+deepseek-event-v1"
PROVIDER = "deepseek"
WHITESPACE = re.compile(r"\s+")
CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

EventType = Literal[
    "activity",
    "appointment",
    "correspondence",
    "meeting",
    "military",
    "speech",
    "visit",
]
ReviewReason = Literal[
    "ambiguous_location",
    "ambiguous_participant",
    "compound_event",
    "source_text_error",
    "other",
]
OrganizationName = Annotated[str, Field(min_length=2, max_length=80)]

SYSTEM_PROMPT = """你是中国近现代史史料的结构化抽取器。只能分析用户给出的单条本地证据，\
不得调用模型记忆，不得补写日期、人物、地点、机构或行动。输出必须是一个 JSON 对象，不要输出\
Markdown 或解释。action_text、location_text、organization_names、participants 中的 mention_text 与\
role_text 都必须是证据中的连续原文（仅允许忽略排版空白差异）。无法确认时使用 null、空数组或\
needs_review=true，绝不能猜测。event_type 只能从 activity、appointment、correspondence、meeting、\
military、speech、visit 中选择。示例：\
{"event_type":"meeting","action_text":"出席中央政治局会议","location_text":null,\
"organization_names":["中央政治局"],"participants":[{"mention_text":"周恩来",\
"role_text":"出席"}],"confidence":0.86,"needs_review":false,"review_reasons":[]}"""


class ModelParticipant(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    mention_text: str = Field(min_length=1, max_length=40)
    role_text: str | None = Field(default=None, min_length=1, max_length=40)


class ModelEventExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    event_type: EventType
    action_text: str = Field(min_length=4, max_length=100)
    location_text: str | None = Field(default=None, min_length=1, max_length=80)
    organization_names: list[OrganizationName] = Field(default_factory=list, max_length=12)
    participants: list[ModelParticipant] = Field(default_factory=list, max_length=20)
    confidence: float = Field(ge=0, le=1)
    needs_review: bool
    review_reasons: list[ReviewReason] = Field(default_factory=list, max_length=5)


class ModelExtractionSummary(BaseModel):
    run_id: str
    provider: str
    model_name: str
    prompt_version: str
    extractor_version: str
    dry_run: bool
    selected: int
    succeeded: int = 0
    invalid: int = 0
    failed: int = 0
    skipped_prior_attempt: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    event_ids: list[str] = Field(default_factory=list)
    attempt_ids: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class EventCandidate:
    event: HistoricalEvent
    updated_at: str
    request_sha256: str
    messages: list[dict[str, str]]


@dataclass(frozen=True)
class DeepSeekResult:
    content: str | None
    response_json: str | None
    error_code: str | None
    usage: dict[str, int]


class ModelOutputError(ValueError):
    """Raised when syntactically valid model output is not grounded in evidence."""


def _compact(value: str) -> str:
    return WHITESPACE.sub(" ", value).strip()


def _grounding_form(value: str) -> str:
    return WHITESPACE.sub("", value)


def _is_grounded(value: str, evidence_text: str) -> bool:
    return _grounding_form(value) in _grounding_form(evidence_text)


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:32]
    return f"{prefix}_{digest}"


def _request_messages(
    event: HistoricalEvent, catalog: PersonCatalog
) -> list[dict[str, str]]:
    evidence = "\n\n".join(
        f"证据{index}（{item.document_id}，PDF第{item.pdf_page_start}页）：\n"
        f"{_compact(item.quote)}"
        for index, item in enumerate(event.evidence, start=1)
    )
    evidence_text = "\n".join(item.quote for item in event.evidence)
    allowed_people = list(
        dict.fromkeys(
            name
            for person in catalog.people
            for name in [person.canonical_name, *(alias.name for alias in person.aliases)]
            if _is_grounded(name, evidence_text)
        )
    )
    allowed = "、".join(allowed_people) if allowed_people else "无"
    user = (
        f"事件ID：{event.event_id}\n"
        f"规则日期原文：{event.start.original_text or '未识别'}\n"
        f"规则事件类型：{event.event_type}\n"
        f"participants 只可使用下列已登记且在证据中出现的原文人名：{allowed}。"
        "清单为‘无’时必须返回空数组。\n"
        f"请只根据下列证据返回一个符合示例字段的 JSON 对象。不要返回日期字段。\n\n"
        f"{evidence}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _request_hash(event: HistoricalEvent) -> str:
    immutable_input = {
        "event_id": event.event_id,
        "start": event.start.model_dump(),
        "end": event.end.model_dump() if event.end else None,
        "evidence": [item.model_dump() for item in event.evidence],
    }
    rendered = json.dumps(immutable_input, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _deepseek_error_code(exc: httpx.HTTPError) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return {
            401: "authentication_failed",
            402: "insufficient_balance",
            429: "rate_limited",
        }.get(exc.response.status_code, f"http_{exc.response.status_code}")
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    return "network_error"


def _call_deepseek_json(settings: Settings, messages: list[dict[str, str]]) -> DeepSeekResult:
    if not settings.llm_enabled:
        return DeepSeekResult(None, None, "not_configured", {})
    assert settings.llm_api_key is not None
    try:
        response = httpx.post(
            settings.llm_base_url.rstrip("/") + "/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.llm_api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.llm_model,
                "messages": messages,
                "response_format": {"type": "json_object"},
                "stream": False,
                "max_tokens": min(settings.llm_max_tokens, 1200),
                "thinking": {"type": "disabled"},
            },
            timeout=settings.llm_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        serialized = json.dumps(payload, ensure_ascii=False)
        choice = payload["choices"][0]
        content = str(choice["message"]["content"]).strip()
        raw_usage = payload.get("usage", {})
        usage = {
            key: int(raw_usage[key])
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if key in raw_usage
        }
        if choice.get("finish_reason") == "length":
            return DeepSeekResult(None, serialized, "max_tokens_exhausted", usage)
        if not content:
            return DeepSeekResult(None, serialized, "empty_response", usage)
        return DeepSeekResult(content, serialized, None, usage)
    except httpx.HTTPError as exc:
        return DeepSeekResult(None, None, _deepseek_error_code(exc), {})
    except (KeyError, IndexError, TypeError, ValueError):
        return DeepSeekResult(None, None, "invalid_response", {})


def _parse_model_output(content: str) -> ModelEventExtraction:
    stripped = CODE_FENCE.sub("", content.strip())
    try:
        return ModelEventExtraction.model_validate_json(stripped)
    except ValidationError as exc:
        raise ModelOutputError(f"schema_validation_failed: {exc}") from exc


def _person_lookup(catalog: PersonCatalog) -> dict[str, list[tuple[str, str]]]:
    lookup: dict[str, list[tuple[str, str]]] = {}
    for person in catalog.people:
        for name in [person.canonical_name, *(alias.name for alias in person.aliases)]:
            lookup.setdefault(normalize_person_name(name), []).append(
                (person.person_id, person.canonical_name)
            )
    return lookup


def _validate_and_resolve(
    output: ModelEventExtraction,
    event: HistoricalEvent,
    catalog: PersonCatalog,
) -> list[EventParticipant]:
    evidence_text = "\n".join(item.quote for item in event.evidence)
    grounded_values: list[tuple[str, str]] = [("action_text", output.action_text)]
    if output.location_text:
        grounded_values.append(("location_text", output.location_text))
    grounded_values.extend(("organization_name", item) for item in output.organization_names)
    for participant in output.participants:
        grounded_values.append(("participant", participant.mention_text))
        if participant.role_text:
            grounded_values.append(("participant_role", participant.role_text))
    for field_name, value in grounded_values:
        if not _is_grounded(value, evidence_text):
            raise ModelOutputError(f"ungrounded_{field_name}: {value}")

    lookup = _person_lookup(catalog)
    resolved: list[EventParticipant] = []
    for participant in output.participants:
        candidates = lookup.get(normalize_person_name(participant.mention_text), [])
        unique = {person_id: canonical for person_id, canonical in candidates}
        if len(unique) != 1:
            error = "unknown" if not unique else "ambiguous"
            raise ModelOutputError(f"{error}_participant: {participant.mention_text}")
        person_id = next(iter(unique))
        resolved.append(
            EventParticipant(
                person_id=person_id,
                role=participant.role_text,
                mention_text=participant.mention_text,
                mention_source="explicit",
            )
        )
    return resolved


def _merge_event(
    source: HistoricalEvent,
    output: ModelEventExtraction,
    resolved_participants: list[EventParticipant],
) -> HistoricalEvent:
    participants: dict[tuple[str, str], EventParticipant] = {
        (item.person_id, item.mention_text): item for item in source.participants
    }
    for item in resolved_participants:
        participants[(item.person_id, item.mention_text)] = item
    subject = next(
        (
            item.mention_text
            for item in source.participants
            if item.mention_source == "chronology_subject"
        ),
        source.participants[0].mention_text,
    )
    organizations = list(
        dict.fromkeys([*source.organization_names, *output.organization_names])
    )
    confidence = min(
        0.9,
        round((source.extraction_confidence + output.confidence) / 2, 4),
    )
    return source.model_copy(
        update={
            "name": f"{subject}：{output.action_text}",
            "event_type": output.event_type,
            "location_text": output.location_text or source.location_text,
            "organization_names": organizations,
            "participants": list(participants.values()),
            "extraction_method": "rule_llm",
            "extraction_confidence": confidence,
            "review_status": "needs_review",
            "extractor_version": EXTRACTOR_VERSION,
        }
    )


def _event_from_connection(
    connection: sqlite3.Connection, event_id: str
) -> tuple[HistoricalEvent, str]:
    row = connection.execute(
        "SELECT * FROM historical_events WHERE event_id = ?", (event_id,)
    ).fetchone()
    if row is None:
        raise ResearchDataError(f"unknown event_id: {event_id}")
    participants = connection.execute(
        "SELECT * FROM event_participants WHERE event_id = ? ORDER BY person_id, mention_text",
        (event_id,),
    ).fetchall()
    evidence_rows = connection.execute(
        """
        SELECT evidence_records.*
        FROM event_evidence
        JOIN evidence_records USING (evidence_id)
        WHERE event_evidence.event_id = ?
        ORDER BY pdf_page_start, evidence_id
        """,
        (event_id,),
    ).fetchall()
    payload: dict[str, Any] = {
        "event_id": row["event_id"],
        "name": row["name"],
        "event_type": row["event_type"],
        "start": {
            "value": row["start_value"],
            "precision": row["start_precision"],
            "certainty": row["start_certainty"],
            "original_text": row["start_original_text"],
        },
        "end": (
            None
            if row["end_precision"] is None
            else {
                "value": row["end_value"],
                "precision": row["end_precision"],
                "certainty": row["end_certainty"],
                "original_text": row["end_original_text"],
            }
        ),
        "location_text": row["location_text"],
        "organization_names": json.loads(row["organization_names_json"]),
        "description": row["description"],
        "participants": [
            {
                "person_id": item["person_id"],
                "role": item["role"],
                "mention_text": item["mention_text"],
                "mention_source": item["mention_source"],
            }
            for item in participants
        ],
        "evidence": [
            {
                "evidence_id": item["evidence_id"],
                "document_id": item["document_id"],
                "chunk_id": item["chunk_id"],
                "pdf_page_start": item["pdf_page_start"],
                "pdf_page_end": item["pdf_page_end"],
                "quote": item["quote"],
                "extraction_methods": json.loads(item["extraction_methods_json"]),
            }
            for item in evidence_rows
        ],
        "extraction_method": row["extraction_method"],
        "extraction_confidence": row["extraction_confidence"],
        "review_status": row["review_status"],
        "extractor_version": row["extractor_version"],
    }
    return HistoricalEvent.model_validate(payload), str(row["updated_at"])


def _candidate_event_ids(
    connection: sqlite3.Connection,
    *,
    document_ids: list[str] | None,
    event_ids: list[str] | None,
) -> list[str]:
    clauses = [
        "(e.extraction_method = 'rule' OR "
        "(e.extraction_method = 'rule_llm' AND e.extractor_version = ?))",
        "e.review_status IN ('unreviewed', 'needs_review')",
    ]
    values: list[object] = [EXTRACTOR_VERSION]
    if not event_ids:
        clauses.append(
            "(e.review_status = 'needs_review' OR e.extraction_confidence < 0.75 "
            "OR e.event_type = 'activity')"
        )
    if document_ids:
        placeholders = ", ".join("?" for _ in document_ids)
        clauses.append(f"er.document_id IN ({placeholders})")
        values.extend(document_ids)
    if event_ids:
        placeholders = ", ".join("?" for _ in event_ids)
        clauses.append(f"e.event_id IN ({placeholders})")
        values.extend(event_ids)
    rows = connection.execute(
        f"""
        SELECT DISTINCT e.event_id, e.extraction_confidence, e.review_status
        FROM historical_events e
        JOIN event_evidence ee ON ee.event_id = e.event_id
        JOIN evidence_records er ON er.evidence_id = ee.evidence_id
        WHERE {' AND '.join(clauses)}
        ORDER BY
            CASE e.review_status WHEN 'needs_review' THEN 0 ELSE 1 END,
            e.extraction_confidence,
            e.event_id
        """,
        values,
    ).fetchall()
    return [str(row["event_id"]) for row in rows]


def _has_prior_attempt(
    connection: sqlite3.Connection,
    candidate: EventCandidate,
    model_name: str,
    retry_failed: bool,
) -> bool:
    statuses = "'succeeded'" if retry_failed else "'succeeded', 'invalid', 'failed'"
    row = connection.execute(
        f"""
        SELECT 1 FROM event_extraction_attempts
        WHERE source_event_id = ? AND provider = ? AND model_name = ?
          AND prompt_version = ? AND request_sha256 = ? AND status IN ({statuses})
        LIMIT 1
        """,
        (
            candidate.event.event_id,
            PROVIDER,
            model_name,
            PROMPT_VERSION,
            candidate.request_sha256,
        ),
    ).fetchone()
    return row is not None


def select_event_candidates(
    database: Database,
    *,
    catalog: PersonCatalog,
    model_name: str,
    limit: int,
    document_ids: list[str] | None = None,
    event_ids: list[str] | None = None,
    retry_failed: bool = False,
) -> tuple[list[EventCandidate], int]:
    if not 1 <= limit <= 50:
        raise ResearchDataError("model extraction limit must be between 1 and 50")
    database.initialize()
    selected: list[EventCandidate] = []
    skipped = 0
    with database.connect() as connection:
        ids = _candidate_event_ids(
            connection,
            document_ids=document_ids,
            event_ids=event_ids,
        )
        for event_id in ids:
            event, updated_at = _event_from_connection(connection, event_id)
            messages = _request_messages(event, catalog)
            candidate = EventCandidate(
                event=event,
                updated_at=updated_at,
                request_sha256=_request_hash(event),
                messages=messages,
            )
            if _has_prior_attempt(connection, candidate, model_name, retry_failed):
                skipped += 1
                continue
            selected.append(candidate)
            if len(selected) >= limit:
                break
    return selected, skipped


def _review_reasons(
    source: HistoricalEvent,
    output: ModelEventExtraction | None,
    error_code: str | None,
) -> list[str]:
    reasons = ["model_assisted"]
    if source.review_status == "needs_review":
        reasons.append("source_needs_review")
    if source.extraction_confidence < 0.75:
        reasons.append("low_rule_confidence")
    if source.event_type == "activity":
        reasons.append("generic_rule_event_type")
    if output is not None:
        if output.needs_review:
            reasons.append("model_requested_review")
        reasons.extend(f"model_{item}" for item in output.review_reasons)
    if error_code:
        reasons.append(error_code)
    return list(dict.fromkeys(reasons))


def _record_attempt(
    database: Database,
    *,
    run_id: str,
    candidate: EventCandidate,
    model_name: str,
    result: DeepSeekResult,
    status: Literal["succeeded", "invalid", "failed"],
    error_code: str | None,
    validated: ModelEventExtraction | None,
    merged: HistoricalEvent | None,
) -> str:
    now = utc_now()
    attempt_id = _stable_id(
        "attempt",
        run_id,
        candidate.event.event_id,
        model_name,
        candidate.request_sha256,
    )
    reasons = _review_reasons(candidate.event, validated, error_code)
    priority = 90 if status != "succeeded" else 70 if validated and validated.needs_review else 50
    queue_id = _stable_id("review", candidate.event.event_id, attempt_id)
    before_json = candidate.event.model_dump_json()
    merged_json = merged.model_dump_json() if merged else None
    with database.connect() as connection:
        current = connection.execute(
            "SELECT extraction_method, review_status, updated_at FROM historical_events "
            "WHERE event_id = ?",
            (candidate.event.event_id,),
        ).fetchone()
        if current is None:
            raise ResearchDataError(f"event disappeared: {candidate.event.event_id}")
        if str(current["updated_at"]) != candidate.updated_at:
            raise ResearchDataError(
                f"event changed during model extraction: {candidate.event.event_id}"
            )
        connection.execute(
            """
            INSERT INTO event_extraction_attempts (
                attempt_id, run_id, source_event_id, provider, model_name,
                prompt_version, extractor_version, request_sha256, response_json,
                validated_json, status, error_code, usage_json, before_event_json,
                merged_event_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                attempt_id,
                run_id,
                candidate.event.event_id,
                PROVIDER,
                model_name,
                PROMPT_VERSION,
                EXTRACTOR_VERSION,
                candidate.request_sha256,
                result.response_json,
                validated.model_dump_json() if validated else None,
                status,
                error_code,
                json.dumps(result.usage, ensure_ascii=False),
                before_json,
                merged_json,
                now,
            ),
        )
        if merged is not None:
            if current["review_status"] not in {"unreviewed", "needs_review"}:
                raise ResearchDataError("reviewed event cannot be updated by a model")
            if current["extraction_method"] not in {"rule", "rule_llm"}:
                raise ResearchDataError("foreign event cannot be updated by a model")
            connection.execute(
                """
                UPDATE historical_events SET
                    name = ?, event_type = ?, location_text = ?,
                    organization_names_json = ?, extraction_method = ?,
                    extraction_confidence = ?, review_status = ?,
                    extractor_version = ?, updated_at = ?
                WHERE event_id = ?
                """,
                (
                    merged.name,
                    merged.event_type,
                    merged.location_text,
                    json.dumps(merged.organization_names, ensure_ascii=False),
                    merged.extraction_method,
                    merged.extraction_confidence,
                    merged.review_status,
                    merged.extractor_version,
                    now,
                    merged.event_id,
                ),
            )
            connection.execute(
                "DELETE FROM event_participants WHERE event_id = ?", (merged.event_id,)
            )
            connection.executemany(
                """
                INSERT INTO event_participants (
                    event_id, person_id, role, mention_text, mention_source
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        merged.event_id,
                        item.person_id,
                        item.role,
                        item.mention_text,
                        item.mention_source,
                    )
                    for item in merged.participants
                ],
            )
        connection.execute(
            """
            INSERT INTO event_review_queue (
                queue_id, event_id, source_attempt_id, reason_codes_json,
                priority, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
            """,
            (
                queue_id,
                candidate.event.event_id,
                attempt_id,
                json.dumps(reasons, ensure_ascii=False),
                priority,
                now,
                now,
            ),
        )
    return attempt_id


def list_review_queue(database: Database, *, limit: int = 20) -> list[dict[str, Any]]:
    if not 1 <= limit <= 200:
        raise ResearchDataError("review queue limit must be between 1 and 200")
    database.initialize()
    with database.connect() as connection:
        rows = connection.execute(
            """
            SELECT q.*, e.name, e.event_type, e.start_value, e.start_original_text,
                   a.model_name, a.prompt_version, a.status AS attempt_status,
                   a.error_code
            FROM event_review_queue q
            JOIN historical_events e ON e.event_id = q.event_id
            JOIN event_extraction_attempts a ON a.attempt_id = q.source_attempt_id
            WHERE q.status = 'pending'
            ORDER BY q.priority DESC, q.created_at, q.queue_id
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {
            **dict(row),
            "reason_codes": json.loads(str(row["reason_codes_json"])),
        }
        for row in rows
    ]


def enrich_events_with_model(
    *,
    database: Database,
    settings: Settings,
    catalog: PersonCatalog,
    reports_dir: Path,
    run_id: str,
    limit: int = 5,
    document_ids: list[str] | None = None,
    event_ids: list[str] | None = None,
    retry_failed: bool = False,
    dry_run: bool = False,
) -> ModelExtractionSummary:
    if not dry_run and not settings.llm_enabled:
        raise ResearchDataError("DeepSeek API key is not configured")
    candidates, skipped = select_event_candidates(
        database,
        catalog=catalog,
        model_name=settings.llm_model,
        limit=limit,
        document_ids=document_ids,
        event_ids=event_ids,
        retry_failed=retry_failed,
    )
    summary = ModelExtractionSummary(
        run_id=run_id,
        provider=PROVIDER,
        model_name=settings.llm_model,
        prompt_version=PROMPT_VERSION,
        extractor_version=EXTRACTOR_VERSION,
        dry_run=dry_run,
        selected=len(candidates),
        skipped_prior_attempt=skipped,
        event_ids=[item.event.event_id for item in candidates],
    )
    if not dry_run:
        for candidate in candidates:
            result = _call_deepseek_json(settings, candidate.messages)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                setattr(summary, key, getattr(summary, key) + result.usage.get(key, 0))
            if result.error_code:
                attempt_id = _record_attempt(
                    database,
                    run_id=run_id,
                    candidate=candidate,
                    model_name=settings.llm_model,
                    result=result,
                    status="failed",
                    error_code=result.error_code,
                    validated=None,
                    merged=None,
                )
                summary.failed += 1
                summary.attempt_ids.append(attempt_id)
                continue
            assert result.content is not None
            try:
                output = _parse_model_output(result.content)
            except ModelOutputError as exc:
                attempt_id = _record_attempt(
                    database,
                    run_id=run_id,
                    candidate=candidate,
                    model_name=settings.llm_model,
                    result=result,
                    status="invalid",
                    error_code=str(exc).split(":", 1)[0],
                    validated=None,
                    merged=None,
                )
                summary.invalid += 1
                summary.attempt_ids.append(attempt_id)
                continue
            try:
                resolved = _validate_and_resolve(output, candidate.event, catalog)
                merged = _merge_event(candidate.event, output, resolved)
            except ModelOutputError as exc:
                attempt_id = _record_attempt(
                    database,
                    run_id=run_id,
                    candidate=candidate,
                    model_name=settings.llm_model,
                    result=result,
                    status="invalid",
                    error_code=str(exc).split(":", 1)[0],
                    validated=output,
                    merged=None,
                )
                summary.invalid += 1
                summary.attempt_ids.append(attempt_id)
                continue
            attempt_id = _record_attempt(
                database,
                run_id=run_id,
                candidate=candidate,
                model_name=settings.llm_model,
                result=result,
                status="succeeded",
                error_code=None,
                validated=output,
                merged=merged,
            )
            summary.succeeded += 1
            summary.attempt_ids.append(attempt_id)

    reports_dir.mkdir(parents=True, exist_ok=True)
    rendered = summary.model_dump_json(indent=2)
    (reports_dir / f"model_event_extraction_{run_id}.json").write_text(
        rendered, encoding="utf-8"
    )
    (reports_dir / "model_event_extraction_latest.json").write_text(
        rendered, encoding="utf-8"
    )
    return summary
