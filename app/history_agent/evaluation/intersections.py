from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from datetime import date
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from history_agent.db import Database
from history_agent.errors import ResearchDataError
from history_agent.research.chronology import PROFILES
from history_agent.research.intersections import RULE_VERSION, get_person_intersections

PLACEHOLDER_REVIEW_REASONS = {
    "connected",
    "connectred",
    "connection",
    "yes",
    "no",
    "true",
    "false",
    "有关联",
    "有联系",
    "是",
    "否",
}


class IntersectionCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source_event_id: str
    document_id: str
    pdf_page: int = Field(ge=1)
    year: int = Field(ge=1, le=9999)
    expected: bool
    reason: str
    person_id: str = "mao_zedong"
    other_person_id: str = "zhou_enlai"
    cohort: Literal["legacy", "multi_pair", "independent"] = "legacy"

    @model_validator(mode="after")
    def distinct_people(self) -> IntersectionCase:
        if not self.person_id.strip() or not self.other_person_id.strip():
            raise ValueError("person IDs cannot be empty")
        if self.person_id == self.other_person_id:
            raise ValueError("intersection cases need two different people")
        return self

    @property
    def pair(self) -> tuple[str, str]:
        first, second = sorted((self.person_id, self.other_person_id))
        return first, second


class CaseOutcome(BaseModel):
    id: str
    person_ids: tuple[str, str]
    source_event_id: str
    expected: bool
    predicted: bool
    cohort: str
    reason: str


class ReviewEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    document_title: str
    volume: str | None = None
    pdf_page_start: int = Field(ge=1)
    pdf_page_end: int = Field(ge=1)
    quote: str


class ReviewAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected: bool | None = None
    reason: str = ""
    reviewed_by: str = ""
    reviewed_at: str = ""


class IntersectionReviewItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    person_id: str
    person_name: str
    other_person_id: str
    other_person_name: str
    source_event_id: str
    event_name: str
    event_type: str
    date: str
    document_id: str
    pdf_page: int = Field(ge=1)
    evidence: list[ReviewEvidence] = Field(min_length=1)
    case_checksum: str = ""
    annotation: ReviewAnnotation = Field(default_factory=ReviewAnnotation)


class IntersectionReviewPacket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal["intersection-independent-review-v1"] = (
        "intersection-independent-review-v1"
    )
    seed: str
    source_event_count: int
    excluded_development_event_count: int
    instructions: list[str]
    cases: list[IntersectionReviewItem] = Field(min_length=1)


def _metrics(outcomes: list[CaseOutcome]) -> dict[str, int | float | None]:
    tp = sum(item.expected and item.predicted for item in outcomes)
    fp = sum(not item.expected and item.predicted for item in outcomes)
    fn = sum(item.expected and not item.predicted for item in outcomes)
    return {
        "sample_count": len(outcomes),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": len(outcomes) - tp - fp - fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }


def _stable_rank(seed: str, *parts: str) -> str:
    return sha256("|".join((seed, *parts)).encode("utf-8")).hexdigest()


def _review_item_checksum(item: IntersectionReviewItem) -> str:
    immutable = item.model_dump(exclude={"annotation", "case_checksum"})
    encoded = json.dumps(
        immutable,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


def _pair_predictions(
    database: Database,
    pair: tuple[str, str],
    *,
    start_year: int,
    end_year: int,
) -> dict[str, tuple[str, int]]:
    predictions: dict[str, tuple[str, int]] = {}
    offset = 0
    while True:
        response = get_person_intersections(
            database,
            person_id=pair[0],
            other_person_id=pair[1],
            start_year=start_year,
            end_year=end_year,
            limit=200,
            offset=offset,
        )
        for item in response.events:
            evidence = {entry.evidence_id: entry for entry in item.event.evidence}
            for proof in item.joint_evidence:
                source = evidence[proof.evidence_id]
                predictions.setdefault(
                    proof.source_event_id,
                    (source.document_id, source.pdf_page_start),
                )
        if not response.has_more:
            return predictions
        offset += 200


def build_intersection_review_packet(
    database: Database,
    development_cases_path: Path,
    *,
    seed: str,
    pair_limit: int = 10,
    per_stratum: int = 2,
    start_year: int = 1921,
    end_year: int = 1978,
) -> IntersectionReviewPacket:
    """Build a deterministic packet whose cases never expose current predictions."""
    if not seed.strip():
        raise ResearchDataError("review packet seed cannot be empty")
    if pair_limit < 1 or per_stratum < 1:
        raise ResearchDataError("pair_limit and per_stratum must be positive")
    if start_year > end_year:
        raise ResearchDataError("review packet start year must not exceed end year")
    development_payload = json.loads(development_cases_path.read_text(encoding="utf-8"))
    development_events = {
        str(item["source_event_id"]) for item in development_payload.get("cases", [])
    }
    with database.connect() as connection:
        people = {
            str(row["person_id"]): str(row["canonical_name"])
            for row in connection.execute(
                "SELECT person_id, canonical_name FROM persons WHERE status='active'"
            ).fetchall()
        }
        rows = connection.execute(
            "SELECT e.event_id, e.name, e.event_type, e.start_value, "
            "er.document_id, d.title AS document_title, d.volume, "
            "er.pdf_page_start, er.pdf_page_end, er.quote, "
            "MIN(er.pdf_page_start) OVER (PARTITION BY e.event_id) AS first_page "
            "FROM historical_events e "
            "JOIN event_evidence ee ON ee.event_id=e.event_id "
            "JOIN evidence_records er ON er.evidence_id=ee.evidence_id "
            "JOIN documents d ON d.document_id=er.document_id "
            "WHERE e.review_status!='rejected' AND e.start_value IS NOT NULL "
            "AND CAST(substr(e.start_value, 1, 4) AS INTEGER) BETWEEN ? AND ? "
            "ORDER BY e.event_id, er.pdf_page_start, er.evidence_id",
            (start_year, end_year),
        ).fetchall()
    evidence_by_event: dict[str, list[ReviewEvidence]] = defaultdict(list)
    anchors: dict[tuple[tuple[str, str], str], sqlite3.Row] = {}
    compact_names = {person_id: re.sub(r"\s+", "", name) for person_id, name in people.items()}
    for row in rows:
        event_id = str(row["event_id"])
        evidence_by_event[event_id].append(
            ReviewEvidence(
                document_id=str(row["document_id"]),
                document_title=str(row["document_title"]),
                volume=str(row["volume"]) if row["volume"] is not None else None,
                pdf_page_start=int(row["pdf_page_start"]),
                pdf_page_end=int(row["pdf_page_end"]),
                quote=str(row["quote"]),
            )
        )
        if event_id in development_events:
            continue
        compact_quote = re.sub(r"\s+", "", str(row["quote"]))
        mentioned = {
            person_id for person_id, name in compact_names.items() if name in compact_quote
        }
        profile = PROFILES.get(str(row["document_id"]))
        if profile is not None and int(row["pdf_page_start"]) == int(row["first_page"]):
            mentioned.add(profile.subject_person_id)
        for first, second in combinations(sorted(mentioned), 2):
            anchors.setdefault(((first, second), event_id), row)
    candidates_by_pair: dict[tuple[str, str], list[str]] = defaultdict(list)
    for pair, event_id in anchors:
        candidates_by_pair[pair].append(event_id)
    selected: list[tuple[tuple[str, str], str, tuple[str, int] | None]] = []
    qualifying_pairs = 0
    for pair in sorted(candidates_by_pair, key=lambda item: (-len(candidates_by_pair[item]), item)):
        predictions = _pair_predictions(
            database,
            pair,
            start_year=start_year,
            end_year=end_year,
        )
        event_ids = candidates_by_pair[pair]
        positives = [event_id for event_id in event_ids if event_id in predictions]
        negatives = [event_id for event_id in event_ids if event_id not in predictions]
        if len(positives) < per_stratum or len(negatives) < per_stratum:
            continue
        positives.sort(key=lambda event_id: _stable_rank(seed, *pair, event_id, "a"))
        negatives.sort(key=lambda event_id: _stable_rank(seed, *pair, event_id, "b"))
        selected.extend(
            (pair, event_id, predictions[event_id])
            for event_id in positives[:per_stratum]
        )
        selected.extend((pair, event_id, None) for event_id in negatives[:per_stratum])
        qualifying_pairs += 1
        if qualifying_pairs == pair_limit:
            break
    if qualifying_pairs < pair_limit:
        raise ResearchDataError(
            f"only {qualifying_pairs} person pairs have enough cases for the requested strata"
        )
    cases: list[IntersectionReviewItem] = []
    for pair, event_id, proof_location in selected:
        row = anchors[(pair, event_id)]
        document_id = str(row["document_id"])
        pdf_page = int(row["pdf_page_start"])
        if proof_location is not None:
            document_id, pdf_page = proof_location
        case_hash = _stable_rank("intersection-blind-case-v1", seed, *pair, event_id)[:16]
        item = IntersectionReviewItem(
            id=f"blind-{case_hash}",
            person_id=pair[0],
            person_name=people[pair[0]],
            other_person_id=pair[1],
            other_person_name=people[pair[1]],
            source_event_id=event_id,
            event_name=str(row["name"]),
            event_type=str(row["event_type"]),
            date=str(row["start_value"]),
            document_id=document_id,
            pdf_page=pdf_page,
            evidence=evidence_by_event[event_id],
        )
        item.case_checksum = _review_item_checksum(item)
        cases.append(item)
    cases.sort(key=lambda item: _stable_rank(seed, item.id, "review-order"))
    return IntersectionReviewPacket(
        seed=seed,
        source_event_count=len(evidence_by_event),
        excluded_development_event_count=len(development_events),
        instructions=[
            "由未参与规则开发的复核者查看列出的原始 PDF 完整页面，不运行交集命令。",
            "判断指定两人是否被该来源条目明确证明共同参与同一动作；通信收件人不等于共同发件人。",
            "在 annotation 中填写 expected、reason、reviewed_by、reviewed_at；不得改动其他字段。",
            "negative 仅表示该来源条目不足以证明共同动作，不表示历史上不存在交集。",
        ],
        cases=cases,
    )


def finalize_intersection_review_packet(packet_path: Path) -> dict[str, object]:
    packet = IntersectionReviewPacket.model_validate_json(packet_path.read_text(encoding="utf-8"))
    cases: list[dict[str, object]] = []
    for item in packet.cases:
        if item.case_checksum != _review_item_checksum(item):
            raise ResearchDataError(f"review case content changed: {item.id}")
        annotation = item.annotation
        if annotation.expected is None:
            raise ResearchDataError(f"review label missing: {item.id}")
        if not annotation.reason.strip():
            raise ResearchDataError(f"review reason missing: {item.id}")
        normalized_reason = re.sub(r"[\s\W_]+", "", annotation.reason).casefold()
        if normalized_reason in PLACEHOLDER_REVIEW_REASONS:
            raise ResearchDataError(
                f"review reason must describe page-specific evidence: {item.id}"
            )
        if not annotation.reviewed_by.strip() or not annotation.reviewed_at.strip():
            raise ResearchDataError(f"reviewer metadata missing: {item.id}")
        try:
            date.fromisoformat(annotation.reviewed_at)
        except ValueError as exc:
            raise ResearchDataError(f"review date must be ISO YYYY-MM-DD: {item.id}") from exc
        if not any(
            evidence.document_id == item.document_id
            and evidence.pdf_page_start <= item.pdf_page <= evidence.pdf_page_end
            for evidence in item.evidence
        ):
            raise ResearchDataError(f"review anchor is outside packet evidence: {item.id}")
        cases.append(
            {
                "id": item.id,
                "person_id": item.person_id,
                "other_person_id": item.other_person_id,
                "cohort": "independent",
                "source_event_id": item.source_event_id,
                "document_id": item.document_id,
                "pdf_page": item.pdf_page,
                "year": int(item.date[:4]),
                "expected": annotation.expected,
                "reason": annotation.reason.strip(),
            }
        )
    reviewers = sorted({item.annotation.reviewed_by.strip() for item in packet.cases})
    return {
        "version": "pdf-independent-review-v1",
        "review_method": "independent_pdf_review",
        "reviewers": reviewers,
        "source_packet_version": packet.version,
        "scope": (
            "独立分层定向样本；标签表示指定来源条目能否证明共同动作，"
            "negative 不表示历史上没有交集。"
        ),
        "cases": cases,
    }


def evaluate_intersections(database: Database, cases_path: Path) -> dict[str, object]:
    """Evaluate source-local proof IDs, not canonical participant unions."""
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = [IntersectionCase.model_validate(item) for item in payload["cases"]]
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ResearchDataError("intersection evaluation requires unique, nonempty cases")
    if len({(case.source_event_id, case.pair) for case in cases}) != len(cases):
        raise ResearchDataError("duplicate source/person pair would inflate evaluation counts")
    with database.connect() as connection:
        for person_id in {person for case in cases for person in case.pair}:
            person = connection.execute(
                "SELECT status FROM persons WHERE person_id=?", (person_id,)
            ).fetchone()
            if person is None or person[0] != "active":
                raise ResearchDataError(f"gold person must be an active stable ID: {person_id}")
        for case in cases:
            found = connection.execute(
                "SELECT 1 FROM event_evidence ee JOIN evidence_records er "
                "ON er.evidence_id=ee.evidence_id WHERE ee.event_id=? "
                "AND er.document_id=? AND er.pdf_page_start<=? AND er.pdf_page_end>=?",
                (case.source_event_id, case.document_id, case.pdf_page, case.pdf_page),
            ).fetchone()
            if found is None:
                raise ResearchDataError(
                    f"gold source/page missing; rebuild matching corpus: {case.id}"
                )
            event = connection.execute(
                "SELECT start_value, COALESCE(end_value, start_value) FROM historical_events "
                "WHERE event_id=?",
                (case.source_event_id,),
            ).fetchone()
            if event is None or not event[0] or not event[1]:
                raise ResearchDataError(f"gold source date unavailable: {case.id}")
            if not int(str(event[0])[:4]) <= case.year <= int(str(event[1])[:4]):
                raise ResearchDataError(f"gold year does not overlap source event: {case.id}")
    predicted: dict[tuple[tuple[str, str], int], set[tuple[str, str, int, int]]] = {}
    for pair, year in sorted({(case.pair, case.year) for case in cases}):
        proofs: set[tuple[str, str, int, int]] = set()
        offset = 0
        while True:
            response = get_person_intersections(
                database,
                person_id=pair[0],
                other_person_id=pair[1],
                start_year=year,
                end_year=year,
                limit=200,
                offset=offset,
            )
            for item in response.events:
                evidence = {entry.evidence_id: entry for entry in item.event.evidence}
                for proof in item.joint_evidence:
                    source = evidence[proof.evidence_id]
                    proofs.add(
                        (
                            proof.source_event_id,
                            source.document_id,
                            source.pdf_page_start,
                            source.pdf_page_end,
                        )
                    )
            if not response.has_more:
                break
            offset += 200
        predicted[pair, year] = proofs
    outcomes = [
        CaseOutcome(
            id=case.id,
            person_ids=case.pair,
            source_event_id=case.source_event_id,
            expected=case.expected,
            predicted=any(
                sid == case.source_event_id
                and doc == case.document_id
                and start <= case.pdf_page <= end
                for sid, doc, start, end in predicted[case.pair, case.year]
            ),
            cohort=case.cohort,
            reason=case.reason,
        )
        for case in cases
    ]
    return {
        "rule_version": RULE_VERSION,
        **_metrics(outcomes),
        "dataset_version": payload.get("version"),
        "review_method": payload.get("review_method"),
        "scope": payload.get("scope"),
        "coverage": {
            "unique_source_events": len({case.source_event_id for case in cases}),
            "unique_document_pages": len({(case.document_id, case.pdf_page) for case in cases}),
            "documents": len({case.document_id for case in cases}),
            "person_pairs": len({case.pair for case in cases}),
            "years": sorted({case.year for case in cases}),
        },
        "by_pair": {
            " + ".join(pair): _metrics([item for item in outcomes if item.person_ids == pair])
            for pair in sorted({case.pair for case in cases})
        },
        "by_cohort": {
            cohort: _metrics([item for item in outcomes if item.cohort == cohort])
            for cohort in sorted({case.cohort for case in cases})
        },
        "cases": [item.model_dump() for item in outcomes],
    }
