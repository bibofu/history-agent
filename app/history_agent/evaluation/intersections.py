from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from history_agent.db import Database
from history_agent.errors import ResearchDataError
from history_agent.research.intersections import RULE_VERSION, get_person_intersections


class IntersectionCase(BaseModel):
    id: str
    source_event_id: str
    document_id: str
    pdf_page: int = Field(ge=1)
    year: int
    expected: bool
    reason: str


def evaluate_intersections(database: Database, cases_path: Path) -> dict[str, object]:
    """Evaluate source-local proof IDs, not canonical participant unions."""
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = [IntersectionCase.model_validate(item) for item in payload["cases"]]
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ResearchDataError("intersection evaluation requires unique, nonempty cases")
    with database.connect() as connection:
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
    predicted: dict[int, set[str]] = {}
    for year in sorted({case.year for case in cases}):
        proof_ids: set[str] = set()
        offset = 0
        while True:
            response = get_person_intersections(
                database,
                person_id="mao_zedong",
                other_person_id="zhou_enlai",
                start_year=year,
                end_year=year,
                limit=200,
                offset=offset,
            )
            proof_ids.update(
                proof.source_event_id for event in response.events for proof in event.joint_evidence
            )
            if not response.has_more:
                break
            offset += 200
        predicted[year] = proof_ids
    outcomes = [
        {
            "id": case.id,
            "expected": case.expected,
            "predicted": case.source_event_id in predicted[case.year],
            "reason": case.reason,
        }
        for case in cases
    ]
    tp = sum(item["expected"] is True and item["predicted"] is True for item in outcomes)
    fp = sum(item["expected"] is False and item["predicted"] is True for item in outcomes)
    fn = sum(item["expected"] is True and item["predicted"] is False for item in outcomes)
    tn = len(cases) - tp - fp - fn
    return {
        "rule_version": RULE_VERSION,
        "sample_count": len(cases),
        "review_method": payload.get("review_method"),
        "scope": payload.get("scope"),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "cases": outcomes,
    }
