from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from history_agent.db import Database
from history_agent.errors import ResearchDataError
from history_agent.research.intersections import RULE_VERSION, get_person_intersections


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
    cohort: Literal["legacy", "multi_pair"] = "legacy"

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
