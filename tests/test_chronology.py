from __future__ import annotations

import json
from pathlib import Path

from history_agent.db import Database
from history_agent.extraction.models import PageRecord
from history_agent.research.catalog import load_person_catalog
from history_agent.research.chronology import (
    extract_chronology_events,
    parse_chronology_date,
)
from history_agent.research.people import sync_person_catalog


def test_chronology_date_rules_preserve_uncertainty() -> None:
    date_range = parse_chronology_date("1962 年7 月28 日－8 月24 日")
    assert date_range.start.value == "1962-07-28"
    assert date_range.end is not None
    assert date_range.end.value == "1962-08-24"
    assert date_range.rule_flags == ["date_range"]

    merged = parse_chronology_date("1962年9月15日、16日、19日")
    assert merged.start.value == "1962-09-15"
    assert merged.start.certainty == "approximate"
    assert merged.end is not None
    assert merged.end.value == "1962-09-19"
    assert "multiple_dates" in merged.rule_flags

    partial = parse_chronology_date("1月5日—7日", year_hint=1948)
    assert partial.start.value == "1948-01-05"
    assert partial.start.certainty == "inferred"
    assert partial.end is not None
    assert partial.end.value == "1948-01-07"

    inherited = parse_chronology_date("在此期间", inherited_from=partial)
    assert inherited.start.value == "1948-01-05"
    assert inherited.start.certainty == "inferred"
    assert "inherited_date" in inherited.rule_flags

    invalid = parse_chronology_date("1948年2月30日")
    assert invalid.start.value == "1948-02"
    assert invalid.start.precision == "month"
    assert "invalid_date_component" in invalid.rule_flags

    reversed_range = parse_chronology_date("3月25日—9日", year_hint=1933)
    assert reversed_range.start.value == "1933-03-25"
    assert reversed_range.end is None
    assert "invalid_date_order" in reversed_range.rule_flags


def _page(document_id: str, pdf_page: int, text: str) -> PageRecord:
    return PageRecord(
        document_id=document_id,
        file_sha256=f"{document_id}-hash",
        pdf_page=pdf_page,
        status="extracted",
        extraction_method="pymupdf_text_layer",
        extractor_version="fixture-v1",
        raw_text=text,
        normalized_text=text,
        text_sha256=f"text-{pdf_page}",
        character_count=len(text),
        cjk_character_count=len(text),
        replacement_character_count=0,
        run_id="fixture-run",
        processed_at="2026-01-01T00:00:00+00:00",
    )


def _write_pages(path: Path, pages: list[PageRecord]) -> None:
    path.write_text(
        "".join(page.model_dump_json() + "\n" for page in pages),
        encoding="utf-8",
    )


def _insert_documents(database: Database) -> None:
    database.initialize()
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
                ) VALUES (?, ?, '[]', 'chronology', NULL, NULL, '[]', 'fixture',
                          1, 'none', 2, NULL, 'now', 'now')
                """,
                (document_id, title),
            )


def test_chronology_extraction_is_page_audited_and_idempotent(work_path: Path) -> None:
    pages_dir = work_path / "pages"
    ocr_dir = work_path / "ocr"
    structure_dir = work_path / "structure"
    events_dir = work_path / "events"
    reports_dir = work_path / "reports"
    for path in (pages_dir, ocr_dir, structure_dir, events_dir, reports_dir):
        path.mkdir()

    _write_pages(
        pages_dir / "zhou_enlai_chronology_1949_1976.jsonl",
        [
            _page(
                "zhou_enlai_chronology_1949_1976",
                1,
                """[周恩来年谱]1954 年（五十六岁）
【1954 年9 月25 日】
△晚，在北京出席毛泽东主持的会议。
△周恩来致信李先念，讨论经费安排，""",
            ),
            _page(
                "zhou_enlai_chronology_1949_1976",
                2,
                """并请有关部门尽快办理。
【在此期间】
△在上海会见有关人员并讨论工作。
【1954年9月25日－30日】
△连续陪同毛泽东接见来华代表团。""",
            ),
        ],
    )
    _write_pages(
        pages_dir / "lin_biao_chronology.jsonl",
        [
            _page(
                "lin_biao_chronology",
                1,
                """1 9 4 8 年 4 1 岁
1月1日 林彪任东北军区司令员兼政治委员。
1月 林彪拟提钟伟为纵队副司令员。
1月2日 林彪令部队向公主屯地区前进，""",
            ),
            _page(
                "lin_biao_chronology",
                2,
                """并切断敌军退路。
1月5日—7日 林彪连续指挥公主屯战斗。""",
            ),
        ],
    )
    structure = [
        {
            "entry_id": "lin_1948_fixture",
            "document_id": "lin_biao_chronology",
            "level": 1,
            "title": "1948年 41岁",
            "pdf_page_start": 1,
            "pdf_page_end": 2,
            "source": "pdf_outline",
        }
    ]
    (structure_dir / "lin_biao_chronology.json").write_text(
        json.dumps(structure, ensure_ascii=False), encoding="utf-8"
    )

    database = Database(work_path / "history_agent.db")
    _insert_documents(database)
    aliases_path = Path.cwd() / "config" / "person_aliases.json"
    sync_person_catalog(database, load_person_catalog(aliases_path))

    first = extract_chronology_events(
        database=database,
        pages_dir=pages_dir,
        ocr_dir=ocr_dir,
        structure_dir=structure_dir,
        events_dir=events_dir,
        reports_dir=reports_dir,
        person_aliases_path=aliases_path,
        run_id="first",
        research_start=1921,
        research_end=1978,
    )
    assert first.totals["candidates"] == 8
    assert first.totals["database_created"] == 8
    assert first.totals["database_updated"] == 0
    assert first.totals["location_candidates"] >= 3

    with database.connect() as connection:
        cross_page = connection.execute(
            """
            SELECT e.start_value, e.end_value, e.review_status,
                   x.pdf_page_start, x.pdf_page_end, p.mention_source
            FROM historical_events e
            JOIN event_evidence ee ON ee.event_id = e.event_id
            JOIN evidence_records x ON x.evidence_id = ee.evidence_id
            JOIN event_participants p ON p.event_id = e.event_id
            WHERE e.description LIKE '%有关部门尽快办理%'
              AND p.person_id = 'zhou_enlai'
            """
        ).fetchone()
        inherited = connection.execute(
            """
            SELECT e.start_value, e.start_certainty, e.review_status
            FROM historical_events e
            WHERE e.description LIKE '%在上海会见%'
            """
        ).fetchone()
    assert dict(cross_page) == {
        "start_value": "1954-09-25",
        "end_value": None,
        "review_status": "unreviewed",
        "pdf_page_start": 1,
        "pdf_page_end": 2,
        "mention_source": "explicit",
    }
    assert dict(inherited) == {
        "start_value": "1954-09-25",
        "start_certainty": "inferred",
        "review_status": "needs_review",
    }

    second = extract_chronology_events(
        database=database,
        pages_dir=pages_dir,
        ocr_dir=ocr_dir,
        structure_dir=structure_dir,
        events_dir=events_dir,
        reports_dir=reports_dir,
        person_aliases_path=aliases_path,
        run_id="second",
        research_start=1921,
        research_end=1978,
    )
    assert second.totals["database_created"] == 0
    assert second.totals["database_updated"] == 0
    assert second.totals["database_skipped"] == 8
