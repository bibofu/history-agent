import json
from pathlib import Path

import pymupdf
from history_agent.extraction.text import build_page_record
from history_agent.processing.chunks import build_document_chunks
from history_agent.retrieval.keyword import build_keyword_index, search_keyword_index


def test_chronology_section_controls_scope_even_with_later_year_mentions(work_path: Path) -> None:
    pdf = pymupdf.open()
    pdf.new_page()
    pdf.new_page()
    pdf.set_toc([[1, "1920年", 1], [1, "1921年", 2]])
    pdf.save(work_path / "fixture.pdf")
    pdf.close()
    pages = work_path / "pages"
    pages.mkdir()
    texts = [
        "毛泽东参加活动。这条1920年的记载在1956年被回忆，不能变成1956年的经历。",
        "毛泽东参加活动。这是1921年研究范围内的事件，有独立的页码证据。",
    ]
    records = [
        build_page_record(
            document_id="fixture", file_sha256="hash", pdf_page=i,
            text=text, image_object_count=0, run_id="test",
        )
        for i, text in enumerate(texts, 1)
    ]
    (pages / "fixture.jsonl").write_text(
        "\n".join(record.model_dump_json() for record in records), encoding="utf-8"
    )
    chunks = work_path / "chunks"
    build_document_chunks(
        document={
            "document_id": "fixture", "title": "毛泽东年谱", "relative_path": "fixture.pdf",
            "filename": "fixture.pdf", "page_count": 2, "sha256": "hash",
            "creators_json": "[]", "source_type": "chronology", "edition": None,
            "volume": None, "verification_status": "checked",
        },
        project_root=work_path, pages_dir=pages, ocr_dir=work_path / "ocr",
        chunks_dir=chunks, structure_dir=work_path / "structure",
        aliases={"毛泽东": []}, research_start=1921, research_end=1978,
    )
    rows = [json.loads(line) for line in (chunks / "fixture.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()]
    assert [row["scope_status"] for row in rows] == ["out_of_scope", "in_scope"]
    assert 1956 in rows[0]["year_mentions"]

    index = work_path / "keyword.db"
    build_keyword_index(
        chunks_dir=chunks, index_path=index, reports_dir=work_path / "reports", run_id="test"
    )
    aliases = Path(__file__).resolve().parents[1] / "config" / "person_aliases.json"
    assert not search_keyword_index(
        index_path=index, aliases_path=aliases, query="毛泽东1956年活动"
    ).hits
    assert search_keyword_index(
        index_path=index, aliases_path=aliases, query="毛泽东1956年活动",
        include_out_of_scope=True,
    ).hits
    assert search_keyword_index(
        index_path=index, aliases_path=aliases, query="毛泽东1921年活动"
    ).hits


def test_vector_year_filter_keeps_research_scope_constraint() -> None:
    from history_agent.retrieval.vector import _query_filter
    from qdrant_client import models

    result = _query_filter(
        models=models, document_ids=None, query_people=[],
        query_year_range=[1956, 1956], include_out_of_scope=False,
    )
    assert result.must
    assert result.must_not
    assert result.must_not[0].key == "scope_status"
    expanded = _query_filter(
        models=models, document_ids=None, query_people=[],
        query_year_range=[1956, 1956], include_out_of_scope=True,
    )
    assert not expanded.must_not
