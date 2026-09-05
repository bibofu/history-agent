import json
from pathlib import Path

import pymupdf
from history_agent.corpus.catalog import load_catalog
from history_agent.corpus.scanner import scan_corpus
from history_agent.db import Database
from history_agent.extraction.benchmark import analyze_text, compact_text
from history_agent.extraction.full import extract_all_text, extract_document
from history_agent.extraction.ocr import parse_paddle_ocr_result
from history_agent.extraction.report import compact_page_ranges
from history_agent.extraction.text import classify_extracted_text, normalize_for_storage
from history_agent.processing.chunks import extract_dates, scope_status, split_text
from history_agent.processing.cleaning import clean_page_text, is_table_of_contents_page
from history_agent.retrieval.keyword import infer_query_intent, infer_year_range, tokenize_query
from pypdf import PdfWriter


def test_blank_page_is_marked_empty() -> None:
    status, flags = classify_extracted_text("   \n")

    assert status == "empty"
    assert "blank_or_vector_page" in flags


def test_image_only_page_routes_to_ocr() -> None:
    status, flags = classify_extracted_text("", image_object_count=1)

    assert status == "ocr_required"
    assert "image_without_text" in flags


def test_short_but_clean_page_keeps_text_layer() -> None:
    status, flags = classify_extracted_text("毛泽东选集")

    assert status == "extracted"
    assert "very_short_text" in flags


def test_normal_chinese_page_uses_text_layer() -> None:
    text = "这是用于测试的中文正文。" * 20
    status, flags = classify_extracted_text(text)

    assert status == "extracted"
    assert flags == []


def test_page_normalization_preserves_line_breaks() -> None:
    assert normalize_for_storage("甲\r\n乙\x00") == "甲\n乙"


def test_page_ranges_are_compacted_for_reports() -> None:
    assert compact_page_ranges([8, 2, 3, 4, 8, 10]) == "2–4, 8, 10"
    assert compact_page_ranges([]) == "—"


def test_paddle_result_is_normalized_without_loading_models() -> None:
    text, confidence = parse_paddle_ocr_result(
        [
            {
                "res": {
                    "rec_texts": [" 第一行 ", "", "第二行"],
                    "rec_scores": [0.9, 0.2, 0.8],
                }
            }
        ]
    )

    assert text == "第一行\n第二行"
    assert confidence == 0.85


def test_cleaning_removes_page_numbers_and_repeated_headers() -> None:
    text = "中国共产党历史\n123\n第一句。\n第二句。"

    assert clean_page_text(text, {"中国共产党历史"}) == "第一句。\n第二句。"


def test_chunk_dates_support_arabic_and_chinese_years() -> None:
    years, dates = extract_dates("一九三五年十月，会议召开。1978年12月18日再次讨论。")

    assert years == [1935, 1978]
    assert dates == ["1935", "1978-12-18"]
    assert scope_status(years, 1921, 1978) == "in_scope"
    assert scope_status([1978, 1979], 1921, 1978) == "mixed"


def test_chunk_split_respects_maximum_size() -> None:
    chunks = split_text("第一句。" * 400, target_chars=100, max_chars=140)

    assert len(chunks) > 1
    assert all(20 <= len(chunk) <= 140 for chunk in chunks)


def test_keyword_query_tokenization_keeps_names_and_years() -> None:
    terms = tokenize_query("周恩来在1956年主要有哪些经历")

    assert "周恩" in terms
    assert "恩来" in terms
    assert "1956" in terms
    assert "1956年" in terms
    assert "主要" not in terms
    assert "来在" not in terms
    assert "经历" not in terms
    assert infer_query_intent("周恩来在1956年主要有哪些经历") == "timeline"
    assert infer_query_intent("毛泽东如何论述调查研究") == "viewpoint"
    assert infer_year_range("长征期间", []) == [1934, 1936]


def test_contents_pages_are_detected_from_dot_leaders() -> None:
    text = "目录\n第一章..........1\n第二章..........20\n第三章..........40"

    assert is_table_of_contents_page(text)


def test_benchmark_analysis_counts_compact_chinese_text() -> None:
    assert compact_text("甲 乙\n丙") == "甲乙丙"
    analysis = analyze_text("甲 乙\n丙")
    assert analysis["character_count"] == 3
    assert analysis["cjk_character_count"] == 3


def test_full_extraction_is_resumable(work_path: Path) -> None:
    docs_dir = work_path / "docs"
    docs_dir.mkdir()
    pdf_path = docs_dir / "fixture.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=400)
    writer.add_blank_page(width=300, height=400)
    with pdf_path.open("wb") as stream:
        writer.write(stream)

    catalog_path = work_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "document_id": "fixture_document",
                        "filename": "fixture.pdf",
                        "title": "Fixture",
                        "source_type": "test",
                        "verification_status": "checked",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    database = Database(work_path / "history_agent.db")
    scan_corpus(
        database=database,
        catalog=load_catalog(catalog_path),
        docs_dir=docs_dir,
        project_root=work_path,
        run_id="scan",
    )

    first = extract_all_text(
        database=database,
        project_root=work_path,
        output_dir=work_path / "pages",
        reports_dir=work_path / "reports",
        run_id="first",
    )
    second = extract_all_text(
        database=database,
        project_root=work_path,
        output_dir=work_path / "pages",
        reports_dir=work_path / "reports",
        run_id="second",
    )

    assert first.totals["pages"] == 2
    assert first.totals["status_empty"] == 2
    assert first.documents[0].processed_pages == 2
    assert second.documents[0].processed_pages == 0
    assert second.documents[0].reused_pages == 2


def test_full_ocr_policy_overrides_existing_text_and_cached_pages(work_path: Path) -> None:
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Existing but unreliable text layer")
    pdf.save(work_path / "fixture.pdf")
    pdf.close()
    file_record = {
        "document_id": "fixture", "relative_path": "fixture.pdf",
        "filename": "fixture.pdf", "sha256": "test-hash",
        "ocr_strategy": "partial_if_needed",
    }
    kwargs = {
        "file_record": file_record, "project_root": work_path,
        "output_dir": work_path / "pages", "run_id": "test",
        "rebuild": False, "parser_name": "pymupdf",
    }
    assert extract_document(**kwargs).status_counts == {"extracted": 1}
    file_record["ocr_strategy"] = "full_required"
    cached = extract_document(**kwargs)
    assert cached.reused_pages == 1
    assert cached.status_counts == {"ocr_required": 1}
    kwargs["rebuild"] = True
    assert extract_document(**kwargs).status_counts == {"ocr_required": 1}
    kwargs["rebuild"] = False
    file_record["ocr_strategy"] = "partial_required"
    assert extract_document(**kwargs).status_counts == {"extracted": 1}
    file_record["ocr_pages"] = [1]
    assert extract_document(**kwargs).status_counts == {"ocr_required": 1}
