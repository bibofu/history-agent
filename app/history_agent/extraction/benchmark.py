from __future__ import annotations

import json
import logging
from collections.abc import Callable
from difflib import SequenceMatcher
from pathlib import Path
from time import perf_counter
from typing import Any

import pdfplumber
import pymupdf
from pypdf import PdfReader

from history_agent.db import Database
from history_agent.errors import ExtractionError
from history_agent.extraction.full import current_files
from history_agent.extraction.text import CJK_PATTERN, normalize_for_storage, utc_now

DEFAULT_BENCHMARK_PAGES: dict[str, int] = {
    "cpc_history_volume_1": 514,
    "cpc_history_volume_2": 510,
    "mao_selected_works_combined": 1624,
    "zhou_enlai_chronology_1949_1976": 737,
    "lin_biao_chronology": 773,
    "deng_xiaoping_selected_works": 466,
    "red_star_over_china": 232,
}


def compact_text(text: str) -> str:
    return "".join(normalize_for_storage(text).split())


def analyze_text(text: str) -> dict[str, Any]:
    compact = compact_text(text)
    return {
        "character_count": len(compact),
        "cjk_character_count": len(CJK_PATTERN.findall(compact)),
        "replacement_character_count": compact.count("\ufffd"),
        "preview": " ".join(text.split())[:160],
    }


def _pypdf_text(path: Path, pdf_page: int) -> str:
    reader = PdfReader(str(path), strict=False)
    return reader.pages[pdf_page - 1].extract_text() or ""


def _pymupdf_text(path: Path, pdf_page: int) -> str:
    with pymupdf.open(path) as document:
        return document[pdf_page - 1].get_text("text") or ""


def _pdfplumber_text(path: Path, pdf_page: int) -> str:
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    with pdfplumber.open(path) as document:
        return document.pages[pdf_page - 1].extract_text() or ""


PARSERS: dict[str, Callable[[Path, int], str]] = {
    "pypdf": _pypdf_text,
    "pymupdf": _pymupdf_text,
    "pdfplumber": _pdfplumber_text,
}


def benchmark_parsers(
    *,
    database: Database,
    project_root: Path,
    reports_dir: Path,
    run_id: str,
    pages: dict[str, int] | None = None,
) -> dict[str, Any]:
    selected_pages = pages or DEFAULT_BENCHMARK_PAGES
    files = current_files(database)
    unknown_ids = sorted(set(selected_pages) - set(files))
    if unknown_ids:
        raise ExtractionError(f"Unknown benchmark document IDs: {', '.join(unknown_ids)}")

    results: list[dict[str, Any]] = []
    for document_id, pdf_page in selected_pages.items():
        file_record = files[document_id]
        path = project_root / str(file_record["relative_path"])
        parser_results: dict[str, Any] = {}
        texts: dict[str, str] = {}
        for parser_name, parser in PARSERS.items():
            started = perf_counter()
            try:
                text = parser(path, pdf_page)
                elapsed_ms = round((perf_counter() - started) * 1000, 3)
                texts[parser_name] = text
                parser_results[parser_name] = {
                    "status": "ok",
                    "elapsed_ms": elapsed_ms,
                    **analyze_text(text),
                }
            except Exception as exc:
                parser_results[parser_name] = {
                    "status": "failed",
                    "elapsed_ms": round((perf_counter() - started) * 1000, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                }

        baseline = compact_text(texts.get("pypdf", ""))
        for parser_name, text in texts.items():
            candidate = compact_text(text)
            parser_results[parser_name]["similarity_to_pypdf"] = round(
                SequenceMatcher(None, baseline, candidate, autojunk=False).ratio(),
                4,
            )
        results.append(
            {
                "document_id": document_id,
                "filename": str(file_record["filename"]),
                "pdf_page": pdf_page,
                "parsers": parser_results,
            }
        )

    payload = {
        "run_id": run_id,
        "created_at": utc_now(),
        "pages": results,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    (reports_dir / f"parser_benchmark_{run_id}.json").write_text(rendered, encoding="utf-8")
    (reports_dir / "parser_benchmark_latest.json").write_text(rendered, encoding="utf-8")
    return payload
