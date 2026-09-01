from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from history_agent.db import Database
from history_agent.errors import ExtractionError
from history_agent.extraction.full import current_files
from history_agent.extraction.models import (
    PageRecord,
    SampleDocumentResult,
    SampleExtractionSummary,
)
from history_agent.extraction.text import (
    build_failed_page_record,
    build_page_record,
    count_page_images,
    utc_now,
)

DEFAULT_SAMPLE_PROFILES: dict[str, list[int]] = {
    "cpc_history_volume_1": [20, 514],
    "deng_xiaoping_selected_works": [397, 466],
    "red_star_over_china": [1, 232],
}


def _current_files(database: Database) -> dict[str, dict[str, str | int]]:
    return current_files(database)


def extract_sample_pages(
    *,
    database: Database,
    project_root: Path,
    output_dir: Path,
    reports_dir: Path,
    run_id: str,
    profiles: dict[str, list[int]] | None = None,
) -> SampleExtractionSummary:
    started_at = utc_now()
    selected_profiles = profiles or DEFAULT_SAMPLE_PROFILES
    files = _current_files(database)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    results: list[SampleDocumentResult] = []

    for document_id, pages in selected_profiles.items():
        file_record = files.get(document_id)
        if file_record is None:
            raise ExtractionError(
                f"No current registered PDF for sample document: {document_id}. "
                "Run `history-agent corpus scan` first."
            )
        path = project_root / str(file_record["relative_path"])
        reader = PdfReader(str(path), strict=False)
        output_path = output_dir / f"{document_id}.jsonl"
        records: list[PageRecord] = []
        for pdf_page in pages:
            if pdf_page > len(reader.pages):
                raise ExtractionError(
                    f"Requested page {pdf_page} exceeds {path.name} ({len(reader.pages)} pages)."
                )
            try:
                page = reader.pages[pdf_page - 1]
                text = page.extract_text() or ""
                record = build_page_record(
                    document_id=document_id,
                    file_sha256=str(file_record["sha256"]),
                    pdf_page=pdf_page,
                    text=text,
                    image_object_count=count_page_images(page),
                    run_id=run_id,
                )
            except Exception as exc:
                record = build_failed_page_record(
                    document_id=document_id,
                    file_sha256=str(file_record["sha256"]),
                    pdf_page=pdf_page,
                    run_id=run_id,
                    exc=exc,
                )
            records.append(record)

        with output_path.open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(record.model_dump_json() + "\n")

        status_counts: dict[str, int] = {}
        for record in records:
            status_counts[record.status] = status_counts.get(record.status, 0) + 1
        results.append(
            SampleDocumentResult(
                document_id=document_id,
                filename=str(file_record["filename"]),
                pages_requested=pages,
                output_path=str(output_path),
                status_counts=status_counts,
            )
        )

    summary = SampleExtractionSummary(
        run_id=run_id,
        started_at=started_at,
        finished_at=utc_now(),
        documents=results,
    )
    rendered = summary.model_dump_json(indent=2)
    (reports_dir / f"sample_extraction_{run_id}.json").write_text(rendered, encoding="utf-8")
    (reports_dir / "sample_extraction_latest.json").write_text(rendered, encoding="utf-8")
    return summary
