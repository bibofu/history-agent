from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from history_agent.corpus.models import CorpusScanSummary
from history_agent.db import Database


def load_manifest(database: Database) -> list[dict[str, Any]]:
    with database.connect() as connection:
        rows = connection.execute(
            """
            SELECT
                d.document_id, d.title, d.creators_json, d.source_type, d.edition,
                d.volume, d.source_series_json, d.verification_status, d.enabled,
                d.ocr_strategy, d.expected_page_count, d.notes,
                f.relative_path, f.filename, f.sha256, f.size_bytes, f.page_count,
                f.modified_at, f.pdf_title, f.pdf_author, f.is_encrypted,
                f.is_present
            FROM documents d
            LEFT JOIN document_files f
              ON f.document_id = d.document_id AND f.is_current = 1
            ORDER BY d.document_id
            """
        ).fetchall()
    manifest: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["creators"] = json.loads(item.pop("creators_json"))
        item["source_series"] = json.loads(item.pop("source_series_json"))
        item["enabled"] = bool(item["enabled"])
        if item["is_encrypted"] is not None:
            item["is_encrypted"] = bool(item["is_encrypted"])
        if item["is_present"] is not None:
            item["is_present"] = bool(item["is_present"])
        manifest.append(item)
    return manifest


def export_manifest(database: Database, reports_dir: Path) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(database)
    json_path = reports_dir / "corpus_manifest.json"
    csv_path = reports_dir / "corpus_manifest.csv"
    json_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fieldnames = sorted({key for item in manifest for key in item})
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for item in manifest:
            row = dict(item)
            row["creators"] = "; ".join(row["creators"])
            row["source_series"] = "; ".join(row["source_series"])
            writer.writerow(row)
    return json_path, csv_path


def export_diff(summary: CorpusScanSummary, reports_dir: Path) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = summary.model_dump()
    payload["counts"] = summary.counts
    run_path = reports_dir / f"corpus_diff_{summary.run_id}.json"
    latest_path = reports_dir / "corpus_diff_latest.json"
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    run_path.write_text(rendered, encoding="utf-8")
    latest_path.write_text(rendered, encoding="utf-8")
    return run_path, latest_path


def load_latest_diff(database: Database) -> dict[str, Any] | None:
    with database.connect() as connection:
        row = connection.execute(
            """
            SELECT summary_json FROM corpus_scan_runs
            ORDER BY finished_at DESC LIMIT 1
            """
        ).fetchone()
    if row is None:
        return None
    summary = CorpusScanSummary.model_validate_json(row["summary_json"])
    payload = summary.model_dump()
    payload["counts"] = summary.counts
    return payload
