import json
from pathlib import Path

from history_agent.corpus.catalog import load_catalog
from history_agent.corpus.exporter import export_manifest, load_manifest
from history_agent.corpus.scanner import scan_corpus
from history_agent.db import Database
from pypdf import PdfWriter


def create_pdf(path: Path, pages: int = 2) -> None:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=300, height=400)
    writer.add_metadata({"/Title": "Fixture PDF", "/Author": "Tests"})
    with path.open("wb") as stream:
        writer.write(stream)


def test_scan_is_idempotent_and_exports_manifest(work_path: Path) -> None:
    docs_dir = work_path / "docs"
    reports_dir = work_path / "reports"
    docs_dir.mkdir()
    create_pdf(docs_dir / "fixture.pdf")
    catalog_path = work_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "documents": [
                    {
                        "document_id": "fixture_document",
                        "filename": "fixture.pdf",
                        "title": "Fixture",
                        "source_type": "test",
                        "verification_status": "checked",
                        "expected_page_count": 2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    database = Database(work_path / "history_agent.db")
    catalog = load_catalog(catalog_path)

    first = scan_corpus(
        database=database,
        catalog=catalog,
        docs_dir=docs_dir,
        project_root=work_path,
        run_id="first",
    )
    second = scan_corpus(
        database=database,
        catalog=catalog,
        docs_dir=docs_dir,
        project_root=work_path,
        run_id="second",
    )

    assert first.counts == {"new": 1}
    assert second.counts == {"unchanged": 1}
    manifest = load_manifest(database)
    assert len(manifest) == 1
    assert manifest[0]["page_count"] == 2
    assert manifest[0]["is_present"] is True
    json_path, csv_path = export_manifest(database, reports_dir)
    assert json_path.is_file()
    assert csv_path.is_file()


def test_changed_file_requires_explicit_acceptance(work_path: Path) -> None:
    docs_dir = work_path / "docs"
    docs_dir.mkdir()
    pdf_path = docs_dir / "fixture.pdf"
    create_pdf(pdf_path, pages=2)
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
    catalog = load_catalog(catalog_path)
    scan_corpus(
        database=database,
        catalog=catalog,
        docs_dir=docs_dir,
        project_root=work_path,
        run_id="initial",
    )

    create_pdf(pdf_path, pages=3)
    blocked = scan_corpus(
        database=database,
        catalog=catalog,
        docs_dir=docs_dir,
        project_root=work_path,
        run_id="blocked-change",
    )
    assert blocked.counts == {"changed": 1}
    assert load_manifest(database)[0]["page_count"] == 2

    accepted = scan_corpus(
        database=database,
        catalog=catalog,
        docs_dir=docs_dir,
        project_root=work_path,
        run_id="accepted-change",
        accept_changes=True,
    )
    assert accepted.counts == {"changed": 1}
    assert load_manifest(database)[0]["page_count"] == 3
