import json
from pathlib import Path

import pytest
from history_agent.corpus.catalog import load_catalog
from history_agent.errors import CatalogError


def test_catalog_rejects_duplicate_document_ids(work_path: Path) -> None:
    document = {
        "document_id": "duplicate_id",
        "filename": "one.pdf",
        "title": "One",
        "source_type": "test",
        "verification_status": "checked",
    }
    path = work_path / "catalog.json"
    path.write_text(
        json.dumps({"schema_version": 1, "documents": [document, document]}),
        encoding="utf-8",
    )

    with pytest.raises(CatalogError, match="duplicate document_id"):
        load_catalog(path)


def test_project_catalog_has_sixteen_unique_documents() -> None:
    root = Path(__file__).resolve().parents[1]
    catalog = load_catalog(root / "config" / "corpus_catalog.json")

    assert len(catalog.documents) == 16
    assert len({document.document_id for document in catalog.documents}) == 16
