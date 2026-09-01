from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from history_agent.errors import ResearchDataError
from history_agent.research.models import PersonCatalog, RelationTypeCatalog


def _load_payload(path: Path) -> object:
    if not path.is_file():
        raise ResearchDataError(f"Research catalog does not exist: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ResearchDataError(f"Invalid JSON catalog {path}: {exc}") from exc


def load_person_catalog(path: Path) -> PersonCatalog:
    payload = _load_payload(path)
    if isinstance(payload, dict) and isinstance(payload.get("people"), dict):
        # Backward-compatible reader for the original name -> aliases mapping.
        records = [
            {
                "person_id": f"legacy_person_{index}",
                "canonical_name": name,
                "aliases": [{"name": alias} for alias in aliases],
            }
            for index, (name, aliases) in enumerate(payload["people"].items(), start=1)
        ]
        payload = {"schema_version": 2, "people": records, "ambiguities": []}
    try:
        return PersonCatalog.model_validate(payload)
    except ValidationError as exc:
        raise ResearchDataError(f"Invalid person catalog {path}: {exc}") from exc


def load_relation_type_catalog(path: Path) -> RelationTypeCatalog:
    payload = _load_payload(path)
    try:
        return RelationTypeCatalog.model_validate(payload)
    except ValidationError as exc:
        raise ResearchDataError(f"Invalid relation-type catalog {path}: {exc}") from exc
