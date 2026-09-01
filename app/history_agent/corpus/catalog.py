from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from history_agent.corpus.models import CorpusCatalog
from history_agent.errors import CatalogError


def load_catalog(path: Path) -> CorpusCatalog:
    if not path.is_file():
        raise CatalogError(f"Corpus catalog not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return CorpusCatalog.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise CatalogError(f"Invalid corpus catalog {path}: {exc}") from exc
