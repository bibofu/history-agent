"""Functional readiness probes for local storage and retrieval backends."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from history_agent.config import Settings
from history_agent.retrieval.keyword import search_keyword_index
from history_agent.retrieval.vector import search_vector_index


def _database_ready(path: Path) -> bool:
    if not path.is_file():
        return False
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        connection.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
        return True
    finally:
        connection.close()


def readiness_snapshot(settings: Settings) -> dict[str, object]:
    components: dict[str, bool] = {}
    try:
        components["database"] = _database_ready(settings.database_path)
    except (OSError, sqlite3.Error):
        components["database"] = False
    try:
        search_keyword_index(
            index_path=settings.keyword_index_path,
            query="中国共产党",
            aliases_path=settings.person_aliases_path,
            top_k=1,
        )
        components["keyword"] = True
    except Exception:
        components["keyword"] = False
    try:
        search_vector_index(
            index_path=settings.vector_index_path,
            model_cache_dir=settings.model_cache_dir / "fastembed",
            aliases_path=settings.person_aliases_path,
            query="中国共产党",
            top_k=1,
        )
        components["vector"] = True
    except Exception:
        components["vector"] = False
    ready = components["database"] and (components["keyword"] or components["vector"])
    return {
        "status": "ready" if ready else "not_ready",
        "components": components,
    }
