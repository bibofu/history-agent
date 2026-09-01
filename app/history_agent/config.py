from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings resolved relative to the repository root."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="HISTORY_AGENT_",
        extra="ignore",
        populate_by_name=True,
    )

    project_root: Path = Field(default_factory=default_project_root)
    docs_dir: Path = Path("docs")
    data_dir: Path = Path("data")
    database_path: Path = Path("data/history_agent.db")
    catalog_path: Path = Path("config/corpus_catalog.json")
    person_aliases_path: Path = Path("config/person_aliases.json")
    relation_types_path: Path = Path("config/relation_types.json")
    log_level: str = "INFO"
    environment: Literal["development", "test", "production"] = "development"
    research_start: date = date(1921, 1, 1)
    research_end: date = date(1978, 12, 31)
    llm_provider: Literal["deepseek"] = "deepseek"
    llm_base_url: str = "https://api.deepseek.com"
    llm_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "llm_api_key",
            "HISTORY_AGENT_LLM_API_KEY",
            "DEEPSEEK_API_KEY",
        ),
    )
    llm_model: Literal["deepseek-v4-pro", "deepseek-v4-flash"] = "deepseek-v4-pro"
    llm_thinking: bool = False
    llm_reasoning_effort: Literal["low", "high", "max"] = "high"
    llm_max_tokens: int = Field(default=2500, ge=256, le=16000)
    llm_timeout_seconds: float = Field(default=120.0, ge=10.0, le=600.0)

    @model_validator(mode="after")
    def resolve_paths_and_validate_dates(self) -> Settings:
        self.project_root = self.project_root.resolve()
        for field_name in (
            "docs_dir",
            "data_dir",
            "database_path",
            "catalog_path",
            "person_aliases_path",
            "relation_types_path",
        ):
            value = getattr(self, field_name)
            if not value.is_absolute():
                setattr(self, field_name, (self.project_root / value).resolve())
            else:
                setattr(self, field_name, value.resolve())
        if self.research_start > self.research_end:
            raise ValueError("research_start must be on or before research_end")
        return self

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def samples_dir(self) -> Path:
        return self.data_dir / "processed" / "samples"

    @property
    def pages_dir(self) -> Path:
        return self.data_dir / "processed" / "pages"

    @property
    def ocr_dir(self) -> Path:
        return self.data_dir / "processed" / "ocr"

    @property
    def structure_dir(self) -> Path:
        return self.data_dir / "processed" / "structure"

    @property
    def chunks_dir(self) -> Path:
        return self.data_dir / "processed" / "chunks"

    @property
    def keyword_index_path(self) -> Path:
        return self.data_dir / "indexes" / "keyword_fts.db"

    @property
    def vector_index_path(self) -> Path:
        return self.data_dir / "indexes" / "qdrant"

    @property
    def model_cache_dir(self) -> Path:
        return self.data_dir / "models"

    def ensure_runtime_dirs(self) -> None:
        for path in (
            self.data_dir,
            self.runs_dir,
            self.reports_dir,
            self.samples_dir,
            self.pages_dir,
            self.ocr_dir,
            self.structure_dir,
            self.chunks_dir,
            self.keyword_index_path.parent,
            self.model_cache_dir,
            self.database_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def llm_enabled(self) -> bool:
        return bool(
            self.llm_api_key and self.llm_api_key.get_secret_value().strip()
        )

    def public_snapshot(self) -> dict[str, Any]:
        return {
            "project_root": str(self.project_root),
            "docs_dir": str(self.docs_dir),
            "data_dir": str(self.data_dir),
            "database_path": str(self.database_path),
            "catalog_path": str(self.catalog_path),
            "person_aliases_path": str(self.person_aliases_path),
            "relation_types_path": str(self.relation_types_path),
            "log_level": self.log_level,
            "environment": self.environment,
            "research_start": self.research_start.isoformat(),
            "research_end": self.research_end.isoformat(),
            "llm_enabled": self.llm_enabled,
            "llm_provider": self.llm_provider,
            "llm_base_url": self.llm_base_url,
            "llm_model": self.llm_model,
            "llm_thinking": self.llm_thinking,
            "llm_reasoning_effort": self.llm_reasoning_effort,
            "llm_max_tokens": self.llm_max_tokens,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
