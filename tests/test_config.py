from datetime import date
from pathlib import Path

import pytest
from history_agent.config import Settings


def test_settings_resolve_paths(work_path: Path) -> None:
    settings = Settings(project_root=work_path)

    assert settings.docs_dir == (work_path / "docs").resolve()
    assert settings.database_path == (work_path / "data" / "history_agent.db").resolve()
    assert settings.research_start == date(1921, 1, 1)
    assert settings.research_end == date(1978, 12, 31)
    assert settings.environment == "development"


def test_invalid_research_range_is_rejected(work_path: Path) -> None:
    with pytest.raises(ValueError, match="research_start"):
        Settings(
            project_root=work_path,
            research_start=date(1979, 1, 1),
            research_end=date(1978, 12, 31),
        )


def test_standard_deepseek_key_environment_is_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    settings = Settings(_env_file=None)

    assert settings.llm_enabled is True
    assert settings.llm_model == "deepseek-v4-pro"
    assert settings.llm_thinking is False
    assert settings.llm_api_key is not None
    assert settings.llm_api_key.get_secret_value() == "sk-test"
