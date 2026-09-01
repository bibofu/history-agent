from __future__ import annotations

import re
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def work_path(request: pytest.FixtureRequest) -> Iterator[Path]:
    """Use a workspace-local test directory without pytest's Windows chmod behavior."""

    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", request.node.name)
    path = Path.cwd() / "data" / "test-work" / safe_name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    yield path
    if path.exists():
        shutil.rmtree(path)
