from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class RunRecord:
    operation: str
    config: dict[str, Any]
    run_id: str = field(default_factory=lambda: uuid4().hex)
    status: str = "running"
    started_at: str = field(default_factory=utc_now)
    finished_at: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)


class RunTracker:
    """Persist a small, auditable record for each batch operation."""

    def __init__(self, runs_dir: Path, operation: str, config: dict[str, Any]):
        self.runs_dir = runs_dir
        self.record = RunRecord(operation=operation, config=config)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._write()

    @property
    def run_id(self) -> str:
        return self.record.run_id

    def finish(self, summary: dict[str, Any] | None = None) -> None:
        self.record.status = "succeeded"
        self.record.finished_at = utc_now()
        self.record.summary = summary or {}
        self._write()

    def fail(self, exc: BaseException, **context: Any) -> None:
        self.record.status = "failed"
        self.record.finished_at = utc_now()
        self.record.errors.append(
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "context": context,
            }
        )
        self._write()

    def _write(self) -> None:
        target = self.runs_dir / f"{self.record.run_id}.json"
        target.write_text(
            json.dumps(asdict(self.record), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
