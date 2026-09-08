from time import monotonic

import httpx
import pytest
from fastapi.testclient import TestClient
from history_agent.answering.runtime import RequestBudget
from history_agent.config import Settings
from history_agent.web import app as web_module


def test_request_budget_rejects_work_after_deadline() -> None:
    budget = RequestBudget(monotonic() - 1)

    with pytest.raises(httpx.TimeoutException):
        budget.timeout(20)


def test_ready_endpoint_returns_service_unavailable_for_failed_probe(monkeypatch) -> None:
    monkeypatch.setattr(
        web_module,
        "readiness_snapshot",
        lambda settings: {
            "status": "not_ready",
            "components": {"database": True, "keyword": False, "vector": False},
        },
    )

    response = TestClient(web_module.create_app(Settings(_env_file=None))).get("/api/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_ready_endpoint_accepts_single_healthy_retrieval_branch(monkeypatch) -> None:
    monkeypatch.setattr(
        web_module,
        "readiness_snapshot",
        lambda settings: {
            "status": "ready",
            "components": {"database": True, "keyword": True, "vector": False},
        },
    )

    response = TestClient(web_module.create_app(Settings(_env_file=None))).get("/api/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
