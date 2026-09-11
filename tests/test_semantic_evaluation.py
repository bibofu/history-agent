import json
from pathlib import Path
from typing import Any

import pytest
from history_agent.answering.models import Citation
from history_agent.config import Settings
from history_agent.evaluation import semantic


def _citation() -> Citation:
    return Citation(
        evidence_id="E1",
        document_id="fixture",
        document="测试文献",
        pdf_page=1,
        section=["测试"],
        quote="会议由周恩来主持，陈毅作外交工作报告。",
        source_type="test",
        verification_status="reviewed",
        extraction_methods=["manual"],
    )


class _Response:
    def __init__(self, content: str) -> None:
        self._content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {"content": self._content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        }


def test_semantic_judge_returns_structured_failure(work_path: Path, monkeypatch: Any) -> None:
    settings = Settings(
        _env_file=None,
        project_root=work_path,
        data_dir=Path("data"),
        llm_api_key="test-key",
    )
    content = json.dumps(
        {
            "verdict": "fail",
            "unsupported_claims": ["周恩来作外交工作报告"],
            "rationale": "引文说作报告者是陈毅。",
        },
        ensure_ascii=False,
    )
    monkeypatch.setattr(semantic.httpx, "post", lambda *args, **kwargs: _Response(content))

    result = semantic.judge_citation_semantics(
        settings,
        question="谁作外交工作报告？",
        answer="周恩来作外交工作报告。[E1]",
        citations=[_citation()],
    )

    assert result["status"] == "used"
    assert result["verdict"] == "fail"
    assert result["unsupported_claims"] == ["周恩来作外交工作报告"]
    assert result["usage"]["total_tokens"] == 120


def test_semantic_judge_rejects_claim_invented_from_question() -> None:
    judgment = semantic.SemanticJudgment(
        verdict="fail",
        unsupported_claims=["毛泽东和林彪共同指挥了行动"],
        rationale="没有共同指挥证据。",
    )

    with pytest.raises(ValueError, match="not copied"):
        semantic._validate_claim_spans(
            judgment,
            "史料只显示林彪列席会议，未显示两人共同指挥。",
        )


def test_semantic_calibration_measures_detection_accuracy(
    work_path: Path, monkeypatch: Any
) -> None:
    case_path = work_path / "semantic.json"
    case_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "case_id": "supported",
                        "question": "谁主持会议？",
                        "answer": "周恩来主持会议。[E1]",
                        "citations": [_citation().model_dump()],
                        "expected_verdict": "pass",
                    },
                    {
                        "case_id": "unsupported",
                        "question": "谁作报告？",
                        "answer": "周恩来作报告。[E1]",
                        "citations": [_citation().model_dump()],
                        "expected_verdict": "fail",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    settings = Settings(
        _env_file=None,
        project_root=work_path,
        data_dir=Path("data"),
        llm_api_key="test-key",
    )

    def fake_judge(*args: Any, answer: str, **kwargs: Any) -> dict[str, Any]:
        verdict = "pass" if "主持" in answer else "fail"
        return {
            "status": "used",
            "verdict": verdict,
            "unsupported_claims": [],
            "rationale": "fixture",
            "latency_ms": 1,
            "usage": None,
        }

    monkeypatch.setattr(semantic, "judge_citation_semantics", fake_judge)
    result = semantic.evaluate_semantic_cases(settings, case_path)

    assert result["coverage"] == 1.0
    assert result["accuracy"] == 1.0


def test_fact_coverage_judge_returns_structured_per_fact_results(
    work_path: Path, monkeypatch: Any
) -> None:
    settings = Settings(
        _env_file=None,
        project_root=work_path,
        data_dir=Path("data"),
        llm_api_key="test-key",
    )
    content = json.dumps(
        {
            "decisions": [
                {"fact_id": "f1", "covered": True, "reason": "回答明确提及。"},
                {"fact_id": "f2", "covered": False, "reason": "回答没有提及。"},
            ]
        },
        ensure_ascii=False,
    )
    monkeypatch.setattr(semantic.httpx, "post", lambda *args, **kwargs: _Response(content))

    result = semantic.judge_fact_coverage(
        settings,
        question="发生了什么？",
        answer="回答只覆盖第一项。",
        facts=[
            {"fact_id": "f1", "claim": "第一项事实"},
            {"fact_id": "f2", "claim": "第二项事实"},
        ],
    )

    assert result["status"] == "used"
    assert result["decisions"] == [
        {"fact_id": "f1", "covered": True, "reason": "回答明确提及。"},
        {"fact_id": "f2", "covered": False, "reason": "回答没有提及。"},
    ]
    assert result["usage"]["total_tokens"] == 120
