"""Offline LLM judge for claim-to-citation entailment.

This module is deliberately evaluation-only. Production answers retain their
low-latency deterministic citation validator; the slower semantic check runs in
explicit benchmark jobs where false attribution can be inspected and measured.
"""

from __future__ import annotations

import re
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field, ValidationError, model_validator

from history_agent.answering.models import Citation
from history_agent.config import Settings

CLAIM_TOKEN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9]")


class SemanticJudgment(BaseModel):
    verdict: Literal["pass", "fail"]
    unsupported_claims: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1, max_length=1200)

    @model_validator(mode="after")
    def validate_consistency(self) -> SemanticJudgment:
        if self.verdict == "pass" and self.unsupported_claims:
            raise ValueError("passing judgment cannot list unsupported claims")
        if self.verdict == "fail" and not self.unsupported_claims:
            raise ValueError("failing judgment must identify unsupported claims")
        return self


class SemanticCitationCase(BaseModel):
    case_id: str
    question: str
    answer: str
    citations: list[Citation] = Field(min_length=1)
    expected_verdict: Literal["pass", "fail"]
    semantic_criteria: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class SemanticCitationCaseSet(BaseModel):
    schema_version: int
    cases: list[SemanticCitationCase] = Field(min_length=1)


class FactCoverageDecision(BaseModel):
    fact_id: str
    covered: bool
    reason: str = Field(min_length=1, max_length=500)


class FactCoverageJudgment(BaseModel):
    decisions: list[FactCoverageDecision] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_fact_ids(self) -> FactCoverageJudgment:
        ids = [item.fact_id for item in self.decisions]
        if len(ids) != len(set(ids)):
            raise ValueError("fact coverage judgment contains duplicate fact IDs")
        return self


def load_semantic_case_set(path: Path) -> SemanticCitationCaseSet:
    return SemanticCitationCaseSet.model_validate_json(path.read_text(encoding="utf-8"))


def _judge_messages(
    question: str,
    answer: str,
    citations: list[Citation],
    semantic_criteria: list[str],
) -> list[dict[str, str]]:
    evidence = "\n\n".join(
        f"[{citation.evidence_id}] {citation.document}，PDF第{citation.pdf_page}页\n"
        f"章节：{' > '.join(citation.section) or '未识别'}\n"
        f"{citation.quote}"
        for citation in citations
    )
    criteria = "\n".join(f"- {item}" for item in semantic_criteria) or "- 无额外标准"
    system = (
        "你是严格的史料回答评测员，只判断回答中的事实性主张是否被它紧邻标注的引文"
        "直接支持，不使用外部知识。姓名同段出现不等于共同参与，名单出现不等于担任同一"
        "职务，时间接近不等于同年发生，原文相关不等于支持更强的因果、评价或互动结论。"
        "回答若明确说证据不足以确认某结论，不应把这种审慎限定误判为错误。允许忠实同义"
        "改写，不因措辞不同而判错；同一段末尾的多个编号可以合并支持该段主张。忽略纯粹的"
        "篇章组织和资料范围提示。‘随后’只表达先后，不自动支持‘因此’‘导致’等因果。"
        "逐条核对后只输出JSON对象，verdict、unsupported_claims、rationale 三者必须一致："
        "pass 时 unsupported_claims 必须为空；fail 时必须列出至少一条具体错误主张。"
        "不得把问题中的待验证说法当成回答已经提出的主张；unsupported_claims 的每一项必须"
        "从回答正文逐字复制，不得改写、补充或省略否定词。输出格式为"
        '{"verdict":"pass|fail","unsupported_claims":["未获支持的完整主张"],'
        '"rationale":"简短判定依据"}。任一事实主张被错误引文挂靠时 verdict 必须为 fail。'
    )
    user = (
        f"问题：{question}\n\n回答：\n{answer}\n\n可用引文：\n{evidence}"
        f"\n\n本题额外判定标准：\n{criteria}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _normalize_claim(value: str) -> str:
    return "".join(CLAIM_TOKEN.findall(value)).casefold()


def _validate_claim_spans(judgment: SemanticJudgment, answer: str) -> None:
    normalized_answer = _normalize_claim(answer)
    if any(
        _normalize_claim(claim) not in normalized_answer for claim in judgment.unsupported_claims
    ):
        raise ValueError("unsupported claim was not copied from the answer")


def _error_code(exc: httpx.HTTPError) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return {
            401: "authentication_failed",
            402: "insufficient_balance",
            429: "rate_limited",
        }.get(exc.response.status_code, f"http_{exc.response.status_code}")
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    return "network_error"


def judge_citation_semantics(
    settings: Settings,
    *,
    question: str,
    answer: str,
    citations: list[Citation],
    semantic_criteria: list[str] | None = None,
) -> dict[str, Any]:
    """Judge whether each cited claim is entailed by the cited local excerpts."""

    started = perf_counter()
    if not settings.llm_enabled:
        return {
            "status": "not_configured",
            "verdict": None,
            "unsupported_claims": [],
            "rationale": "语义评测需要配置 LLM。",
            "latency_ms": 0,
            "usage": None,
        }
    assert settings.llm_api_key is not None
    usage: dict[str, int] = {}
    judgment: SemanticJudgment | None = None
    for attempt in range(2):
        try:
            response = httpx.post(
                settings.llm_base_url.rstrip("/") + "/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.llm_api_key.get_secret_value()}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.llm_model,
                    "messages": _judge_messages(
                        question, answer, citations, semantic_criteria or []
                    ),
                    "response_format": {"type": "json_object"},
                    "stream": False,
                    "max_tokens": 1200,
                    "thinking": {"type": "disabled"},
                },
                timeout=settings.llm_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            choice = payload["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ValueError("judge output reached token limit")
            judgment = SemanticJudgment.model_validate_json(choice["message"]["content"])
            _validate_claim_spans(judgment, answer)
            raw_usage = payload.get("usage", {})
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if key in raw_usage:
                    usage[key] = usage.get(key, 0) + int(raw_usage[key])
            break
        except httpx.HTTPError as exc:
            return {
                "status": _error_code(exc),
                "verdict": None,
                "unsupported_claims": [],
                "rationale": "语义评测请求失败。",
                "latency_ms": round((perf_counter() - started) * 1000),
                "usage": usage or None,
            }
        except (KeyError, IndexError, TypeError, ValueError, ValidationError):
            if attempt == 0:
                continue
    if judgment is None:
        return {
            "status": "invalid_response",
            "verdict": None,
            "unsupported_claims": [],
            "rationale": "语义评测连续两次返回无效结构。",
            "latency_ms": round((perf_counter() - started) * 1000),
            "usage": usage or None,
        }
    return {
        "status": "used",
        "verdict": judgment.verdict,
        "unsupported_claims": judgment.unsupported_claims,
        "rationale": judgment.rationale,
        "latency_ms": round((perf_counter() - started) * 1000),
        "usage": usage or None,
    }


def _fact_judge_messages(
    question: str,
    answer: str,
    facts: list[dict[str, str]],
) -> list[dict[str, str]]:
    fact_lines = "\n".join(f"- {item['fact_id']}: {item['claim']}" for item in facts)
    system = (
        "你是严格的历史问答覆盖度评测员。只判断回答正文是否明确表达了每条给定事实；"
        "允许忠实同义改写，但不得使用外部知识、问题中的暗示或引文原文替回答补全事实。"
        "对每个 fact_id 都必须返回一项，不能增加、删除或重复 ID。只输出 JSON 对象，格式为"
        '{"decisions":[{"fact_id":"f1","covered":true,'
        '"reason":"回答中对应的简短依据"}]}。reason 必须解释正文为何覆盖或未覆盖。'
    )
    user = f"问题：{question}\n\n回答：\n{answer}\n\n待核对事实：\n{fact_lines}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def judge_fact_coverage(
    settings: Settings,
    *,
    question: str,
    answer: str,
    facts: list[dict[str, str]],
) -> dict[str, Any]:
    """Semantically judge gold facts that deterministic matching could not resolve."""

    started = perf_counter()
    if not facts:
        return {
            "status": "not_applicable",
            "decisions": [],
            "latency_ms": 0,
            "usage": None,
        }
    if not settings.llm_enabled:
        return {
            "status": "not_configured",
            "decisions": [],
            "latency_ms": 0,
            "usage": None,
        }
    expected_ids = {item["fact_id"] for item in facts}
    if len(expected_ids) != len(facts):
        raise ValueError("fact coverage input requires unique fact IDs")
    assert settings.llm_api_key is not None
    usage: dict[str, int] = {}
    judgment: FactCoverageJudgment | None = None
    for attempt in range(2):
        try:
            response = httpx.post(
                settings.llm_base_url.rstrip("/") + "/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.llm_api_key.get_secret_value()}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.llm_model,
                    "messages": _fact_judge_messages(question, answer, facts),
                    "response_format": {"type": "json_object"},
                    "stream": False,
                    "max_tokens": 1200,
                    "thinking": {"type": "disabled"},
                },
                timeout=settings.llm_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            choice = payload["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ValueError("fact judge output reached token limit")
            candidate = FactCoverageJudgment.model_validate_json(
                choice["message"]["content"]
            )
            if {item.fact_id for item in candidate.decisions} != expected_ids:
                raise ValueError("fact judge did not return the requested fact IDs")
            judgment = candidate
            raw_usage = payload.get("usage", {})
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if key in raw_usage:
                    usage[key] = usage.get(key, 0) + int(raw_usage[key])
            break
        except httpx.HTTPError as exc:
            return {
                "status": _error_code(exc),
                "decisions": [],
                "latency_ms": round((perf_counter() - started) * 1000),
                "usage": usage or None,
            }
        except (KeyError, IndexError, TypeError, ValueError, ValidationError):
            if attempt == 0:
                continue
    if judgment is None:
        return {
            "status": "invalid_response",
            "decisions": [],
            "latency_ms": round((perf_counter() - started) * 1000),
            "usage": usage or None,
        }
    return {
        "status": "used",
        "decisions": [item.model_dump() for item in judgment.decisions],
        "latency_ms": round((perf_counter() - started) * 1000),
        "usage": usage or None,
    }


def evaluate_semantic_cases(settings: Settings, case_set_path: Path) -> dict[str, Any]:
    case_set = load_semantic_case_set(case_set_path)
    results: list[dict[str, Any]] = []
    evaluated = 0
    correct = 0
    for case in case_set.cases:
        judgment = judge_citation_semantics(
            settings,
            question=case.question,
            answer=case.answer,
            citations=case.citations,
            semantic_criteria=case.semantic_criteria,
        )
        verdict = judgment["verdict"]
        is_correct = verdict == case.expected_verdict if verdict is not None else None
        evaluated += int(verdict is not None)
        correct += int(is_correct is True)
        results.append(
            {
                "case_id": case.case_id,
                "expected_verdict": case.expected_verdict,
                "actual_verdict": verdict,
                "correct": is_correct,
                "tags": case.tags,
                "judgment": judgment,
            }
        )
    return {
        "case_set": str(case_set_path),
        "cases": len(case_set.cases),
        "evaluated": evaluated,
        "accuracy": round(correct / evaluated, 6) if evaluated else 0.0,
        "coverage": round(evaluated / len(case_set.cases), 6),
        "results": results,
    }
