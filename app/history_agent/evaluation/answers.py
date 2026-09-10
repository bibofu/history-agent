from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from history_agent import __version__
from history_agent.answering.models import AnswerResponse, Citation, QuestionRequest
from history_agent.answering.service import PROMPT_VERSION, answer_question
from history_agent.answering.validation import validate_grounded_answer
from history_agent.config import Settings
from history_agent.evaluation.retrieval import RetrievalQuestion, load_question_set
from history_agent.evaluation.semantic import (
    evaluate_semantic_cases,
    judge_citation_semantics,
)
from history_agent.extraction.report import load_page_records
from history_agent.processing.chunks import effective_pages

TEXT_TOKEN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9]")
CATEGORY_MINIMUMS = {
    "timeline": 8,
    "intersection": 5,
    "viewpoint": 6,
    "event": 5,
    "organization": 3,
    "refusal": 3,
    "complex": 3,
    "intersection_negative": 2,
}


def _normalize_text(value: str) -> str:
    return "".join(TEXT_TOKEN.findall(value)).casefold()


def quote_matches_page(quote: str, page_text: str) -> bool:
    normalized_quote = _normalize_text(quote)
    normalized_page = _normalize_text(page_text)
    if len(normalized_quote) < 12 or not normalized_page:
        return False
    if normalized_quote in normalized_page:
        return True
    anchor_size = min(24, max(12, len(normalized_quote) // 5))
    anchors = [
        normalized_quote[start : start + anchor_size]
        for start in range(0, len(normalized_quote) - anchor_size + 1, anchor_size)
    ]
    return bool(anchors) and sum(anchor in normalized_page for anchor in anchors) / len(
        anchors
    ) >= 0.6


def _load_effective_page_texts(settings: Settings) -> dict[tuple[str, int], str]:
    texts: dict[tuple[str, int], str] = {}
    for page_path in sorted(settings.pages_dir.glob("*.jsonl")):
        document_id = page_path.stem
        base_records = load_page_records(page_path)
        ocr_path = settings.ocr_dir / page_path.name
        ocr_records = load_page_records(ocr_path) if ocr_path.is_file() else {}
        for pdf_page, record in effective_pages(base_records, ocr_records).items():
            texts[(document_id, pdf_page)] = record.normalized_text
    return texts


def _gold_pages(item: RetrievalQuestion) -> set[tuple[str, int]]:
    return {
        (evidence.document_id, page)
        for evidence in item.expected_evidence
        for page in evidence.pdf_pages
    }


def _fact_coverage(item: RetrievalQuestion, response: AnswerResponse) -> tuple[int, int]:
    answer_text = _normalize_text(response.answer)
    covered = sum(
        any(_normalize_text(term) in answer_text for term in alternatives)
        for alternatives in item.required_fact_terms
    )
    return covered, len(item.required_fact_terms)


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def _latency_summary(values: list[int]) -> dict[str, int]:
    return {
        "count": len(values),
        "mean_ms": round(mean(values)) if values else 0,
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "max_ms": max(values, default=0),
    }


def _sum_usage(results: list[dict[str, Any]], field: str) -> dict[str, int]:
    usages = [result.get(field) for result in results]
    keys = {key for usage in usages if isinstance(usage, dict) for key in usage}
    return {
        key: sum(int(usage.get(key, 0)) for usage in usages if isinstance(usage, dict))
        for key in sorted(keys)
    }


def _citation_check(
    citation: Citation, page_texts: dict[tuple[str, int], str]
) -> dict[str, object]:
    page_text = page_texts.get((citation.document_id, citation.pdf_page), "")
    matched = quote_matches_page(citation.quote, page_text)
    return {
        "evidence_id": citation.evidence_id,
        "document_id": citation.document_id,
        "document": citation.document,
        "pdf_page": citation.pdf_page,
        "section": citation.section,
        "quote": citation.quote,
        "quote_matches_page": matched,
    }


def _index_versions(settings: Settings) -> dict[str, object]:
    versions: dict[str, object] = {}
    for name in ("keyword_index_latest.json", "vector_index_latest.json"):
        path = settings.reports_dir / name
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        versions[name.removesuffix("_latest.json")] = {
            key: payload.get(key)
            for key in ("run_id", "index_version", "model_name", "chunks")
            if payload.get(key) is not None
        }
    return versions


def _render_markdown(payload: dict[str, Any]) -> str:
    passed = "通过" if payload["passed"] else "未通过"
    metrics = payload["metrics"]
    lines = [
        "# MVP 回答评估报告",
        "",
        f"- 结论：**{passed}**",
        f"- 运行 ID：`{payload['run_id']}`",
        f"- 题目数：{payload['questions']}",
        f"- 回答模式：{payload['answer_mode']}",
        f"- 题集 SHA-256：`{payload['question_set_sha256']}`",
        "",
        "## 核心指标",
        "",
        f"- 关键页命中率：{metrics['gold_page_hit_rate']:.2%}",
        f"- 引用页文本一致率：{metrics['citation_page_accuracy']:.2%}",
        f"- 核心事实引用校验通过率：{metrics['grounding_pass_rate']:.2%}",
        f"- 引用语义支持率：{metrics['semantic_grounding_pass_rate']:.2%}",
        f"- 语义评测覆盖率：{metrics['semantic_judge_coverage']:.2%}",
        f"- 语义错误校准集准确率：{metrics['semantic_calibration_accuracy']:.2%}",
        f"- 必要事实覆盖率：{metrics['required_fact_coverage']:.2%}",
        f"- 拒答正确率：{metrics['refusal_accuracy']:.2%}",
        f"- 综合可回答性准确率：{metrics['answerability_accuracy']:.2%}",
        f"- 有证据回答的 LLM 生成率：{metrics['llm_generation_rate']:.2%}",
        f"- 端到端延迟 P50 / P95：{metrics['latency_p50_ms']} / {metrics['latency_p95_ms']} ms",
        f"- 复杂问题延迟 P50 / P95：{metrics['complex_latency_p50_ms']} / "
        f"{metrics['complex_latency_p95_ms']} ms",
        "",
        "## 验收闸门",
        "",
    ]
    for gate, value in payload["gates"].items():
        lines.append(f"- [{'x' if value else ' '}] {gate}")
    failures = [result for result in payload["results"] if not result["success"]]
    lines.extend(["", "## 失败样本", ""])
    if not failures:
        lines.append("无。")
    else:
        for result in failures:
            reasons = "；".join(result["failure_reasons"])
            lines.append(f"- `{result['question_id']}`：{reasons}")
    lines.append("")
    return "\n".join(lines)


def evaluate_answers(
    *,
    settings: Settings,
    question_set_path: Path,
    top_k: int,
    run_id: str,
    use_llm: bool = False,
    semantic_judge: bool = False,
    semantic_case_set_path: Path | None = None,
) -> dict[str, Any]:
    question_set = load_question_set(question_set_path)
    page_texts = _load_effective_page_texts(settings)
    active_settings = (
        settings if use_llm else settings.model_copy(update={"llm_api_key": None})
    )
    results: list[dict[str, Any]] = []
    citation_checks: list[dict[str, object]] = []
    gold_hits = 0
    gold_questions = 0
    grounding_passes = 0
    grounding_questions = 0
    fact_covered = 0
    fact_total = 0
    refusal_correct = 0
    refusal_total = 0
    answerability_correct = 0
    category_counts: Counter[str] = Counter()
    semantic_judged = 0
    semantic_passes = 0
    latencies: list[int] = []
    complex_latencies: list[int] = []
    standard_latencies: list[int] = []

    for item in question_set.questions:
        category_counts.update(item.tags)
        started = perf_counter()
        response = answer_question(
            active_settings,
            QuestionRequest(question=item.question, top_k=top_k),
        )
        latency_ms = round((perf_counter() - started) * 1000)
        latencies.append(latency_ms)
        if "complex" in item.tags:
            complex_latencies.append(latency_ms)
        else:
            standard_latencies.append(latency_ms)
        returned_pages = {
            (citation.document_id, citation.pdf_page) for citation in response.citations
        }
        expected_pages = _gold_pages(item)
        gold_hit = None if not expected_pages else bool(returned_pages & expected_pages)
        if gold_hit is not None:
            gold_questions += 1
            gold_hits += int(gold_hit)

        current_checks = [
            _citation_check(citation, page_texts) for citation in response.citations
        ]
        citation_checks.extend(current_checks)
        grounding_valid = True
        if response.citations:
            grounding_questions += 1
            grounding_valid = validate_grounded_answer(
                response.answer, response.citations
            ).valid
            grounding_passes += int(grounding_valid)

        semantic_result: dict[str, Any] | None = None
        if semantic_judge and response.citations:
            semantic_result = judge_citation_semantics(
                active_settings,
                question=item.question,
                answer=response.answer,
                citations=response.citations,
                semantic_criteria=item.semantic_criteria,
            )
            if semantic_result["verdict"] is not None:
                semantic_judged += 1
                semantic_passes += int(semantic_result["verdict"] == "pass")

        covered, required = _fact_coverage(item, response)
        fact_covered += covered
        fact_total += required
        forbidden_found = [
            term for term in item.forbidden_answer_terms if term in response.answer
        ]
        has_answer = response.evidence_status != "no_evidence" and bool(response.citations)
        correct_answerability = has_answer == item.answerable
        answerability_correct += int(correct_answerability)
        if not item.answerable:
            refusal_total += 1
            refusal_correct += int(not has_answer)

        failure_reasons: list[str] = []
        if not correct_answerability:
            failure_reasons.append("可回答性判断错误")
        if gold_hit is False:
            failure_reasons.append("未命中人工登记的关键页")
        if any(not check["quote_matches_page"] for check in current_checks):
            failure_reasons.append("引用摘录与对应 PDF 页文本不一致")
        if not grounding_valid:
            failure_reasons.append("核心事实引用校验未通过")
        if semantic_result is not None and semantic_result["verdict"] == "fail":
            failure_reasons.append("回答中的事实主张未被所标引文语义支持")
        if semantic_result is not None and semantic_result["verdict"] is None:
            failure_reasons.append("引用语义评测未完成")
        if covered < required:
            failure_reasons.append(f"必要事实仅覆盖 {covered}/{required}")
        if forbidden_found:
            failure_reasons.append("出现禁止内容：" + "、".join(forbidden_found))
        results.append(
            {
                "question_id": item.question_id,
                "question": item.question,
                "answerable": item.answerable,
                "success": not failure_reasons,
                "failure_reasons": failure_reasons,
                "evidence_status": response.evidence_status,
                "generator_mode": response.generator_mode,
                "llm_status": response.llm_status,
                "llm_error_code": response.llm_error_code,
                "llm_usage": response.llm_usage,
                "query_planner_status": response.query_planner_status,
                "query_planner_usage": response.query_planner_usage,
                "retrieval_reflection_status": response.retrieval_reflection_status,
                "retrieval_rounds": response.retrieval_rounds,
                "retrieval_missing_aspects": response.retrieval_missing_aspects,
                "retrieval_reflection_usage": response.retrieval_reflection_usage,
                "retrieval_reflection_error_code": (
                    response.retrieval_reflection_error_code
                ),
                "gold_page_hit": gold_hit,
                "required_facts_covered": covered,
                "required_facts_total": required,
                "forbidden_terms_found": forbidden_found,
                "answer": response.answer,
                "citation_count": len(response.citations),
                "citations": current_checks,
                "semantic_judgment": semantic_result,
                "latency_ms": latency_ms,
            }
        )

    total_questions = len(question_set.questions)
    citation_matches = sum(
        int(bool(check["quote_matches_page"])) for check in citation_checks
    )
    citation_total = len(citation_checks)
    evidence_backed = sum(result["citation_count"] > 0 for result in results)
    llm_generated = sum(result["generator_mode"] == "llm" for result in results)
    semantic_expected = evidence_backed if semantic_judge else 0
    semantic_calibration = None
    if semantic_judge and semantic_case_set_path is not None:
        semantic_calibration = evaluate_semantic_cases(active_settings, semantic_case_set_path)
    latency = _latency_summary(latencies)
    complex_latency = _latency_summary(complex_latencies)
    standard_latency = _latency_summary(standard_latencies)
    metrics = {
        "gold_page_hit_rate": gold_hits / gold_questions if gold_questions else 0.0,
        "citation_page_accuracy": (
            citation_matches / citation_total if citation_total else 0.0
        ),
        "grounding_pass_rate": (
            grounding_passes / grounding_questions if grounding_questions else 0.0
        ),
        "semantic_grounding_pass_rate": (
            semantic_passes / semantic_judged if semantic_judged else 0.0
        ),
        "semantic_judge_coverage": (
            semantic_judged / semantic_expected if semantic_expected else 0.0
        ),
        "semantic_calibration_accuracy": (
            float(semantic_calibration["accuracy"]) if semantic_calibration is not None else 0.0
        ),
        "required_fact_coverage": fact_covered / fact_total if fact_total else 0.0,
        "refusal_accuracy": refusal_correct / refusal_total if refusal_total else 0.0,
        "answerability_accuracy": answerability_correct / total_questions,
        "llm_generation_rate": llm_generated / evidence_backed if evidence_backed else 0.0,
        "latency_mean_ms": latency["mean_ms"],
        "latency_p50_ms": latency["p50_ms"],
        "latency_p95_ms": latency["p95_ms"],
        "latency_max_ms": latency["max_ms"],
        "complex_latency_p50_ms": complex_latency["p50_ms"],
        "complex_latency_p95_ms": complex_latency["p95_ms"],
        "retrieval_reflection_retry_rate": (
            sum(result["retrieval_rounds"] > 1 for result in results) / total_questions
            if total_questions
            else 0.0
        ),
    }
    gates = {
        "至少 30 个固定问题": total_questions >= 30,
        "题型数量满足需求基线": all(
            category_counts[tag] >= minimum
            for tag, minimum in CATEGORY_MINIMUMS.items()
        ),
        "所有可回答问题已登记关键页": gold_questions
        == sum(item.answerable for item in question_set.questions),
        "关键页命中率不低于 85%": metrics["gold_page_hit_rate"] >= 0.85,
        "引用页文本一致率不低于 95%": metrics["citation_page_accuracy"] >= 0.95,
        "核心事实引用校验通过率为 100%": metrics["grounding_pass_rate"] == 1.0,
        "必要事实覆盖率不低于 80%": metrics["required_fact_coverage"] >= 0.8,
        "拒答正确率为 100%": metrics["refusal_accuracy"] == 1.0,
    }
    if semantic_judge:
        gates.update(
            {
                "所有 LLM 回答均完成引用语义评测": metrics["semantic_judge_coverage"] == 1.0,
                "LLM 回答引用语义支持率不低于 95%": metrics["semantic_grounding_pass_rate"] >= 0.95,
                "语义错误校准集准确率为 100%": metrics["semantic_calibration_accuracy"] == 1.0,
                "所有有证据回答均由 LLM 生成": metrics["llm_generation_rate"] == 1.0,
                "所有问答均在请求时间预算内完成": latency["max_ms"]
                <= round(settings.request_timeout_seconds * 1000),
            }
        )
    question_bytes = question_set_path.read_bytes()
    payload: dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "project_version": __version__,
        "prompt_version": PROMPT_VERSION,
        "question_set": str(question_set_path),
        "question_set_sha256": hashlib.sha256(question_bytes).hexdigest(),
        "answer_mode": "llm" if use_llm else "extractive",
        "semantic_judge_enabled": semantic_judge,
        "semantic_judge_model": (settings.llm_model if semantic_judge else None),
        "top_k": top_k,
        "questions": total_questions,
        "category_counts": dict(sorted(category_counts.items())),
        "index_versions": _index_versions(settings),
        "metrics": {key: round(value, 6) for key, value in metrics.items()},
        "latency": {
            "all": latency,
            "complex": complex_latency,
            "standard": standard_latency,
            "request_timeout_ms": round(settings.request_timeout_seconds * 1000),
        },
        "usage": {
            "answer_generation": _sum_usage(results, "llm_usage"),
            "query_planner": _sum_usage(results, "query_planner_usage"),
            "retrieval_reflection": _sum_usage(
                results, "retrieval_reflection_usage"
            ),
            "semantic_judge": _sum_usage(
                [
                    {"usage": result["semantic_judgment"]["usage"]}
                    for result in results
                    if result["semantic_judgment"] is not None
                ],
                "usage",
            ),
        },
        "semantic_calibration": semantic_calibration,
        "gates": gates,
        "passed": all(gates.values()),
        "results": results,
    }
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    (settings.reports_dir / f"answer_eval_{run_id}.json").write_text(
        rendered, encoding="utf-8"
    )
    (settings.reports_dir / "answer_eval_latest.json").write_text(
        rendered, encoding="utf-8"
    )
    markdown = _render_markdown(payload)
    (settings.reports_dir / f"mvp_eval_{run_id}.md").write_text(
        markdown, encoding="utf-8"
    )
    (settings.reports_dir / "mvp_eval_latest.md").write_text(
        markdown, encoding="utf-8"
    )
    return payload
