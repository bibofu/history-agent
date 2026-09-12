"""Deterministic, narrow chat routes for auditable research records."""

from __future__ import annotations

import re
import sqlite3

from history_agent.answering.models import (
    AnswerResponse,
    Citation,
    QueryPlan,
    QuestionRequest,
)
from history_agent.config import Settings
from history_agent.db import Database
from history_agent.errors import ResearchDataError
from history_agent.research.intersections import get_person_intersections
from history_agent.research.people import resolve_person
from history_agent.research.timeline import TimelineEvidence, get_person_timeline

_RAW_TIMELINE = re.compile(r"列出|时间线|逐条|原文|明细|记录|清单|参加[过了]哪些会议")
MAX_STRUCTURED_SUMMARY_RECORDS = 36


def _response(
    request: QuestionRequest,
    intent: str,
    answer: str,
    citations: list[Citation] | None = None,
    limitations: list[str] | None = None,
) -> AnswerResponse:
    return AnswerResponse(
        question=request.question,
        answer=answer,
        evidence_status="partial" if citations else "no_evidence",
        generator_mode="extractive",
        llm_status="not_applicable",
        retrieval_mode=f"structured_{intent}",
        query_intent=intent,
        citations=citations or [],
        retrieved_evidence_count=len(citations or []),
        limitations=limitations or [],
    )


def _citation(
    evidence: TimelineEvidence,
    number: int,
    quote: str,
    *,
    section: list[str] | None = None,
) -> Citation:
    return Citation(
        evidence_id=f"E{number}",
        document_id=evidence.document_id,
        document=evidence.document_title,
        volume=evidence.volume,
        pdf_page=evidence.pdf_page_start,
        pdf_page_end=evidence.pdf_page_end,
        section=section or [],
        quote=quote,
        source_type=evidence.source_type,
        verification_status=evidence.verification_status,
        extraction_methods=evidence.extraction_methods,
    )


def requires_structured_generation(response: AnswerResponse) -> bool:
    """Require LLM generation for every structured answer backed by evidence."""

    return bool(response.citations)


def _structured_result_limit(request: QuestionRequest, start: int, end: int) -> int:
    span = end - start + 1
    return min(MAX_STRUCTURED_SUMMARY_RECORDS, max(request.top_k, span * 3))


def answer_structured_question(
    settings: Settings, request: QuestionRequest, plan: QueryPlan | None
) -> AnswerResponse | None:
    """Run an audited timeline/intersection lookup selected by the LLM query plan."""

    if plan is None or plan.retrieval_route != "structured":
        return None
    if plan.intent not in {"timeline", "intersection"}:
        return None
    intent = plan.intent
    question = re.sub(r"\s+", "", request.question).rstrip("？?。！!")
    lower, upper = settings.research_start.year, settings.research_end.year
    start, end = plan.start_year, plan.end_year
    if start is None or end is None:
        return None
    if not lower <= start <= end <= upper:
        return _response(
            request, intent, f"研究范围为 {lower}—{upper} 年，请提供范围内且起止顺序正确的年份。"
        )
    if not settings.database_path.is_file():
        return _response(request, intent, "结构化研究库尚未就绪，请先初始化并抽取年谱事件。")
    database = Database(settings.database_path)
    try:
        with database.connect() as connection:
            forms = {
                str(row[0])
                for row in connection.execute(
                    "SELECT canonical_name FROM persons UNION "
                    "SELECT alias_text FROM person_aliases WHERE is_active=1"
                ).fetchall()
                if len(str(row[0])) >= 2
            }
        if not forms:
            return _response(request, intent, "人物主数据尚未就绪，请先初始化研究库。")
        person_entities = [entity for entity in plan.entities if entity.type == "person"]
        expected_count = 2 if intent == "intersection" else 1
        if len(person_entities) != expected_count:
            return None
        person_ids = []
        for entity in person_entities:
            resolution = resolve_person(database, entity.canonical)
            if resolution.status != "resolved" and entity.text != entity.canonical:
                resolution = resolve_person(database, entity.text)
            if resolution.status != "resolved":
                return None
            person = resolution.candidates[0]
            person_ids.append(person.merged_into_person_id or person.person_id)
        if len(set(person_ids)) != expected_count:
            return _response(
                request, intent, "交集查询需要两位不同人物；两个称呼可能是同一人的别名。"
            )
        citations: list[Citation] = []
        lines: list[str] = []
        event_types = ["meeting"] if "会议" in question else None
        result_limit = _structured_result_limit(request, start, end)
        if intent == "intersection":
            intersections = get_person_intersections(
                database,
                person_id=person_ids[0],
                other_person_id=person_ids[1],
                start_year=start,
                end_year=end,
                event_types=event_types,
                limit=result_limit,
            )
            total, shown = intersections.total, len(intersections.events)
            for item in intersections.events:
                # Cite the actual proof, never the canonical representative's unrelated text.
                proof = item.joint_evidence[0]
                evidence = next(
                    e for e in item.event.evidence if e.evidence_id == proof.evidence_id
                )
                quote = proof.supporting_text[:420] + (
                    "……" if len(proof.supporting_text) > 420 else ""
                )
                citation = _citation(evidence, len(citations) + 1, quote)
                citations.append(citation)
                roles = "、".join(
                    f"{name}：{proof.roles[pid]}"
                    for name, pid in (
                        (intersections.canonical_name, person_ids[0]),
                        (intersections.other_canonical_name, person_ids[1]),
                    )
                )
                # Canonical dates may differ across sources; label them as index fields.
                lines.append(
                    f"- 记录日期 {item.event.start.value or '不明确'}"
                    f"（{item.event.start.certainty}）；"
                    f"{roles}；待复核。原文：{quote} [{citation.evidence_id}]"
                )
            limitations = [
                intersections.limitation,
                "角色和日期来自规则/索引，尚未逐条核实；同一史实可能保留多条来源候选。",
            ]
            lead = (
                f"当前规则找到 {total} 条交集候选，按时间展示前 {shown} 条；"
                "不是完整或已确认的交集清单。"
            )
            empty = "当前规则未找到可展示的共同动作候选；这不代表两人没有交集，复杂句式仍可能漏检。"
        else:
            synthesize = _RAW_TIMELINE.search(question) is None
            timeline = get_person_timeline(
                database,
                person_id=person_ids[0],
                start_year=start,
                end_year=end,
                event_types=event_types,
                limit=result_limit if synthesize else request.top_k,
                sample_across_range=synthesize,
                subject_only=True,
            )
            total, shown = timeline.total, len(timeline.events)
            for event in timeline.events:
                evidence = event.evidence[0]
                quote = evidence.quote[:420] + ("……" if len(evidence.quote) > 420 else "")
                citation = _citation(
                    evidence,
                    len(citations) + 1,
                    quote,
                    section=[f"结构化索引日期：{event.start.value or '不明确'}"],
                )
                citations.append(citation)
                lines.append(
                    f"- 记录日期 {event.start.value or '不明确'}（{event.start.certainty}）；"
                    f"记录复核状态：{event.review_status}。原文：{quote} [{citation.evidence_id}]"
                )
            limitations = [
                "时间线只纳入目标人物被标记为年谱主体的记录；动作归属仍须结合原文核对。",
                (
                    "记录按月份/时段抽样并优先采用较高复核状态，不等同于重要性排名；"
                    "日期为索引字段，须结合原文精度和来源差异核对。"
                    if synthesize
                    else "按时间展示而非重要性排序；日期为索引字段，须结合原文精度和来源差异核对。"
                ),
            ]
            lead = (
                f"找到 {total} 条与{timeline.canonical_name}相关的事件记录，"
                + (
                    f"从所问时段抽取 {shown} 条代表性记录供综合，不是完整经历结论。"
                    if synthesize
                    else f"按时间展示前 {shown} 条，不是完整经历结论。"
                )
            )
            empty = "当前结构化研究库未找到符合条件的记录，不能据此断言该时期没有活动。"
        return _response(
            request,
            intent,
            lead + "\n\n" + "\n".join(lines) if lines else empty,
            citations,
            limitations,
        )
    except (ResearchDataError, sqlite3.Error):
        # A failed structured lookup must not become a broad RAG claim about joint participation.
        return _response(
            request,
            intent,
            "结构化研究查询暂不可用，请检查研究库初始化状态；本次没有用人名共现结果替代回答。",
        )
