"""Deterministic, narrow chat routes for auditable research records."""

from __future__ import annotations

import re
import sqlite3

from history_agent.answering.models import AnswerResponse, Citation, QuestionRequest
from history_agent.config import Settings
from history_agent.db import Database
from history_agent.errors import ResearchDataError
from history_agent.research.intersections import get_person_intersections
from history_agent.research.people import resolve_person
from history_agent.research.timeline import TimelineEvidence, get_person_timeline

_YEAR = re.compile(r"(?P<start>\d{4})年?(?:(?:至|到|—|–|-|~|～)(?P<end>\d{4})年?)?")
_INTERSECTION = re.compile(r"交集|共同(?:事件|活动|经历|参加|参与|出席)")
_TIMELINE = re.compile(r"时间线|经历[？?。]*$|经历有哪些|有哪些活动|做了什么|参加[过了]哪些会议")
_ALLOWED = {
    "intersection": re.compile(
        r"(?:有哪些|有什么|有过哪些)?(?:交集|共同事件|共同活动|共同经历)"
        r"|共同参加过哪些会议|共同参与过哪些事件"
    ),
    "timeline": re.compile(
        r"(?:主要)?(?:有哪些|有什么)?(?:主要)?(?:经历|活动)"
        r"|(?:主要)?经历有哪些|时间线|做了什么|参加[过了]哪些会议"
    ),
}


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
        limitations=limitations or [],
    )


def _citation(evidence: TimelineEvidence, number: int, quote: str) -> Citation:
    return Citation(
        evidence_id=f"E{number}",
        document_id=evidence.document_id,
        document=evidence.document_title,
        volume=evidence.volume,
        pdf_page=evidence.pdf_page_start,
        pdf_page_end=evidence.pdf_page_end,
        section=[],
        quote=quote,
        source_type=evidence.source_type,
        verification_status=evidence.verification_status,
        extraction_methods=evidence.extraction_methods,
    )


def answer_structured_question(
    settings: Settings, request: QuestionRequest
) -> AnswerResponse | None:
    question = re.sub(r"\s+", "", request.question).rstrip("？?。！!")
    intent = (
        "intersection"
        if _INTERSECTION.search(question)
        else "timeline"
        if _TIMELINE.search(question)
        else None
    )
    if intent is None:
        previous = next(
            (message.content for message in reversed(request.history) if message.role == "user"), ""
        )
        if re.fullmatch(r"(?:那|那么)?(?:\d{4}年)?(?:呢|他呢|他们呢|继续|下一页)", question):
            intent = (
                "intersection"
                if _INTERSECTION.search(previous)
                else "timeline"
                if _TIMELINE.search(previous)
                else None
            )
        if intent is None:
            return None
    clarify = (
        "请明确人物和年份，例如“毛泽东在1949年有哪些经历”或"
        "“毛泽东与周恩来在1949年有哪些交集”。当前结构化查询支持整年或年份区间，"
        "不会忽略地点、月份等附加条件，也不会自动继承上一轮人物。"
    )
    years = list(_YEAR.finditer(question))
    if len(years) != 1:
        return _response(request, intent, clarify)
    start = int(years[0]["start"])
    end = int(years[0]["end"] or start)
    lower, upper = settings.research_start.year, settings.research_end.year
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
        name_pattern = re.compile(
            "|".join(re.escape(form) for form in sorted(forms, key=len, reverse=True))
        )
        mentions = list(name_pattern.finditer(question))
        expected_count = 2 if intent == "intersection" else 1
        if len(mentions) != expected_count:
            return _response(request, intent, clarify)
        # Removing only recognized names/year/function words prevents silently dropping
        # constraints (e.g. an unknown third person, a month, place, or negation).
        remainder = name_pattern.sub("", _YEAR.sub("", question))
        remainder = re.sub(r"^(?:(?:请问|请|帮我|查询|列出|梳理|一下|看看))+", "", remainder)
        remainder = re.sub(r"[和与及、在于的]", "", remainder)
        if _ALLOWED[intent].fullmatch(remainder) is None:
            return _response(request, intent, clarify)
        person_ids = []
        for mention in mentions:
            resolution = resolve_person(database, mention.group())
            if resolution.status != "resolved":
                return _response(
                    request, intent, f"人物“{mention.group()}”未能唯一解析，请使用完整姓名。"
                )
            person = resolution.candidates[0]
            person_ids.append(person.merged_into_person_id or person.person_id)
        if len(set(person_ids)) != expected_count:
            return _response(
                request, intent, "交集查询需要两位不同人物；两个称呼可能是同一人的别名。"
            )
        citations: list[Citation] = []
        lines: list[str] = []
        event_types = ["meeting"] if "会议" in question else None
        if intent == "intersection":
            intersections = get_person_intersections(
                database,
                person_id=person_ids[0],
                other_person_id=person_ids[1],
                start_year=start,
                end_year=end,
                event_types=event_types,
                limit=request.top_k,
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
            timeline = get_person_timeline(
                database,
                person_id=person_ids[0],
                start_year=start,
                end_year=end,
                event_types=event_types,
                limit=request.top_k,
            )
            total, shown = timeline.total, len(timeline.events)
            for event in timeline.events:
                evidence = event.evidence[0]
                quote = evidence.quote[:420] + ("……" if len(evidence.quote) > 420 else "")
                citation = _citation(evidence, len(citations) + 1, quote)
                citations.append(citation)
                lines.append(
                    f"- 记录日期 {event.start.value or '不明确'}（{event.start.certainty}）；"
                    f"记录复核状态：{event.review_status}。原文：{quote} [{citation.evidence_id}]"
                )
            limitations = [
                "时间线包含年谱主体与原文提及记录，不保证本人参与了每条事件。",
                "按时间展示而非重要性排序；日期为索引字段，须结合原文精度和来源差异核对。",
            ]
            lead = (
                f"找到 {total} 条与{timeline.canonical_name}相关的事件记录，"
                f"按时间展示前 {shown} 条，不是完整经历结论。"
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
