"""Conservative, source-local joint-action candidates; never a mere name join."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from history_agent.db import Database
from history_agent.errors import ResearchDataError
from history_agent.research.chronology import PROFILES
from history_agent.research.timeline import (
    TimelineEvent,
    get_person_timeline,
)

RULE_VERSION = "joint-action-rules-v4"
_ACTION_ROLES = {
    "出席": "共同出席者",
    "参加": "共同参加者",
    "主持": "共同主持者",
    "会见": "共同会见者",
    "接见": "共同接见者",
    "致电": "共同发电者",
    "签署": "共同签署者",
    "署名": "共同署名者",
    "联名": "共同署名者",
}
_ACTIONS = "|".join(_ACTION_ROLES)
# Only consume recognizable chronology dates/time labels, never arbitrary prose.
_PREFIX = re.compile(
    r"^(?:(?:\d{4}年)?(?:\d{1,2}月)?\d{1,2}日|同日|当日|本日)?"
    r"(?:上午|下午|晚上|晚间|晚|晨|凌晨|中午)?(?:\d{1,2}时(?:\d{1,2}分)?)?"
)
_NON_ASSERTION = re.compile(
    r"回忆|追述|曾经|曾于|据说|谈到|提到|如果|假如|没有|未|不|拟|准备|计划|将|可能"
    r"|建议|提议|要求|希望|应当|决定|是否|取消|否认"
)


class JointActionEvidence(BaseModel):
    evidence_id: str
    source_event_id: str
    action: str
    supporting_text: str
    roles: dict[str, str]
    subject_basis: Literal["explicit", "chronology_subject"]
    match_method: Literal[
        "single_clause", "adjacent_attendance", "grouped_attendance"
    ] = "single_clause"


class IntersectionEvent(BaseModel):
    event: TimelineEvent
    joint_evidence: list[JointActionEvidence] = Field(min_length=1)
    verification_status: Literal["needs_review"] = "needs_review"


class PersonIntersectionResponse(BaseModel):
    person_id: str
    other_person_id: str
    canonical_name: str
    other_canonical_name: str
    start_year: int | None
    end_year: int | None
    rule_version: str = RULE_VERSION
    total: int
    co_mention_total: int
    offset: int
    limit: int
    has_more: bool
    events: list[IntersectionEvent]
    limitation: str = (
        "仅返回同一来源原文中命中保守共同动作规则的候选，不等于已核实共同参与；"
        "未命中不代表没有交集，未跨页拼接动作，不从人名共现推断组织关系。"
    )


def _unquoted_clauses(text: str) -> list[str]:
    # Keep boundaries: removing a quotation must not join the prose on either side.
    masked: list[str] = []
    stack: list[str] = []
    closing = {"“": "”", "‘": "’", "「": "」", "『": "』", '"': '"'}
    for character in text:
        if stack and character == stack[-1]:
            stack.pop()
            masked.append(";")
        elif character in closing:
            stack.append(closing[character])
            masked.append(";")
        else:
            masked.append(";" if stack else character)
    # Colons introduce reported speech; conservatively discard their whole sentence.
    sentences = re.split(r"[。！？!?；;]", "".join(masked))
    return [
        clause
        for sentence in sentences
        if not re.search(r"[:：]", sentence)
        for clause in re.split(r"[，,]", sentence)
        if clause.strip()
    ]


def _match_action(
    text: str,
    first: str,
    second: str,
    subject_name: str | None,
    known_names: Sequence[str],
) -> tuple[str, dict[str, str], Literal["explicit", "chronology_subject"]] | None:
    compact = re.sub(r"\s+", "", text)
    compact = _PREFIX.sub("", compact)
    if _NON_ASSERTION.search(compact):
        return None
    names = (
        "(?:"
        + "|".join(re.escape(name) for name in sorted(known_names, key=len, reverse=True))
        + ")"
    )
    basis: Literal["explicit", "chronology_subject"] = "explicit"
    joint_text = compact
    if subject_name in {first, second} and compact.startswith(("和", "与", "同")):
        joint_text = str(subject_name) + compact
        basis = "chronology_subject"
    joint = re.match(
        rf"(?P<names>{names}(?:[、和与同及]{names})+)(?:等)?"
        rf"(?:在[\u4e00-\u9fff]{{1,12}}?)?(?:一起|共同|联名)?(?P<action>{_ACTIONS})",
        joint_text,
    )
    if joint and {first, second} <= set(re.split(r"[、和与同及]", joint["names"])):
        action = joint["action"]
        return action, {first: _ACTION_ROLES[action], second: _ACTION_ROLES[action]}, basis
    for left, right in ((first, second), (second, first)):
        match = re.match(
            rf"{re.escape(left)}[、和与同及]{re.escape(right)}(?:等)?"
            rf"(?:一起|共同|联名)?(?P<action>{_ACTIONS})",
            compact,
        )
        if match:
            action = match["action"]
            return action, {left: _ACTION_ROLES[action], right: _ACTION_ROLES[action]}, "explicit"
        match = re.match(
            rf"{re.escape(left)}(?P<action>会见|接见){re.escape(right)}(?:同志)?(?=$|[^\u4e00-\u9fff]|并|等)",
            compact,
        )
        if match:
            return match["action"], {left: "会见方", right: "被会见方"}, "explicit"
        if subject_name == left:
            if re.match(rf"出席{re.escape(right)}主持的", compact):
                return "出席", {left: "参会者", right: "主持者"}, "chronology_subject"
            match = re.match(
                rf"[和与同]{re.escape(right)}(?:等)?(?:一起|共同|联名)?"
                rf"(?P<action>{_ACTIONS})",
                compact,
            )
            if match:
                action = match["action"]
                return (
                    action,
                    {left: _ACTION_ROLES[action], right: _ACTION_ROLES[action]},
                    "chronology_subject",
                )
    return None


def _match_adjacent_attendance(
    text: str,
    first: str,
    second: str,
    subject_name: str | None,
) -> tuple[str, str, dict[str, str], Literal["explicit", "chronology_subject"]] | None:
    """Bind only the opening activity to its immediately following explicit roster.

    The returned span contains BOTH sentences verbatim and fits the chat quote budget.
    No look-back over quotations, footnotes, intervening events, or pages is allowed.
    """
    sentences = re.match(r"\A\s*(?P<lead>[^。！？!?；;]+)。\s*(?P<roster>[^。！？!?；;]+)。", text)
    if sentences is None:
        return None
    supporting_text = sentences.group().strip()
    if len(supporting_text) > 420:
        return None
    compact = re.sub(r"\s+", "", supporting_text)
    if (
        _NON_ASSERTION.search(compact)
        or re.search(r'[“”‘’「」『』"\[\]〔〕（）()《》]', compact)
        or re.search(r"次日|翌日|后来|另[一场次]|随后|此前|之后|改由|委托|代为", compact)
    ):
        return None
    lead = re.sub(r"\s+", "", sentences["lead"])
    lead = re.sub(r"^(?:(?:\d{4}年)?(?:\d{1,2}月)?\d{1,2}日|同日|当日)?", "", lead)
    lead = re.sub(
        r"^(?:上午|下午|晚上|晚间|晚|晨|中午)?"
        r"(?:[一二三四五六七八九十两\d]{1,3}时(?:[一二三四五六七八九十\d]{1,3}分)?)?[，,]?",
        "",
        lead,
    )
    basis: Literal["explicit", "chronology_subject"] = "chronology_subject"
    actor = subject_name
    for name in (first, second):
        if lead.startswith(name):
            actor, lead, basis = name, lead[len(name) :], "explicit"
            break
    if actor is None:
        return None
    activity = re.fullmatch(
        r"(?:(?:在|于)[\u4e00-\u9fff]{2,16}?)?"
        r"(?P<action>会见|接见|主持(?:召开)?)(?P<body>[^：:]+)",
        lead,
    )
    if activity is None:
        return None
    action = activity["action"]
    noun = "会议" if action.startswith("主持") else action
    if noun == "会议" and activity["body"].count("会议") != 1:
        return None
    # A second activity in the lead would make the following roster ambiguous.
    if re.search(r"会见|接见|主持|另行|转赴", activity["body"]):
        return None
    roster = re.sub(r"\s+", "", sentences["roster"])
    roster_match = re.fullmatch(
        rf"(?:参加|出席){noun}(?:和宴请)?的(?:还)?有[：:]?(?P<names>.+)",
        roster,
    )
    if roster_match is None:
        return None
    if "和宴请" in roster and "宴请" not in activity["body"]:
        return None
    names = roster_match["names"].removesuffix("等")
    # Other unregistered names are allowed only as clean 2-4 character list items.
    if re.fullmatch(r"[\u4e00-\u9fff]{2,4}(?:[、和与][\u4e00-\u9fff]{2,4})*", names) is None:
        return None
    participants = set(re.split(r"[、和与]", names))
    if actor in {first, second}:
        other = second if actor == first else first
        if other not in participants:
            return None
        roles = (
            {actor: "主持者", other: "参会者"}
            if noun == "会议"
            else {actor: _ACTION_ROLES[action], other: _ACTION_ROLES[action]}
        )
    elif {first, second} <= participants:
        attendee_role = "参会者" if noun == "会议" else _ACTION_ROLES[action]
        roles = {first: attendee_role, second: attendee_role}
    else:
        return None
    assert actor is not None
    return supporting_text, "主持" if noun == "会议" else action, roles, basis


def _match_grouped_attendance(
    text: str,
    first: str,
    second: str,
    subject_name: str | None,
) -> tuple[str, str, dict[str, str], Literal["chronology_subject"]] | None:
    """Bind a chronology subject to an explicit, adjacent domestic delegation roster."""
    sentences = re.match(r"\A\s*(?P<lead>[^。！？!?；;]+)。\s*(?P<roster>[^。！？!?；;]+)。", text)
    if sentences is None or subject_name is None:
        return None
    supporting_text = sentences.group().strip()
    if len(supporting_text) > 420:
        return None
    compact = re.sub(r"\s+", "", supporting_text)
    if (
        _NON_ASSERTION.search(compact)
        or re.search(r'[“”‘’「」『』"\[\]〔〕（）()《》]', compact)
        or re.search(r"次日|翌日|后来|另[一场次]|随后|此前|之后|改由|委托|代为", compact)
    ):
        return None
    lead = re.sub(r"\s+", "", sentences["lead"])
    lead = re.sub(r"^(?:(?:\d{4}年)?(?:\d{1,2}月)?\d{1,2}日|同日|当日)?", "", lead)
    lead = re.sub(
        r"^(?:上午|下午|晚上|晚间|晚|晨|中午)?"
        r"(?:[一二三四五六七八九十两\d]{1,3}时(?:[一二三四五六七八九十\d]{1,3}分)?)?[，,]?",
        "",
        lead,
    )
    activity = re.fullmatch(
        r"去[\u4e00-\u9fff]{1,16}参加[\u4e00-\u9fff]{2,40}(?:大会|活动)前[，,]"
        r"(?:在|于)[\u4e00-\u9fff]{2,16}(?P<action>会见|接见)(?P<body>[^：:]+)",
        lead,
    )
    if activity is None or re.search(r"会见|接见|主持|另行|转赴", activity["body"]):
        return None
    roster = re.sub(r"\s+", "", sentences["roster"])
    roster_match = re.fullmatch(
        r"参加(?P<action>会见|接见)的[，,]中方有(?P<domestic>[^，,]+)[，,]"
        r"苏方有(?P<foreign>[^，,]+)",
        roster,
    )
    if roster_match is None or roster_match["action"] != activity["action"]:
        return None
    list_pattern = r"[\u4e00-\u9fff]{2,4}(?:[、和与][\u4e00-\u9fff]{2,4})*"
    if re.fullmatch(list_pattern, roster_match["domestic"]) is None:
        return None
    if re.fullmatch(list_pattern, roster_match["foreign"]) is None:
        return None
    domestic = set(re.split(r"[、和与]", roster_match["domestic"]))
    role = _ACTION_ROLES[activity["action"]]
    if subject_name in {first, second}:
        other = second if subject_name == first else first
        if other not in domestic:
            return None
    elif not {first, second} <= domestic:
        return None
    return (
        supporting_text,
        activity["action"],
        {first: role, second: role},
        "chronology_subject",
    )


def get_person_intersections(
    database: Database,
    *,
    person_id: str,
    other_person_id: str,
    start_year: int | None = None,
    end_year: int | None = None,
    event_types: Sequence[str] | None = None,
    review_statuses: Sequence[str] | None = None,
    limit: int = 50,
    offset: int = 0,
) -> PersonIntersectionResponse:
    if person_id == other_person_id:
        raise ResearchDataError("intersection requires two different people")
    # Validate both identities and all public filters before scanning candidates.
    other = get_person_timeline(database, person_id=other_person_id, limit=1)
    page = get_person_timeline(
        database,
        person_id=person_id,
        other_person_id=other_person_id,
        start_year=start_year,
        end_year=end_year,
        event_types=event_types,
        review_statuses=review_statuses,
        limit=limit,
        offset=offset,
    )
    first_name, second_name = page.canonical_name, other.canonical_name
    results: list[IntersectionEvent] = []
    co_mention_total = page.total
    # Matching precedes pagination; no matches disappear at raw candidate page edges.
    for scan_offset in range(0, co_mention_total, 200):
        batch = get_person_timeline(
            database,
            person_id=person_id,
            other_person_id=other_person_id,
            start_year=start_year,
            end_year=end_year,
            event_types=event_types,
            review_statuses=review_statuses,
            limit=200,
            offset=scan_offset,
        )
        with database.connect() as connection:
            known_names = [
                str(row[0])
                for row in connection.execute(
                    "SELECT canonical_name FROM persons WHERE status='active'"
                ).fetchall()
            ]
            source_ids = list({sid for event in batch.events for sid in event.source_event_ids})
            subjects: dict[str, str] = {}
            eligible_sources: set[str] = set()
            if source_ids:
                eligible_sources = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT event_id FROM historical_events WHERE review_status != 'rejected' "
                        "AND event_id IN (" + ",".join("?" for _ in source_ids) + ")",
                        source_ids,
                    ).fetchall()
                }
                rows = connection.execute(
                    "SELECT ep.event_id, p.canonical_name FROM event_participants ep "
                    "JOIN persons p ON p.person_id=ep.person_id "
                    "WHERE ep.mention_source='chronology_subject' AND ep.event_id IN ("
                    + ",".join("?" for _ in source_ids)
                    + ")",
                    source_ids,
                ).fetchall()
                for row in rows:
                    subjects[str(row["event_id"])] = str(row["canonical_name"])
                # A chronology owner can also be explicitly named later in the entry.
                # Only trust our recognized extractors and a unique document profile.
                owner_rows = connection.execute(
                    "SELECT DISTINCT e.event_id, er.document_id FROM historical_events e "
                    "JOIN event_evidence ee ON ee.event_id=e.event_id "
                    "JOIN evidence_records er ON er.evidence_id=ee.evidence_id "
                    "WHERE e.extractor_version IN "
                    "('chronology-rules-v1', 'mao-chronology-rules-v1') "
                    "AND e.event_id IN (" + ",".join("?" for _ in source_ids) + ")",
                    source_ids,
                ).fetchall()
                owners: dict[str, set[str]] = {}
                for row in owner_rows:
                    profile = PROFILES.get(str(row["document_id"]))
                    if profile is not None:
                        owners.setdefault(str(row["event_id"]), set()).add(profile.subject_name)
                for source_id, owner_names in owners.items():
                    if len(owner_names) == 1:
                        subjects[source_id] = next(iter(owner_names))
        for event in batch.events:
            proofs = []
            for evidence in event.evidence:
                if evidence.source_event_id not in eligible_sources:
                    continue
                first_page = min(
                    e.pdf_page_start
                    for e in event.evidence
                    if e.source_event_id == evidence.source_event_id
                )
                if evidence.pdf_page_start == first_page == evidence.pdf_page_end:
                    grouped = _match_grouped_attendance(
                        evidence.quote,
                        first_name,
                        second_name,
                        subjects.get(evidence.source_event_id),
                    )
                    if grouped is not None:
                        span, action, named_roles, grouped_basis = grouped
                        proofs.append(
                            JointActionEvidence(
                                evidence_id=evidence.evidence_id,
                                source_event_id=evidence.source_event_id,
                                action=action,
                                supporting_text=span,
                                roles={
                                    person_id: named_roles[first_name],
                                    other_person_id: named_roles[second_name],
                                },
                                subject_basis=grouped_basis,
                                match_method="grouped_attendance",
                            )
                        )
                    adjacent = _match_adjacent_attendance(
                        evidence.quote,
                        first_name,
                        second_name,
                        subjects.get(evidence.source_event_id),
                    )
                    if adjacent is not None:
                        span, action, named_roles, adjacent_basis = adjacent
                        proofs.append(
                            JointActionEvidence(
                                evidence_id=evidence.evidence_id,
                                source_event_id=evidence.source_event_id,
                                action=action,
                                supporting_text=span,
                                roles={
                                    person_id: named_roles[first_name],
                                    other_person_id: named_roles[second_name],
                                },
                                subject_basis=adjacent_basis,
                                match_method="adjacent_attendance",
                            )
                        )
                for clause in _unquoted_clauses(evidence.quote):
                    matched = _match_action(
                        clause,
                        first_name,
                        second_name,
                        (
                            subjects.get(evidence.source_event_id)
                            if evidence.pdf_page_start == first_page
                            and evidence.quote.lstrip().startswith(clause.strip())
                            else None
                        ),
                        known_names,
                    )
                    if matched is None:
                        continue
                    action, named_roles, action_basis = matched
                    proofs.append(
                        JointActionEvidence(
                            evidence_id=evidence.evidence_id,
                            source_event_id=evidence.source_event_id,
                            action=action,
                            supporting_text=clause.strip(),
                            roles={
                                person_id: named_roles[first_name],
                                other_person_id: named_roles[second_name],
                            },
                            subject_basis=action_basis,
                        )
                    )
            if proofs:
                results.append(IntersectionEvent(event=event, joint_evidence=proofs))
    return PersonIntersectionResponse(
        person_id=person_id,
        other_person_id=other_person_id,
        canonical_name=first_name,
        other_canonical_name=second_name,
        start_year=start_year,
        end_year=end_year,
        total=len(results),
        co_mention_total=co_mention_total,
        offset=offset,
        limit=limit,
        has_more=offset + limit < len(results),
        events=results[offset : offset + limit],
    )
