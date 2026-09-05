from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from history_agent.db import Database
from history_agent.errors import ExtractionError
from history_agent.extraction.models import PageRecord
from history_agent.extraction.report import load_page_records
from history_agent.processing.chunks import effective_pages
from history_agent.processing.cleaning import detect_repeated_marginal_lines
from history_agent.processing.models import StructureEntry
from history_agent.research.catalog import load_person_catalog
from history_agent.research.models import (
    DateCertainty,
    EventParticipant,
    EvidenceReference,
    HistoricalEvent,
    PersonCatalog,
    ReviewStatus,
    TemporalPoint,
)
from history_agent.research.store import ResearchStore

CHRONOLOGY_EXTRACTOR_VERSION = "chronology-rules-v1"
DEFAULT_CHRONOLOGIES = (
    "zhou_enlai_chronology_1949_1976",
    "lin_biao_chronology",
    *(f"mao_zedong_chronology_volume_{volume}" for volume in range(1, 10)),
)

ZHOU_DATE_MARKER = re.compile(r"【(?P<label>[^】]{2,48})】")
ZHOU_YEAR_HEADING = re.compile(
    r"^\[周恩来年谱\]\s*(?:18|19)\d{2}\s*年\s*[（(][^）)]*岁[^）)]*[）)]\s*$"
)
LIN_YEAR_HEADING = re.compile(r"^(?:\d\s*){4}年\s*[^\n]{0,20}岁\s*$")
PAGE_NUMBER_LINE = re.compile(r"^[\s—–-]*\d{1,4}[\s—–-]*$")
YEAR_IN_TITLE = re.compile(r"((?:18|19|20)\d{2})年")
DATE_RANGE_SEPARATOR = re.compile(r"(?:－|—|–|~|～|至|(?<=日)一(?=\d))")
DATE_LIST_SEPARATOR = re.compile(r"[、,，]")
DATE_NOTE = re.compile(r"[（(][^）)\n]{0,50}[）)]")
DATE_APPROXIMATE_SUFFIX = re.compile(r"(?:前|后|左右|上下)$")
DATE_QUALIFIER = r"上旬|中旬|下旬|中下旬|初|月初|末|底"
LIN_DIGIT = r"\d(?:\s*\d)?"
LIN_DATE_MARKER = re.compile(
    rf"(?m)^[ \t\u3000]*(?P<label>"
    rf"(?:{LIN_DIGIT}\s*月\s*{LIN_DIGIT}\s*日?"
    rf"(?:\s*(?:[—－–~～至一、,，])\s*"
    rf"(?:{LIN_DIGIT}\s*月\s*)?{LIN_DIGIT}\s*日?)*"
    rf"|{LIN_DIGIT}\s*月\s*(?:{DATE_QUALIFIER})"
    rf"|{LIN_DIGIT}\s*月"
    rf"|同日|翌日|当日|本月|是月)"
    rf"(?:[（(][^）)\n]{{1,50}}[）)])?"
    rf")[ \t\u3000]+"
)

EVENT_TYPE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("correspondence", ("致电", "致信", "电告", "复信", "来函", "签发")),
    ("meeting", ("会议", "大会", "全会", "座谈会", "开会", "出席", "主持")),
    ("speech", ("讲话", "演说", "发言", "报告", "发表", "撰写", "起草")),
    ("visit", ("访问", "考察", "视察", "前往", "抵达", "到达", "返回", "赴")),
    (
        "military",
        ("战役", "战斗", "进攻", "歼灭", "作战", "伏击", "攻克", "撤退", "率部"),
    ),
    ("appointment", ("任命", "当选", "任职", "担任", "兼任", "被选为")),
)

COMMON_LOCATIONS = (
    "北京",
    "北平",
    "上海",
    "广州",
    "武汉",
    "南京",
    "天津",
    "重庆",
    "延安",
    "西安",
    "南昌",
    "瑞金",
    "井冈山",
    "太原",
    "沈阳",
    "长春",
    "哈尔滨",
    "公主屯",
    "平型关",
    "山海关",
    "巴黎",
    "里昂",
    "伦敦",
    "爱丁堡",
    "莫斯科",
    "苏联",
    "法国",
    "德国",
    "印度",
    "朝鲜",
    "越南",
    "蒙古",
)
LOCATION_PATTERN = re.compile(
    r"(?:抵达|到达|前往|返回|来到|进驻|移驻|撤至|占领|攻克|进入|进至|"
    r"开往|移防|离开|撤出|集中|在|于|赴|从|经|向|到)"
    r"(?P<place>[\u4e00-\u9fff·]{2,16}?)"
    r"(?=召开|举行|出席|参加|会见|接见|主持|访问|考察|视察|工作|学习|"
    r"休养|前进|起飞|，|。|；)"
)
LOCATION_SUFFIXES = (
    "省",
    "市",
    "县",
    "区",
    "镇",
    "村",
    "山",
    "关",
    "岭",
    "沟",
    "屯",
    "堡",
    "车站",
    "饭店",
    "大学",
)
SPATIAL_END_MARKERS = (
    "附近地区",
    "机场",
    "车站",
    "饭店",
    "大学",
    "境内",
    "地区",
    "附近",
    "以北",
    "以南",
    "以东",
    "以西",
)


class ParsedChronologyDate(BaseModel):
    expression: str
    start: TemporalPoint
    end: TemporalPoint | None = None
    rule_flags: list[str] = Field(default_factory=list)


class ChronologyEventCandidate(BaseModel):
    event: HistoricalEvent
    date_expression: str
    rule_flags: list[str] = Field(default_factory=list)
    source_file_sha256: str


class ChronologyDocumentResult(BaseModel):
    document_id: str
    subject_person_id: str
    pages_processed: int
    raw_entries: int
    candidates: int
    database_created: int
    database_updated: int
    database_skipped: int
    skipped_short: int
    skipped_out_of_scope: int
    skipped_context: int = 0
    date_precision_counts: dict[str, int]
    rule_flag_counts: dict[str, int]
    event_type_counts: dict[str, int]
    location_candidates: int
    output_path: str


class ChronologyExtractionSummary(BaseModel):
    run_id: str
    dry_run: bool = False
    extractor_version: str = CHRONOLOGY_EXTRACTOR_VERSION
    documents: list[ChronologyDocumentResult]
    samples: list[dict[str, object]] = Field(default_factory=list)

    @property
    def totals(self) -> dict[str, int]:
        return {
            "documents": len(self.documents),
            "pages_processed": sum(item.pages_processed for item in self.documents),
            "raw_entries": sum(item.raw_entries for item in self.documents),
            "candidates": sum(item.candidates for item in self.documents),
            "database_created": sum(item.database_created for item in self.documents),
            "database_updated": sum(item.database_updated for item in self.documents),
            "database_skipped": sum(item.database_skipped for item in self.documents),
            "skipped_short": sum(item.skipped_short for item in self.documents),
            "skipped_context": sum(item.skipped_context for item in self.documents),
            "skipped_out_of_scope": sum(
                item.skipped_out_of_scope for item in self.documents
            ),
            "location_candidates": sum(
                item.location_candidates for item in self.documents
            ),
        }


@dataclass(frozen=True)
class ChronologyProfile:
    document_id: str
    subject_person_id: str
    subject_name: str
    layout: Literal["zhou", "lin", "mao"]


@dataclass
class RawChronologyEntry:
    document_id: str
    subject_person_id: str
    subject_name: str
    temporal: ParsedChronologyDate
    file_sha256: str
    text_parts: list[str] = field(default_factory=list)
    page_methods: dict[int, str] = field(default_factory=dict)
    page_text_parts: dict[int, list[str]] = field(default_factory=dict)
    extractor_version: str = CHRONOLOGY_EXTRACTOR_VERSION
    rule_flags: list[str] = field(default_factory=list)

    def append(self, text: str, page: PageRecord) -> None:
        value = text.strip()
        if not value:
            return
        self.text_parts.append(value)
        self.page_methods[page.pdf_page] = page.extraction_method
        self.page_text_parts.setdefault(page.pdf_page, []).append(value)

    @property
    def text(self) -> str:
        return "\n".join(part for part in self.text_parts if part).strip()

    @property
    def pages(self) -> list[int]:
        return sorted(self.page_methods)


PROFILES = {
    "zhou_enlai_chronology_1949_1976": ChronologyProfile(
        document_id="zhou_enlai_chronology_1949_1976",
        subject_person_id="zhou_enlai",
        subject_name="周恩来",
        layout="zhou",
    ),
    "lin_biao_chronology": ChronologyProfile(
        document_id="lin_biao_chronology",
        subject_person_id="lin_biao",
        subject_name="林彪",
        layout="lin",
    ),
}

PROFILES.update({
    document_id: ChronologyProfile(
        document_id=document_id, subject_person_id="mao_zedong",
        subject_name="毛泽东", layout="mao",
    )
    for document_id in DEFAULT_CHRONOLOGIES if document_id.startswith("mao_zedong_")
})


def _compact_date_text(value: str) -> str:
    return re.sub(r"\s+", "", value.strip())


def _point_with_expression(
    point: TemporalPoint,
    expression: str,
    *,
    certainty: DateCertainty | None = None,
) -> TemporalPoint:
    update: dict[str, str] = {"original_text": expression}
    if certainty is not None:
        update["certainty"] = certainty
    return point.model_copy(update=update)


def _date_fragment(
    fragment: str,
    *,
    expression: str,
    year_hint: int | None,
    month_hint: int | None,
) -> tuple[TemporalPoint | None, int | None, int | None, bool]:
    value = _compact_date_text(fragment)
    year_qualifier = re.fullmatch(
        r"(?P<year>(?:18|19|20)\d{2})年(?P<qualifier>春|夏|秋|冬|年初|年底|上半年|下半年)",
        value,
    )
    if year_qualifier:
        year = int(year_qualifier.group("year"))
        return (
            TemporalPoint(
                value=f"{year:04d}",
                precision="year",
                certainty="approximate",
                original_text=expression,
            ),
            year,
            None,
            False,
        )

    match = re.fullmatch(
        rf"(?:(?P<year>(?:18|19|20)\d{{2}})年)?"
        rf"(?P<month>\d{{1,2}})月"
        rf"(?:(?P<day>\d{{1,2}})日?|(?P<qualifier>{DATE_QUALIFIER}))?",
        value,
    )
    if match:
        explicit_year = match.group("year") is not None
        parsed_year_value = int(match.group("year")) if explicit_year else year_hint
        month = int(match.group("month"))
        day = int(match.group("day")) if match.group("day") else None
        qualifier = match.group("qualifier")
        certainty: DateCertainty = "exact" if explicit_year else "inferred"
        if parsed_year_value is None:
            return None, None, month, False
        if not 1 <= month <= 12:
            return (
                TemporalPoint(
                    value=f"{parsed_year_value:04d}",
                    precision="year",
                    certainty="approximate",
                    original_text=expression,
                ),
                parsed_year_value,
                None,
                True,
            )
        if day is not None:
            try:
                parsed = date(parsed_year_value, month, day)
            except ValueError:
                return (
                    TemporalPoint(
                        value=f"{parsed_year_value:04d}-{month:02d}",
                        precision="month",
                        certainty="approximate",
                        original_text=expression,
                    ),
                    parsed_year_value,
                    month,
                    True,
                )
            return (
                TemporalPoint(
                    value=parsed.isoformat(),
                    precision="day",
                    certainty=certainty,
                    original_text=expression,
                ),
                parsed_year_value,
                month,
                False,
            )
        return (
            TemporalPoint(
                value=f"{parsed_year_value:04d}-{month:02d}",
                precision="month",
                certainty="approximate" if qualifier else certainty,
                original_text=expression,
            ),
            parsed_year_value,
            month,
            False,
        )

    year_match = re.fullmatch(r"(?P<year>(?:18|19|20)\d{2})年", value)
    if year_match:
        year = int(year_match.group("year"))
        return (
            TemporalPoint(
                value=f"{year:04d}",
                precision="year",
                certainty="exact",
                original_text=expression,
            ),
            year,
            None,
            False,
        )

    day_match = re.fullmatch(r"(?P<day>\d{1,2})日?", value)
    if day_match and year_hint is not None and month_hint is not None:
        day = int(day_match.group("day"))
        try:
            parsed = date(year_hint, month_hint, day)
        except ValueError:
            return None, year_hint, month_hint, True
        return (
            TemporalPoint(
                value=parsed.isoformat(),
                precision="day",
                certainty="inferred",
                original_text=expression,
            ),
            year_hint,
            month_hint,
            False,
        )
    return None, year_hint, month_hint, False


def parse_chronology_date(
    expression: str,
    *,
    year_hint: int | None = None,
    inherited_from: ParsedChronologyDate | None = None,
) -> ParsedChronologyDate:
    """Parse chronology headers while retaining every non-exact inference as a flag."""

    original = expression.strip()
    core = DATE_NOTE.sub("", original)
    core = _compact_date_text(core)
    flags: list[str] = []
    if DATE_APPROXIMATE_SUFFIX.search(core):
        core = DATE_APPROXIMATE_SUFFIX.sub("", core)
        flags.append("approximate_date")

    has_date_component = bool(re.search(r"\d|年|月|日|春|夏|秋|冬", core))
    if not has_date_component:
        flags.append("inherited_date")
        if inherited_from is not None:
            start = _point_with_expression(
                inherited_from.start, original, certainty="inferred"
            )
            end = (
                _point_with_expression(
                    inherited_from.end, original, certainty="inferred"
                )
                if inherited_from.end is not None
                else None
            )
            return ParsedChronologyDate(
                expression=original,
                start=start,
                end=end,
                rule_flags=flags,
            )
        if year_hint is not None:
            return ParsedChronologyDate(
                expression=original,
                start=TemporalPoint(
                    value=f"{year_hint:04d}",
                    precision="year",
                    certainty="inferred",
                    original_text=original,
                ),
                rule_flags=[*flags, "year_only_fallback"],
            )
        return ParsedChronologyDate(
            expression=original,
            start=TemporalPoint(original_text=original),
            rule_flags=[*flags, "unknown_date"],
        )

    list_parts = [part for part in DATE_LIST_SEPARATOR.split(core) if part]
    if len(list_parts) > 1:
        points: list[TemporalPoint] = []
        current_year = year_hint
        current_month: int | None = None
        invalid = False
        for part in list_parts:
            point, parsed_year, parsed_month, fragment_invalid = _date_fragment(
                part,
                expression=original,
                year_hint=current_year,
                month_hint=current_month,
            )
            invalid = invalid or fragment_invalid
            if parsed_year is not None:
                current_year = parsed_year
            if parsed_month is not None:
                current_month = parsed_month
            if point is not None:
                points.append(point)
        if points:
            flags.append("multiple_dates")
            if invalid:
                flags.append("invalid_date_component")
            start = _point_with_expression(points[0], original, certainty="approximate")
            end = _point_with_expression(points[-1], original, certainty="approximate")
            if start.value is not None and end.value is not None and end.value < start.value:
                flags.append("invalid_date_order")
                end = None
            return ParsedChronologyDate(
                expression=original,
                start=start,
                end=end,
                rule_flags=flags,
            )

    range_parts = [part for part in DATE_RANGE_SEPARATOR.split(core, maxsplit=1) if part]
    if len(range_parts) == 2:
        range_start, parsed_year, parsed_month, start_invalid = _date_fragment(
            range_parts[0],
            expression=original,
            year_hint=year_hint,
            month_hint=None,
        )
        end, _, _, end_invalid = _date_fragment(
            range_parts[1],
            expression=original,
            year_hint=parsed_year or year_hint,
            month_hint=parsed_month,
        )
        if range_start is not None:
            flags.append("date_range")
            if start_invalid or end_invalid:
                flags.append("invalid_date_component")
            if (
                end is not None
                and range_start.value is not None
                and end.value is not None
                and end.value < range_start.value
            ):
                flags.append("invalid_date_order")
                end = None
            return ParsedChronologyDate(
                expression=original,
                start=range_start,
                end=end,
                rule_flags=flags,
            )

    single_start, _, _, invalid = _date_fragment(
        core,
        expression=original,
        year_hint=year_hint,
        month_hint=None,
    )
    if single_start is not None:
        if single_start.certainty == "approximate" and "approximate_date" not in flags:
            flags.append("approximate_date")
        if invalid:
            flags.append("invalid_date_component")
        return ParsedChronologyDate(
            expression=original,
            start=single_start,
            rule_flags=flags,
        )

    fallback = (
        TemporalPoint(
            value=f"{year_hint:04d}",
            precision="year",
            certainty="inferred",
            original_text=original,
        )
        if year_hint is not None
        else TemporalPoint(original_text=original)
    )
    return ParsedChronologyDate(
        expression=original,
        start=fallback,
        rule_flags=[*flags, "unknown_date"],
    )


def _prepare_page_text(
    record: PageRecord, repeated_lines: set[str], *, layout: str
) -> str:
    lines: list[str] = []
    for raw_line in record.normalized_text.splitlines():
        line = re.sub(r"[ \t\u3000]+", " ", raw_line).strip()
        if not line or line in repeated_lines or PAGE_NUMBER_LINE.fullmatch(line):
            continue
        if layout == "zhou" and ZHOU_YEAR_HEADING.fullmatch(line):
            continue
        if layout == "lin" and LIN_YEAR_HEADING.fullmatch(line):
            continue
        lines.append(line)
    return "\n".join(lines)


def _append_zhou_block(
    block: str,
    *,
    page: PageRecord,
    profile: ChronologyProfile,
    temporal: ParsedChronologyDate,
    current: RawChronologyEntry | None,
    entries: list[RawChronologyEntry],
) -> RawChronologyEntry | None:
    pieces = block.split("△")
    prefix = pieces[0].strip()
    if prefix:
        if current is None:
            current = RawChronologyEntry(
                document_id=profile.document_id,
                subject_person_id=profile.subject_person_id,
                subject_name=profile.subject_name,
                temporal=temporal,
                file_sha256=page.file_sha256,
            )
        current.append(prefix, page)
    for piece in pieces[1:]:
        if current is not None and current.text:
            entries.append(current)
        current = RawChronologyEntry(
            document_id=profile.document_id,
            subject_person_id=profile.subject_person_id,
            subject_name=profile.subject_name,
            temporal=temporal,
            file_sha256=page.file_sha256,
        )
        current.append(piece, page)
    return current


def _extract_zhou_entries(
    pages: dict[int, PageRecord],
    repeated_lines: set[str],
    profile: ChronologyProfile,
) -> list[RawChronologyEntry]:
    entries: list[RawChronologyEntry] = []
    current: RawChronologyEntry | None = None
    previous_temporal: ParsedChronologyDate | None = None
    for _, page in sorted(pages.items()):
        text = _prepare_page_text(page, repeated_lines, layout="zhou")
        markers = list(ZHOU_DATE_MARKER.finditer(text))
        if not markers:
            if current is not None:
                current.append(text, page)
            continue
        prefix = text[: markers[0].start()]
        if current is not None:
            current.append(prefix, page)
        for index, marker in enumerate(markers):
            if current is not None and current.text:
                entries.append(current)
                current = None
            temporal = parse_chronology_date(
                marker.group("label"), inherited_from=previous_temporal
            )
            previous_temporal = temporal
            block_end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
            current = _append_zhou_block(
                text[marker.end() : block_end],
                page=page,
                profile=profile,
                temporal=temporal,
                current=current,
                entries=entries,
            )
    if current is not None and current.text:
        entries.append(current)
    return entries


def _load_structure(path: Path) -> list[StructureEntry]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [StructureEntry.model_validate(item) for item in payload]
    except (OSError, ValueError, TypeError) as exc:
        raise ExtractionError(f"Cannot load chronology structure {path}: {exc}") from exc


def _year_by_page(
    entries: list[StructureEntry], *, research_start: int, research_end: int
) -> dict[int, int]:
    result: dict[int, int] = {}
    for entry in entries:
        match = YEAR_IN_TITLE.match(entry.title.strip())
        if match is None:
            continue
        year = int(match.group(1))
        if not research_start <= year <= research_end:
            continue
        for pdf_page in range(entry.pdf_page_start, entry.pdf_page_end + 1):
            result[pdf_page] = year
    return result


def _extract_lin_entries(
    pages: dict[int, PageRecord],
    repeated_lines: set[str],
    profile: ChronologyProfile,
    year_pages: dict[int, int],
) -> list[RawChronologyEntry]:
    entries: list[RawChronologyEntry] = []
    current: RawChronologyEntry | None = None
    previous_temporal: ParsedChronologyDate | None = None
    current_year: int | None = None
    for pdf_page, page in sorted(pages.items()):
        year = year_pages.get(pdf_page)
        if year is None:
            continue
        if year != current_year:
            if current is not None and current.text:
                entries.append(current)
            current = None
            previous_temporal = None
            current_year = year
        text = _prepare_page_text(page, repeated_lines, layout="lin")
        markers = list(LIN_DATE_MARKER.finditer(text))
        if not markers:
            if current is not None:
                current.append(text, page)
            continue
        prefix = text[: markers[0].start()]
        if current is not None:
            current.append(prefix, page)
        for index, marker in enumerate(markers):
            if current is not None and current.text:
                entries.append(current)
            temporal = parse_chronology_date(
                marker.group("label"),
                year_hint=year,
                inherited_from=previous_temporal,
            )
            previous_temporal = temporal
            block_end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
            current = RawChronologyEntry(
                document_id=profile.document_id,
                subject_person_id=profile.subject_person_id,
                subject_name=profile.subject_name,
                temporal=temporal,
                file_sha256=page.file_sha256,
            )
            current.append(text[marker.end() : block_end], page)
    if current is not None and current.text:
        entries.append(current)
    return entries


class PersonMentionResolver:
    def __init__(self, catalog: PersonCatalog):
        self.people = {person.person_id: person for person in catalog.people}
        forms: dict[str, list[str]] = {}
        for person in catalog.people:
            for form in (person.canonical_name, *(alias.name for alias in person.aliases)):
                forms.setdefault(form, []).append(person.person_id)
        self.unique_forms = {
            form: person_ids[0]
            for form, person_ids in forms.items()
            if len(set(person_ids)) == 1
        }

    def participants(
        self, text: str, *, subject_person_id: str, subject_name: str
    ) -> list[EventParticipant]:
        matches: dict[str, tuple[int, str]] = {}
        for form, person_id in self.unique_forms.items():
            index = text.find(form)
            if index < 0:
                continue
            current = matches.get(person_id)
            if current is None or index < current[0] or (
                index == current[0] and len(form) > len(current[1])
            ):
                matches[person_id] = (index, form)
        result: list[EventParticipant] = []
        subject_match = matches.pop(subject_person_id, None)
        if subject_match is not None:
            result.append(
                EventParticipant(
                    person_id=subject_person_id,
                    role="年谱主体",
                    mention_text=subject_match[1],
                    mention_source="explicit",
                )
            )
        else:
            result.append(
                EventParticipant(
                    person_id=subject_person_id,
                    role="年谱主体",
                    mention_text=subject_name,
                    mention_source="chronology_subject",
                )
            )
        for person_id, (_, form) in sorted(matches.items(), key=lambda item: item[1][0]):
            result.append(
                EventParticipant(
                    person_id=person_id,
                    mention_text=form,
                    mention_source="explicit",
                )
            )
        return result


def _event_type(text: str) -> str:
    for event_type, terms in EVENT_TYPE_RULES:
        if any(term in text for term in terms):
            return event_type
    return "activity"


def _location_text(text: str) -> str | None:
    compact = re.sub(r"\s+", "", text)
    found: list[tuple[int, str]] = []
    for match in LOCATION_PATTERN.finditer(compact):
        location = match.group("place").rstrip("的")
        if location.startswith(("会议", "会上", "工作", "中央")):
            continue
        normalized: str | None = None
        for marker in SPATIAL_END_MARKERS:
            marker_index = location.find(marker)
            if marker_index >= 0:
                normalized = location[: marker_index + len(marker)]
                break
        if normalized is None:
            normalized = next(
                (
                    known
                    for known in sorted(COMMON_LOCATIONS, key=len, reverse=True)
                    if location.startswith(known)
                ),
                None,
            )
        if normalized is None and location.endswith(LOCATION_SUFFIXES):
            normalized = location
        if normalized is not None:
            found.append((match.start("place"), normalized))
    unique: list[str] = []
    for _, location in sorted(found):
        overlapping = next(
            (index for index, prior in enumerate(unique) if prior in location), None
        )
        if overlapping is not None:
            unique[overlapping] = location
        elif location not in unique and not any(location in prior for prior in unique):
            unique.append(location)
        if len(unique) == 3:
            break
    return "；".join(unique) if unique else None


def _event_name(subject_name: str, text: str) -> str:
    compact = re.sub(r"\s+", "", text)
    clause = re.split(r"[。！？；]", compact, maxsplit=1)[0]
    if len(clause) > 46:
        clause = clause[:46] + "…"
    return f"{subject_name}：{clause}"


def _confidence(entry: RawChronologyEntry, flags: list[str]) -> float:
    value = 0.94 if entry.document_id.startswith("zhou_enlai") else 0.89
    deductions = {
        "multiple_dates": 0.12,
        "inherited_date": 0.15,
        "unknown_date": 0.25,
        "year_only_fallback": 0.12,
        "invalid_date_component": 0.20,
        "invalid_date_order": 0.20,
        "approximate_date": 0.06,
        "cross_page": 0.04,
    }
    for flag in set(flags):
        value -= deductions.get(flag, 0.0)
    if entry.temporal.start.precision == "month":
        value -= 0.03
    elif entry.temporal.start.precision == "year":
        value -= 0.08
    if "ocr" in entry.page_methods.values():
        value -= 0.08
    return round(max(0.4, value), 2)


def _stable_ids(entry: RawChronologyEntry) -> tuple[str, str]:
    body_hash = hashlib.sha256(entry.text.encode("utf-8")).hexdigest()
    identity = "|".join(
        (
            entry.document_id,
            entry.file_sha256,
            entry.temporal.expression,
            str(entry.pages[0]),
            str(entry.pages[-1]),
            body_hash,
            entry.extractor_version,
        )
    )
    if entry.document_id.startswith("mao_zedong_chronology_"):
        identity += "|" + entry.temporal.start.model_dump_json()
        identity += "|" + (entry.temporal.end.model_dump_json() if entry.temporal.end else "")
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:28]
    return f"evt_{digest}", f"evd_{digest}"


def _entry_year(entry: RawChronologyEntry) -> int | None:
    value = entry.temporal.start.value
    if value is None:
        return None
    return int(value[:4])


def _build_candidate(
    entry: RawChronologyEntry, resolver: PersonMentionResolver
) -> ChronologyEventCandidate | None:
    text = entry.text
    if len(re.sub(r"\s+", "", text)) < 12 or not entry.pages:
        return None
    flags = [*entry.temporal.rule_flags, *entry.rule_flags]
    if len(entry.pages) > 1:
        flags.append("cross_page")
    participants = resolver.participants(
        text,
        subject_person_id=entry.subject_person_id,
        subject_name=entry.subject_name,
    )
    if participants[0].mention_source == "chronology_subject":
        flags.append("subject_from_chronology")
    event_id, evidence_id = _stable_ids(entry)
    quote = text[:1200]
    review_status: ReviewStatus = (
        "needs_review"
        if {
            "multiple_dates",
            "inherited_date",
            "unknown_date",
            "invalid_date_component",
            "invalid_date_order",
        }.intersection(flags)
        else "unreviewed"
    )
    evidence = [
        EvidenceReference(
            evidence_id=evidence_id,
            document_id=entry.document_id,
            pdf_page_start=entry.pages[0], pdf_page_end=entry.pages[-1],
            quote=quote, extraction_methods=sorted(set(entry.page_methods.values())),
        )
    ]
    if entry.document_id.startswith("mao_zedong_chronology_"):
        review_status = "needs_review"
        evidence = [
            EvidenceReference(
                evidence_id=f"{evidence_id}_p{page}", document_id=entry.document_id,
                pdf_page_start=(
                    page if len("\n".join(entry.page_text_parts[page])) >= 12 else entry.pages[0]
                ),
                pdf_page_end=(
                    page if len("\n".join(entry.page_text_parts[page])) >= 12 else entry.pages[-1]
                ),
                quote=(
                    "\n".join(entry.page_text_parts[page])[:1200]
                    if len("\n".join(entry.page_text_parts[page])) >= 12 else quote
                ),
                extraction_methods=(
                    [entry.page_methods[page]]
                    if len("\n".join(entry.page_text_parts[page])) >= 12
                    else sorted(set(entry.page_methods.values()))
                ),
            ) for page in entry.pages
        ]
    event = HistoricalEvent(
        event_id=event_id,
        name=_event_name(entry.subject_name, text),
        event_type=_event_type(text),
        start=entry.temporal.start,
        end=entry.temporal.end,
        location_text=_location_text(text),
        description=text,
        participants=participants,
        evidence=evidence,
        extraction_method="rule",
        extraction_confidence=_confidence(entry, flags),
        review_status=review_status,
        extractor_version=entry.extractor_version,
    )
    return ChronologyEventCandidate(
        event=event,
        date_expression=entry.temporal.expression,
        rule_flags=sorted(set(flags)),
        source_file_sha256=entry.file_sha256,
    )


def _write_jsonl(path: Path, candidates: list[ChronologyEventCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for candidate in candidates:
            stream.write(candidate.model_dump_json() + "\n")
    os.replace(temporary, path)


def _document_samples(
    document_id: str, candidates: list[ChronologyEventCandidate], *, count: int = 5
) -> list[dict[str, object]]:
    if not candidates:
        return []
    if len(candidates) <= count:
        selected = candidates
    else:
        indexes = {
            round(index * (len(candidates) - 1) / (count - 1)) for index in range(count)
        }
        selected = [candidates[index] for index in sorted(indexes)]
    return [
        {
            "document_id": document_id,
            "event_id": item.event.event_id,
            "date_expression": item.date_expression,
            "start": item.event.start.model_dump(),
            "end": item.event.end.model_dump() if item.event.end else None,
            "pdf_pages": [
                item.event.evidence[0].pdf_page_start,
                item.event.evidence[-1].pdf_page_end,
            ],
            "subject": item.event.participants[0].model_dump(),
            "event_type": item.event.event_type,
            "location_text": item.event.location_text,
            "rule_flags": item.rule_flags,
            "preview": re.sub(r"\s+", "", item.event.description)[:180],
        }
        for item in selected
    ]


def _extract_document(
    *,
    database: Database,
    profile: ChronologyProfile,
    pages_dir: Path,
    ocr_dir: Path,
    structure_dir: Path,
    events_dir: Path,
    resolver: PersonMentionResolver,
    research_start: int,
    research_end: int,
    dry_run: bool = False,
) -> tuple[ChronologyDocumentResult, list[ChronologyEventCandidate]]:
    base_path = pages_dir / f"{profile.document_id}.jsonl"
    if not base_path.is_file():
        raise ExtractionError(f"Missing extracted chronology pages: {base_path}")
    base_records = load_page_records(base_path)
    ocr_path = ocr_dir / f"{profile.document_id}.jsonl"
    ocr_records = load_page_records(ocr_path) if ocr_path.is_file() else {}
    pages = effective_pages(base_records, ocr_records)
    repeated_lines = detect_repeated_marginal_lines(pages)
    if profile.layout == "zhou":
        raw_entries = _extract_zhou_entries(pages, repeated_lines, profile)
    else:
        structure = _load_structure(structure_dir / f"{profile.document_id}.json")
        year_pages = _year_by_page(
            structure,
            research_start=research_start,
            research_end=research_end,
        )
        if profile.layout == "mao":
            from history_agent.research.mao_chronology import extract_mao_entries

            raw_entries = extract_mao_entries(pages, profile, year_pages)
        else:
            raw_entries = _extract_lin_entries(
                pages, repeated_lines, profile, year_pages
            )

    candidates: list[ChronologyEventCandidate] = []
    skipped_short = 0
    skipped_out_of_scope = 0
    skipped_context = 0
    for entry in raw_entries:
        year = _entry_year(entry)
        if year is None or not research_start <= year <= research_end:
            skipped_out_of_scope += 1
            continue
        if profile.layout == "mao":
            from history_agent.research.mao_chronology import is_subject_entry

            if not is_subject_entry(entry):
                skipped_context += 1
                continue
        candidate = _build_candidate(entry, resolver)
        if candidate is None:
            skipped_short += 1
            continue
        candidates.append(candidate)

    output_path = events_dir / f"{profile.document_id}.jsonl"
    created = updated = skipped = 0
    if not dry_run:
        _write_jsonl(output_path, candidates)
        created, updated, skipped = ResearchStore(database).sync_generated_events(
            [candidate.event for candidate in candidates]
        )
    precision_counts = Counter(
        candidate.event.start.precision for candidate in candidates
    )
    flag_counts = Counter(
        flag for candidate in candidates for flag in candidate.rule_flags
    )
    event_type_counts = Counter(candidate.event.event_type for candidate in candidates)
    result = ChronologyDocumentResult(
        document_id=profile.document_id,
        subject_person_id=profile.subject_person_id,
        pages_processed=len(pages),
        raw_entries=len(raw_entries),
        candidates=len(candidates),
        database_created=created,
        database_updated=updated,
        database_skipped=skipped,
        skipped_short=skipped_short,
        skipped_out_of_scope=skipped_out_of_scope,
        skipped_context=skipped_context,
        date_precision_counts=dict(sorted(precision_counts.items())),
        rule_flag_counts=dict(sorted(flag_counts.items())),
        event_type_counts=dict(sorted(event_type_counts.items())),
        location_candidates=sum(
            candidate.event.location_text is not None for candidate in candidates
        ),
        output_path=str(output_path),
    )
    return result, candidates


def extract_chronology_events(
    *,
    database: Database,
    pages_dir: Path,
    ocr_dir: Path,
    structure_dir: Path,
    events_dir: Path,
    reports_dir: Path,
    person_aliases_path: Path,
    run_id: str,
    research_start: int,
    research_end: int,
    document_ids: list[str] | None = None,
    dry_run: bool = False,
) -> ChronologyExtractionSummary:
    selected = list(document_ids or DEFAULT_CHRONOLOGIES)
    unsupported = sorted(set(selected) - set(PROFILES))
    if unsupported:
        raise ExtractionError(
            "Unsupported chronology document IDs: " + ", ".join(unsupported)
        )
    catalog = load_person_catalog(person_aliases_path)
    known_people = {person.person_id for person in catalog.people}
    missing_subjects = [
        PROFILES[document_id].subject_person_id
        for document_id in selected
        if PROFILES[document_id].subject_person_id not in known_people
    ]
    if missing_subjects:
        raise ExtractionError(
            "Chronology subjects are missing from the person catalog: "
            + ", ".join(missing_subjects)
        )
    resolver = PersonMentionResolver(catalog)
    results: list[ChronologyDocumentResult] = []
    samples: list[dict[str, object]] = []
    for document_id in selected:
        result, candidates = _extract_document(
            database=database,
            profile=PROFILES[document_id],
            pages_dir=pages_dir,
            ocr_dir=ocr_dir,
            structure_dir=structure_dir,
            events_dir=events_dir,
            resolver=resolver,
            research_start=research_start,
            research_end=research_end,
            dry_run=dry_run,
        )
        results.append(result)
        samples.extend(_document_samples(document_id, candidates))
    summary = ChronologyExtractionSummary(
        run_id=run_id,
        dry_run=dry_run,
        documents=results,
        samples=samples,
    )
    payload = summary.model_dump()
    payload["totals"] = summary.totals
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"chronology_extraction_{run_id}.json"
    latest_path = reports_dir / "chronology_extraction_latest.json"
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    report_path.write_text(rendered, encoding="utf-8")
    latest_path.write_text(rendered, encoding="utf-8")
    return summary
