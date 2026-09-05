"""Date-entry segmentation for the nine-volume Mao chronology.

Only dated body sections are consumed. All output remains reviewable candidates;
names in the body are mentions, not proof that people acted together.
"""
from __future__ import annotations

import re
from datetime import date, timedelta

from history_agent.extraction.models import PageRecord
from history_agent.research.chronology import (
    ChronologyProfile,
    ParsedChronologyDate,
    RawChronologyEntry,
    parse_chronology_date,
)

MAO_EXTRACTOR_VERSION = "mao-chronology-rules-v1"
DIGIT = r"\d(?:[ \t\n]*\d)?"
DAY = rf"{DIGIT}\s*月\s*{DIGIT}\s*日"
TAIL = rf"(?:{DIGIT}\s*月\s*)?{DIGIT}\s*日"
QUALIFIER = r"中下旬|上旬|中旬|下旬|月初|初|末|底"
MARKER = re.compile(
    rf"(?m)^[ \t\u3000]*(?P<label>"
    rf"{DAY}(?:\s*[—－–~～至一、,，]\s*{TAIL})*"
    rf"(?:前后|左右|前|后)?"
    rf"|{DIGIT}\s*月\s*(?:{QUALIFIER})"
    rf"|{DIGIT}\s*月(?=[ \t\u3000\n])"
    rf"|同日|翌日|次日|当日|本月|是月|本年|是年|[春夏秋冬](?=[ \t\n]))"
)
FOOTNOTE = re.compile(r"^[ \t]*[〔\[（(【C]\s*[\dlI]{1,2}\s*[〕\]）)】Jj}]")
YEAR_HEADER = re.compile(
    r"^\s*(?:[\dlIO][\s.]*){4}年[ \t]*(?:\n[ \t]*)?"
    r"(?:[一二三四五六七八九十百〇零\d\s]{1,15}岁"
    r"|(?:\d\s*){1,2}月(?=[ \t]*\n))?"
)


def is_subject_entry(entry: RawChronologyEntry) -> bool:
    """Do not turn contextual world events into actions by the chronology subject."""
    if "毛泽东" in entry.text or "毛主席" in entry.text:
        return True
    body = re.sub(r"\s+", "", entry.text)
    label = re.sub(r"\s+", "", entry.temporal.expression)
    body = body.removeprefix(label)
    body = re.sub(
        r"^(?:(?:上午|下午|晚上|晚间|晚|早晨|凌晨|清晨|晨|夜|中午|午后|深夜)"
        r"[一二三四五六七八九十〇零两\d点时分半]*[，,、]?)+", "", body,
    )
    return body.startswith((
        "和", "与", "同", "在", "为", "致", "复", "阅", "审阅", "起草", "出席",
        "主持", "会见", "接见", "到", "离开", "返回", "抵", "得知", "修改", "签",
        "听", "写", "作", "读", "指示", "批", "约", "收到", "转", "电", "赴",
        "谈", "继续", "参加", "召集", "就", "关于", "要求", "应", "向", "陪", "率",
        "发表", "离", "回", "决定", "提出",
    ))


def body_text(page: PageRecord) -> tuple[str, bool]:
    text = page.normalized_text.strip()
    # Remove only the leading physical/printed page number, never a day number
    # appearing on its own line inside a split date label.
    text = re.sub(r"^\d{1,4}[ \t]*\n", "", text, count=1)
    text = YEAR_HEADER.sub("", text, count=1).lstrip()
    text = re.sub(r"^\d{1,4}[ \t]*\n", "", text, count=1)
    lines: list[str] = []
    removed = False
    for line in text.splitlines():
        if FOOTNOTE.match(line):
            removed = True
            break
        lines.append(line.strip())
    return "\n".join(lines).strip(), removed


def _date(
    label: str, year: int, previous: ParsedChronologyDate | None,
) -> ParsedChronologyDate:
    compact = re.sub(r"\s+", "", label)
    if compact in {"同日", "当日", "翌日", "次日", "本月", "是月"}:
        result = parse_chronology_date("在此期间", year_hint=year, inherited_from=previous)
        result.expression = label
        result.start.original_text = label
        if result.end is not None:
            result.end.original_text = label
        if compact in {"本月", "是月"} and result.start.value:
            result.start.value = result.start.value[:7]
            result.start.precision = "month" if len(result.start.value) == 7 else "year"
            result.end = None
        if compact in {"翌日", "次日"}:
            if result.start.precision == "day" and result.start.value and result.end is None:
                result.start.value = (
                    date.fromisoformat(result.start.value) + timedelta(days=1)
                ).isoformat()
            else:
                result = parse_chronology_date("日期不明", year_hint=year)
                result.expression = label
                result.start.original_text = label
                result.rule_flags.append("relative_date_unresolved")
        return result
    if compact in {"本年", "是年"}:
        result = parse_chronology_date(f"{year}年")
        result.expression = label
        result.start.original_text = label
        result.start.certainty = "inferred"
        return result
    return parse_chronology_date(label, year_hint=year)


def extract_mao_entries(
    pages: dict[int, PageRecord], profile: ChronologyProfile, year_pages: dict[int, int],
) -> list[RawChronologyEntry]:
    entries: list[RawChronologyEntry] = []
    current: RawChronologyEntry | None = None
    previous: ParsedChronologyDate | None = None
    current_year: int | None = None
    last_page: int | None = None
    for number, page in sorted(pages.items()):
        year = year_pages.get(number)
        # A missing page or non-body section must not silently join two events.
        if year != current_year or (last_page is not None and number != last_page + 1):
            if current is not None and current.text:
                if last_page is not None and number != last_page + 1:
                    current.rule_flags.append("page_gap")
                entries.append(current)
            current = None
            previous = None
        current_year = year
        last_page = number
        if year is None:
            continue
        text, removed = body_text(page)
        markers = list(MARKER.finditer(text))
        prefix = text[:markers[0].start()] if markers else text
        if current is not None:
            current.append(prefix, page)
            if removed:
                current.rule_flags.append("footnotes_excluded")
        for i, marker in enumerate(markers):
            if current is not None and current.text:
                entries.append(current)
            temporal = _date(marker.group("label"), year, previous)
            previous = temporal
            end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
            current = RawChronologyEntry(
                document_id=profile.document_id,
                subject_person_id=profile.subject_person_id, subject_name=profile.subject_name,
                temporal=temporal, file_sha256=page.file_sha256,
                extractor_version=MAO_EXTRACTOR_VERSION,
                rule_flags=["mao_candidate", *(["footnotes_excluded"] if removed else [])],
            )
            # Keep the date anchor in the evidence even when its body starts
            # on the following page.
            current.append(marker.group("label"), page)
            current.append(text[marker.end():end], page)
    if current is not None and current.text:
        entries.append(current)
    return entries
