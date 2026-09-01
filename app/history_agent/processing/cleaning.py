from __future__ import annotations

import re
from collections import Counter

from history_agent.extraction.models import PageRecord

CLEANER_VERSION = "page-cleaner-v1"
PAGE_NUMBER_LINE = re.compile(r"^[\s—–-]*\d{1,4}[\s—–-]*$")
HEADING_PATTERNS = (
    re.compile(r"^第[一二三四五六七八九十百零〇0-9]+[章节编部卷篇].{0,40}$"),
    re.compile(r"^(?:19|20)\d{2}年(?:\d{1,2}月(?:\d{1,2}日)?)?.{0,30}$"),
    re.compile(r"^(目录|前言|序言|引言|后记|附录|出版说明|编者说明)$"),
)
PARAGRAPH_END = re.compile(r"[。！？；：…）”』】]$")
DOT_LEADER = re.compile(r"\.{5,}|…{5,}")


def detect_repeated_marginal_lines(pages: dict[int, PageRecord]) -> set[str]:
    counts: Counter[str] = Counter()
    for record in pages.values():
        lines = [line.strip() for line in record.normalized_text.splitlines() if line.strip()]
        for line in set(lines[:2] + lines[-2:]):
            if 1 < len(line) <= 50 and not PAGE_NUMBER_LINE.fullmatch(line):
                counts[line] += 1
    threshold = max(5, round(len(pages) * 0.04))
    return {line for line, count in counts.items() if count >= threshold}


def is_heading(line: str) -> bool:
    return any(pattern.fullmatch(line) for pattern in HEADING_PATTERNS)


def is_table_of_contents_page(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    first_lines = "".join(lines[:12])
    leader_lines = sum(bool(DOT_LEADER.search(line)) for line in lines)
    return "目录" in first_lines and leader_lines >= 3


def clean_page_text(text: str, repeated_lines: set[str]) -> str:
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"[ \t\u3000]+", " ", raw_line).strip()
        if line in repeated_lines or PAGE_NUMBER_LINE.fullmatch(line):
            continue
        cleaned_lines.append(line)

    paragraphs: list[str] = []
    buffer = ""
    for line in cleaned_lines:
        if not line:
            if buffer:
                paragraphs.append(buffer)
                buffer = ""
            continue
        if is_heading(line):
            if buffer:
                paragraphs.append(buffer)
                buffer = ""
            paragraphs.append(line)
            continue
        buffer += line
        if PARAGRAPH_END.search(line):
            paragraphs.append(buffer)
            buffer = ""
    if buffer:
        paragraphs.append(buffer)
    return "\n".join(paragraph for paragraph in paragraphs if paragraph)
