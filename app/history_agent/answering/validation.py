from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode

from history_agent.answering.models import Citation

EVIDENCE_MARKER = re.compile(r"\[(E\d+)\]")
MARKDOWN = MarkdownIt("commonmark").enable("table")
CITATION_ONLY = re.compile(r"(?:\s*\[E\d+\][\s,，、;；。.]*)+")
# Only nominal topic labels are exempt; a factual heading still needs a citation.
TOPIC_LABEL = re.compile(
    r"(?:[一二三四五六七八九十0-9]+[、.)．]\s*)?"
    r"(?:(?:共同)?(?:参加|参与|出席)(?:会议|活动)(?:情况)?|"
    r"(?:成立|召开)(?:背景|过程)|(?:参与|出席|参加)(?:者|人员)(?:个例|情况|构成)?|"
    r"主要讲话|会议决定)[:：]?"
)
EVIDENCE_LIMIT = re.compile(
    r"^(?:(?:现有|当前|本次|所提供的|检索到的|已提供的)*(?:资料|材料|史料|证据|片段)"
    r"(?:尚|仍|还)?(?:不足以|无法|不能|未能|未|没有)|"
    r"(?:尚无法|尚不能|无法|不能|不足以))"
    r"(?:确认|证实|判断|确定|证明|推断)"
)
EVIDENCE_SCOPE_NOTE = re.compile(
    r"(?:证据包|证据|资料|材料|史料).*(?:无更多材料|不足|无法进一步说明|"
    r"无直接关联|不予采入|未予采入|不纳入|未纳入|"
    r"(?:仅|只)(?:能)?覆盖[^。！？；]{0,80}(?:(?:几个|若干|部分|少数)"
    r"(?:时间点|时期|年代|方面)|(?:有限的)?(?:时间|时期)?范围))"
)
CLAUSE_BOUNDARY = re.compile(r"[，,。！？；;\n]")
ASSERTION_TRANSITION = re.compile(r"但|然而|不过|实际|事实上|而且|并且|随后|因此|所以")
CHINESE_YEAR = r"[一二三四五六七八九〇零]{4}年"
DATE_SIGNAL = re.compile(rf"(?:(?:18|19|20)\d{{2}}年|{CHINESE_YEAR}|\d{{1,2}}月\d{{1,2}}日)")
CORE_FACT_SIGNAL = re.compile(
    rf"(?:"
    rf"(?:18|19|20)\d{{2}}年|{CHINESE_YEAR}|\d{{1,2}}月\d{{1,2}}日|"
    r"担任|任命|出任|兼任|调任|任职|主持|参加|参与|出席|会见|访问|考察|"
    r"领导|负责|指挥|汇报|讲话|发言|提出|指出|认为|主张|决定|通过|召开|"
    r"成立|开展|发动|抵达|到达|前往|赴|离开|返回|逝世|撤职|免职|"
    r"上级|下属|同事|领导关系|组织关系|记载|发生|签署|发布|执行"
    r")"
)
SOURCE_PAGE_REFERENCE = re.compile(
    r"(?:(?:《(?P<document>[^》\n]{1,80})》)[^\n。！？；]{0,24})?"
    r"PDF\s*第\s*(?P<page>\d+)\s*页",
    re.IGNORECASE,
)
DOCUMENT_NORMALIZATION = re.compile(r"[^\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9]")


@dataclass(frozen=True)
class AnswerValidationResult:
    valid: bool
    error_code: str | None = None
    used_evidence_ids: tuple[str, ...] = ()
    uncited_claims: tuple[str, ...] = ()
    citation_mismatches: tuple[str, ...] = ()


@dataclass
class _ClaimBlock:
    text: str
    start_line: int | None
    end_line: int | None


def _claim_text(line: str) -> str:
    return EVIDENCE_MARKER.sub("", line).strip()


def _node_text(node: SyntaxTreeNode) -> str:
    if node.type in {"softbreak", "hardbreak"}:
        return " "
    if node.children:
        separator = " | " if node.type == "tr" else ""
        return separator.join(_node_text(child) for child in node.children)
    return node.content


def _claim_blocks(answer: str) -> list[_ClaimBlock]:
    """Respect Markdown paragraph, quotation, list-item and table-row boundaries."""
    blocks: list[_ClaimBlock] = []

    def visit(container: SyntaxTreeNode) -> None:
        preceding_paragraph: int | None = None
        for child in container.children:
            if child.type in {"paragraph", "heading", "tr", "fence", "code_block", "html_block"}:
                text = _node_text(child).strip()
                if (
                    child.type == "paragraph"
                    and preceding_paragraph is not None
                    and CITATION_ONLY.fullmatch(text)
                ):
                    # A citation on its own line belongs only to the immediately
                    # preceding paragraph in this container, never another list item.
                    blocks[preceding_paragraph].text += " " + text
                    if child.map is not None:
                        blocks[preceding_paragraph].end_line = child.map[1]
                    continue
                is_undated_heading = child.type == "heading" and not DATE_SIGNAL.search(text)
                is_nested_list_label = container.type == "list_item" and any(
                    item.type in {"bullet_list", "ordered_list"} for item in container.children
                )
                is_table_header = container.type == "thead" and all(
                    TOPIC_LABEL.fullmatch(_node_text(cell))
                    or not _is_core_fact_block(_node_text(cell))
                    for cell in child.children
                )
                if (
                    is_undated_heading
                    or (is_nested_list_label and TOPIC_LABEL.fullmatch(text))
                    or is_table_header
                ):
                    preceding_paragraph = None
                    continue
                start_line, end_line = child.map if child.map is not None else (None, None)
                blocks.append(_ClaimBlock(text, start_line, end_line))
                preceding_paragraph = len(blocks) - 1 if child.type == "paragraph" else None
            else:
                preceding_paragraph = None
                visit(child)

    visit(SyntaxTreeNode(MARKDOWN.parse(answer)))
    return blocks


def _is_core_fact_block(block: str) -> bool:
    claim = _claim_text(block)
    if not ASSERTION_TRANSITION.search(claim) and (
        EVIDENCE_LIMIT.search(claim) or EVIDENCE_SCOPE_NOTE.search(claim)
    ):
        return False
    for clause in CLAUSE_BOUNDARY.split(claim):
        clause = clause.strip()
        if CORE_FACT_SIGNAL.search(clause) and not (
            EVIDENCE_LIMIT.match(clause) and not ASSERTION_TRANSITION.search(clause)
        ):
            return True
    return False


def _normalize_document(value: str) -> str:
    return DOCUMENT_NORMALIZATION.sub("", value).casefold()


def _document_matches(claimed: str | None, actual: str) -> bool:
    if claimed is None:
        return True
    normalized_claimed = _normalize_document(claimed)
    normalized_actual = _normalize_document(actual)
    return bool(normalized_claimed) and (
        normalized_claimed in normalized_actual or normalized_actual in normalized_claimed
    )


def validate_grounded_answer(answer: str, citations: list[Citation]) -> AnswerValidationResult:
    """Check reference syntax, metadata and coverage, not semantic entailment."""

    citation_by_id = {citation.evidence_id: citation for citation in citations}
    if len(citation_by_id) != len(citations):
        return AnswerValidationResult(valid=False, error_code="invalid_citation_bundle")

    blocks = _claim_blocks(answer)
    # Markdown can decode escaped brackets/entities into visible evidence IDs.
    # Validate those too before looking them up while checking source metadata.
    markers = EVIDENCE_MARKER.findall(answer) + EVIDENCE_MARKER.findall(
        "\n".join(block.text for block in blocks)
    )
    used_evidence_ids = tuple(dict.fromkeys(markers))
    if not markers:
        return AnswerValidationResult(
            valid=False,
            error_code="missing_evidence_markers",
        )

    unknown_markers = [marker for marker in used_evidence_ids if marker not in citation_by_id]
    if unknown_markers:
        return AnswerValidationResult(
            valid=False,
            error_code="invalid_evidence_marker",
            used_evidence_ids=used_evidence_ids,
        )

    uncited_claims: list[str] = []
    citation_mismatches: list[str] = []
    for block in blocks:
        block_markers = EVIDENCE_MARKER.findall(block.text)
        if _is_core_fact_block(block.text) and not block_markers:
            uncited_claims.append(_claim_text(block.text))
        for match in SOURCE_PAGE_REFERENCE.finditer(block.text):
            claimed_page = int(match.group("page"))
            claimed_document = match.group("document")
            matching_citation = any(
                citation_by_id[marker].pdf_page == claimed_page
                and _document_matches(claimed_document, citation_by_id[marker].document)
                for marker in block_markers
            )
            if not matching_citation:
                citation_mismatches.append(match.group(0))

    if citation_mismatches:
        return AnswerValidationResult(
            valid=False,
            error_code="citation_metadata_mismatch",
            used_evidence_ids=used_evidence_ids,
            uncited_claims=tuple(uncited_claims),
            citation_mismatches=tuple(citation_mismatches),
        )
    if uncited_claims:
        return AnswerValidationResult(
            valid=False,
            error_code="uncited_core_claim",
            used_evidence_ids=used_evidence_ids,
            uncited_claims=tuple(uncited_claims),
        )
    return AnswerValidationResult(
        valid=True,
        used_evidence_ids=used_evidence_ids,
    )


def remove_uncited_claim_blocks(answer: str, uncited_claims: tuple[str, ...]) -> str | None:
    """Remove only source blocks rejected for missing citations.

    The caller must validate the returned Markdown again before using it. Returning
    ``None`` keeps the safe extractive fallback for structures without source maps.
    """

    remaining = Counter(uncited_claims)
    removed_lines: set[int] = set()
    for block in _claim_blocks(answer):
        claim = _claim_text(block.text)
        if not remaining[claim]:
            continue
        if block.start_line is None or block.end_line is None:
            return None
        removed_lines.update(range(block.start_line, block.end_line))
        remaining[claim] -= 1
    if any(remaining.values()) or not removed_lines:
        return None
    lines = answer.splitlines(keepends=True)
    candidate = "".join(line for index, line in enumerate(lines) if index not in removed_lines)
    candidate = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", candidate).strip()
    return candidate or None
