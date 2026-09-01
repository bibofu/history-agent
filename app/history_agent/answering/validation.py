from __future__ import annotations

import re
from dataclasses import dataclass

from history_agent.answering.models import Citation

EVIDENCE_MARKER = re.compile(r"\[(E\d+)\]")
MARKDOWN_LIST_PREFIX = re.compile(r"^\s*(?:[-+*>]|\d+[.)、])\s*")
CHINESE_YEAR = r"[一二三四五六七八九〇零]{4}年"
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


def _claim_text(line: str) -> str:
    text = MARKDOWN_LIST_PREFIX.sub("", line.strip())
    text = text.replace("**", "").replace("`", "")
    return EVIDENCE_MARKER.sub("", text).strip()


def _is_core_fact_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return False
    claim = _claim_text(stripped)
    if not claim or claim.endswith(("：", ":")):
        return False
    return CORE_FACT_SIGNAL.search(claim) is not None


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


def validate_grounded_answer(
    answer: str, citations: list[Citation]
) -> AnswerValidationResult:
    """Validate evidence IDs, explicit source metadata, and factual-line coverage."""

    citation_by_id = {citation.evidence_id: citation for citation in citations}
    if len(citation_by_id) != len(citations):
        return AnswerValidationResult(valid=False, error_code="invalid_citation_bundle")

    markers = EVIDENCE_MARKER.findall(answer)
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
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line_markers = EVIDENCE_MARKER.findall(line)
        if _is_core_fact_line(line) and not line_markers:
            uncited_claims.append(_claim_text(line))
        for match in SOURCE_PAGE_REFERENCE.finditer(line):
            claimed_page = int(match.group("page"))
            claimed_document = match.group("document")
            matching_citation = any(
                citation_by_id[marker].pdf_page == claimed_page
                and _document_matches(claimed_document, citation_by_id[marker].document)
                for marker in line_markers
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
