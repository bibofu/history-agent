from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from history_agent.errors import IndexBuildError, RetrievalError
from history_agent.processing.chunks import load_person_aliases
from history_agent.processing.models import ChunkRecord
from history_agent.retrieval.models import KeywordIndexSummary, SearchHit, SearchResponse

INDEX_VERSION = "sqlite-fts5-cjk-bigram-v1"
CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
LATIN_OR_NUMBER = re.compile(r"[A-Za-z]+(?:[-_][A-Za-z0-9]+)*|\d+(?:\.\d+)?")
YEAR = re.compile(r"(?<!\d)((?:18|19|20)\d{2})(?!\d)")
QUERY_STOP_TERMS = {
    "哪些",
    "什么",
    "怎么",
    "如何",
    "主要",
    "有哪",
    "都有",
    "请问",
    "一下",
    "这个",
    "那个",
    "是否",
    "可以",
    "人物",
    "经历",
    "观点",
    "交集",
    "期间",
}
QUERY_STOP_EDGE_CHARACTERS = set("的了在和与及是有为于把被从对")
PERIOD_RANGES = {
    "长征": (1934, 1936),
    "抗日战争": (1937, 1945),
    "全面抗战": (1937, 1945),
    "解放战争": (1945, 1949),
    "抗美援朝": (1950, 1953),
    "大跃进": (1958, 1960),
    "文化大革命": (1966, 1976),
    "文革": (1966, 1976),
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def cjk_bigrams(value: str) -> list[str]:
    if len(value) <= 2:
        return [value]
    return [value[index : index + 2] for index in range(len(value) - 1)]


def tokenize_for_index(text: str) -> list[str]:
    tokens: list[str] = []
    for match in CJK_RUN.finditer(text):
        tokens.extend(cjk_bigrams(match.group()))
    tokens.extend(match.group().casefold() for match in LATIN_OR_NUMBER.finditer(text))
    for match in YEAR.finditer(text):
        tokens.extend([match.group(1), f"{match.group(1)}年"])
    return tokens


def tokenize_query(query: str) -> list[str]:
    tokens = tokenize_for_index(query)
    filtered: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if (
            token in QUERY_STOP_TERMS
            or token in seen
            or len(token.strip()) < 2
            or (
                CJK_RUN.fullmatch(token)
                and (
                    token[0] in QUERY_STOP_EDGE_CHARACTERS
                    or token[-1] in QUERY_STOP_EDGE_CHARACTERS
                )
            )
        ):
            continue
        seen.add(token)
        filtered.append(token)
    if not filtered:
        raise RetrievalError(
            "The query does not contain searchable Chinese, Latin, or numeric terms."
        )
    return filtered


def infer_query_intent(query: str) -> str:
    if any(term in query for term in ("交集", "共同", "一起", "关系")):
        return "intersection"
    if any(term in query for term in ("记述", "描述")):
        return "observation"
    if any(
        term in query
        for term in ("观点", "论述", "主张", "看法", "如何看", "怎样分析", "如何分析")
    ):
        return "viewpoint"
    if YEAR.search(query) and any(
        term in query
        for term in ("经历", "做了什么", "活动", "任职", "担任", "职务", "主要做")
    ):
        return "timeline"
    return "general"


def hard_filter_people(query_people: list[str], query_intent: str) -> list[str]:
    """Return people that must occur in a chunk for the current query intent."""

    # A person's own selected works often omit the author's name from the body.
    # Viewpoint queries therefore use creator/source ranking bonuses instead.
    if query_intent in {"viewpoint", "observation"}:
        return []
    return query_people


def infer_year_range(query: str, explicit_years: list[int]) -> list[int]:
    if explicit_years:
        return [min(explicit_years), max(explicit_years)]
    ranges = [year_range for name, year_range in PERIOD_RANGES.items() if name in query]
    if not ranges:
        return []
    return [min(item[0] for item in ranges), max(item[1] for item in ranges)]


def load_chunks(chunks_dir: Path) -> list[ChunkRecord]:
    paths = sorted(chunks_dir.glob("*.jsonl"))
    if not paths:
        raise IndexBuildError(f"No chunk JSONL files found in {chunks_dir}")
    records: list[ChunkRecord] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(ChunkRecord.model_validate_json(line))
                except Exception as exc:
                    raise IndexBuildError(
                        f"Invalid chunk record in {path} line {line_number}: {exc}"
                    ) from exc
    return records


def _initialize(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        CREATE TABLE chunk_metadata (
            chunk_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            title TEXT NOT NULL,
            filename TEXT NOT NULL,
            volume TEXT,
            source_type TEXT NOT NULL,
            verification_status TEXT NOT NULL,
            creators_json TEXT NOT NULL,
            pdf_page_start INTEGER NOT NULL,
            pdf_page_end INTEGER NOT NULL,
            section_path_json TEXT NOT NULL,
            text TEXT NOT NULL,
            year_mentions_json TEXT NOT NULL,
            scope_status TEXT NOT NULL,
            people_json TEXT NOT NULL,
            extraction_methods_json TEXT NOT NULL
        );
        CREATE INDEX idx_chunk_metadata_document ON chunk_metadata(document_id);
        CREATE INDEX idx_chunk_metadata_scope ON chunk_metadata(scope_status);
        CREATE VIRTUAL TABLE chunk_fts USING fts5(
            chunk_id UNINDEXED,
            terms,
            tokenize = 'unicode61 remove_diacritics 2'
        );
        """
    )


def build_keyword_index(
    *, chunks_dir: Path, index_path: Path, reports_dir: Path, run_id: str
) -> KeywordIndexSummary:
    started_at = utc_now()
    records = load_chunks(chunks_dir)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = index_path.with_suffix(index_path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary_path)
    try:
        _initialize(connection)
        metadata_rows = []
        fts_rows = []
        for record in records:
            metadata_rows.append(
                (
                    record.chunk_id,
                    record.document_id,
                    record.title,
                    record.filename,
                    record.volume,
                    record.source_type,
                    record.verification_status,
                    json.dumps(record.creators, ensure_ascii=False),
                    record.pdf_page_start,
                    record.pdf_page_end,
                    json.dumps(record.section_path, ensure_ascii=False),
                    record.text,
                    json.dumps(record.year_mentions),
                    record.scope_status,
                    json.dumps(record.people, ensure_ascii=False),
                    json.dumps(record.extraction_methods),
                )
            )
            fts_rows.append((record.chunk_id, " ".join(tokenize_for_index(record.search_text))))
        connection.executemany(
            """
            INSERT INTO chunk_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            metadata_rows,
        )
        connection.executemany(
            "INSERT INTO chunk_fts (chunk_id, terms) VALUES (?, ?)", fts_rows
        )
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or integrity[0] != "ok":
            raise IndexBuildError(f"Keyword index integrity check failed: {integrity}")
    except Exception:
        connection.close()
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        if connection:
            connection.close()
    os.replace(temporary_path, index_path)
    summary = KeywordIndexSummary(
        run_id=run_id,
        started_at=started_at,
        finished_at=utc_now(),
        index_version=INDEX_VERSION,
        documents=len({record.document_id for record in records}),
        chunks=len(records),
        index_path=str(index_path),
        size_bytes=index_path.stat().st_size,
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    rendered = summary.model_dump_json(indent=2)
    (reports_dir / f"keyword_index_{run_id}.json").write_text(rendered, encoding="utf-8")
    (reports_dir / "keyword_index_latest.json").write_text(rendered, encoding="utf-8")
    return summary


def _match_expression(tokens: list[str]) -> str:
    escaped = [token.replace('"', '""') for token in tokens]
    return " OR ".join(f'"{token}"' for token in escaped)


def search_keyword_index(
    *,
    index_path: Path,
    query: str,
    aliases_path: Path,
    top_k: int = 10,
    document_ids: list[str] | None = None,
    include_out_of_scope: bool = False,
) -> SearchResponse:
    if not index_path.is_file():
        raise RetrievalError(f"Keyword index does not exist: {index_path}")
    terms = tokenize_query(query)
    query_intent = infer_query_intent(query)
    query_years = sorted({int(match.group(1)) for match in YEAR.finditer(query)})
    query_year_range = infer_year_range(query, query_years)
    aliases = load_person_aliases(aliases_path)
    query_people = sorted(
        person
        for person, names in aliases.items()
        if person in query or any(alias in query for alias in names)
    )
    filter_people = hard_filter_people(query_people, query_intent)
    conditions = ["chunk_fts MATCH ?"]
    parameters: list[object] = [_match_expression(terms)]
    if document_ids:
        placeholders = ", ".join("?" for _ in document_ids)
        conditions.append(f"m.document_id IN ({placeholders})")
        parameters.extend(document_ids)
    if query_years:
        year_conditions = []
        for year in query_years:
            year_conditions.append(
                "EXISTS (SELECT 1 FROM json_each(m.year_mentions_json) WHERE value = ?)"
            )
            parameters.append(year)
        conditions.append("(" + " OR ".join(year_conditions) + ")")
    elif query_year_range:
        conditions.append(
            "EXISTS (SELECT 1 FROM json_each(m.year_mentions_json) WHERE value BETWEEN ? AND ?)"
        )
        parameters.extend(query_year_range)
    elif not include_out_of_scope:
        conditions.append("m.scope_status != 'out_of_scope'")
    for person in filter_people:
        conditions.append(
            "EXISTS (SELECT 1 FROM json_each(m.people_json) WHERE value = ?)"
        )
        parameters.append(person)
    score_expression = "bm25(chunk_fts)"
    score_parameters: list[object] = []
    if query_intent == "timeline":
        score_expression += " - CASE WHEN m.source_type = 'chronology' THEN 4.0 ELSE 0 END"
    elif query_intent == "viewpoint":
        score_expression += " - CASE WHEN m.source_type = 'selected_works' THEN 4.0 ELSE 0 END"
    if query_people:
        creator_placeholders = ", ".join("?" for _ in query_people)
        score_expression += (
            " - CASE WHEN EXISTS (SELECT 1 FROM json_each(m.creators_json) "
            f"WHERE value IN ({creator_placeholders})) THEN 2.0 ELSE 0 END"
        )
        score_parameters.extend(query_people)
    sql = f"""
        SELECT m.*, {score_expression} AS raw_score
        FROM chunk_fts
        JOIN chunk_metadata m ON m.chunk_id = chunk_fts.chunk_id
        WHERE {' AND '.join(conditions)}
        ORDER BY raw_score
        LIMIT ?
    """
    connection = sqlite3.connect(f"file:{index_path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(sql, [*score_parameters, *parameters, top_k]).fetchall()
    except sqlite3.Error as exc:
        raise RetrievalError(f"Keyword search failed: {exc}") from exc
    finally:
        connection.close()
    hits = [
        SearchHit(
            rank=rank,
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            title=row["title"],
            filename=row["filename"],
            volume=row["volume"],
            source_type=row["source_type"],
            verification_status=row["verification_status"],
            pdf_page_start=row["pdf_page_start"],
            pdf_page_end=row["pdf_page_end"],
            section_path=json.loads(row["section_path_json"]),
            text=row["text"],
            year_mentions=json.loads(row["year_mentions_json"]),
            people=json.loads(row["people_json"]),
            extraction_methods=json.loads(row["extraction_methods_json"]),
            score=round(-float(row["raw_score"]), 6),
            matched_terms=terms,
        )
        for rank, row in enumerate(rows, start=1)
    ]
    return SearchResponse(
        query=query,
        query_intent=query_intent,
        query_terms=terms,
        query_years=query_years,
        query_year_range=query_year_range,
        query_people=query_people,
        document_filters=document_ids or [],
        include_out_of_scope=include_out_of_scope,
        hits=hits,
    )
