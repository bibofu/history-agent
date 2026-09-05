from __future__ import annotations

import os
import shutil
import uuid
from datetime import UTC, datetime
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from history_agent.errors import IndexBuildError, RetrievalError
from history_agent.processing.chunks import load_person_aliases
from history_agent.processing.models import ChunkRecord
from history_agent.retrieval.keyword import (
    YEAR,
    hard_filter_people,
    infer_query_intent,
    infer_year_range,
    load_chunks,
)
from history_agent.retrieval.models import (
    SearchHit,
    SearchResponse,
    VectorIndexSummary,
)

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
VECTOR_SIZE = 512
COLLECTION_NAME = "history_chunks"
INDEX_VERSION = "qdrant-local-fastembed-v1"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _load_vector_dependencies() -> tuple[Any, Any, Any]:
    try:
        from fastembed import TextEmbedding
        from qdrant_client import QdrantClient, models
    except ImportError as exc:
        raise IndexBuildError(
            "Vector dependencies are missing. Run: uv sync --extra vector --group dev"
        ) from exc
    return TextEmbedding, QdrantClient, models


@lru_cache(maxsize=2)
def _embedding_model(model_cache_dir: Path) -> Any:
    TextEmbedding, _, _ = _load_vector_dependencies()
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    return TextEmbedding(
        model_name=MODEL_NAME,
        cache_dir=str(model_cache_dir),
        threads=max(1, min(8, os.cpu_count() or 1)),
    )


def _embedding_text(record: ChunkRecord) -> str:
    section = " > ".join(record.section_path)
    return f"{record.title}\n{section}\n{record.text}".strip()


def _payload(record: ChunkRecord) -> dict[str, Any]:
    return {
        "chunk_id": record.chunk_id,
        "document_id": record.document_id,
        "title": record.title,
        "filename": record.filename,
        "volume": record.volume,
        "source_type": record.source_type,
        "verification_status": record.verification_status,
        "creators": record.creators,
        "pdf_page_start": record.pdf_page_start,
        "pdf_page_end": record.pdf_page_end,
        "section_path": record.section_path,
        "text": record.text,
        "year_mentions": record.year_mentions,
        "scope_status": record.scope_status,
        "people": record.people,
        "extraction_methods": record.extraction_methods,
    }


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _replace_directory(source: Path, destination: Path) -> None:
    backup = destination.with_name(destination.name + ".backup")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        os.replace(destination, backup)
    try:
        os.replace(source, destination)
    except Exception:
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def build_vector_index(
    *,
    chunks_dir: Path,
    index_path: Path,
    model_cache_dir: Path,
    reports_dir: Path,
    run_id: str,
    batch_size: int = 64,
) -> VectorIndexSummary:
    started_at = utc_now()
    records = load_chunks(chunks_dir)
    if not records:
        raise IndexBuildError("No chunks are available for vector indexing.")
    _, QdrantClient, models = _load_vector_dependencies()
    embedding_model = _embedding_model(model_cache_dir)
    temporary_path = index_path.with_name(index_path.name + ".tmp")
    if temporary_path.exists():
        shutil.rmtree(temporary_path)
    temporary_path.parent.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(temporary_path))
    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            vectors = list(
                embedding_model.embed(
                    [_embedding_text(record) for record in batch],
                    batch_size=batch_size,
                )
            )
            points = [
                models.PointStruct(
                    id=str(uuid.UUID(record.chunk_id)),
                    vector=vector.tolist(),
                    payload=_payload(record),
                )
                for record, vector in zip(batch, vectors, strict=True)
            ]
            client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
        count = client.count(collection_name=COLLECTION_NAME, exact=True).count
        if count != len(records):
            raise IndexBuildError(
                f"Vector index count mismatch: expected {len(records)}, found {count}."
            )
    finally:
        client.close()
    _replace_directory(temporary_path, index_path)
    try:
        fastembed_version = version("fastembed")
        qdrant_version = version("qdrant-client")
    except PackageNotFoundError:
        fastembed_version = qdrant_version = "unknown"
    summary = VectorIndexSummary(
        run_id=run_id,
        started_at=started_at,
        finished_at=utc_now(),
        index_version=f"{INDEX_VERSION}; fastembed={fastembed_version}; qdrant={qdrant_version}",
        model_name=MODEL_NAME,
        vector_size=VECTOR_SIZE,
        documents=len({record.document_id for record in records}),
        chunks=len(records),
        collection_name=COLLECTION_NAME,
        index_path=str(index_path),
        size_bytes=_directory_size(index_path),
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    rendered = summary.model_dump_json(indent=2)
    (reports_dir / f"vector_index_{run_id}.json").write_text(rendered, encoding="utf-8")
    (reports_dir / "vector_index_latest.json").write_text(rendered, encoding="utf-8")
    return summary


def _query_filter(
    *,
    models: Any,
    document_ids: list[str] | None,
    query_people: list[str],
    query_year_range: list[int],
    include_out_of_scope: bool,
) -> Any:
    must: list[Any] = []
    must_not: list[Any] = []
    if document_ids:
        must.append(
            models.FieldCondition(
                key="document_id", match=models.MatchAny(any=document_ids)
            )
        )
    for person in query_people:
        must.append(
            models.FieldCondition(key="people", match=models.MatchValue(value=person))
        )
    if query_year_range:
        must.append(
            models.FieldCondition(
                key="year_mentions",
                range=models.Range(gte=query_year_range[0], lte=query_year_range[1]),
            )
        )
    if not include_out_of_scope:
        must_not.append(
            models.FieldCondition(
                key="scope_status", match=models.MatchValue(value="out_of_scope")
            )
        )
    return models.Filter(must=must or None, must_not=must_not or None)


def search_vector_index(
    *,
    index_path: Path,
    model_cache_dir: Path,
    aliases_path: Path,
    query: str,
    top_k: int = 10,
    document_ids: list[str] | None = None,
    include_out_of_scope: bool = False,
) -> SearchResponse:
    if not index_path.is_dir():
        raise RetrievalError(f"Vector index does not exist: {index_path}")
    aliases = load_person_aliases(aliases_path)
    query_people = sorted(
        person
        for person, names in aliases.items()
        if person in query or any(alias in query for alias in names)
    )
    query_intent = infer_query_intent(query)
    filter_people = hard_filter_people(query_people, query_intent)
    query_years = sorted({int(match.group(1)) for match in YEAR.finditer(query)})
    query_year_range = infer_year_range(query, query_years)
    _, QdrantClient, models = _load_vector_dependencies()
    embedding_model = _embedding_model(model_cache_dir)
    query_vector = next(iter(embedding_model.query_embed(query)))
    # Chronologies often mention nearby years in the body. Fetch a wider candidate
    # pool for exact-year timeline questions, then prefer the year heading in the
    # recovered document structure over incidental year mentions.
    candidate_limit = top_k
    prefer_section_year = bool(query_years and query_intent == "timeline")
    if prefer_section_year:
        candidate_limit = min(500, max(100, top_k * 20))
    client = QdrantClient(path=str(index_path))
    try:
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector.tolist(),
            query_filter=_query_filter(
                models=models,
                document_ids=document_ids,
                query_people=filter_people,
                query_year_range=query_year_range,
                include_out_of_scope=include_out_of_scope,
            ),
            limit=candidate_limit,
            with_payload=True,
        )
    finally:
        client.close()
    points = list(response.points)
    if prefer_section_year:
        section_matches: list[Any] = []
        other_matches: list[Any] = []
        expected = set(query_years)
        for point in points:
            payload = point.payload or {}
            section_path = [str(item) for item in payload.get("section_path", [])]
            section_years = {
                int(match.group(1))
                for section in section_path
                for match in YEAR.finditer(section)
            }
            target = (
                section_matches
                if expected.intersection(section_years)
                else other_matches
            )
            target.append(point)
        points = [*section_matches, *other_matches]
    hits: list[SearchHit] = []
    for rank, point in enumerate(points[:top_k], start=1):
        payload = point.payload or {}
        hits.append(
            SearchHit(
                rank=rank,
                chunk_id=str(payload["chunk_id"]),
                document_id=str(payload["document_id"]),
                title=str(payload["title"]),
                filename=str(payload["filename"]),
                volume=payload.get("volume"),
                source_type=str(payload["source_type"]),
                verification_status=str(payload["verification_status"]),
                pdf_page_start=int(payload["pdf_page_start"]),
                pdf_page_end=int(payload["pdf_page_end"]),
                section_path=list(payload.get("section_path", [])),
                text=str(payload["text"]),
                year_mentions=list(payload.get("year_mentions", [])),
                people=list(payload.get("people", [])),
                extraction_methods=list(payload.get("extraction_methods", [])),
                score=round(float(point.score), 6),
                matched_terms=[],
            )
        )
    return SearchResponse(
        query=query,
        query_intent=query_intent,
        query_terms=[],
        query_years=query_years,
        query_year_range=query_year_range,
        query_people=query_people,
        document_filters=document_ids or [],
        include_out_of_scope=include_out_of_scope,
        hits=hits,
        retrieval_mode="vector",
    )
