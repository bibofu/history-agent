from __future__ import annotations

from importlib.resources import files
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from history_agent import __version__
from history_agent.answering.models import AnswerResponse, QuestionRequest
from history_agent.answering.service import answer_question
from history_agent.config import Settings, get_settings
from history_agent.db import Database
from history_agent.errors import ResearchDataError, RetrievalError
from history_agent.research.intersections import (
    PersonIntersectionResponse,
    get_person_intersections,
)
from history_agent.research.organization import (
    OrganizationRelationResponse,
    get_organization_relationships,
)
from history_agent.research.timeline import (
    PersonTimelineResponse,
    TimelineReviewStatus,
    get_person_timeline,
)


def create_app(settings: Settings | None = None) -> FastAPI:
    active_settings = settings or get_settings()
    static_dir = files("history_agent.web").joinpath("static")
    api = FastAPI(
        title="近现代史研究 Agent",
        version=__version__,
        description="Local evidence-grounded RAG for the 1921-1978 corpus.",
    )

    @api.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(str(static_dir.joinpath("index.html")))

    @api.get("/app.css", include_in_schema=False)
    def stylesheet() -> FileResponse:
        return FileResponse(str(static_dir.joinpath("app.css")), media_type="text/css")

    @api.get("/app.js", include_in_schema=False)
    def javascript() -> FileResponse:
        return FileResponse(str(static_dir.joinpath("app.js")), media_type="application/javascript")

    @api.get("/api/health")
    def health() -> dict[str, object]:
        indexes = {
            "keyword": active_settings.keyword_index_path.is_file(),
            "vector": active_settings.vector_index_path.is_dir(),
        }
        return {
            "status": "ok" if all(indexes.values()) else "degraded",
            "version": __version__,
            "indexes": indexes,
            "llm_enabled": active_settings.llm_enabled,
            "llm_provider": active_settings.llm_provider,
            "llm_model": active_settings.llm_model,
            "llm_thinking": active_settings.llm_thinking,
            "research_range": [
                active_settings.research_start.year,
                active_settings.research_end.year,
            ],
        }

    @api.post("/api/questions", response_model=AnswerResponse)
    def question(request: QuestionRequest) -> AnswerResponse:
        try:
            return answer_question(active_settings, request)
        except RetrievalError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @api.get("/api/people/{person_id}/timeline", response_model=PersonTimelineResponse)
    def person_timeline(
        person_id: str,
        start_year: Annotated[int | None, Query(ge=1, le=9999)] = None,
        end_year: Annotated[int | None, Query(ge=1, le=9999)] = None,
        event_type: Annotated[list[str] | None, Query()] = None,
        review_status: Annotated[list[TimelineReviewStatus] | None, Query()] = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> PersonTimelineResponse:
        lower = active_settings.research_start.year
        upper = active_settings.research_end.year
        if start_year is not None and not lower <= start_year <= upper:
            raise HTTPException(
                status_code=422,
                detail=f"start_year must be within the research range {lower}-{upper}",
            )
        if end_year is not None and not lower <= end_year <= upper:
            raise HTTPException(
                status_code=422,
                detail=f"end_year must be within the research range {lower}-{upper}",
            )
        try:
            return get_person_timeline(
                Database(active_settings.database_path),
                person_id=person_id,
                start_year=start_year,
                end_year=end_year,
                event_types=event_type,
                review_statuses=review_status,
                limit=limit,
                offset=offset,
            )
        except ResearchDataError as exc:
            status_code = 404 if str(exc).startswith("unknown person_id") else 400
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    @api.get(
        "/api/people/{person_id}/intersections/{other_person_id}",
        response_model=PersonIntersectionResponse,
    )
    def person_intersections(
        person_id: str,
        other_person_id: str,
        start_year: Annotated[int | None, Query(ge=1, le=9999)] = None,
        end_year: Annotated[int | None, Query(ge=1, le=9999)] = None,
        event_type: Annotated[list[str] | None, Query()] = None,
        review_status: Annotated[list[TimelineReviewStatus] | None, Query()] = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> PersonIntersectionResponse:
        lower, upper = active_settings.research_start.year, active_settings.research_end.year
        for value in (start_year, end_year):
            if value is not None and not lower <= value <= upper:
                raise HTTPException(422, f"year must be within the research range {lower}-{upper}")
        try:
            return get_person_intersections(
                Database(active_settings.database_path),
                person_id=person_id,
                other_person_id=other_person_id,
                start_year=start_year if start_year is not None else lower,
                end_year=end_year if end_year is not None else upper,
                event_types=event_type,
                review_statuses=review_status,
                limit=limit,
                offset=offset,
            )
        except ResearchDataError as exc:
            status_code = 404 if str(exc).startswith("unknown person_id") else 400
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    @api.get(
        "/api/people/{person_id}/relationships",
        response_model=OrganizationRelationResponse,
    )
    def person_relationships(
        person_id: str,
        at: Annotated[str | None, Query()] = None,
        relation_type: Annotated[list[str] | None, Query()] = None,
        review_status: Annotated[list[str] | None, Query()] = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> OrganizationRelationResponse:
        if at is not None:
            try:
                year = int(at[:4])
            except (TypeError, ValueError) as exc:
                raise HTTPException(422, "at must begin with a four-digit year") from exc
            lower, upper = active_settings.research_start.year, active_settings.research_end.year
            if not lower <= year <= upper:
                raise HTTPException(422, f"at must be within the research range {lower}-{upper}")
        try:
            return get_organization_relationships(
                Database(active_settings.database_path),
                person_id=person_id,
                at=at,
                relation_types=relation_type,
                review_statuses=review_status,
                limit=limit,
                offset=offset,
            )
        except ResearchDataError as exc:
            status_code = 404 if str(exc).startswith("unknown person_id") else 400
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    return api


app = create_app()
