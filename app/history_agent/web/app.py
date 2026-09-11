from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager
from importlib.resources import files
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from history_agent import __version__
from history_agent.answering.context import ConversationStore, build_prompt_context
from history_agent.answering.models import AnswerResponse, QuestionRequest
from history_agent.answering.runtime import LLMRuntime, RequestBudget
from history_agent.answering.service import answer_question_async
from history_agent.answering.streaming import AnswerStreamEvent, stream_answer_question
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
from history_agent.web.readiness import readiness_snapshot

LOGGER = logging.getLogger(__name__)
SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{16,64}$")


def create_app(settings: Settings | None = None) -> FastAPI:
    active_settings = settings or get_settings()
    static_dir = files("history_agent.web").joinpath("static")
    conversations = ConversationStore(active_settings.conversation_database_path)

    def with_server_context(request: QuestionRequest) -> QuestionRequest:
        if request.session_id is None:
            return request
        try:
            persisted = conversations.messages(request.session_id)
        except Exception:
            LOGGER.exception("conversation history could not be loaded")
            persisted = []
        history = build_prompt_context(
            persisted,
            max_messages=active_settings.context_max_messages,
            max_chars=active_settings.context_max_chars,
        )
        return request.model_copy(update={"history": history})

    def save_exchange(session_id: str, question: str, answer: str) -> None:
        try:
            conversations.append_exchange(session_id, question, answer)
        except Exception:
            # Context persistence is auxiliary and must never discard a completed answer.
            LOGGER.exception("conversation exchange could not be persisted")

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        runtime = LLMRuntime(active_settings.llm_max_concurrency)
        application.state.llm_runtime = runtime
        try:
            yield
        finally:
            await runtime.aclose()

    api = FastAPI(
        title="近现代史研究 Agent",
        version=__version__,
        description="Local evidence-grounded RAG for the 1921-1978 corpus.",
        lifespan=lifespan,
    )
    api.mount("/assets", StaticFiles(directory=str(static_dir)), name="assets")

    @api.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(
            str(static_dir.joinpath("index.html")), headers={"Cache-Control": "no-store"}
        )

    @api.get("/app.css", include_in_schema=False)
    def stylesheet() -> FileResponse:
        return FileResponse(
            str(static_dir.joinpath("app.css")),
            media_type="text/css",
            headers={"Cache-Control": "no-store"},
        )

    @api.get("/app.js", include_in_schema=False)
    def javascript() -> FileResponse:
        return FileResponse(
            str(static_dir.joinpath("app.js")),
            media_type="application/javascript",
            headers={"Cache-Control": "no-store"},
        )

    @api.get("/api/health")
    def health() -> dict[str, object]:
        indexes = {
            "keyword": active_settings.keyword_index_path.is_file(),
            "vector": active_settings.vector_index_path.is_dir(),
        }
        return {
            "status": "ok",
            "version": __version__,
            "rag_framework": "llamaindex",
            "rag_execution_paths": ["llamaindex", "native"],
            "llamaindex_components": [
                "workflow",
                "query_bundle",
                "retriever",
                "node_postprocessor",
                "llm_adapter",
                "callback_manager",
                "output_parser",
                "prompt_template",
            ],
            "indexes": indexes,
            "llm_enabled": active_settings.llm_enabled,
            "llm_provider": active_settings.llm_provider,
            "llm_model": active_settings.llm_model,
            "llm_thinking": active_settings.llm_thinking,
            "llm_query_planning": active_settings.llm_query_planning,
            "llm_query_planner_model": active_settings.llm_query_planner_model,
            "llm_retrieval_reflection": active_settings.llm_retrieval_reflection,
            "llm_retrieval_reflection_max_rounds": (
                active_settings.llm_retrieval_reflection_max_rounds
            ),
            "research_range": [
                active_settings.research_start.year,
                active_settings.research_end.year,
            ],
        }

    @api.get("/api/ready")
    def ready() -> JSONResponse:
        snapshot = readiness_snapshot(active_settings)
        status_code = 200 if snapshot["status"] == "ready" else 503
        return JSONResponse(snapshot, status_code=status_code)

    @api.post("/api/questions", response_model=AnswerResponse)
    async def question(request: QuestionRequest) -> AnswerResponse:
        try:
            contextual_request = with_server_context(request)
            runtime = getattr(api.state, "llm_runtime", None)
            if runtime is None:
                response = await answer_question_async(active_settings, contextual_request)
            else:
                response = await answer_question_async(
                    active_settings,
                    contextual_request,
                    runtime,
                    RequestBudget.start(active_settings.request_timeout_seconds),
                )
            if request.session_id is not None:
                save_exchange(request.session_id, request.question, response.answer)
            return response
        except RetrievalError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @api.post("/api/questions/stream")
    async def question_stream(request: QuestionRequest) -> StreamingResponse:
        async def events() -> AsyncIterator[str]:
            try:
                contextual_request = with_server_context(request)
                runtime = getattr(api.state, "llm_runtime", None)
                answers = (
                    stream_answer_question(active_settings, contextual_request)
                    if runtime is None
                    else stream_answer_question(
                        active_settings,
                        contextual_request,
                        runtime,
                        RequestBudget.start(active_settings.request_timeout_seconds),
                    )
                )
                async with aclosing(answers) as answers:
                    async for event in answers:
                        if event.event == "done" and request.session_id is not None:
                            save_exchange(
                                request.session_id,
                                request.question,
                                str(event.data["answer"]),
                            )
                        yield event.encode()
            except RetrievalError:
                yield AnswerStreamEvent(
                    "error", {"message": "本地检索暂不可用，请检查索引后重试。"}
                ).encode()
            except Exception:
                yield AnswerStreamEvent(
                    "error", {"message": "问答服务暂时不可用，请重试。"}
                ).encode()

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
        )

    @api.get("/api/sessions/{session_id}")
    def conversation(session_id: str) -> dict[str, object]:
        if SESSION_ID.fullmatch(session_id) is None:
            raise HTTPException(status_code=422, detail="invalid session_id")
        return {
            "session_id": session_id,
            "messages": [item.model_dump() for item in conversations.messages(session_id)],
        }

    @api.delete("/api/sessions/{session_id}")
    def clear_conversation(session_id: str) -> dict[str, object]:
        if SESSION_ID.fullmatch(session_id) is None:
            raise HTTPException(status_code=422, detail="invalid session_id")
        conversations.clear(session_id)
        return {"session_id": session_id, "cleared": True}

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
