from __future__ import annotations

from importlib.resources import files

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from history_agent import __version__
from history_agent.answering.models import AnswerResponse, QuestionRequest
from history_agent.answering.service import answer_question
from history_agent.config import Settings, get_settings
from history_agent.errors import RetrievalError


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
        return FileResponse(
            str(static_dir.joinpath("app.js")), media_type="application/javascript"
        )

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

    return api


app = create_app()
