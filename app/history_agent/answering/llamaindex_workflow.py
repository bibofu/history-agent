"""Serializable, asynchronous Agentic RAG retrieval workflow."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Literal

from llama_index.core.callbacks import CallbackManager
from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step
from pydantic import BaseModel, ConfigDict

from history_agent.answering.models import QueryPlan, QuestionRequest
from history_agent.answering.retrieval_reflection import (
    EvidenceAssessment,
    ReflectionResult,
)
from history_agent.answering.runtime import LLMRuntime, RequestBudget
from history_agent.config import Settings
from history_agent.errors import RetrievalError
from history_agent.retrieval.llamaindex import SearchBackend, aretrieve_with_llamaindex
from history_agent.retrieval.models import RetrievalPlan, SearchResponse

AssessRetrieval = Callable[..., ReflectionResult]
MergeRetrieval = Callable[..., SearchResponse]
ReflectionStatus = Literal["disabled", "skipped", "sufficient", "retried", "fallback"]
STATE_KEY = "retrieval_state"
INITIAL_RETRIEVAL_KEY = "initial_retrieval"


class WorkflowRetrievalResult(BaseModel):
    """Serializable result returned by the LlamaIndex Workflow."""

    model_config = ConfigDict(frozen=True)

    retrieval: SearchResponse
    reflection_status: ReflectionStatus
    retrieval_rounds: int
    missing_aspects: tuple[str, ...]
    reflection_usage: dict[str, int] | None
    reflection_error_code: str | None


class RetrievalStartEvent(StartEvent):
    request: QuestionRequest
    plan: QueryPlan | None = None
    primary_query: str
    additional_queries: list[str]
    retrieval_plan: RetrievalPlan | None = None
    retrieval_limit: int


class InitialEvidenceEvent(Event):
    retrieval: SearchResponse


class RetryEvidenceEvent(Event):
    followup_queries: list[str]
    assessment: EvidenceAssessment | None = None
    usage: dict[str, int] | None = None
    error_code: str | None = None


class RetrievalWorkflowState(BaseModel):
    """Serializable per-run state persisted through LlamaIndex Context."""

    request: QuestionRequest
    plan: QueryPlan | None = None
    retrieval_plan: RetrievalPlan | None = None
    retrieval_limit: int


def _search_kwargs(settings: Settings, retrieval_plan: RetrievalPlan | None) -> dict[str, object]:
    return {
        "keyword_index_path": settings.keyword_index_path,
        "vector_index_path": settings.vector_index_path,
        "model_cache_dir": settings.model_cache_dir / "fastembed",
        "aliases_path": settings.person_aliases_path,
        "plan": retrieval_plan,
    }


class HistoryRetrievalWorkflow(Workflow):
    """Bounded retrieve-assess-retrieve workflow with serializable events."""

    def __init__(
        self,
        *,
        settings: Settings,
        search_backend: SearchBackend,
        assess_retrieval: AssessRetrieval,
        merge_retrieval: MergeRetrieval,
        runtime: LLMRuntime | None,
        budget: RequestBudget | None,
        max_chunks: int,
        callback_manager: CallbackManager | None = None,
    ) -> None:
        super().__init__(
            timeout=settings.request_timeout_seconds,
            workflow_name="history-agent-retrieval",
        )
        self._settings = settings
        self._search_backend = search_backend
        self._assess_retrieval = assess_retrieval
        self._merge_retrieval = merge_retrieval
        self._llm_runtime = runtime
        self._budget = budget
        self._max_chunks = max_chunks
        self._callback_manager = callback_manager or CallbackManager()

    @step
    async def retrieve(self, ctx: Context, ev: RetrievalStartEvent) -> InitialEvidenceEvent:
        state = RetrievalWorkflowState(
            request=ev.request,
            plan=ev.plan,
            retrieval_plan=ev.retrieval_plan,
            retrieval_limit=ev.retrieval_limit,
        )
        await ctx.store.set(STATE_KEY, state)
        retrieval = await aretrieve_with_llamaindex(
            search_backend=self._search_backend,
            query=ev.primary_query,
            additional_queries=ev.additional_queries,
            top_k=ev.retrieval_limit,
            search_kwargs=_search_kwargs(self._settings, ev.retrieval_plan),
            callback_manager=self._callback_manager,
        )
        retrieval = retrieval.model_copy(update={"query": ev.request.question})
        await ctx.store.set(INITIAL_RETRIEVAL_KEY, retrieval)
        return InitialEvidenceEvent(retrieval=retrieval)

    @step
    async def assess(
        self, ctx: Context, ev: InitialEvidenceEvent
    ) -> RetryEvidenceEvent | StopEvent:
        state = RetrievalWorkflowState.model_validate(await ctx.store.get(STATE_KEY))
        assessment = await asyncio.to_thread(
            self._assess_retrieval,
            self._settings,
            state.request,
            state.plan,
            ev.retrieval,
            self._llm_runtime,
            self._budget,
        )
        if assessment.status != "retry":
            return StopEvent(
                result=WorkflowRetrievalResult(
                    retrieval=ev.retrieval,
                    reflection_status=assessment.status,
                    retrieval_rounds=1,
                    missing_aspects=(
                        tuple(assessment.assessment.missing_aspects)
                        if assessment.assessment
                        else ()
                    ),
                    reflection_usage=assessment.usage,
                    reflection_error_code=assessment.error_code,
                )
            )
        return RetryEvidenceEvent(
            followup_queries=list(assessment.followup_queries),
            assessment=assessment.assessment,
            usage=assessment.usage,
            error_code=assessment.error_code,
        )

    @step
    async def retry(self, ctx: Context, ev: RetryEvidenceEvent) -> StopEvent:
        state = RetrievalWorkflowState.model_validate(await ctx.store.get(STATE_KEY))
        initial = SearchResponse.model_validate(await ctx.store.get(INITIAL_RETRIEVAL_KEY))
        try:
            retry_limit = min(12, max(6, len(ev.followup_queries) * 4))
            retry = await aretrieve_with_llamaindex(
                search_backend=self._search_backend,
                query=ev.followup_queries[0],
                additional_queries=ev.followup_queries[1:],
                top_k=retry_limit,
                search_kwargs=_search_kwargs(self._settings, state.retrieval_plan),
                callback_manager=self._callback_manager,
            )
            retrieval = self._merge_retrieval(
                initial,
                retry,
                query=state.request.question,
                limit=min(
                    self._max_chunks,
                    max(state.retrieval_limit, state.request.top_k + retry_limit),
                ),
            ).model_copy(update={"rag_framework": "llamaindex"})
            status: Literal["retried", "fallback"] = "retried"
            rounds = 2
            error_code = ev.error_code
        except RetrievalError:
            retrieval = initial
            status = "fallback"
            rounds = 1
            error_code = "retry_retrieval_failed"
        return StopEvent(
            result=WorkflowRetrievalResult(
                retrieval=retrieval,
                reflection_status=status,
                retrieval_rounds=rounds,
                missing_aspects=(tuple(ev.assessment.missing_aspects) if ev.assessment else ()),
                reflection_usage=ev.usage,
                reflection_error_code=error_code,
            )
        )


async def run_retrieval_workflow(
    *,
    settings: Settings,
    request: QuestionRequest,
    plan: QueryPlan | None,
    primary_query: str,
    additional_queries: list[str],
    retrieval_plan: RetrievalPlan | None,
    retrieval_limit: int,
    search_backend: SearchBackend,
    assess_retrieval: AssessRetrieval,
    merge_retrieval: MergeRetrieval,
    runtime: LLMRuntime | None,
    budget: RequestBudget | None,
    max_chunks: int,
    callback_manager: CallbackManager | None = None,
) -> WorkflowRetrievalResult:
    workflow = HistoryRetrievalWorkflow(
        settings=settings,
        search_backend=search_backend,
        assess_retrieval=assess_retrieval,
        merge_retrieval=merge_retrieval,
        runtime=runtime,
        budget=budget,
        max_chunks=max_chunks,
        callback_manager=callback_manager,
    )
    ctx = Context(workflow)
    result = await workflow.run(
        ctx=ctx,
        start_event=RetrievalStartEvent(
            request=request,
            plan=plan,
            primary_query=primary_query,
            additional_queries=additional_queries,
            retrieval_plan=retrieval_plan,
            retrieval_limit=retrieval_limit,
        ),
    )
    if not isinstance(result, WorkflowRetrievalResult):
        raise TypeError("LlamaIndex retrieval workflow returned an invalid result")
    return result
