"""Agentic retrieval loop implemented as a LlamaIndex Workflow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step

from history_agent.answering.models import QueryPlan, QuestionRequest
from history_agent.answering.query_understanding import QueryExecution
from history_agent.answering.retrieval_reflection import ReflectionResult
from history_agent.answering.runtime import LLMRuntime, RequestBudget
from history_agent.config import Settings
from history_agent.errors import RetrievalError
from history_agent.retrieval.llamaindex import SearchBackend, retrieve_with_llamaindex
from history_agent.retrieval.models import SearchResponse

AssessRetrieval = Callable[..., ReflectionResult]
MergeRetrieval = Callable[..., SearchResponse]


@dataclass(frozen=True)
class WorkflowRetrievalResult:
    retrieval: SearchResponse
    reflection_status: Literal[
        "disabled", "skipped", "sufficient", "retried", "fallback"
    ]
    retrieval_rounds: int
    missing_aspects: tuple[str, ...]
    reflection_usage: dict[str, int] | None
    reflection_error_code: str | None


class RetrievalStartEvent(StartEvent):
    settings: Any
    request: Any
    plan: Any
    execution: Any
    retrieval_limit: int
    search_backend: Any
    assess_retrieval: Any
    merge_retrieval: Any
    runtime: Any = None
    budget: Any = None
    max_chunks: int


class InitialEvidenceEvent(Event):
    settings: Any
    request: Any
    plan: Any
    execution: Any
    retrieval_limit: int
    retrieval: Any
    search_backend: Any
    assess_retrieval: Any
    merge_retrieval: Any
    runtime: Any = None
    budget: Any = None
    max_chunks: int


class RetryEvidenceEvent(Event):
    initial: Any
    reflection: Any
    settings: Any
    request: Any
    execution: Any
    retrieval_limit: int
    search_backend: Any
    merge_retrieval: Any
    max_chunks: int


def _search_kwargs(settings: Settings, execution: QueryExecution) -> dict[str, Any]:
    return {
        "keyword_index_path": settings.keyword_index_path,
        "vector_index_path": settings.vector_index_path,
        "model_cache_dir": settings.model_cache_dir / "fastembed",
        "aliases_path": settings.person_aliases_path,
        "plan": execution.retrieval_plan,
    }


class HistoryRetrievalWorkflow(Workflow):
    """Bounded retrieve-assess-retrieve workflow with typed LlamaIndex events."""

    @step
    async def retrieve(self, ev: RetrievalStartEvent) -> InitialEvidenceEvent:
        settings: Settings = ev.settings
        execution: QueryExecution = ev.execution
        retrieval = retrieve_with_llamaindex(
            search_backend=ev.search_backend,
            query=execution.primary_query,
            additional_queries=list(execution.additional_queries),
            top_k=ev.retrieval_limit,
            search_kwargs=_search_kwargs(settings, execution),
        ).model_copy(update={"query": ev.request.question})
        return InitialEvidenceEvent(
            settings=settings,
            request=ev.request,
            plan=ev.plan,
            execution=execution,
            retrieval_limit=ev.retrieval_limit,
            retrieval=retrieval,
            search_backend=ev.search_backend,
            assess_retrieval=ev.assess_retrieval,
            merge_retrieval=ev.merge_retrieval,
            runtime=ev.runtime,
            budget=ev.budget,
            max_chunks=ev.max_chunks,
        )

    @step
    async def assess(self, ev: InitialEvidenceEvent) -> RetryEvidenceEvent | StopEvent:
        assessment: ReflectionResult = ev.assess_retrieval(
            ev.settings,
            ev.request,
            ev.plan,
            ev.retrieval,
            ev.runtime,
            ev.budget,
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
            initial=ev.retrieval,
            reflection=assessment,
            settings=ev.settings,
            request=ev.request,
            execution=ev.execution,
            retrieval_limit=ev.retrieval_limit,
            search_backend=ev.search_backend,
            merge_retrieval=ev.merge_retrieval,
            max_chunks=ev.max_chunks,
        )

    @step
    async def retry(self, ev: RetryEvidenceEvent) -> StopEvent:
        reflection: ReflectionResult = ev.reflection
        queries = list(reflection.followup_queries)
        try:
            retry_limit = min(12, max(6, len(queries) * 4))
            retry = retrieve_with_llamaindex(
                search_backend=ev.search_backend,
                query=queries[0],
                additional_queries=queries[1:],
                top_k=retry_limit,
                search_kwargs=_search_kwargs(ev.settings, ev.execution),
            )
            retrieval = ev.merge_retrieval(
                ev.initial,
                retry,
                query=ev.request.question,
                limit=min(
                    ev.max_chunks,
                    max(ev.retrieval_limit, ev.request.top_k + retry_limit),
                ),
            ).model_copy(update={"rag_framework": "llamaindex"})
            status: Literal["retried", "fallback"] = "retried"
            rounds = 2
            error_code = reflection.error_code
        except RetrievalError:
            retrieval = ev.initial
            status = "fallback"
            rounds = 1
            error_code = "retry_retrieval_failed"
        return StopEvent(
            result=WorkflowRetrievalResult(
                retrieval=retrieval,
                reflection_status=status,
                retrieval_rounds=rounds,
                missing_aspects=(
                    tuple(reflection.assessment.missing_aspects)
                    if reflection.assessment
                    else ()
                ),
                reflection_usage=reflection.usage,
                reflection_error_code=error_code,
            )
        )


async def run_retrieval_workflow(
    *,
    settings: Settings,
    request: QuestionRequest,
    plan: QueryPlan | None,
    execution: QueryExecution,
    retrieval_limit: int,
    search_backend: SearchBackend,
    assess_retrieval: AssessRetrieval,
    merge_retrieval: MergeRetrieval,
    runtime: LLMRuntime | None,
    budget: RequestBudget | None,
    max_chunks: int,
) -> WorkflowRetrievalResult:
    workflow = HistoryRetrievalWorkflow(timeout=settings.request_timeout_seconds)
    result = await workflow.run(
        start_event=RetrievalStartEvent(
            settings=settings,
            request=request,
            plan=plan,
            execution=execution,
            retrieval_limit=retrieval_limit,
            search_backend=search_backend,
            assess_retrieval=assess_retrieval,
            merge_retrieval=merge_retrieval,
            runtime=runtime,
            budget=budget,
            max_chunks=max_chunks,
        )
    )
    if not isinstance(result, WorkflowRetrievalResult):
        raise TypeError("LlamaIndex retrieval workflow returned an invalid result")
    return result
