"""Thread management endpoints.

GET  /api/threads: list persisted threads.
GET  /api/threads/{id}/state: raw state dump.
GET  /api/threads/{id}/history: transcript messages.
POST /api/threads/{id}/end: end session and summarize.
"""

from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from api.dependencies import get_llm_client, get_runtime_selection
from api.models import (
    ApiMemoryMode,
    EndSessionRequest,
    MessageResponse,
    SessionArcResponse,
    ThreadSessionStatusResponse,
    ThreadSummaryResponse,
)
from api.session_end import end_runtime_session
from llm.base import BaseLLMClient

router = APIRouter(prefix="/threads", tags=["threads"])


@router.get("", response_model=list[ThreadSummaryResponse])
async def list_threads(
    limit: int = 20,
    memory_mode: ApiMemoryMode | None = Query(default=None),
) -> list[ThreadSummaryResponse]:
    """List persisted threads, most recent first.

    Args:
        limit: Maximum number of threads to return.
        memory_mode: Optional runtime selector for thread lookup.

    Returns:
        Persisted thread summaries.
    """

    selection = get_runtime_selection(memory_mode)
    summaries = await selection.runtime.list_threads(limit=limit)
    return [
        ThreadSummaryResponse(
            thread_id=s.thread_id,
            turn_count=s.turn_count,
            message_count=s.message_count,
            has_context=s.has_context,
        )
        for s in summaries
    ]


@router.get("/{thread_id}/state")
async def get_thread_state(
    thread_id: str,
    memory_mode: ApiMemoryMode | None = Query(default=None),
) -> dict:
    """Return the raw persisted runtime state for a thread.

    This is the API equivalent of the CLI's ``/debug state``
    command. Returns the full state dict including diagnostics,
    routing, memory, crisis assessment, and transcript.

    Returns 404 if the thread has no persisted state.

    Args:
        thread_id: Thread identifier to inspect.
        memory_mode: Optional runtime selector for thread lookup.

    Returns:
        JSON-serializable persisted runtime state.
    """

    selection = get_runtime_selection(memory_mode)
    state = await selection.runtime.get_state(thread_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"No state for thread {thread_id}")

    # The state may contain pydantic models (for example, CrisisAssessment)
    # that don't serialize to JSON directly. Convert via str fallback
    # for any non-serializable values — same approach as the CLI's
    # /debug state which uses json.dumps(state, default=str).
    import json

    return json.loads(json.dumps(state, default=str))


@router.get("/{thread_id}/history", response_model=list[MessageResponse])
async def get_thread_history(
    thread_id: str,
    memory_mode: ApiMemoryMode | None = Query(default=None),
) -> list[MessageResponse]:
    """Return the transcript for a thread.

    Each entry includes role (user/assistant), content, and an
    optional mode annotation for assistant turns (which therapeutic
    mode shaped that reply).

    Returns an empty list if the thread has no history.

    Args:
        thread_id: Thread identifier to inspect.
        memory_mode: Optional runtime selector for thread lookup.

    Returns:
        Transcript messages for the thread.
    """

    selection = get_runtime_selection(memory_mode)
    messages = await selection.runtime.get_history(thread_id)
    return [
        MessageResponse(
            role=m.role.value,
            content=m.content,
            response_style=m.response_style,
        )
        for m in messages
    ]


@router.get("/{thread_id}/session-status", response_model=ThreadSessionStatusResponse)
async def get_thread_session_status(
    thread_id: str,
    memory_mode: ApiMemoryMode | None = Query(default=None),
) -> ThreadSessionStatusResponse:
    """Return whether this thread currently has an active session.

    Args:
        thread_id: Thread identifier to inspect.
        memory_mode: Optional runtime selector for thread lookup.

    Returns:
        Active-session status for the thread.
    """

    selection = get_runtime_selection(memory_mode)
    return ThreadSessionStatusResponse(
        has_active_session=await selection.runtime.has_active_session(thread_id)
    )


@router.post("/{thread_id}/end")
async def end_session(
    thread_id: str,
    body: EndSessionRequest = Body(default_factory=EndSessionRequest),
    llm_client: BaseLLMClient | None = Depends(get_llm_client),
) -> SessionArcResponse | dict:
    """End the session for a thread and produce an episodic summary.

    Triggers the session summarizer which reads the full transcript
    and writes a ``StoredSessionArc`` to episodic memory. Returns
    the arc data on success, or a message explaining why no summary
    was produced (too few turns, no LLM client, incognito mode).

    Optional ``body.feedback`` accepts a feedback label
    (``"positive"``, ``"negative"``, ``"skip"``). When provided, it
    is written to the session-feedback store via
    :meth:`PersistentAgentRuntime.record_session_feedback` BEFORE
    summarization runs, with ``source="api_end"``. Clients POSTing
    with no body or ``{"feedback": null}`` skip the feedback step
    and summarization runs unchanged. Feedback write failures are
    best-effort and never block summarization.

    The response shape is unchanged from prior versions: feedback
    write status is not surfaced. Feedback persistence is orthogonal
    to summarization.

    Args:
        thread_id: Thread identifier to end.
        body: Optional session-ending request body.
        llm_client: Optional control-plane LLM client.

    Returns:
        Session summary response, or an explanatory detail payload when
        no summary was produced.
    """

    selection = get_runtime_selection(body.memory_mode)
    result = await end_runtime_session(
        runtime=selection.runtime,
        thread_id=thread_id,
        feedback=body.feedback,
        llm_client=llm_client,
        memory_mode=selection.memory_mode,
    )

    if not result.finalized:
        return {
            "summary": None,
            "detail": result.detail,
        }

    assert result.summary is not None
    assert result.mood_opened is not None
    assert result.mood_closed is not None
    assert result.turn_count is not None
    return SessionArcResponse(
        summary=result.summary,
        themes=result.themes,
        mood_opened=result.mood_opened,
        mood_closed=result.mood_closed,
        turn_count=result.turn_count,
        open_loops=result.open_loops,
        resolved_threads=result.resolved_threads,
    )
