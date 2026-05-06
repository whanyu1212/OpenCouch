"""Thread management endpoints.

GET  /api/threads: list persisted threads.
GET  /api/threads/{id}/state: raw state dump.
GET  /api/threads/{id}/history: transcript messages.
POST /api/threads/{id}/end: end session and summarize.
"""

from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException

from agent.persistence import PersistentAgentRuntime
from api.dependencies import get_llm_client, get_runtime
from api.models import (
    EndSessionRequest,
    MessageResponse,
    SessionArcResponse,
    ThreadSessionStatusResponse,
    ThreadSummaryResponse,
)
from services.base import BaseLLMClient

router = APIRouter(prefix="/threads", tags=["threads"])


@router.get("", response_model=list[ThreadSummaryResponse])
async def list_threads(
    limit: int = 20,
    runtime: PersistentAgentRuntime = Depends(get_runtime),
) -> list[ThreadSummaryResponse]:
    """List persisted threads, most recent first.

    Args:
        limit: Maximum number of threads to return.
        runtime: Shared persistent agent runtime.

    Returns:
        Persisted thread summaries.
    """

    summaries = await runtime.list_threads(limit=limit)
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
    runtime: PersistentAgentRuntime = Depends(get_runtime),
) -> dict:
    """Return the raw graph state for a thread.

    This is the API equivalent of the CLI's ``/debug state``
    command. Returns the full state dict including diagnostics,
    routing, memory, crisis assessment, and transcript.

    Returns 404 if the thread has no persisted state.

    Args:
        thread_id: Thread identifier to inspect.
        runtime: Shared persistent agent runtime.

    Returns:
        JSON-serializable graph state.
    """

    state = await runtime.get_state(thread_id)
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
    runtime: PersistentAgentRuntime = Depends(get_runtime),
) -> list[MessageResponse]:
    """Return the transcript for a thread.

    Each entry includes role (user/assistant), content, and an
    optional mode annotation for assistant turns (which therapeutic
    mode shaped that reply).

    Returns an empty list if the thread has no history.

    Args:
        thread_id: Thread identifier to inspect.
        runtime: Shared persistent agent runtime.

    Returns:
        Transcript messages for the thread.
    """

    messages = await runtime.get_history(thread_id)
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
    runtime: PersistentAgentRuntime = Depends(get_runtime),
) -> ThreadSessionStatusResponse:
    """Return whether this thread currently has an active session.

    Args:
        thread_id: Thread identifier to inspect.
        runtime: Shared persistent agent runtime.

    Returns:
        Active-session status for the thread.
    """

    return ThreadSessionStatusResponse(
        has_active_session=await runtime.has_active_session(thread_id)
    )


@router.post("/{thread_id}/end")
async def end_session(
    thread_id: str,
    body: EndSessionRequest = Body(default_factory=EndSessionRequest),
    runtime: PersistentAgentRuntime = Depends(get_runtime),
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
        runtime: Shared persistent agent runtime.
        llm_client: Optional control-plane LLM client.

    Returns:
        Session summary response, or an explanatory detail payload when
        no summary was produced.
    """

    # Optional best-effort feedback capture. Runtime outages do not block summary.
    if body.feedback is not None:
        await runtime.record_session_feedback(
            thread_id,
            label=body.feedback,
            source="api_end",
        )

    # Existing summarization flow.
    arc = await runtime.end_session(thread_id, llm_client=llm_client)

    if arc is None:
        return {
            "summary": None,
            "detail": "No summary produced (session too short, no LLM, or incognito mode).",
        }

    return SessionArcResponse(
        summary=arc.summary,
        themes=list(arc.primary_themes),
        mood_opened=arc.mood_arc.opened,
        mood_closed=arc.mood_arc.closed,
        turn_count=arc.turn_count,
        open_loops=list(arc.open_loops),
        resolved_threads=list(arc.resolved_threads),
    )
