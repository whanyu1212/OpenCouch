"""Thread management endpoints.

GET  /api/threads — list persisted threads
GET  /api/threads/{id}/state — raw state dump
GET  /api/threads/{id}/history — transcript messages
POST /api/threads/{id}/end — end session and summarize
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from agent.persistence import PersistentAgentRuntime
from api.dependencies import get_llm_client, get_runtime
from api.models import (
    MessageResponse,
    SessionArcResponse,
    ThreadSummaryResponse,
)
from services.llm.base import BaseLLMClient

router = APIRouter(prefix="/threads", tags=["threads"])


@router.get("", response_model=list[ThreadSummaryResponse])
async def list_threads(
    limit: int = 20,
    runtime: PersistentAgentRuntime = Depends(get_runtime),
) -> list[ThreadSummaryResponse]:
    """List persisted threads, most recent first."""

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
    """

    state = await runtime.get_state(thread_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"No state for thread {thread_id}")

    # The state may contain pydantic models (e.g., CrisisAssessment)
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
    """

    messages = await runtime.get_history(thread_id)
    return [
        MessageResponse(
            role=m.role.value,
            content=m.content,
            mode=m.mode,
        )
        for m in messages
    ]


@router.post("/{thread_id}/end")
async def end_session(
    thread_id: str,
    runtime: PersistentAgentRuntime = Depends(get_runtime),
    llm_client: BaseLLMClient | None = Depends(get_llm_client),
) -> SessionArcResponse | dict:
    """End the session for a thread and produce an episodic summary.

    Triggers the session summarizer which reads the full transcript
    and writes a ``StoredSessionArc`` to episodic memory. Returns
    the arc data on success, or a message explaining why no summary
    was produced (too few turns, no LLM client, incognito mode).
    """

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
