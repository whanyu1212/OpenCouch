"""Shared API session finalization helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.feedback.models import FeedbackLabel, FeedbackModality
from agent.runtime import PersistentAgentRuntime
from api.models import ApiMemoryMode
from llm.base import BaseLLMClient

_GENERIC_NO_SUMMARY_DETAIL = (
    "No summary produced (session too short, no LLM, or incognito mode)."
)
_INCOGNITO_NO_SUMMARY_DETAIL = "Incognito session ended without durable finalization."


@dataclass(frozen=True)
class SessionEndResult:
    """Normalized result for text and voice session-end routes."""

    finalized: bool
    summary: str | None
    detail: str
    themes: list[str] = field(default_factory=list)
    mood_opened: str | None = None
    mood_closed: str | None = None
    turn_count: int | None = None
    open_loops: list[str] = field(default_factory=list)
    resolved_threads: list[str] = field(default_factory=list)


async def end_runtime_session(
    *,
    runtime: PersistentAgentRuntime,
    thread_id: str,
    feedback: FeedbackLabel | None,
    llm_client: BaseLLMClient | None,
    memory_mode: ApiMemoryMode,
    modality: FeedbackModality = "text",
) -> SessionEndResult:
    """Record optional feedback, finalize the runtime session, and normalize output.

    When feedback accompanies the end request, both steps run under one thread
    lock so the recorded ``turn_count_at_end`` describes the same session
    window that gets finalized. Without feedback this is a plain finalize.
    """

    if feedback is not None:
        _, arc = await runtime.end_session_with_feedback(
            thread_id,
            label=feedback,
            source="api_end",
            modality=modality,
            llm_client=llm_client,
        )
    else:
        arc = await runtime.end_session(thread_id, llm_client=llm_client)
    if arc is None:
        detail = (
            _INCOGNITO_NO_SUMMARY_DETAIL
            if memory_mode is ApiMemoryMode.INCOGNITO
            else _GENERIC_NO_SUMMARY_DETAIL
        )
        return SessionEndResult(finalized=False, summary=None, detail=detail)

    return SessionEndResult(
        finalized=True,
        summary=arc.summary,
        detail="Session summary produced.",
        themes=list(arc.primary_themes),
        mood_opened=arc.mood_arc.opened,
        mood_closed=arc.mood_arc.closed,
        turn_count=arc.turn_count,
        open_loops=list(arc.open_loops),
        resolved_threads=list(arc.resolved_threads),
    )
