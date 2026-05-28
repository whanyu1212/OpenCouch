"""Shared API helpers for explicit session feedback."""

from __future__ import annotations

from agent.feedback.models import FeedbackLabel, FeedbackModality
from agent.runtime import PersistentAgentRuntime


async def record_runtime_feedback(
    *,
    runtime: PersistentAgentRuntime,
    thread_id: str,
    feedback: FeedbackLabel,
    modality: FeedbackModality,
) -> bool:
    """Record one API-originated session feedback label."""

    record = await runtime.record_session_feedback(
        thread_id,
        label=feedback,
        source="api_end",
        modality=modality,
    )
    return record is not None


__all__ = ["record_runtime_feedback"]
