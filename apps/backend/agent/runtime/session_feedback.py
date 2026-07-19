"""Runtime policy for writing explicit session feedback records."""

from __future__ import annotations

import logging
from uuid import uuid4

from agent.feedback.models import (
    FeedbackLabel,
    FeedbackModality,
    FeedbackSource,
    SessionFeedbackRecord,
)
from agent.feedback.session_feedback import SessionFeedbackBackend
from agent.memory.hashing import hash_session_id, iso_now
from agent.memory.modes import MemoryMode
from agent.runtime.session import turn_count_from_state
from agent.state import AgentState

logger = logging.getLogger(__name__)


async def record_session_feedback(
    *,
    backend: SessionFeedbackBackend,
    thread_id: str,
    state: AgentState | None,
    memory_mode: MemoryMode,
    label: FeedbackLabel,
    source: FeedbackSource,
    modality: FeedbackModality = "text",
) -> SessionFeedbackRecord | None:
    """Build and persist one trusted end-of-session feedback record.

    Runtime lifecycle code owns this policy because the fields are derived from
    app-owned state and trusted session-ending surfaces, not from model output.
    """

    try:
        turn_count = turn_count_from_state(state)
        if memory_mode == MemoryMode.INCOGNITO:
            user_id: str | None = None
        elif state is not None:
            user_id = state.get("user_id")
        else:
            user_id = None

        record = SessionFeedbackRecord(
            id=str(uuid4()),
            session_id_opaque=hash_session_id(thread_id),
            user_id_or_null=user_id,
            recorded_at=iso_now(),
            label=label,
            turn_count_at_end=turn_count,
            source=source,
            modality=modality,
            schema_version=1,
        )

        await backend.aappend(record)
        return record
    except Exception:
        logger.warning(
            "session feedback write failed for thread %s",
            thread_id,
            exc_info=True,
        )
        return None
