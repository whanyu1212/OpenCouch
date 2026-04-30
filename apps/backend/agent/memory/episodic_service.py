"""Episodic memory service helpers."""

from __future__ import annotations

import logging
from datetime import datetime
from uuid import uuid4

from agent.memory.hashing import iso_now as _iso_now
from agent.memory.models import SessionArc, StoredSessionArc
from agent.memory.store import MemoryStore

logger = logging.getLogger(__name__)


def prepare_session_summary_metadata(
    *,
    started_at: str,
    ended_at: str,
    transcript: list[dict[str, object]],
) -> tuple[int, int]:
    """Return duration seconds and user-turn count for a session summary.

    Args:
        started_at: Session start timestamp.
        ended_at: Session end timestamp.
        transcript: Full session transcript.

    Returns:
        tuple[int, int]: Duration seconds and user-turn count.
    """

    duration_seconds = 0
    try:
        start_dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
        duration_seconds = max(0, int((end_dt - start_dt).total_seconds()))
    except (ValueError, AttributeError):
        logger.warning(
            "run_summarize_session: could not parse started_at/ended_at; "
            "duration will be 0. started_at=%r ended_at=%r",
            started_at,
            ended_at,
        )

    user_turn_count = sum(1 for turn in transcript if turn.get("role") == "user")
    return duration_seconds, user_turn_count


def session_arc_to_stored(
    arc: SessionArc,
    *,
    owner_id: str,
    crisis_level_max: int = 0,
) -> StoredSessionArc:
    """Convert an LLM-produced SessionArc to a stored shape."""

    now = _iso_now()
    return StoredSessionArc(
        **arc.model_dump(),
        id=str(uuid4()),
        owner_id=owner_id,
        created_at=now,
        last_referenced_at=now,
        user_visible=True,
        write_timing="session_end",
        write_reason="session-end episodic summary written from completed session transcript",
        policy_version="phase5_v1",
        crisis_level_max=crisis_level_max,  # type: ignore[arg-type]
    )


async def write_session_arc(
    store: MemoryStore,
    *,
    owner_id: str,
    stored_arc: StoredSessionArc,
    embedding: list[float] | None = None,
    embedding_model: str | None = None,
) -> None:
    """Persist a stored session arc to the episodic namespace."""

    namespace = (owner_id, "episodic")
    await store.aput(
        namespace,
        key=stored_arc.id,
        value=stored_arc.model_dump(mode="json"),
        embedding=embedding,
        embedding_model=embedding_model,
    )
