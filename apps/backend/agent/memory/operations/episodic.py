"""Episodic memory helpers for session-end arc writes.

Episodic memory stores **one record per session arc** in the
``(owner_id, "episodic")`` namespace. Each arc summarizes what happened
during a single session — themes, mood trajectory, approach used —
and is written exactly once at session end by ``summarize_session_node``.

Module layout:

- :func:`prepare_session_summary_metadata` — derives duration seconds and
  user-turn count from raw timestamps and transcript. Pure helper used by
  the summarization node before it calls the LLM.
- :func:`session_arc_to_stored` — converts an LLM-produced
  :class:`SessionArc` into a :class:`StoredSessionArc` by attaching the
  fields the store layer needs (id, owner, timestamps, write provenance).
- :func:`write_session_arc` — persists a stored arc into the episodic
  namespace using the standard ``(owner_id, "episodic")`` shape.

Episodic writes are append-only on the hot path; reconciliation between
arcs (merging or superseding old session summaries) is not done here.
The retrieval-side helpers that read these arcs live in
:mod:`agent.memory.retrieval.service`.
"""

from __future__ import annotations

import logging
from datetime import datetime
from uuid import uuid4

from agent.memory.hashing import iso_now as _iso_now
from agent.memory.types import SessionArc, StoredSessionArc
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
    """Convert an LLM-produced SessionArc into a stored shape.

    Args:
        arc (SessionArc): LLM-produced session arc with summary and themes.
        owner_id (str): Owner whose episodic namespace receives the arc.
        crisis_level_max (int): Highest crisis level observed during the
            session (0 when no crisis was detected).

    Returns:
        StoredSessionArc: Arc enriched with id, owner, timestamps, and
            write provenance ready for the episodic namespace.
    """

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
    """Persist a stored session arc into the episodic namespace.

    Args:
        store (MemoryStore): Memory store to write into.
        owner_id (str): Owner whose episodic namespace receives the arc.
        stored_arc (StoredSessionArc): Arc to persist; its ``id`` is used
            as the record key.
        embedding (list[float] | None): Optional precomputed summary
            embedding for hybrid retrieval.
        embedding_model (str | None): Optional embedding model identifier.

    Returns:
        None: Persists the arc record under ``(owner_id, "episodic")``.
    """

    namespace = (owner_id, "episodic")
    await store.aput(
        namespace,
        key=stored_arc.id,
        value=stored_arc.model_dump(mode="json"),
        embedding=embedding,
        embedding_model=embedding_model,
    )
