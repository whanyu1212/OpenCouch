"""Memory side effects for guided therapeutic exercises."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from agent.memory.modes import MemoryMode
from agent.memory.types import EntityRef, SemanticFact
from agent.memory.store import MemoryStore
from agent.state import AgentState, resolve_owner_id

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExerciseCompletionMemoryRequest:
    """Framework-neutral input for an exercise-completion memory write."""

    owner_id: str
    session_id: str
    turn_count: int
    exercise_type: str
    display_name: str


async def _write_exercise_completion_fact(
    *,
    state: AgentState,
    exercise_type: str,
    display_name: str,
    memory_store: MemoryStore | None,
    memory_mode: MemoryMode | None,
) -> None:
    """Write a semantic fact recording that the user completed an exercise.

    This is a deterministic write — no LLM involved. The fact is
    written as a coping_strategy with predicate USES, which the
    retrieval system will surface on future turns when the user's
    context overlaps with coping strategies.

    Skips silently when:
    - memory_store is None (no store configured)
    - memory_mode is INCOGNITO (no persistent writes allowed)
    - any error occurs (logged, never raised)

    Args:
        state: Current runtime state.
        exercise_type: Completed exercise identifier.
        display_name: Human-readable exercise name.
        memory_store: Memory store used for the write, if configured.
        memory_mode: Current memory mode.

    Returns:
        None.
    """

    if memory_store is None or memory_mode == MemoryMode.INCOGNITO:
        return

    owner_id = resolve_owner_id(state)
    session_id = str(state.get("session_id") or owner_id)
    session_progress = state.get("session_progress", {}) or {}
    raw_turn_count = (
        session_progress.get("turn_count", 0)
        if isinstance(session_progress, dict)
        else 0
    )
    turn_count = raw_turn_count if isinstance(raw_turn_count, int) else 0

    await write_exercise_completion_fact(
        request=ExerciseCompletionMemoryRequest(
            owner_id=owner_id,
            session_id=session_id,
            turn_count=turn_count,
            exercise_type=exercise_type,
            display_name=display_name,
        ),
        memory_store=memory_store,
        memory_mode=memory_mode,
    )


async def write_exercise_completion_fact(
    *,
    request: ExerciseCompletionMemoryRequest,
    memory_store: MemoryStore | None,
    memory_mode: MemoryMode | None,
) -> None:
    """Write a semantic fact for an exercise completion from neutral input."""

    if memory_store is None or memory_mode == MemoryMode.INCOGNITO:
        return

    now = datetime.now(timezone.utc).isoformat()
    fact = SemanticFact(
        id=str(uuid4()),
        category="coping_strategy",
        subject=EntityRef(type="User", identifier=request.owner_id),
        predicate="USES",
        object=EntityRef(type="CopingStrategy", identifier=request.exercise_type),
        evidence_quote=f"Completed {request.display_name} exercise.",
        confidence="high",
        source_session_id=request.session_id,
        source_turn_index=request.turn_count,
        created_at=now,
        last_referenced_at=now,
        dormant_at=None,
        superseded_by=None,
        user_visible=True,
    )

    try:
        namespace = (request.owner_id, "semantic")
        await memory_store.aput(
            namespace,
            key=fact.id,
            value=fact.model_dump(mode="json"),
        )
        logger.info(
            "Wrote exercise completion fact: exercise_type=%s owner=%s",
            request.exercise_type,
            request.owner_id,
        )
    except Exception:
        logger.warning(
            "Failed to write exercise completion fact; skipping.",
            exc_info=True,
        )
