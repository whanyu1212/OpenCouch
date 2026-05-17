"""Pure session-state helpers for the persistent agent runtime."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from agent.models import CrisisAssessment
from agent.state import AgentState

EXERCISE_STATE_FIELDS = (
    "exercise_type",
    "exercise_step",
    "exercise_step_id",
    "exercise_version",
    "exercise_therapeutic_approach",
)


def transcript_length(state: AgentState | None) -> int:
    """Return the durable transcript length for a thread state.

    Args:
        state (AgentState | None): Thread state snapshot.

    Returns:
        int: Transcript length, or ``0`` when state is absent.
    """

    if state is None:
        return 0
    return len(state.get("transcript", []) or [])


def slice_state_to_active_session(
    state: AgentState,
    *,
    transcript_start_index: int,
) -> AgentState:
    """Slice a state snapshot to the active-session transcript window.

    Args:
        state (AgentState): Full thread state.
        transcript_start_index (int): Transcript index where the active session
            begins.

    Returns:
        AgentState: Shallow state copy limited to the active session window.
    """

    transcript = list(state.get("transcript", []) or [])
    start = min(max(transcript_start_index, 0), len(transcript))
    windowed = cast(AgentState, dict(state))
    windowed["transcript"] = transcript[start:]

    if "history" in state:
        history = list(state.get("history", []) or [])
        history_start = min(start, len(history))
        windowed["history"] = history[history_start:]

    return windowed


def session_continuity_clear_delta(state: AgentState | None) -> dict[str, Any]:
    """Build a delta that clears session-scoped continuity fields.

    Args:
        state (AgentState | None): Current runtime state, if any.

    Returns:
        dict[str, Any]: Partial state update that clears stale session
            continuity.
    """

    if state is None:
        return {}

    delta: dict[str, Any] = {}
    exercise_state = state.get("exercise_state", {}) or {}
    if any(exercise_state.get(field) is not None for field in EXERCISE_STATE_FIELDS):
        delta["exercise_state"] = {
            "exercise_type": None,
            "exercise_step": None,
            "exercise_step_id": None,
            "exercise_version": None,
            "exercise_therapeutic_approach": None,
        }

    if state.get("therapeutic_approach") is not None:
        delta["therapeutic_approach"] = None

    return delta


def turn_count_from_state(state: AgentState | None) -> int:
    """Extract the persisted turn count from a runtime state snapshot.

    Args:
        state (AgentState | None): Runtime state snapshot, if any.

    Returns:
        int: Persisted turn count.
    """

    if state is None:
        return 0
    session_progress = state.get("session_progress", {}) or {}
    return int(session_progress.get("turn_count", 0) or 0)


def active_transcript_length(
    state: AgentState,
    *,
    transcript_start_index: int,
) -> int:
    """Return transcript length within the active session window.

    Args:
        state (AgentState): Thread state snapshot.
        transcript_start_index (int): Transcript index where the active session
            begins.

    Returns:
        int: Non-negative active-session transcript length.
    """

    return max(0, transcript_length(state) - transcript_start_index)


def crisis_level_from_state(state: AgentState) -> int:
    """Extract the crisis level from a graph state.

    Args:
        state (AgentState): Graph state snapshot.

    Returns:
        int: Crisis level, defaulting to ``0`` when absent or unrecognized.
    """

    crisis = state.get("crisis")
    if isinstance(crisis, CrisisAssessment):
        return crisis.level
    if isinstance(crisis, Mapping):
        return int(crisis.get("level", 0) or 0)
    return 0
