"""Pure session-state helpers for the persistent agent runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from agent.models import CrisisAssessment
from agent.state import AgentState, cleared_exercise_state

EXERCISE_STATE_FIELDS = (
    "exercise_type",
    "exercise_step",
    "exercise_step_id",
    "exercise_version",
    "exercise_therapeutic_approach",
)

ACTIVE_FLOWS = {"none", "guided_exercise", "pending_memory_action"}
ACTIVE_FLOW_ACTIONS = {"none", "start", "continue", "preserve", "resume", "clear"}


@dataclass(frozen=True)
class TurnLifecycleDecision:
    """Resolved active-flow lifecycle metadata for the current turn."""

    active_flow: str
    action: str
    state_delta: dict[str, object]


def get_transcript(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return transcript turns from runtime state."""

    transcript = state.get("transcript")
    if isinstance(transcript, list):
        return [turn for turn in transcript if isinstance(turn, dict)]

    history = state.get("history")
    if isinstance(history, list):
        return [turn for turn in history if isinstance(turn, dict)]

    return []


def get_recent_history(
    state: Mapping[str, Any], *, limit: int = 6
) -> list[dict[str, Any]]:
    """Return recent transcript turns for prompt/context use."""

    if limit <= 0:
        return []
    return get_transcript(state)[-limit:]


def format_recent_history(
    state: Mapping[str, Any],
    *,
    limit: int = 6,
    empty: str = "(no prior history)",
) -> str:
    """Format recent transcript turns for prompt injection."""

    lines = []
    for turn in get_recent_history(state, limit=limit):
        content = str(turn.get("content", "") or "").strip()
        if not content:
            continue
        role = str(turn.get("role", "unknown") or "unknown")
        lines.append(f"{role}: {content}")
    return "\n".join(lines) or empty


def current_turn_lifecycle(state: AgentState) -> TurnLifecycleDecision:
    """Read the current turn's active-flow lifecycle decision from state."""

    raw = state.get("turn_lifecycle")
    if not isinstance(raw, Mapping):
        raise ValueError("Missing or invalid turn_lifecycle state.")

    active_flow = raw.get("active_flow")
    action = raw.get("action")
    if active_flow not in ACTIVE_FLOWS or action not in ACTIVE_FLOW_ACTIONS:
        raise ValueError(f"Malformed turn_lifecycle state: {raw!r}.")
    return TurnLifecycleDecision(str(active_flow), str(action), {})


def clear_all_active_flows_delta() -> dict[str, object]:
    """Return a delta that clears guided exercise and pending memory actions."""

    return {
        "exercise_state": cleared_exercise_state(),
        "memory_control": {"pending_action": None},
        "turn_lifecycle": {"active_flow": "none", "action": "none"},
    }


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
    """Extract the crisis level from a runtime state.

    Args:
        state (AgentState): Runtime state snapshot.

    Returns:
        int: Crisis level, defaulting to ``0`` when absent or unrecognized.
    """

    crisis = state.get("crisis")
    if isinstance(crisis, CrisisAssessment):
        return crisis.level
    if isinstance(crisis, Mapping):
        return int(crisis.get("level", 0) or 0)
    return 0
