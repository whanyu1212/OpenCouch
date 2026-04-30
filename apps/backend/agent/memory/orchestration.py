"""Shared orchestration helpers for memory side-effect nodes."""

from __future__ import annotations

from agent.memory.small_talk_gate import is_small_talk
from agent.state import AgentState


def should_skip_memory_extraction(state: AgentState) -> str | None:
    """Return the extraction skip reason for operational or small-talk turns.

    Args:
        state: Current graph state.

    Returns:
        Skip reason without the ``"skipped: "`` prefix, or ``None`` when
        memory extraction should proceed.
    """

    route = state.get("route")
    if route == "crisis":
        return "crisis_path"
    if route == "memory_control":
        return "memory_control_path"
    if route == "grounded_lookup":
        return "grounded_lookup_path"

    if is_small_talk(state["message"]):
        return "small_talk_gate"

    return None


def get_session_turn_index(state: AgentState) -> int:
    """Return the zero-based user turn index for memory provenance.

    Args:
        state: Current graph state.

    Returns:
        Zero-based user turn index.
    """

    session_progress = state.get("session_progress", {})
    turn_count = int(session_progress.get("turn_count", 1))
    return max(0, turn_count - 1)
