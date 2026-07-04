"""Tests for shared runtime state merge helpers."""

from __future__ import annotations

from agent.models import Channel
from agent.runtime.state_ops import apply_state_delta, build_effective_turn_state
from agent.state import AgentState, AgentTurnInputState


def test_build_effective_turn_state_preserves_grouped_channel_siblings() -> None:
    prior_state: AgentState = {
        "message": "previous",
        "channel": Channel.WEB,
        "user_id": "user-1",
        "session_id": "thread-1",
        "installed_skills": [],
        "working_memory": [],
        "session_memory": {
            "summary": "prior summary",
            "current_goal": "sleep better",
        },
        "session_progress": {"turn_count": 4, "is_guest": False},
        "transcript": [{"role": "user", "content": "earlier"}],
    }
    initial_state: AgentTurnInputState = {
        "message": "new",
        "channel": Channel.WEB,
        "user_id": "user-1",
        "session_id": "thread-1",
        "installed_skills": [],
        "working_memory": [],
        "session_memory": {"summary": "new summary"},
        "session_progress": {"turn_count": 5},
        "transcript": [{"role": "user", "content": "new"}],
    }

    state = build_effective_turn_state(prior_state, initial_state)

    assert state["session_memory"] == {
        "summary": "new summary",
        "current_goal": "sleep better",
    }
    assert state["session_progress"] == {"turn_count": 5, "is_guest": False}
    assert state["transcript"] == [
        {"role": "user", "content": "earlier"},
        {"role": "user", "content": "new"},
    ]


def test_build_effective_turn_state_preserves_turn_lifecycle_clarification() -> None:
    prior_state: AgentState = {
        "turn_lifecycle": {
            "active_flow": "none",
            "action": "none",
            "tentative_route": "guided_exercise",
            "triage_confidence": "medium",
        }
    }
    initial_state: AgentTurnInputState = {
        "turn_lifecycle": {"active_flow": "none", "action": "clear"}
    }

    state = build_effective_turn_state(prior_state, initial_state)

    assert state["turn_lifecycle"] == {
        "active_flow": "none",
        "action": "clear",
        "tentative_route": "guided_exercise",
        "triage_confidence": "medium",
    }


def test_build_effective_turn_state_can_preserve_prior_grouped_channel_values() -> None:
    prior_state: AgentState = {
        "procedural_profile": {"proactive_recall_enabled": True},
    }
    initial_state: AgentTurnInputState = {
        "procedural_profile": {"proactive_recall_enabled": False},
    }

    state = build_effective_turn_state(
        prior_state,
        initial_state,
        prior_state_wins=True,
    )

    assert state["procedural_profile"]["proactive_recall_enabled"] is True


def test_build_effective_turn_state_can_preserve_prior_scalar_and_list_values() -> None:
    prior_state: AgentState = {
        "installed_skills": ["guided_exercise"],
        "message": "prior message",
    }
    initial_state: AgentTurnInputState = {
        "installed_skills": [],
        "message": "new message",
        "user_id": "user-1",
    }

    state = build_effective_turn_state(
        prior_state,
        initial_state,
        prior_state_wins=True,
    )

    assert state["installed_skills"] == ["guided_exercise"]
    assert state["message"] == "prior message"
    assert state["user_id"] == "user-1"


def test_apply_state_delta_preserves_grouped_channel_siblings() -> None:
    state: AgentState = {
        "memory_control": {
            "pending_action": {"kind": "delete"},
            "action": {"kind": "request"},
        }
    }

    apply_state_delta(state, {"memory_control": {"action": {"kind": "confirm"}}})

    assert state["memory_control"] == {
        "pending_action": {"kind": "delete"},
        "action": {"kind": "confirm"},
    }
