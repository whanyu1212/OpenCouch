"""Unit tests for deterministic Graphiti retrieval and write heuristics."""

from agent.memory_graph import (
    build_graph_episode_payload,
    build_graph_memory_query,
    should_record_graph_episode,
    should_retrieve_graph_memory,
)
from agent.models import Channel, CrisisAssessment, ModeType
from agent.state import AgentState


def _base_state(**overrides) -> AgentState:
    """Return a minimal agent state fixture for graph-memory tests.

    Args:
        **overrides: Partial state overrides for the test case.

    Returns:
        A fully-formed agent state.
    """

    state: AgentState = {
        "message": "Why does this keep happening with my partner?",
        "channel": Channel.TEST,
        "user_id": "user-123",
        "session_id": "session-123",
        "installed_skills": [],
        "history": [],
        "working_memory": [],
        "session_summary": "Recent user themes: anxiety and relationship stress.",
        "active_concerns": ["relationship strain", "anxiety or rumination"],
        "open_loops": [],
        "current_goal": "Understand the relationship pattern.",
        "session_intent": "reflection_and_pattern_finding",
        "session_intent_source": "explicit",
        "session_stage": "exploration",
        "session_stage_source": "deterministic",
        "turn_count": 4,
        "crisis": CrisisAssessment(
            level=0,
            confidence="low",
            reason="",
            needs_crisis_response=False,
            needs_clarification=False,
        ),
        "mode": "pattern_reflection",
        "mode_source": "keyword",
        "mode_type": ModeType.THERAPEUTIC,
        "response_text": "Let’s look at the pattern together.",
    }
    state.update(overrides)
    return state


def test_should_retrieve_graph_memory_skips_orientation_message() -> None:
    """Orientation/capability questions should not trigger Graphiti retrieval."""

    assert (
        should_retrieve_graph_memory(
            message="Hi, what can you do for me?",
            prior_state=None,
        )
        is False
    )


def test_should_retrieve_graph_memory_for_pattern_follow_up() -> None:
    """Pattern-oriented follow-ups should trigger Graphiti retrieval."""

    prior_state = _base_state()
    assert (
        should_retrieve_graph_memory(
            message="Why does this keep happening with us?",
            prior_state=prior_state,
        )
        is True
    )


def test_build_graph_memory_query_uses_prior_state_context() -> None:
    """Graphiti queries should include compact prior concerns and intent context."""

    query = build_graph_memory_query(
        message="Why does this keep happening with us?",
        prior_state=_base_state(),
    )

    assert "Current user message: Why does this keep happening with us?" in query
    assert "Active concerns: relationship strain, anxiety or rumination" in query
    assert "Need prior recurring patterns or similar episodes." in query


def test_should_record_graph_episode_skips_low_value_support_turn() -> None:
    """Short, low-information support turns should not become graph memory."""

    state = _base_state(
        message="Both",
        session_intent=None,
        active_concerns=["anxiety or rumination"],
        current_goal=None,
        mode="supportive_conversation",
        response_text="Which part feels louder right now?",
    )

    assert should_record_graph_episode(state) is False


def test_build_graph_episode_payload_is_curated() -> None:
    """Curated episodes should include structured state, not raw assistant text."""

    name, body, source_description = build_graph_episode_payload(_base_state())

    assert name.startswith("opencouch-session-123-turn-4-")
    assert "User shared: Why does this keep happening with my partner?" in body
    assert "Active concerns: relationship strain, anxiety or rumination" in body
    assert "Therapeutic mode: pattern reflection" in body
    assert "Durable cues:" in body
    assert "assistant:" not in body.lower()
    assert source_description == "OpenCouch curated therapeutic memory episode"
