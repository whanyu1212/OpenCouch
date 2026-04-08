from agent.graph import build_initial_state
from agent.models import AgentInput, Message, MessageRole
from agent.prompts.builders import format_session_context


def test_build_initial_state_populates_session_context_and_trims_history() -> None:
    """Initial state building should derive context fields and trim history."""

    history = [
        Message(role=MessageRole.USER, content="I'm new here."),
        Message(role=MessageRole.ASSISTANT, content="What feels most important today?"),
        Message(
            role=MessageRole.USER,
            content="Work has been overwhelming and I feel drained.",
        ),
        Message(role=MessageRole.ASSISTANT, content="That sounds exhausting."),
        Message(
            role=MessageRole.USER,
            content="I can't switch off after work and my sleep is getting worse.",
        ),
        Message(
            role=MessageRole.ASSISTANT,
            content="When did that start feeling most intense?",
        ),
        Message(
            role=MessageRole.USER,
            content="A couple of months ago, and I keep spiraling at night.",
        ),
        Message(
            role=MessageRole.ASSISTANT,
            content="What do you notice right before the spiral starts?",
        ),
        Message(
            role=MessageRole.USER, content="Usually after messages from my manager."
        ),
        Message(
            role=MessageRole.ASSISTANT,
            content="That helps narrow the trigger a little.",
        ),
    ]

    state = build_initial_state(
        AgentInput(
            message="Can you help me understand why I keep spiraling like this?",
            history=history,
        )
    )

    assert len(state["history"]) == 8
    assert state["turn_count"] == 6
    assert "overwhelm or stress" in state["active_concerns"]
    assert "anxiety or rumination" in state["active_concerns"]
    assert state["current_goal"] == "understand a recurring pattern"
    assert state["open_loops"]
    assert "Active concerns include" in state["session_summary"]


def test_format_session_context_includes_structured_fields() -> None:
    """Formatted session context should surface the core structured fields."""

    state = build_initial_state(
        AgentInput(
            message="Can you help me understand why this keeps happening with work?",
            history=[
                Message(
                    role=MessageRole.USER,
                    content="Work has been making me anxious for weeks.",
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    content="What feels most stuck about it?",
                ),
            ],
        )
    )

    context_block = format_session_context(state)

    assert "Turn count:" in context_block
    assert "Session summary:" in context_block
    assert "Active concerns:" in context_block
    assert "Open loops:" in context_block
    assert "Current goal:" in context_block
    assert "Session intent:" in context_block
    assert "understand a recurring pattern" in context_block


def test_format_session_context_includes_working_memory() -> None:
    """Formatted session context should surface retrieved long-term memory."""

    state = build_initial_state(
        AgentInput(
            message="I'm feeling anxious again.",
            working_memory=[
                "Support preference: Sometimes wants space before advice.",
                "Recurring concern: anxiety or rumination",
            ],
        )
    )

    context_block = format_session_context(state)

    assert "Long-term memory:" in context_block
    assert "Support preference: Sometimes wants space before advice." in context_block


def test_build_initial_state_sets_explicit_session_intent() -> None:
    """Explicit intent language should be captured during initial state building."""

    state = build_initial_state(
        AgentInput(message="I want to do a CBT thought record about this fear.")
    )

    assert state["session_intent"] == "guided_cbt_work"
    assert state["session_intent_source"] == "explicit"


def test_meta_orientation_turn_does_not_pollute_session_context() -> None:
    """Capability questions should not become concerns, goals, or open loops."""

    state = build_initial_state(AgentInput(message="Hi, what can you do for me?"))

    assert state["active_concerns"] == []
    assert state["current_goal"] is None
    assert state["open_loops"] == []
    assert "Hi, what can you do for me" not in state["session_summary"]
    assert state["session_intent"] is None


def test_reflection_intent_stays_sticky_on_emotional_follow_up() -> None:
    """Generic support language should not erase an ongoing reflection arc."""

    state = build_initial_state(
        AgentInput(
            message="That sounds right, and it makes me sad to hear it that clearly.",
            history=[
                Message(
                    role=MessageRole.USER,
                    content="I want help understanding why I keep ending up in the same pattern.",
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    content="What pattern feels most present to you?",
                ),
            ],
        )
    )

    assert state["session_intent"] == "reflection_and_pattern_finding"
