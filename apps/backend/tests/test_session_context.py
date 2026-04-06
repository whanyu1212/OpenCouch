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


def test_build_initial_state_sets_explicit_session_intent() -> None:
    """Explicit intent language should be captured during initial state building."""

    state = build_initial_state(
        AgentInput(message="I want to do a CBT thought record about this fear.")
    )

    assert state["session_intent"] == "guided_cbt_work"
    assert state["session_intent_source"] == "explicit"
