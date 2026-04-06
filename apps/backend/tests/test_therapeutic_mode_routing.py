import pytest

from agent.graph import build_initial_state
from agent.models import AgentInput, Message, MessageRole
from agent.subgraphs.therapeutic import run_therapeutic_subgraph, select_therapeutic_mode


@pytest.mark.asyncio
async def test_orientation_mode_routes_for_first_time_orientation_message() -> None:
    state = build_initial_state(AgentInput(message="How does this work? I'm new here."))

    assert select_therapeutic_mode(state) == "orientation"

    state = await run_therapeutic_subgraph(state)
    assert state["mode"] == "orientation"
    assert "I can help you talk through difficult moments" in state["response_text"]


@pytest.mark.asyncio
async def test_guided_exercise_mode_routes_for_grounding_request() -> None:
    state = build_initial_state(
        AgentInput(message="Can you give me a grounding exercise to help me calm down?")
    )

    assert select_therapeutic_mode(state) == "guided_exercise"

    state = await run_therapeutic_subgraph(state)
    assert state["mode"] == "guided_exercise"
    assert "5 things you can see" in state["response_text"]


@pytest.mark.asyncio
async def test_reflection_mode_routes_for_pattern_request() -> None:
    state = build_initial_state(
        AgentInput(message="Can you help me understand why I keep getting stuck in this pattern?")
    )

    assert select_therapeutic_mode(state) == "reflection"

    state = await run_therapeutic_subgraph(state)
    assert state["mode"] == "reflection"
    assert "A pattern I notice" in state["response_text"]


@pytest.mark.asyncio
async def test_out_of_scope_mode_routes_for_diagnosis_request() -> None:
    state = build_initial_state(
        AgentInput(message="Can you diagnose me and tell me what medication I should take?")
    )

    assert select_therapeutic_mode(state) == "out_of_scope"

    state = await run_therapeutic_subgraph(state)
    assert state["mode"] == "out_of_scope"
    assert "can't diagnose" in state["response_text"]


@pytest.mark.asyncio
async def test_realignment_mode_routes_when_user_says_reply_missed() -> None:
    state = build_initial_state(
        AgentInput(message="That's not what I meant. You misunderstood me.")
    )

    assert select_therapeutic_mode(state) == "realignment"

    state = await run_therapeutic_subgraph(state)
    assert state["mode"] == "realignment"
    assert "missed the point" in state["response_text"]


def test_session_intent_biases_ambiguous_follow_up_toward_guided_exercise() -> None:
    state = build_initial_state(
        AgentInput(
            message="Okay, let's do that.",
            history=[
                Message(role=MessageRole.USER, content="I want to do CBT for this negative thought."),
                Message(role=MessageRole.ASSISTANT, content="We can work through that together."),
            ],
        )
    )

    state["session_intent"] = "guided_cbt_work"
    state["session_intent_source"] = "explicit"

    assert select_therapeutic_mode(state) == "guided_exercise"


def test_explicit_intent_shift_is_allowed_mid_session() -> None:
    state = build_initial_state(
        AgentInput(
            message="Actually I just want to vent today. I don't want advice.",
            history=[
                Message(role=MessageRole.USER, content="I want to do CBT for this negative thought."),
                Message(role=MessageRole.ASSISTANT, content="We can work through that together."),
            ],
        )
    )

    assert state["session_intent"] == "just_need_to_vent"
    assert state["session_intent_source"] == "explicit"
    assert select_therapeutic_mode(state) == "support"
