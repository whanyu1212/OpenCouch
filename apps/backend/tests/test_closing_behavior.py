import pytest

from agent.graph import build_initial_state
from agent.models import AgentInput
from agent.nodes.guided_exercise import run_guided_exercise_response
from agent.nodes.reflection import run_reflection_response
from agent.nodes.therapeutic import run_therapeutic_response
from agent.prompts.builders import build_therapeutic_response_prompt


@pytest.mark.asyncio
async def test_support_fallback_changes_when_session_is_closing() -> None:
    state = build_initial_state(
        AgentInput(message="I think that's enough for today.")
    )
    state["session_stage"] = "closing"

    state = await run_therapeutic_response(state)

    assert "most important thing" in state["response_text"].lower()
    assert "whenever you want" in state["response_text"].lower()
    assert "tell me a bit more" not in state["response_text"].lower()


@pytest.mark.asyncio
async def test_reflection_fallback_lands_gently_when_session_is_closing() -> None:
    state = build_initial_state(
        AgentInput(message="Before we wrap up, what pattern do you notice?")
    )
    state["session_stage"] = "closing"

    state = await run_reflection_response(state)

    assert "most important from this session" in state["response_text"].lower()
    assert "we can return to it" in state["response_text"].lower()


@pytest.mark.asyncio
async def test_guided_exercise_fallback_avoids_starting_full_new_exercise_when_closing() -> None:
    state = build_initial_state(
        AgentInput(message="Can you give me one last exercise before we end?")
    )
    state["session_stage"] = "closing"

    state = await run_guided_exercise_response(state)

    assert "before we end" in state["response_text"].lower()
    assert "do not need to do a full exercise" in state["response_text"].lower()


def test_closing_prompt_includes_stage_specific_guidance() -> None:
    state = build_initial_state(
        AgentInput(message="I think that's enough for today.")
    )
    state["session_stage"] = "closing"

    prompt = build_therapeutic_response_prompt(state)

    assert "Treat this as a closing-phase reply." in prompt
    assert "Do not open a broad new exploration." in prompt
