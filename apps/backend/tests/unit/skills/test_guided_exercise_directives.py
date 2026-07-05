"""Tests for guided-exercise directive rendering."""

from __future__ import annotations

from typing import Any, cast

import pytest

from agent.skills.guided_exercises.lifecycle.responses import (
    _build_advance_delta,
    _build_exit_delta,
    _build_start_delta,
)
from agent.skills.guided_exercises.catalog.registry import (
    EXERCISE_BOX_BREATHING,
    get_exercise_display_name,
    get_exercise_steps,
)
from agent.skills.guided_exercises.rendering.directives import (
    GuidedExerciseDirective,
    render_full_guided_exercise_directive,
    render_tool_forced_guided_exercise_directive,
)
from agent.state import AgentState


class _PromptCaptureLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> Any:
        del system_instruction
        self.prompts.append(prompt)
        yield "guided reply"


def _state(
    message: str,
    exercise_type: str | None = EXERCISE_BOX_BREATHING,
    exercise_step: int | None = 0,
) -> AgentState:
    exercise_state: dict[str, Any] = {}
    if exercise_type is not None:
        exercise_state["exercise_type"] = exercise_type
    if exercise_step is not None:
        exercise_state["exercise_step"] = exercise_step
    return cast(
        AgentState,
        {
            "message": message,
            "session_id": "test-guided-directive",
            "history": [],
            "session_progress": {"turn_count": 1},
            "exercise_state": exercise_state,
            "turn_lifecycle": {"active_flow": "guided_exercise", "action": "none"},
        },
    )


def test_full_guided_exercise_directive_renders_skill_context() -> None:
    directive = GuidedExerciseDirective(
        exercise_type=EXERCISE_BOX_BREATHING,
        runtime_action="start",
        current_step_index=0,
        runtime_task="Start the guided breathing exercise.",
    )

    rendered = render_full_guided_exercise_directive(directive)

    assert rendered.startswith("Exercise skill:")
    assert f"- skill_id: {EXERCISE_BOX_BREATHING}" in rendered
    assert "- runtime_action: start" in rendered
    assert "Current runtime step:" in rendered
    assert "- step_id: inhale" in rendered
    assert rendered.endswith("Runtime task:\nStart the guided breathing exercise.")


def test_tool_forced_guided_exercise_directive_renders_required_tool_call() -> None:
    directive = GuidedExerciseDirective(
        exercise_type=EXERCISE_BOX_BREATHING,
        runtime_action="advance",
        current_step_index=1,
        runtime_task="Move to the next breathing step.",
    )

    rendered = render_tool_forced_guided_exercise_directive(directive)

    assert rendered.startswith("Exercise skill:\n")
    assert "Required tool: load_guided_exercise_skill" in rendered
    assert '"current_step_index": 1' in rendered
    assert f'"exercise_type": "{EXERCISE_BOX_BREATHING}"' in rendered
    assert '"runtime_action": "advance"' in rendered
    assert (
        "Use only the returned skill_context plus the Runtime task below." in rendered
    )
    assert rendered.endswith("Runtime task:\nMove to the next breathing step.")


@pytest.mark.asyncio
async def test_response_builders_preserve_start_advance_and_exit_runtime_tasks() -> (
    None
):
    steps = get_exercise_steps(EXERCISE_BOX_BREATHING)
    assert steps is not None
    display_name = get_exercise_display_name(EXERCISE_BOX_BREATHING)
    llm = _PromptCaptureLLM()

    await _build_start_delta(
        _state("start a breathing exercise", exercise_type=None, exercise_step=None),
        llm_client=llm,  # type: ignore[arg-type]
        exercise_type=EXERCISE_BOX_BREATHING,
    )
    start_prompt = llm.prompts[-1]
    assert "Exercise skill:" in start_prompt
    assert f"- skill_id: {EXERCISE_BOX_BREATHING}" in start_prompt
    assert "- runtime_action: start" in start_prompt
    assert "Current runtime step:" in start_prompt
    assert (
        "Runtime task:\n"
        f"Start the guided exercise {display_name}. "
        "Briefly name the exercise and invite the user into step 0.\n"
        f'Step 0 instruction: "{steps[0].instruction}"\n'
        "Rephrase naturally in your own words. Do NOT present a menu or "
        "ask whether they want a different exercise."
    ) in start_prompt

    await _build_advance_delta(
        state=_state("done", EXERCISE_BOX_BREATHING, 0),
        llm_client=llm,  # type: ignore[arg-type]
        exercise_type=EXERCISE_BOX_BREATHING,
        next_step_index=1,
    )
    advance_prompt = llm.prompts[-1]
    assert f"- skill_id: {EXERCISE_BOX_BREATHING}" in advance_prompt
    assert "- runtime_action: advance" in advance_prompt
    assert "Current runtime step:" in advance_prompt
    assert (
        "Runtime task:\n"
        "The user completed step 0 of 3. Briefly acknowledge what they shared, "
        "then move to step 1.\n"
        f'Step 1 instruction: "{steps[1].instruction}"\n'
        "Rephrase naturally in your own words — do NOT repeat this "
        "instruction verbatim. Do NOT repeat any earlier step."
    ) in advance_prompt

    await _build_exit_delta(
        _state("I want to stop", EXERCISE_BOX_BREATHING, 1),
        llm_client=llm,  # type: ignore[arg-type]
    )
    exit_prompt = llm.prompts[-1]
    assert f"- skill_id: {EXERCISE_BOX_BREATHING}" in exit_prompt
    assert "- runtime_action: exit" in exit_prompt
    assert (
        "Runtime task:\n"
        "The user wants to stop or leave the current guided exercise. "
        "Briefly acknowledge that choice, do not continue the exercise, and "
        "ask what would feel most helpful now."
    ) in exit_prompt
