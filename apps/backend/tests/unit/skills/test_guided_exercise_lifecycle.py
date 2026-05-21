"""Tests for the guided exercise skill lifecycle service."""

from __future__ import annotations

from typing import Any, cast

import pytest

from agent.memory.modes import MemoryMode
from agent.skills.guided_exercises.lifecycle import GuidedExerciseSkillService
from agent.skills.guided_exercises.registry import (
    EXERCISE_BOX_BREATHING,
    EXERCISE_SELF_COMPASSION,
)
from agent.state import AgentState


class _StepClassifierLLM:
    """Fake LLM for deterministic guided-exercise lifecycle tests."""

    def __init__(
        self,
        *,
        step_state: str = "complete",
        exercise_type: str = EXERCISE_BOX_BREATHING,
        response_text: str = "next step",
        selection_confidence: str = "high",
    ) -> None:
        self.step_state = step_state
        self.exercise_type = exercise_type
        self.response_text = response_text
        self.selection_confidence = selection_confidence
        self.structured_calls = 0

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: Any,
        system_instruction: str | None = None,
    ) -> Any:
        self.structured_calls += 1
        if response_schema.__name__ == "ExerciseSelectionDecision":
            return response_schema(
                exercise_type=self.exercise_type,
                reasoning="fake exercise selection",
                confidence=self.selection_confidence,
            )
        return response_schema(
            step_state=self.step_state,
            reasoning="fake step classification",
            confidence="high",
        )

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> Any:
        yield self.response_text


def _service(llm: _StepClassifierLLM) -> GuidedExerciseSkillService:
    return GuidedExerciseSkillService(
        classifier_llm=llm,  # type: ignore[arg-type]
        response_llm=llm,  # type: ignore[arg-type]
        memory_store=None,
        memory_mode=MemoryMode.INCOGNITO,
    )


def _state(
    message: str,
    exercise_type: str | None = None,
    exercise_step: int | None = None,
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
            "session_id": "test-exercise",
            "history": [],
            "session_progress": {"turn_count": 1},
            "exercise_state": exercise_state,
            "turn_lifecycle": {"active_flow": "none", "action": "none"},
        },
    )


@pytest.mark.asyncio
async def test_lifecycle_starts_selected_skill() -> None:
    llm = _StepClassifierLLM(
        exercise_type=EXERCISE_SELF_COMPASSION,
        response_text="Let's start with a self-compassion break.",
    )

    delta = await _service(llm).run_turn(_state("Can we do something for this?"))

    assert delta["exercise_state"]["exercise_type"] == EXERCISE_SELF_COMPASSION
    assert delta["exercise_state"]["exercise_step"] == 0
    assert delta["exercise_state"]["exercise_step_id"] == "acknowledge_suffering"
    assert delta["response_style"] == "guided_exercise"


@pytest.mark.asyncio
async def test_lifecycle_advances_completed_step() -> None:
    llm = _StepClassifierLLM(
        step_state="complete",
        response_text="Good. Now hold the breath for four counts.",
    )

    delta = await _service(llm).run_turn(_state("done", EXERCISE_BOX_BREATHING, 0))

    assert delta["exercise_state"]["exercise_step"] == 1
    assert delta["exercise_state"]["exercise_step_id"] == "hold_full"
    assert "hold" in delta["response_text"].lower()


@pytest.mark.asyncio
async def test_lifecycle_exit_clears_active_skill_state() -> None:
    llm = _StepClassifierLLM(
        step_state="exit",
        response_text="We can stop. What would help now?",
    )

    delta = await _service(llm).run_turn(
        _state("I don't want to do this", EXERCISE_BOX_BREATHING, 1)
    )

    assert delta["exercise_state"]["exercise_type"] is None
    assert delta["exercise_state"]["exercise_step"] is None
    assert delta["exercise_state"]["exercise_step_id"] is None
    assert "stop" in delta["response_text"].lower()
