"""Tests for prompt-local guided exercise skill rendering."""

from __future__ import annotations

import pytest

from agent.skills.guided_exercises.catalog.registry import (
    EXERCISE_5_4_3_2_1,
    EXERCISE_BOX_BREATHING,
)
from agent.skills.guided_exercises.rendering.skill_context import (
    build_exercise_skill,
    render_exercise_skill_context,
)


def test_build_exercise_skill_from_registered_definition() -> None:
    skill = build_exercise_skill(EXERCISE_BOX_BREATHING)

    assert skill.skill_id == EXERCISE_BOX_BREATHING
    assert skill.version == 1
    assert skill.name == "a box breathing cycle"
    assert skill.steps[0].step_id == "inhale"
    assert skill.steps[0].completion_mode == "confirmation"


def test_render_exercise_skill_context_pins_current_runtime_step() -> None:
    rendered = render_exercise_skill_context(
        EXERCISE_5_4_3_2_1,
        current_step_index=1,
        runtime_action="advance",
    )

    assert "Exercise skill:" in rendered
    assert f"- skill_id: {EXERCISE_5_4_3_2_1}" in rendered
    assert "- runtime_action: advance" in rendered
    assert "Current runtime step:" in rendered
    assert "- step_id: hear" in rendered
    assert "Step map:" in rendered


def test_build_exercise_skill_rejects_unknown_exercise() -> None:
    with pytest.raises(KeyError):
        build_exercise_skill("unknown_exercise")
