"""Tests for guided-exercise routing helpers."""

from __future__ import annotations

import pytest

from agent.flows.guided_exercise import routing as guided_exercise_routing
from agent.skills.guided_exercises.catalog.registry import (
    available_exercise_definitions,
)
from agent.skills.guided_exercises.catalog.types import ExerciseDefinition, ExerciseStep


def _definition(
    exercise_id: str,
    *,
    required_capability: str | None = None,
) -> ExerciseDefinition:
    return ExerciseDefinition(
        id=exercise_id,
        display_name=exercise_id.replace("_", " ").title(),
        selection_use_case=f"{exercise_id} support",
        steps=(
            ExerciseStep(
                instruction="Try one small step.",
                id="step",
                completion_mode="confirmation",
            ),
        ),
        selection_aliases=(exercise_id.replace("_", " "),),
        required_capability=required_capability,
    )


def test_available_aliases_respect_installed_capability_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    basic = _definition("basic")
    gated = _definition(
        "gated",
        required_capability="advanced_exercises",
    )
    definitions = (basic, gated)

    def fake_available_exercise_definitions(
        **kwargs: object,
    ) -> tuple[ExerciseDefinition, ...]:
        return available_exercise_definitions(
            definitions=definitions,
            installed_skills=kwargs.get("installed_skills", ()),
            channel=kwargs.get("channel", "text"),
            therapeutic_approach=kwargs.get("therapeutic_approach"),
        )

    monkeypatch.setattr(
        guided_exercise_routing,
        "available_exercise_definitions",
        fake_available_exercise_definitions,
    )

    without_capability = guided_exercise_routing.available_exercise_aliases_for_state(
        {"installed_skills": [], "channel": "text"}
    )
    assert "basic" in without_capability
    assert "gated" not in without_capability

    with_capability = guided_exercise_routing.available_exercise_aliases_for_state(
        {"installed_skills": ["advanced_exercises"], "channel": "text"}
    )
    assert "basic" in with_capability
    assert "gated" in with_capability
