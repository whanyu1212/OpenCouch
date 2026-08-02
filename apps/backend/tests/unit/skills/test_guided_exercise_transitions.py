"""Tests for transport-neutral guided-exercise transitions."""

from __future__ import annotations

import pytest

from agent.skills.guided_exercises.catalog.registry import (
    EXERCISE_5_4_3_2_1,
    EXERCISE_BOX_BREATHING,
    get_exercise_definition,
)
from agent.skills.guided_exercises.lifecycle.transitions import (
    InvalidGuidedExerciseState,
    progress_guided_exercise_transition,
    start_guided_exercise_transition,
)


def _definition(exercise_type: str):
    definition = get_exercise_definition(exercise_type)
    assert definition is not None
    return definition


def _active_state(exercise_type: str = EXERCISE_BOX_BREATHING) -> dict[str, object]:
    transition = start_guided_exercise_transition(
        _definition(exercise_type),
        therapeutic_approach="cbt",
    )
    assert transition.exercise_state is not None
    return dict(transition.exercise_state)


def test_start_transition_initializes_registered_first_step() -> None:
    definition = _definition(EXERCISE_BOX_BREATHING)

    transition = start_guided_exercise_transition(
        definition,
        therapeutic_approach="cbt",
    )

    assert transition.action == "start"
    assert transition.current_step_id == "inhale"
    assert transition.next_step_id == "inhale"
    assert transition.exercise_state == {
        "exercise_type": EXERCISE_BOX_BREATHING,
        "exercise_step": 0,
        "exercise_step_id": "inhale",
        "exercise_version": definition.version,
        "exercise_therapeutic_approach": "cbt",
    }


def test_start_transition_uses_grounding_default_approach() -> None:
    transition = start_guided_exercise_transition(
        _definition(EXERCISE_5_4_3_2_1),
        therapeutic_approach=None,
    )

    assert transition.exercise_state is not None
    assert transition.exercise_state["exercise_therapeutic_approach"] == "dbt_skills"


@pytest.mark.parametrize(
    ("outcome", "action"),
    [
        ("hold", "hold"),
        ("stuck", "simplify"),
        ("exit", "cancel"),
        ("unsafe", "crisis"),
    ],
)
def test_non_advancing_transitions_preserve_or_clear_state(
    outcome: str,
    action: str,
) -> None:
    transition = progress_guided_exercise_transition(
        _definition(EXERCISE_BOX_BREATHING),
        exercise_state=_active_state(),
        outcome=outcome,  # type: ignore[arg-type]
    )

    assert transition.action == action
    assert transition.previous_step_id == "inhale"
    if outcome == "exit":
        assert transition.exercise_state == {
            "exercise_type": None,
            "exercise_step": None,
            "exercise_step_id": None,
            "exercise_version": None,
            "exercise_therapeutic_approach": None,
        }
    else:
        assert transition.exercise_state is None
        assert transition.current_step_id == "inhale"


def test_complete_transition_advances_and_preserves_continuity() -> None:
    definition = _definition(EXERCISE_BOX_BREATHING)
    state = _active_state()
    state["exercise_version"] = definition.version
    state["exercise_therapeutic_approach"] = "dbt_skills"

    transition = progress_guided_exercise_transition(
        definition,
        exercise_state=state,
        outcome="complete",
    )

    assert transition.action == "advance"
    assert transition.previous_step_id == "inhale"
    assert transition.current_step_id == "hold_full"
    assert transition.next_step_id == "hold_full"
    assert transition.exercise_state == {
        "exercise_type": EXERCISE_BOX_BREATHING,
        "exercise_step": 1,
        "exercise_step_id": "hold_full",
        "exercise_version": definition.version,
        "exercise_therapeutic_approach": "dbt_skills",
    }


def test_complete_transition_clears_final_step() -> None:
    definition = _definition(EXERCISE_BOX_BREATHING)
    state = _active_state()
    final_index = len(definition.steps) - 1
    state["exercise_step"] = final_index
    state["exercise_step_id"] = definition.steps[final_index].id

    transition = progress_guided_exercise_transition(
        definition,
        exercise_state=state,
        outcome="complete",
    )

    assert transition.action == "complete"
    assert transition.previous_step_id == definition.steps[final_index].id
    assert transition.exercise_state == {
        "exercise_type": None,
        "exercise_step": None,
        "exercise_step_id": None,
        "exercise_version": None,
        "exercise_therapeutic_approach": None,
    }


@pytest.mark.parametrize(
    "update",
    [
        {"exercise_type": EXERCISE_5_4_3_2_1},
        {"exercise_step": -1},
        {"exercise_step": 99},
        {"exercise_step_id": "stale-step"},
    ],
)
def test_progress_transition_rejects_invalid_active_state(
    update: dict[str, object],
) -> None:
    state = _active_state()
    state.update(update)

    with pytest.raises(InvalidGuidedExerciseState):
        progress_guided_exercise_transition(
            _definition(EXERCISE_BOX_BREATHING),
            exercise_state=state,
            outcome="complete",
        )
