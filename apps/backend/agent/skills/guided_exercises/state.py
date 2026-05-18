"""State-delta helpers for guided therapeutic exercises."""

from __future__ import annotations

from typing import Any

from agent.state import AgentState, cleared_exercise_state
from agent.skills.guided_exercises.registry import (
    get_exercise_definition,
    get_exercise_steps,
)
from agent.skills.guided_exercises.types import ExerciseStep


# ── State delta helpers ────────────────────────────────────────────────


def _start_exercise_delta(
    state: AgentState,
    *,
    exercise_type: str,
) -> dict[str, Any]:
    """Return the exercise-state delta that starts a new exercise at step 0.

    Captures the current ``therapeutic_approach`` as
    ``exercise_therapeutic_approach`` so the prompt builder can use a stable
    approach for the entire exercise lifetime, immune to mid-exercise
    side-turn drift.

    Args:
        state: Current runtime state.
        exercise_type: Exercise identifier to start.

    Returns:
        State delta that starts the exercise at step 0.
    """

    approach = state.get("therapeutic_approach")
    definition = get_exercise_definition(exercise_type)
    steps = definition.steps if definition is not None else ()
    return {
        "exercise_state": {
            "exercise_type": exercise_type,
            "exercise_step": 0,
            "exercise_step_id": steps[0].id if steps else None,
            "exercise_version": definition.version if definition is not None else None,
            "exercise_therapeutic_approach": approach,
        },
    }


def _advance_step_delta(state: AgentState) -> dict[str, Any]:
    """Return the exercise-state delta that bumps the exercise step index.

    Args:
        state: Current runtime state.

    Returns:
        State delta that advances the exercise step by one.
    """

    exercise_state = state.get("exercise_state", {})
    current = exercise_state.get("exercise_step") or 0
    exercise_type = exercise_state.get("exercise_type")
    next_index = current + 1
    next_step = _get_current_step(exercise_type, next_index)
    return {
        "exercise_state": {
            "exercise_step": next_index,
            "exercise_step_id": next_step.id if next_step is not None else None,
        },
    }


def clear_exercise_delta(state: AgentState) -> dict[str, Any]:
    """Return the exercise-state delta that clears exercise state.

    Used on both exit and natural completion. Setting all continuity fields
    to ``None`` is the marker for "no exercise running" that the
    OpenAI text runtime checks for.

    Args:
        state (AgentState): Current runtime state.

    Returns:
        dict[str, Any]: State delta that clears active exercise fields.
    """

    return {"exercise_state": cleared_exercise_state()}


def _get_current_step(
    exercise_type: str | None,
    step_index: int | None,
) -> ExerciseStep | None:
    """Return the current exercise step.

    Args:
        exercise_type: Active exercise identifier.
        step_index: Current step index.

    Returns:
        Current ``ExerciseStep``, or ``None`` if the state is invalid.
    """

    if exercise_type is None or step_index is None:
        return None
    steps = get_exercise_steps(exercise_type)
    if steps is None:
        return None
    if step_index < 0 or step_index >= len(steps):
        return None
    return steps[step_index]


def _is_last_step(exercise_type: str, step_index: int) -> bool:
    """Return whether the given step is the last one in the exercise.

    Args:
        exercise_type: Active exercise identifier.
        step_index: Current step index.

    Returns:
        Whether ``step_index`` points at the final step.
    """

    steps = get_exercise_steps(exercise_type)
    if steps is None:
        return False
    return step_index >= len(steps) - 1
