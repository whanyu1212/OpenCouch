"""State-delta helpers for guided therapeutic exercises."""

from __future__ import annotations

from typing import Any

from agent.state import AgentState, cleared_exercise_state
from agent.skills.guided_exercises.catalog.registry import get_exercise_steps
from agent.skills.guided_exercises.catalog.types import ExerciseStep


# ── State delta helpers ────────────────────────────────────────────────


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
