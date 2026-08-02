"""Transport-neutral guided-exercise state transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agent.state import ExerciseState, cleared_exercise_state
from agent.skills.guided_exercises.catalog.registry import EXERCISE_5_4_3_2_1
from agent.skills.guided_exercises.catalog.types import ExerciseDefinition

GuidedExerciseOutcome = Literal["complete", "hold", "stuck", "exit", "unsafe"]
GuidedExerciseAction = Literal[
    "start",
    "advance",
    "hold",
    "simplify",
    "complete",
    "cancel",
    "crisis",
]


class InvalidGuidedExerciseState(ValueError):
    """Raised when active guided-exercise state cannot be transitioned."""


@dataclass(frozen=True, slots=True)
class GuidedExerciseTransition:
    """One deterministic guided-exercise lifecycle transition."""

    action: GuidedExerciseAction
    exercise_state: ExerciseState | None
    previous_step_id: str | None = None
    current_step_id: str | None = None
    next_step_id: str | None = None


def start_guided_exercise_transition(
    definition: ExerciseDefinition,
    *,
    therapeutic_approach: str | None,
) -> GuidedExerciseTransition:
    """Return the initial state for a registered exercise definition."""

    if not definition.steps:
        raise InvalidGuidedExerciseState(
            f"Guided exercise {definition.id!r} has no registered steps."
        )

    approach = therapeutic_approach
    if approach in (None, "", "none") and definition.id == EXERCISE_5_4_3_2_1:
        approach = "dbt_skills"
    first_step = definition.steps[0]
    return GuidedExerciseTransition(
        action="start",
        exercise_state={
            "exercise_type": definition.id,
            "exercise_step": 0,
            "exercise_step_id": first_step.id,
            "exercise_version": definition.version,
            "exercise_therapeutic_approach": approach,
        },
        current_step_id=first_step.id,
        next_step_id=first_step.id,
    )


def progress_guided_exercise_transition(
    definition: ExerciseDefinition,
    *,
    exercise_state: ExerciseState,
    outcome: GuidedExerciseOutcome,
) -> GuidedExerciseTransition:
    """Return the deterministic transition for one active exercise outcome."""

    step_index, current_step_id = _validated_active_step(
        definition,
        exercise_state=exercise_state,
    )

    if outcome == "unsafe":
        return GuidedExerciseTransition(
            action="crisis",
            exercise_state=None,
            previous_step_id=current_step_id,
            current_step_id=current_step_id,
        )
    if outcome == "exit":
        return GuidedExerciseTransition(
            action="cancel",
            exercise_state=cleared_exercise_state(),
            previous_step_id=current_step_id,
        )
    if outcome == "hold":
        return GuidedExerciseTransition(
            action="hold",
            exercise_state=None,
            previous_step_id=current_step_id,
            current_step_id=current_step_id,
        )
    if outcome == "stuck":
        return GuidedExerciseTransition(
            action="simplify",
            exercise_state=None,
            previous_step_id=current_step_id,
            current_step_id=current_step_id,
        )
    if outcome != "complete":
        raise ValueError(f"Unsupported guided exercise outcome: {outcome!r}")

    next_index = step_index + 1
    if next_index >= len(definition.steps):
        return GuidedExerciseTransition(
            action="complete",
            exercise_state=cleared_exercise_state(),
            previous_step_id=current_step_id,
        )

    next_step = definition.steps[next_index]
    return GuidedExerciseTransition(
        action="advance",
        exercise_state=_active_exercise_state(
            definition,
            exercise_state=exercise_state,
            step_index=next_index,
        ),
        previous_step_id=current_step_id,
        current_step_id=next_step.id,
        next_step_id=next_step.id,
    )


def _validated_active_step(
    definition: ExerciseDefinition,
    *,
    exercise_state: ExerciseState,
) -> tuple[int, str]:
    """Return the validated active step index and catalog step id."""

    if not definition.steps:
        raise InvalidGuidedExerciseState(
            f"Guided exercise {definition.id!r} has no registered steps."
        )
    if exercise_state.get("exercise_type") != definition.id:
        raise InvalidGuidedExerciseState(
            "Active guided-exercise type does not match the requested definition."
        )

    step_index = exercise_state.get("exercise_step")
    if type(step_index) is not int or not 0 <= step_index < len(definition.steps):
        raise InvalidGuidedExerciseState("Active guided-exercise step is invalid.")

    current_step_id = definition.steps[step_index].id
    stored_step_id = exercise_state.get("exercise_step_id")
    if stored_step_id not in (None, "", current_step_id):
        raise InvalidGuidedExerciseState(
            "Active guided-exercise step id does not match the registered step."
        )
    return step_index, current_step_id


def _active_exercise_state(
    definition: ExerciseDefinition,
    *,
    exercise_state: ExerciseState,
    step_index: int,
) -> ExerciseState:
    """Return full active state at ``step_index``, preserving continuity metadata."""

    step = definition.steps[step_index]
    return {
        "exercise_type": definition.id,
        "exercise_step": step_index,
        "exercise_step_id": step.id,
        "exercise_version": exercise_state.get("exercise_version"),
        "exercise_therapeutic_approach": exercise_state.get(
            "exercise_therapeutic_approach"
        ),
    }


__all__ = [
    "GuidedExerciseAction",
    "GuidedExerciseOutcome",
    "GuidedExerciseTransition",
    "InvalidGuidedExerciseState",
    "progress_guided_exercise_transition",
    "start_guided_exercise_transition",
]
