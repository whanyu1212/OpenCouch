"""Exercise catalog and derived indexes for guided therapeutic exercises."""

from __future__ import annotations

from agent.therapeutic.exercises.definitions.act_values import (
    EXERCISE_LEAVES_ON_STREAM,
    EXERCISE_VALUES_COMPASS,
    LEAVES_ON_STREAM_DEFINITION,
    VALUES_COMPASS_DEFINITION,
)
from agent.therapeutic.exercises.definitions.activation import (
    EXERCISE_TINY_ACTION,
    TINY_ACTION_DEFINITION,
)
from agent.therapeutic.exercises.definitions.emotion_regulation import (
    EXERCISE_GRATITUDE,
    EXERCISE_IMPROVE,
    EXERCISE_SELF_COMPASSION,
    GRATITUDE_DEFINITION,
    IMPROVE_DEFINITION,
    SELF_COMPASSION_DEFINITION,
)
from agent.therapeutic.exercises.definitions.grounding import (
    BOX_BREATHING_DEFINITION,
    EXERCISE_5_4_3_2_1,
    EXERCISE_BOX_BREATHING,
    EXERCISE_MUSCLE_RELAXATION,
    EXERCISE_STOP_TECHNIQUE,
    GROUNDING_5_4_3_2_1_DEFINITION,
    MUSCLE_RELAXATION_DEFINITION,
    STOP_TECHNIQUE_DEFINITION,
)
from agent.therapeutic.exercises.definitions.thought_work import (
    BEHAVIORAL_EXPERIMENT_DEFINITION,
    CONTINUUM_DEFINITION,
    EXERCISE_BEHAVIORAL_EXPERIMENT,
    EXERCISE_CONTINUUM,
    EXERCISE_THOUGHT_RECORD,
    THOUGHT_RECORD_DEFINITION,
)
from agent.therapeutic.exercises.types import ExerciseDefinition, ExerciseStep

__all__ = [
    "ALL_EXERCISE_DEFINITIONS",
    "EXERCISE_5_4_3_2_1",
    "EXERCISE_BEHAVIORAL_EXPERIMENT",
    "EXERCISE_BOX_BREATHING",
    "EXERCISE_CONTINUUM",
    "EXERCISE_GRATITUDE",
    "EXERCISE_IMPROVE",
    "EXERCISE_LEAVES_ON_STREAM",
    "EXERCISE_MUSCLE_RELAXATION",
    "EXERCISE_SELF_COMPASSION",
    "EXERCISE_STOP_TECHNIQUE",
    "EXERCISE_THOUGHT_RECORD",
    "EXERCISE_TINY_ACTION",
    "EXERCISE_VALUES_COMPASS",
    "_DEFAULT_EXERCISE_OPTIONS",
    "_EXERCISE_DEFINITIONS_BY_ID",
    "_EXERCISE_DISPLAY_NAMES",
    "_EXERCISE_REGISTRY",
    "_EXERCISE_SELECTION_USE_CASES",
    "_EXERCISE_SELECTORS",
    "_VOICE_EXERCISE_IDS",
]


ALL_EXERCISE_DEFINITIONS: tuple[ExerciseDefinition, ...] = (
    GROUNDING_5_4_3_2_1_DEFINITION,
    BOX_BREATHING_DEFINITION,
    STOP_TECHNIQUE_DEFINITION,
    THOUGHT_RECORD_DEFINITION,
    TINY_ACTION_DEFINITION,
    LEAVES_ON_STREAM_DEFINITION,
    MUSCLE_RELAXATION_DEFINITION,
    BEHAVIORAL_EXPERIMENT_DEFINITION,
    SELF_COMPASSION_DEFINITION,
    IMPROVE_DEFINITION,
    VALUES_COMPASS_DEFINITION,
    GRATITUDE_DEFINITION,
    CONTINUUM_DEFINITION,
)

_DEFAULT_EXERCISE_OPTIONS: tuple[str, ...] = (
    EXERCISE_5_4_3_2_1,
    EXERCISE_BOX_BREATHING,
    EXERCISE_SELF_COMPASSION,
)


def _validate_catalog(definitions: tuple[ExerciseDefinition, ...]) -> None:
    """Validate exercise catalog integrity at import time.

    Args:
        definitions: Exercise definitions to validate.

    Returns:
        None.
    """

    ids = [definition.id for definition in definitions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate exercise ids in guided exercise catalog")

    for definition in definitions:
        if not definition.id:
            raise ValueError("Exercise definition has empty id")
        if not definition.display_name:
            raise ValueError(f"Exercise {definition.id} has empty display name")
        if not definition.selection_use_case:
            raise ValueError(f"Exercise {definition.id} has empty use case")
        if not definition.steps:
            raise ValueError(f"Exercise {definition.id} has no steps")
        for step in definition.steps:
            if not step.prompt_fallback:
                raise ValueError(f"Exercise {definition.id} has an empty step prompt")

    missing_defaults = set(_DEFAULT_EXERCISE_OPTIONS).difference(ids)
    if missing_defaults:
        raise ValueError(
            f"Default exercise options are not registered: {missing_defaults}"
        )


_validate_catalog(ALL_EXERCISE_DEFINITIONS)

_EXERCISE_DEFINITIONS_BY_ID: dict[str, ExerciseDefinition] = {
    definition.id: definition for definition in ALL_EXERCISE_DEFINITIONS
}

_EXERCISE_REGISTRY: dict[str, tuple[ExerciseStep, ...]] = {
    definition.id: definition.steps for definition in ALL_EXERCISE_DEFINITIONS
}

_EXERCISE_DISPLAY_NAMES: dict[str, str] = {
    definition.id: definition.display_name for definition in ALL_EXERCISE_DEFINITIONS
}

_EXERCISE_SELECTION_USE_CASES: dict[str, str] = {
    definition.id: definition.selection_use_case
    for definition in ALL_EXERCISE_DEFINITIONS
}

_EXERCISE_SELECTORS: tuple[tuple[tuple[str, ...], str], ...] = tuple(
    (selector_group.keywords, definition.id)
    for definition, selector_group in sorted(
        (
            (definition, selector_group)
            for definition in ALL_EXERCISE_DEFINITIONS
            for selector_group in definition.selector_groups
        ),
        key=lambda item: item[1].priority,
    )
)

_VOICE_EXERCISE_IDS: tuple[str, ...] = tuple(
    definition.id
    for definition in ALL_EXERCISE_DEFINITIONS
    if definition.voice_supported
)
