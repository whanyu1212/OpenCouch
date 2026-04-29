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
    "fallback_suggestion_options",
    "get_exercise_definition",
    "get_exercise_display_name",
    "get_exercise_steps",
    "is_valid_exercise_type",
    "iter_exercise_definitions",
    "iter_exercise_selection_aliases",
    "iter_exercise_selectors",
    "voice_exercise_ids",
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
        if not definition.selection_aliases:
            raise ValueError(f"Exercise {definition.id} has no selection aliases")
        if (
            definition.fallback_suggestion_rank is not None
            and definition.fallback_suggestion_rank < 0
        ):
            raise ValueError(
                f"Exercise {definition.id} has a negative fallback suggestion rank"
            )
        for alias in definition.selection_aliases:
            if not alias.strip():
                raise ValueError(f"Exercise {definition.id} has an empty alias")
        for step in definition.steps:
            if not step.prompt_fallback:
                raise ValueError(f"Exercise {definition.id} has an empty step prompt")

    fallback_options = [
        definition.id
        for definition in definitions
        if definition.fallback_suggestion_rank is not None
    ]
    if len(fallback_options) < 2:
        raise ValueError("At least two fallback suggestion exercises are required")


_validate_catalog(ALL_EXERCISE_DEFINITIONS)

_EXERCISE_DEFINITIONS_BY_ID: dict[str, ExerciseDefinition] = {
    definition.id: definition for definition in ALL_EXERCISE_DEFINITIONS
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

_FALLBACK_SUGGESTION_OPTIONS: tuple[str, ...] = tuple(
    definition.id
    for definition in sorted(
        (
            definition
            for definition in ALL_EXERCISE_DEFINITIONS
            if definition.fallback_suggestion_rank is not None
        ),
        key=lambda definition: definition.fallback_suggestion_rank or 0,
    )
)


def iter_exercise_definitions() -> tuple[ExerciseDefinition, ...]:
    """Return all registered exercise definitions in catalog order.

    Returns:
        Tuple of registered exercise definitions.
    """

    return ALL_EXERCISE_DEFINITIONS


def get_exercise_definition(exercise_type: str) -> ExerciseDefinition | None:
    """Return the definition for an exercise type.

    Args:
        exercise_type: Exercise identifier to look up.

    Returns:
        Matching exercise definition, or ``None`` when unregistered.
    """

    return _EXERCISE_DEFINITIONS_BY_ID.get(exercise_type)


def get_exercise_steps(exercise_type: str) -> tuple[ExerciseStep, ...] | None:
    """Return the ordered steps for an exercise type.

    Args:
        exercise_type: Exercise identifier to look up.

    Returns:
        Tuple of exercise steps, or ``None`` when unregistered.
    """

    definition = get_exercise_definition(exercise_type)
    if definition is None:
        return None
    return definition.steps


def get_exercise_display_name(
    exercise_type: str,
    *,
    default: str | None = None,
) -> str:
    """Return the user-facing display name for an exercise type.

    Args:
        exercise_type: Exercise identifier to look up.
        default: Fallback value for unregistered exercise identifiers. When
            omitted, the identifier itself is returned.

    Returns:
        Exercise display name or fallback value.
    """

    definition = get_exercise_definition(exercise_type)
    if definition is None:
        return exercise_type if default is None else default
    return definition.display_name


def is_valid_exercise_type(exercise_type: str | None) -> bool:
    """Return whether an exercise identifier is registered.

    Args:
        exercise_type: Exercise identifier to check.

    Returns:
        ``True`` when the identifier exists in the catalog.
    """

    return exercise_type in _EXERCISE_DEFINITIONS_BY_ID


def fallback_suggestion_options(limit: int = 3) -> tuple[str, ...]:
    """Return deterministic fallback options for broad exercise requests.

    Args:
        limit: Maximum number of fallback options to return.

    Returns:
        Tuple of fallback exercise identifiers ordered by catalog rank.
    """

    if limit <= 0:
        return ()
    return _FALLBACK_SUGGESTION_OPTIONS[:limit]


def iter_exercise_selectors() -> tuple[tuple[tuple[str, ...], str], ...]:
    """Return deterministic exercise selectors in priority order.

    Returns:
        Tuple of ``(keyword_patterns, exercise_type)`` selector groups.
    """

    return _EXERCISE_SELECTORS


def iter_exercise_selection_aliases() -> tuple[tuple[str, str], ...]:
    """Return exercise selection aliases in catalog order.

    Returns:
        Tuple of ``(alias, exercise_type)`` pairs.
    """

    return tuple(
        (alias, definition.id)
        for definition in ALL_EXERCISE_DEFINITIONS
        for alias in definition.selection_aliases
    )


def voice_exercise_ids() -> tuple[str, ...]:
    """Return exercise identifiers marked as suitable for voice delivery.

    Returns:
        Tuple of voice-supported exercise identifiers.
    """

    return _VOICE_EXERCISE_IDS
