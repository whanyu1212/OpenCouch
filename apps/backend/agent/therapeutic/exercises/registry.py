"""Exercise catalog and derived indexes for guided therapeutic exercises."""

from __future__ import annotations

from typing import Any

from agent.therapeutic.exercises.definitions.act_values import (
    DEFINITIONS as ACT_VALUES_DEFINITIONS,
    EXERCISE_LEAVES_ON_STREAM,
    EXERCISE_VALUES_COMPASS,
)
from agent.therapeutic.exercises.definitions.activation import (
    DEFINITIONS as ACTIVATION_DEFINITIONS,
    EXERCISE_TINY_ACTION,
)
from agent.therapeutic.exercises.definitions.emotion_regulation import (
    DEFINITIONS as EMOTION_REGULATION_DEFINITIONS,
    EXERCISE_GRATITUDE,
    EXERCISE_IMPROVE,
    EXERCISE_SELF_COMPASSION,
)
from agent.therapeutic.exercises.definitions.grounding import (
    DEFINITIONS as GROUNDING_DEFINITIONS,
    EXERCISE_5_4_3_2_1,
    EXERCISE_BOX_BREATHING,
    EXERCISE_MUSCLE_RELAXATION,
    EXERCISE_STOP_TECHNIQUE,
)
from agent.therapeutic.exercises.definitions.thought_work import (
    DEFINITIONS as THOUGHT_WORK_DEFINITIONS,
    EXERCISE_BEHAVIORAL_EXPERIMENT,
    EXERCISE_CONTINUUM,
    EXERCISE_THOUGHT_RECORD,
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
    "available_exercise_definitions",
    "get_exercise_definition",
    "get_exercise_display_name",
    "get_exercise_steps",
    "iter_exercise_definitions",
    "iter_exercise_selection_aliases",
    "voice_exercise_ids",
]


ALL_EXERCISE_DEFINITIONS: tuple[ExerciseDefinition, ...] = (
    *GROUNDING_DEFINITIONS,
    *THOUGHT_WORK_DEFINITIONS,
    *ACTIVATION_DEFINITIONS,
    *ACT_VALUES_DEFINITIONS,
    *EMOTION_REGULATION_DEFINITIONS,
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
        if definition.version < 1:
            raise ValueError(f"Exercise {definition.id} has invalid version")
        if not definition.category.strip():
            raise ValueError(f"Exercise {definition.id} has empty category")
        if definition.duration_seconds is not None and definition.duration_seconds < 1:
            raise ValueError(f"Exercise {definition.id} has invalid duration")
        if not definition.channels:
            raise ValueError(f"Exercise {definition.id} has no supported channels")
        if (
            definition.required_skill is not None
            and not definition.required_skill.strip()
        ):
            raise ValueError(f"Exercise {definition.id} has an empty required skill")
        for alias in definition.selection_aliases:
            if not alias.strip():
                raise ValueError(f"Exercise {definition.id} has an empty alias")
        for tag in definition.tags:
            if not tag.strip():
                raise ValueError(f"Exercise {definition.id} has an empty tag")
        for approach in definition.approaches:
            if not approach.strip():
                raise ValueError(f"Exercise {definition.id} has an empty approach")
        for channel in definition.channels:
            if channel not in {"text", "voice"}:
                raise ValueError(
                    f"Exercise {definition.id} has unsupported channel {channel!r}"
                )
        step_ids = [step.id for step in definition.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError(f"Exercise {definition.id} has duplicate step ids")
        for step in definition.steps:
            if not step.id.strip():
                raise ValueError(f"Exercise {definition.id} has an empty step id")
            if not step.instruction.strip():
                raise ValueError(
                    f"Exercise {definition.id} has an empty step instruction"
                )
            if step.completion_mode == "items":
                if step.min_items is None or step.min_items < 1:
                    raise ValueError(
                        f"Exercise {definition.id} has an item step without min_items"
                    )
                if step.target_items is not None and step.target_items < step.min_items:
                    raise ValueError(
                        f"Exercise {definition.id} has target_items below min_items"
                    )
            elif step.min_items is not None or step.target_items is not None:
                raise ValueError(
                    f"Exercise {definition.id} has item counts on a "
                    f"{step.completion_mode!r} step"
                )
            if step.completion_mode == "llm_judged" and not step.completion_criteria:
                raise ValueError(
                    f"Exercise {definition.id} has an LLM-judged step without "
                    f"completion criteria"
                )


_validate_catalog(ALL_EXERCISE_DEFINITIONS)

_EXERCISE_DEFINITIONS_BY_ID: dict[str, ExerciseDefinition] = {
    definition.id: definition for definition in ALL_EXERCISE_DEFINITIONS
}

_VOICE_EXERCISE_IDS: tuple[str, ...] = tuple(
    definition.id
    for definition in ALL_EXERCISE_DEFINITIONS
    if definition.voice_supported
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


def _normalize_channel(channel: Any) -> str:
    """Normalize external channel values to exercise delivery modes.

    Args:
        channel: Raw channel value from graph state or a caller.

    Returns:
        ``"voice"`` for voice turns; ``"text"`` for all text-like channels.
    """

    value = getattr(channel, "value", channel)
    return "voice" if value == "voice" else "text"


def available_exercise_definitions(
    *,
    installed_skills: list[str] | tuple[str, ...] = (),
    channel: Any = "text",
    therapeutic_approach: str | None = None,
    definitions: tuple[ExerciseDefinition, ...] = ALL_EXERCISE_DEFINITIONS,
) -> tuple[ExerciseDefinition, ...]:
    """Return exercise definitions available for the current capabilities.

    Args:
        installed_skills: Capability keys available to this user/session.
        channel: Raw graph channel or exercise delivery mode.
        therapeutic_approach: Current therapeutic approach selected by routing.
        definitions: Catalog to filter. Defaults to the full registry.

    Returns:
        Tuple of available exercise definitions in catalog order.
    """

    skill_set = set(installed_skills)
    delivery_mode = _normalize_channel(channel)
    return tuple(
        definition
        for definition in definitions
        if _is_definition_available(
            definition,
            installed_skills=skill_set,
            delivery_mode=delivery_mode,
            therapeutic_approach=therapeutic_approach,
        )
    )


def _is_definition_available(
    definition: ExerciseDefinition,
    *,
    installed_skills: set[str],
    delivery_mode: str,
    therapeutic_approach: str | None,
) -> bool:
    """Return whether one exercise definition is available.

    Args:
        definition: Exercise definition to evaluate.
        installed_skills: Capability keys available to this user/session.
        delivery_mode: Normalized exercise delivery mode.
        therapeutic_approach: Current therapeutic approach selected by routing.

    Returns:
        ``True`` when the definition can be offered.
    """

    if (
        definition.required_skill is not None
        and definition.required_skill not in installed_skills
    ):
        return False

    if delivery_mode == "voice":
        if not (definition.voice_supported or "voice" in definition.channels):
            return False
    elif delivery_mode not in definition.channels:
        return False

    if definition.approaches and therapeutic_approach not in definition.approaches:
        return False

    return True


def iter_exercise_selection_aliases(
    *,
    definitions: tuple[ExerciseDefinition, ...] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return exercise selection aliases in catalog order.

    Args:
        definitions: Optional catalog to read aliases from. Defaults to all
            registered exercises.

    Returns:
        Tuple of ``(alias, exercise_type)`` pairs.
    """

    catalog = ALL_EXERCISE_DEFINITIONS if definitions is None else definitions
    return tuple(
        (alias, definition.id)
        for definition in catalog
        for alias in definition.selection_aliases
    )


def voice_exercise_ids() -> tuple[str, ...]:
    """Return exercise identifiers marked as suitable for voice delivery.

    Returns:
        Tuple of voice-supported exercise identifiers.
    """

    return _VOICE_EXERCISE_IDS
