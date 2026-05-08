"""Exercise selection for guided therapeutic exercises."""

from __future__ import annotations

from typing import Any

from agent.conversation import format_recent_history
from agent.state import AgentState
from agent.therapeutic.exercises.registry import (
    available_exercise_definitions,
    iter_exercise_selection_aliases,
)
from agent.therapeutic.exercises.types import (
    ExerciseDefinition,
    ExerciseSelectionDecision,
)


def _available_definitions_for_state(
    state: AgentState,
) -> tuple[ExerciseDefinition, ...]:
    """Return the exercise catalog available for this turn.

    Args:
        state: Current graph state.

    Returns:
        Available exercise definitions in catalog order.
    """

    available_definitions = available_exercise_definitions(
        installed_skills=tuple(state.get("installed_skills") or ()),
        channel=state.get("channel", "text"),
        therapeutic_approach=state.get("therapeutic_approach"),
    )
    return available_definitions


def _available_exercises_for_prompt(
    definitions: tuple[ExerciseDefinition, ...],
) -> str:
    """Return a compact exercise catalog for selection prompts.

    Args:
        definitions: Exercise definitions available for this turn.

    Returns:
        Newline-separated exercise ids, display names, and use cases.
    """

    rows: list[str] = []
    for definition in definitions:
        rows.append(
            "- "
            f"{definition.id}: {definition.display_name} — "
            f"{definition.selection_use_case}"
        )
    return "\n".join(rows)


def _selection_aliases_for_prompt(
    definitions: tuple[ExerciseDefinition, ...],
) -> str:
    """Return a compact, deduplicated alias list for selector prompts.

    Args:
        definitions: Exercise definitions available for this turn.

    Returns:
        Comma-separated exercise aliases in catalog order.
    """

    aliases: list[str] = []
    seen: set[str] = set()
    for alias, _ in iter_exercise_selection_aliases(definitions=definitions):
        normalized = alias.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        aliases.append(alias)
    return ", ".join(aliases)


def _build_exercise_selection_prompt(
    state: AgentState,
    definitions: tuple[ExerciseDefinition, ...],
) -> str:
    """Build the LLM prompt for choosing a guided exercise.

    Args:
        state: Current graph state.
        definitions: Exercise definitions available for this turn.

    Returns:
        Prompt asking for a structured exercise-selection decision.
    """

    recent_history = format_recent_history(state, limit=6, empty="(none)")
    alias_hint = _selection_aliases_for_prompt(definitions)
    return (
        "Choose the single best guided exercise for the user's current need. "
        "If the user explicitly names a supported exercise or technique family "
        f"such as {alias_hint}, treat that as the selected exercise. "
        "If several exercises could fit, choose the best one based on the "
        "current message and recent context. Do not return options or a menu. "
        "Return only a supported exercise_type from the available list.\n\n"
        "Available exercises:\n"
        f"{_available_exercises_for_prompt(definitions)}\n\n"
        "Recent conversation:\n"
        f"{recent_history}\n\n"
        f'Current user message: "{state.get("message", "")}"'
    )


async def _select_exercise_llm_primary(
    state: AgentState,
    *,
    classifier_llm: Any,
) -> str:
    """Select a guided exercise with the control-plane LLM.

    Args:
        state: Current graph state.
        classifier_llm: Control-plane LLM client.

    Returns:
        Selected exercise identifier.

    Raises:
        RuntimeError: If no classifier LLM is configured.
        ValueError: If the classifier returns an unavailable or low-confidence
            exercise.
    """

    if classifier_llm is None:
        raise RuntimeError("Guided exercise selection requires a classifier LLM.")

    available_definitions = _available_definitions_for_state(state)
    decision: ExerciseSelectionDecision = await classifier_llm.generate_structured(
        prompt=_build_exercise_selection_prompt(state, available_definitions),
        response_schema=ExerciseSelectionDecision,
        system_instruction=(
            "You are a strict classifier that selects one supported guided "
            "exercise. Do not write user-facing text. Return structured output "
            "only."
        ),
    )

    available_ids = {definition.id for definition in available_definitions}
    if decision.confidence == "low":
        raise ValueError("Guided exercise classifier returned low confidence.")
    if decision.exercise_type not in available_ids:
        raise ValueError(
            f"Guided exercise classifier returned unavailable exercise "
            f"{decision.exercise_type!r}."
        )
    return decision.exercise_type
