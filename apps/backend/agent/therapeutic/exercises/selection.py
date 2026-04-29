"""Exercise selection for guided therapeutic exercises."""

from __future__ import annotations

import logging
import re
from typing import Any

from agent.conversation import format_recent_history, get_recent_history
from agent.state import AgentState
from agent.therapeutic.exercises.registry import (
    EXERCISE_5_4_3_2_1,
    EXERCISE_THOUGHT_RECORD,
    fallback_suggestion_options,
    get_exercise_definition,
    is_valid_exercise_type,
    iter_exercise_definitions,
    iter_exercise_selection_aliases,
    iter_exercise_selectors,
)
from agent.therapeutic.exercises.types import (
    ExerciseOptionChoiceDecision,
    ExerciseSelectionDecision,
    ExerciseSelectionResult,
)

logger = logging.getLogger(__name__)


# ── Exercise selection ─────────────────────────────────────────────────


def _select_exercise(
    message: str,
    *,
    history: list[dict[str, str]] | None = None,
) -> str | None:
    """Pick an exercise type based on the current message plus recent context.

    The selector still prioritizes explicit keywords in the current message,
    but short acceptance turns like "can we work through that?" need a small
    amount of recent user-turn context to avoid falling back to generic
    grounding when the prior turn clearly named a cognitive belief pattern.

    Returns the exercise_type constant when there is deterministic evidence.
    Returns ``None`` when no keyword or context marker matches; callers choose
    their own fallback behavior.

    Args:
        message: Current user message.
        history: Recent conversation history used for short acceptance turns.

    Returns:
        Selected exercise type, or ``None`` when no deterministic match exists.
    """

    lowered = message.lower()
    for keywords, exercise_type in iter_exercise_selectors():
        for kw in keywords:
            if re.search(kw, lowered):
                return exercise_type

    if _matches_selection_alias(message, EXERCISE_5_4_3_2_1):
        return EXERCISE_5_4_3_2_1

    if re.search(r"\bwork through (?:that|this|it)\b", lowered):
        recent_user_text = " ".join(
            turn.get("content", "")
            for turn in (history or [])[-6:]
            if turn.get("role") == "user" and turn.get("content")
        ).lower()
        if any(
            marker in recent_user_text
            for marker in (
                "i always assume",
                "one mistake means",
                "i'm incompetent",
                "im incompetent",
                "everyone will see",
                "i'm about to fail",
                "im about to fail",
                "this belief",
                "this thought",
            )
        ):
            return EXERCISE_THOUGHT_RECORD

    return None


def _matches_selection_alias(message: str, exercise_type: str) -> bool:
    """Return whether a message contains a catalog alias for an exercise.

    Args:
        message: Current user message.
        exercise_type: Exercise identifier whose aliases should be checked.

    Returns:
        ``True`` when a normalized alias appears in the normalized message.
    """

    definition = get_exercise_definition(exercise_type)
    if definition is None:
        return False

    normalized = f" {_normalize_selection_text(message)} "
    for alias in definition.selection_aliases:
        normalized_alias = _normalize_selection_text(alias)
        if normalized_alias and f" {normalized_alias} " in normalized:
            return True
    return False


def _normalize_selection_text(text: str) -> str:
    """Normalize selection text for alias containment checks.

    Args:
        text: User text or catalog alias.

    Returns:
        Lowercase alphanumeric tokens separated by single spaces.
    """

    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _valid_exercise_options(
    option_types: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    """Return valid unique exercise options in display order.

    Args:
        option_types: Exercise identifiers proposed by a classifier.

    Returns:
        Deduplicated exercise identifiers that exist in the registry.
    """

    valid: list[str] = []
    for exercise_type in option_types:
        if is_valid_exercise_type(exercise_type) and exercise_type not in valid:
            valid.append(exercise_type)
    return tuple(valid[:3])


def _available_exercises_for_prompt() -> str:
    """Return a compact exercise catalog for selection prompts.

    Returns:
        Newline-separated exercise ids, display names, and use cases.
    """

    rows: list[str] = []
    for definition in iter_exercise_definitions():
        rows.append(
            "- "
            f"{definition.id}: {definition.display_name} — "
            f"{definition.selection_use_case}"
        )
    return "\n".join(rows)


def _selection_aliases_for_prompt() -> str:
    """Return a compact, deduplicated alias list for selector prompts.

    Returns:
        Comma-separated exercise aliases in catalog order.
    """

    aliases: list[str] = []
    seen: set[str] = set()
    for alias, _ in iter_exercise_selection_aliases():
        normalized = alias.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        aliases.append(alias)
    return ", ".join(aliases)


def _build_exercise_selection_prompt(state: AgentState) -> str:
    """Build the LLM prompt for choosing a guided exercise.

    Args:
        state: Current graph state.

    Returns:
        Prompt asking for a structured exercise-selection decision.
    """

    recent_history = format_recent_history(state, limit=6, empty="(none)")
    alias_hint = _selection_aliases_for_prompt()
    return (
        "Choose the best guided exercise for the user's current need. "
        "If the user explicitly names a supported exercise or technique family "
        f"such as {alias_hint}, treat that as a clear selected exercise. "
        "If one exercise is clearly best, return selection_kind='selected' "
        "with that exercise_type. Return selection_kind='ambiguous' only when "
        "multiple exercises are plausible or the user's need is too broad. "
        "When ambiguous, include 2 or 3 option_types the user can choose from. "
        "Choose ambiguous options based on the user's current need and recent "
        "context, not a fixed menu. Do not default to grounding just because "
        "the request is broad.\n\n"
        "Available exercises:\n"
        f"{_available_exercises_for_prompt()}\n\n"
        "Recent conversation:\n"
        f"{recent_history}\n\n"
        f'Current user message: "{state.get("message", "")}"'
    )


def _resolve_pending_exercise_choice(
    message: str,
    options: tuple[str, ...],
) -> str | None:
    """Resolve a user's follow-up choice from pending exercise options.

    Args:
        message: Current user message.
        options: Pending exercise identifiers offered on the prior turn.

    Returns:
        Selected exercise identifier, or ``None`` if the message is not a
        clear choice.
    """

    lowered = message.lower().strip()
    match = re.match(r"^(?:option\s*)?(?P<index>[1-9])(?:[.)])?\b", lowered)
    if match is not None:
        index = int(match.group("index")) - 1
        if 0 <= index < len(options):
            return options[index]

    ordinal_map = {
        "one": 0,
        "first": 0,
        "two": 1,
        "second": 1,
        "three": 2,
        "third": 2,
    }
    for word, index in ordinal_map.items():
        if re.search(rf"\b{word}\b", lowered) and 0 <= index < len(options):
            return options[index]

    selected = _select_exercise(message)
    if selected in options:
        return selected
    return None


def _build_pending_exercise_choice_prompt(
    message: str,
    options: tuple[str, ...],
) -> str:
    """Build the LLM prompt for resolving a pending exercise choice.

    Args:
        message: Current user message.
        options: Pending exercise options from the prior assistant turn.

    Returns:
        Prompt asking for a structured option-choice decision.
    """

    option_rows = []
    for index, exercise_type in enumerate(options, start=1):
        definition = get_exercise_definition(exercise_type)
        if definition is None:
            continue
        option_rows.append(
            f"{index}. {exercise_type}: {definition.display_name} - "
            f"{definition.selection_use_case}"
        )
    return (
        "The assistant previously offered these guided-exercise options. "
        "Decide whether the user's latest reply chooses exactly one of them. "
        "Use choice_kind='unclear' when the reply asks a question, rejects the "
        "options, or does not clearly select one.\n\n"
        "Options:\n"
        f"{chr(10).join(option_rows)}\n\n"
        f'User reply: "{message}"'
    )


async def _resolve_pending_exercise_choice_llm_primary(
    message: str,
    options: tuple[str, ...],
    *,
    classifier_llm: Any,
) -> str | None:
    """Resolve a pending exercise-option choice with an LLM primary path.

    Args:
        message: Current user message.
        options: Pending exercise options from the prior assistant turn.
        classifier_llm: Control-plane LLM client, if configured.

    Returns:
        Selected exercise identifier, or ``None`` when the choice is unclear.
    """

    if classifier_llm is None:
        return None

    try:
        decision: ExerciseOptionChoiceDecision = (
            await classifier_llm.generate_structured(
                prompt=_build_pending_exercise_choice_prompt(message, options),
                response_schema=ExerciseOptionChoiceDecision,
                system_instruction=(
                    "You are a strict classifier for a pending guided-exercise "
                    "option choice. Return structured output only."
                ),
            )
        )
    except Exception:
        logger.warning(
            "Guided exercise option-choice classifier failed; keeping options.",
            exc_info=True,
        )
        return None

    if (
        decision.choice_kind == "selected"
        and decision.exercise_type in options
        and decision.confidence != "low"
    ):
        return decision.exercise_type
    return None


async def _select_exercise_llm_primary(
    state: AgentState,
    *,
    classifier_llm: Any,
) -> ExerciseSelectionResult:
    """Select a guided exercise with deterministic guards and an LLM path.

    Args:
        state: Current graph state.
        classifier_llm: Control-plane LLM client, if available.

    Returns:
        Exercise selection result. ``exercise_type`` is set when the node should
        start an exercise immediately; otherwise ``options`` contains choices
        to offer the user.
    """

    exercise_state = state.get("exercise_state", {}) or {}
    pending_options = _valid_exercise_options(
        tuple(exercise_state.get("exercise_selection_options") or ())
    )
    if pending_options:
        choice = _resolve_pending_exercise_choice(
            state.get("message", ""), pending_options
        )
        if choice is not None:
            return ExerciseSelectionResult(exercise_type=choice)
        choice = await _resolve_pending_exercise_choice_llm_primary(
            state.get("message", ""),
            pending_options,
            classifier_llm=classifier_llm,
        )
        if choice is not None:
            return ExerciseSelectionResult(exercise_type=choice)
        return ExerciseSelectionResult(exercise_type=None, options=pending_options)

    selected = _select_exercise(
        state.get("message", ""),
        history=get_recent_history(state, limit=6),
    )
    if selected is not None:
        return ExerciseSelectionResult(exercise_type=selected)

    if classifier_llm is None:
        return ExerciseSelectionResult(
            exercise_type=None,
            options=fallback_suggestion_options(),
        )

    try:
        decision: ExerciseSelectionDecision = await classifier_llm.generate_structured(
            prompt=_build_exercise_selection_prompt(state),
            response_schema=ExerciseSelectionDecision,
            system_instruction=(
                "You are a strict classifier that selects among supported "
                "guided exercises. Do not write user-facing text. Return a "
                "structured selection only."
            ),
        )
    except Exception:
        logger.warning(
            "Guided exercise selection classifier failed; offering choices.",
            exc_info=True,
        )
        return ExerciseSelectionResult(
            exercise_type=None,
            options=fallback_suggestion_options(),
        )

    if (
        decision.selection_kind == "selected"
        and is_valid_exercise_type(decision.exercise_type)
        and decision.confidence != "low"
    ):
        return ExerciseSelectionResult(exercise_type=decision.exercise_type)

    options = _valid_exercise_options(tuple(decision.option_types))
    if len(options) < 2:
        options = fallback_suggestion_options()
    return ExerciseSelectionResult(exercise_type=None, options=options)
