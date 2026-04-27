"""Exercise selection for guided therapeutic exercises."""

from __future__ import annotations

import logging
import re
from typing import Any

from agent.state import AgentState
from agent.therapeutic.exercises.registry import (
    EXERCISE_5_4_3_2_1,
    EXERCISE_THOUGHT_RECORD,
    _DEFAULT_EXERCISE_OPTIONS,
    _EXERCISE_DISPLAY_NAMES,
    _EXERCISE_REGISTRY,
    _EXERCISE_SELECTORS,
    _EXERCISE_SELECTION_USE_CASES,
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
) -> str:
    """Pick an exercise type based on the current message plus recent context.

    The selector still prioritizes explicit keywords in the current message,
    but short acceptance turns like "can we work through that?" need a small
    amount of recent user-turn context to avoid falling back to generic
    grounding when the prior turn clearly named a cognitive belief pattern.

    Returns the exercise_type constant. Falls back to 5-4-3-2-1
    grounding when no keyword matches — the most established
    exercise and the safest default.

    Args:
        message: Current user message.
        history: Recent conversation history used for short acceptance turns.

    Returns:
        Selected exercise type.
    """

    lowered = message.lower()
    for keywords, exercise_type in _EXERCISE_SELECTORS:
        for kw in keywords:
            if re.search(kw, lowered):
                return exercise_type

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

    return EXERCISE_5_4_3_2_1


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
        if exercise_type in _EXERCISE_REGISTRY and exercise_type not in valid:
            valid.append(exercise_type)
    return tuple(valid[:3])


def _available_exercises_for_prompt() -> str:
    """Return a compact exercise catalog for selection prompts.

    Returns:
        Newline-separated exercise ids, display names, and use cases.
    """

    rows: list[str] = []
    for exercise_type in _EXERCISE_REGISTRY:
        rows.append(
            "- "
            f"{exercise_type}: {_EXERCISE_DISPLAY_NAMES[exercise_type]} — "
            f"{_EXERCISE_SELECTION_USE_CASES[exercise_type]}"
        )
    return "\n".join(rows)


def _build_exercise_selection_prompt(state: AgentState) -> str:
    """Build the LLM prompt for choosing a guided exercise.

    Args:
        state: Current graph state.

    Returns:
        Prompt asking for a structured exercise-selection decision.
    """

    history_lines = []
    for turn in (state.get("history", []) or [])[-6:]:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        if content:
            history_lines.append(f"{role}: {content}")
    recent_history = "\n".join(history_lines) or "(none)"
    return (
        "Choose the best guided exercise for the user's current need. "
        "If the user explicitly names a supported exercise or technique family "
        "such as breathing, box breathing, muscle relaxation, STOP, a thought "
        "record, behavioral experiment, self-compassion, values, defusion, or "
        "gratitude, treat that as a clear selected exercise. "
        "If one exercise is clearly best, return selection_kind='selected' "
        "with that exercise_type. Return selection_kind='ambiguous' only when "
        "multiple exercises are plausible or the user's need is too broad. "
        "When ambiguous, include 2 or 3 option_types the user can choose from. "
        "Do not default to grounding just because the request is broad.\n\n"
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
        option_rows.append(
            f"{index}. {exercise_type}: {_EXERCISE_DISPLAY_NAMES[exercise_type]} - "
            f"{_EXERCISE_SELECTION_USE_CASES[exercise_type]}"
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
    """Select a guided exercise with an LLM primary path.

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

    if classifier_llm is None:
        return ExerciseSelectionResult(
            exercise_type=_select_exercise(
                state.get("message", ""),
                history=state.get("history", []),
            )
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
            exercise_type=None, options=_DEFAULT_EXERCISE_OPTIONS
        )

    if (
        decision.selection_kind == "selected"
        and decision.exercise_type in _EXERCISE_REGISTRY
        and decision.confidence != "low"
    ):
        return ExerciseSelectionResult(exercise_type=decision.exercise_type)

    options = _valid_exercise_options(tuple(decision.option_types))
    if len(options) < 2:
        options = _DEFAULT_EXERCISE_OPTIONS
    return ExerciseSelectionResult(exercise_type=None, options=options)
