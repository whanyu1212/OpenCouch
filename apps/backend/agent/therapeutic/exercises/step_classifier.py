"""Classify progress within an active guided exercise step."""

from __future__ import annotations

import logging
import re
from typing import Any

from agent.state import AgentState
from agent.therapeutic.exercises.registry import get_exercise_display_name
from agent.therapeutic.exercises.types import (
    ExerciseStep,
    ExerciseStepDecision,
    StepState,
)

logger = logging.getLogger(__name__)


# ── Step-state classifier ──────────────────────────────────────────────

# Explicit exit signals. The user wants to STOP the exercise.
_EXIT_PATTERNS: tuple[str, ...] = (
    r"\b(?:stop|quit|cancel|never[\s-]?mind|nvm)\b",
    r"\bi (?:don'?t|do not) (?:want to|wanna)\b",
    r"\b(?:this|it) (?:isn'?t|is not|ain'?t) helping\b",
    r"\b(?:can|could) we just talk\b",
    r"\b(?:i need to|i have to|i should) (?:go|stop|step away)\b",
    r"\bnot (?:in the mood|into this|feeling (?:this|it))\b",
)

_RESUME_PATTERNS: tuple[str, ...] = (
    r"\b(?:go|get|come) back to (?:the )?(?:grounding|exercise|step)\b",
    r"\b(?:resume|continue) (?:the )?(?:grounding|exercise|step)\b",
    r"\blet'?s (?:go|get|come) back\b.{0,40}\b(?:grounding|exercise|step)\b",
)

# Explicit "I can't" stuck signals. The user wants help with the step
# itself but can't engage — the escalation ladder should offer to
# rephrase or (after multiple turns) exit.
_STUCK_PATTERNS: tuple[str, ...] = (
    r"\bi can'?t\b",
    r"\bi (?:don'?t|do not) know\b",
    r"\b(?:nothing|none) (?:comes to mind|stands out)\b",
    r"\b(?:this is|that is|it'?s) (?:stupid|pointless|not working)\b",
    r"\bi (?:am|'?m) stuck\b",
)


# User confirmation patterns. Used for steps where the user does
# something (breathe, visualize, pause) and confirms they did it.
# These are intentionally strict: bare "ok" / "done" are full-message
# matches; longer confirmations require specific phrasing. This avoids
# false positives where "ok" appears mid-sentence (e.g., "ok but this
# isn't working" — which hits STUCK first anyway via pattern priority).
_CONFIRMATION_PATTERNS: tuple[str, ...] = (
    r"^\s*(?:ok|okay|done|yes|yeah|yep|yup|got it|did it|ready|"
    r"finished|mhm|mhmm|alright|sure|done that|did that)\s*[.!]?\s*$",
    r"\b(?:i did|i'?ve done|i'?m done|i'?m ready|done with that|"
    r"i did that|that'?s done)\b",
    r"\b(?:took (?:a |the )?breath|breathed?|exhaled?|inhaled?|"
    r"held it|paused|i (?:can |do )?(?:see|picture|imagine) it)\b",
)


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    """Return whether the text matches any of the patterns.

    Args:
        text: Text to search.
        patterns: Regex patterns to test.

    Returns:
        Whether any pattern matches.
    """

    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def _count_listed_items(message: str) -> int:
    """Count how many distinct items the user listed in their message.

    Used by the step-state classifier to decide whether a counting-based
    step is complete. The heuristic is simple: split on common list
    delimiters (commas, "and", newlines) and count non-trivial tokens.

    Examples:
        "a lamp, a plant, and my coffee cup" → 3
        "I see my keys and a book" → 2
        "um, a plant?" → 1
        "just a chair" → 1
        "" → 0

    This is deliberately lenient — users phrase lists in many different
    ways, and being too strict about the counting pattern means missing
    valid completions. The classifier uses this count + the step's
    ``min_count_for_completion`` to decide COMPLETE vs HOLD.

    Args:
        message: Current user message.

    Returns:
        Heuristic count of listed items in the message.
    """

    if not message.strip():
        return 0

    # Strip filler words and question marks so "um, a plant?" doesn't
    # get counted as 2 items ("um" and "a plant?").
    cleaned = re.sub(
        r"\b(?:um|uh|hmm+|well|maybe|like|just|i see|i hear|i feel|i smell|i taste)\b",
        " ",
        message,
        flags=re.IGNORECASE,
    )

    # Split on commas, semicolons, " and ", " & ", newlines. These are
    # the canonical list separators; anything else is either punctuation
    # noise or a single-item phrase.
    parts = re.split(r"(?:,|;|\band\b|&|\n)", cleaned, flags=re.IGNORECASE)

    # A part counts if it has at least one non-trivial word (length >= 2
    # after stripping whitespace and punctuation).
    non_trivial = 0
    for part in parts:
        stripped = re.sub(r"[^\w\s]", "", part).strip()
        if len(stripped) >= 2:
            non_trivial += 1

    return non_trivial


def _is_clear_completion(message: str, current_step: ExerciseStep) -> bool:
    """Return whether local heuristics clearly show step completion.

    Args:
        message: Current user message.
        current_step: Exercise step the user is responding to.

    Returns:
        ``True`` when deterministic evidence is strong enough to advance
        without consulting the LLM classifier.
    """

    if current_step.completion_mode == "user_confirmation":
        return _matches_any(message, _CONFIRMATION_PATTERNS)

    item_count = _count_listed_items(message)
    return item_count >= current_step.min_count_for_completion


def _classify_step_state(
    message: str,
    current_step: ExerciseStep,
) -> StepState:
    """Classify the user's message as complete / hold / stuck / exit.

    Order of checks matters:
    1. EXIT first — explicit exit signals dominate everything else,
       even if the message happens to name items. If the user says
       "I can't, this isn't helping, let's stop" we exit, period.
    2. STUCK second — explicit "I can't" signals. The user is engaging
       but needs help, not advancement.
    3. COMPLETE third — did the user name at least
       ``min_count_for_completion`` items?
    4. HOLD as the fallback — the user is engaging tentatively or
       sharing something off-step. Give space, don't advance, don't
       escalate.

    Note: HOLD is the "safe default" — if the classifier is uncertain,
    it should HOLD rather than advance or exit. Advancing incorrectly
    rushes the user through a step they didn't finish; exiting
    incorrectly abandons an exercise they were engaging with. Holding
    wastes at most one turn of prompting.

    Args:
        message: Current user message.
        current_step: Exercise step the user is responding to.

    Returns:
        Step-state classification for the current exercise turn.
    """

    if _matches_any(message, _RESUME_PATTERNS):
        return "hold"

    if _matches_any(message, _EXIT_PATTERNS):
        return "exit"

    if _matches_any(message, _STUCK_PATTERNS):
        return "stuck"

    if _is_clear_completion(message, current_step):
        return "complete"

    return "hold"


def _build_step_classifier_prompt(
    *,
    state: AgentState,
    exercise_type: str,
    step_index: int,
    current_step: ExerciseStep,
) -> str:
    """Build the LLM prompt for guided-exercise step classification.

    Args:
        state: Current graph state.
        exercise_type: Active exercise identifier.
        step_index: Current zero-based exercise step.
        current_step: Exercise step the user is responding to.

    Returns:
        Prompt asking for a structured step-state decision.
    """

    message = state.get("message", "")
    exercise_name = get_exercise_display_name(exercise_type)
    return (
        "Classify the user's latest reply to the current guided-exercise step. "
        "Return exactly one step_state:\n"
        "- complete: the user appears to have completed the requested step, "
        "including natural confirmations like 'done that' or equivalent wording.\n"
        "- hold: the user is tentative, partial, off-step, or still engaging but "
        "has not clearly completed the step.\n"
        "- stuck: the user says they cannot do the step, nothing comes to mind, "
        "or the exercise feels confusing/frustrating.\n"
        "- exit: the user wants to stop, cancel, switch away, or just talk.\n\n"
        "If uncertain between complete and hold, choose hold. If the reply "
        "clearly opts out, choose exit.\n\n"
        f"Exercise: {exercise_name} ({exercise_type})\n"
        f"Current step index: {step_index}\n"
        f"Completion mode: {current_step.completion_mode}\n"
        f"Expected item count: {current_step.expected_count}\n"
        f"Minimum item count for completion: {current_step.min_count_for_completion}\n"
        f'Step instruction: "{current_step.prompt_fallback}"\n'
        f'User reply: "{message}"'
    )


async def _classify_step_state_llm_primary(
    *,
    state: AgentState,
    classifier_llm: Any,
    exercise_type: str,
    step_index: int,
    current_step: ExerciseStep,
) -> StepState:
    """Classify step progress with an LLM primary path and regex fallback.

    Args:
        state: Current graph state.
        classifier_llm: Control-plane LLM client, if available.
        exercise_type: Active exercise identifier.
        step_index: Current zero-based exercise step.
        current_step: Exercise step the user is responding to.

    Returns:
        Step-state classification for the current exercise turn.
    """

    message = state.get("message", "")

    # High-precision local overrides keep explicit exits and stuck states
    # immediate even if the classifier is unavailable or permissive.
    if _matches_any(message, _RESUME_PATTERNS):
        return "hold"
    if _matches_any(message, _EXIT_PATTERNS):
        return "exit"
    if _matches_any(message, _STUCK_PATTERNS):
        return "stuck"
    if _is_clear_completion(message, current_step):
        return "complete"

    if classifier_llm is None:
        return _classify_step_state(message, current_step)

    try:
        decision: ExerciseStepDecision = await classifier_llm.generate_structured(
            prompt=_build_step_classifier_prompt(
                state=state,
                exercise_type=exercise_type,
                step_index=step_index,
                current_step=current_step,
            ),
            response_schema=ExerciseStepDecision,
            system_instruction=(
                "You are a strict state classifier for a therapeutic guided "
                "exercise. Do not write user-facing text. Classify only the "
                "latest user reply against the current step."
            ),
        )
        return decision.step_state
    except Exception:
        logger.warning(
            "Guided exercise step classifier failed; using deterministic fallback.",
            exc_info=True,
        )
        return _classify_step_state(message, current_step)
