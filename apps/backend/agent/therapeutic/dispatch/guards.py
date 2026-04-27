"""State-aware dispatch guards built on the regex catalog."""

from __future__ import annotations

import re

from agent.state import AgentState
from agent.therapeutic.dispatch.regex_catalog import (
    ACCEPTANCE_PATTERNS,
    ANAPHORIC_GUIDANCE_PATTERNS,
    COPING_ADVICE_REQUEST_PATTERNS,
    EXERCISE_CONSENT_PATTERNS,
    EXERCISE_OFFER_PATTERNS,
    INFORMATIONAL_WALKTHROUGH_NOUN_PATTERN,
    INFORMATIONAL_WALKTHROUGH_PATTERN,
    _ACTIVE_EXERCISE_CLARIFICATION_PATTERNS,
    _BARE_ACKNOWLEDGMENT_PATTERNS,
    _OPEN_QUESTION_PATTERNS,
)


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    """Return whether the text matches any regex pattern.

    Args:
        text: The text to test.
        patterns: The regex patterns to evaluate.

    Returns:
        ``True`` when any pattern matches ``text``.
    """

    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _word_count(text: str) -> int:
    """Count words in a message.

    Args:
        text: The input text to tokenize.

    Returns:
        The number of word-like tokens in ``text``.
    """

    return len([w for w in re.findall(r"\w+", text) if w])


def _has_active_exercise(state: AgentState) -> bool:
    """Return whether an exercise is active in exercise state.

    Args:
        state: The current agent state.

    Returns:
        ``True`` when both ``exercise_type`` and ``exercise_step`` are set.
    """

    exercise_state = state.get("exercise_state", {}) or {}
    return (
        exercise_state.get("exercise_type") is not None
        and exercise_state.get("exercise_step") is not None
    )


def _has_pending_exercise_selection(state: AgentState) -> bool:
    """Return whether the user has pending guided-exercise options.

    Args:
        state: The current agent state.

    Returns:
        ``True`` when the prior guided-exercise turn offered selectable options.
    """

    exercise_state = state.get("exercise_state", {}) or {}
    return bool(exercise_state.get("exercise_selection_options"))


def _looks_like_pending_exercise_choice(message: str) -> bool:
    """Return whether a message looks like an exercise-option choice.

    Args:
        message: The current user message.

    Returns:
        ``True`` for short numeric/ordinal choices or explicit exercise names.
    """

    lowered = message.lower().strip()
    if re.match(r"^(?:option\s*)?[1-9](?:[.)])?\s*$", lowered):
        return True
    return _matches_any(
        lowered,
        (
            r"\b(?:one|two|three|first|second|third)\b",
            r"\b(?:grounding|ground me|5-4-3-2-1)\b",
            r"\b(?:breath|breathe|breathing|box breathing)\b",
            r"\b(?:self.?compassion|compassion break|kinder to myself)\b",
            r"\b(?:thought record|thought check|belief)\b",
            r"\b(?:values|what matters|purpose|compass)\b",
            r"\b(?:gratitude|grateful|thankful)\b",
            r"\b(?:muscle|relaxation|pmr)\b",
            r"\b(?:stop technique|s\.t\.o\.p)\b",
            r"\b(?:improve|overwhelmed|too much)\b",
            r"\b(?:defusion|leaves|let go)\b",
            r"\b(?:behavioral experiment|test this belief)\b",
            r"\b(?:continuum|all.or.nothing)\b",
        ),
    )


def _active_exercise_modality(state: AgentState) -> str | None:
    """Return the pinned modality for an active exercise.

    Args:
        state: The current agent state.

    Returns:
        The exercise modality when present, otherwise the current
        turn's top-level ``therapeutic_approach``.
    """

    exercise_state = state.get("exercise_state", {}) or {}
    modality = exercise_state.get("exercise_modality")
    if modality:
        return modality
    return state.get("therapeutic_approach")


def _is_coping_advice_without_exercise_consent(message: str) -> bool:
    """Return whether a message asks for advice rather than guided practice.

    Args:
        message: The current user message.

    Returns:
        ``True`` when the user is asking for tips/options/strategies
        and has not explicitly opted into a structured exercise.
    """

    lowered = message.lower()
    return _matches_any(
        lowered,
        COPING_ADVICE_REQUEST_PATTERNS,
    ) and not _matches_any(lowered, EXERCISE_CONSENT_PATTERNS)


def _message_is_acceptance_of_offer(state: AgentState, message: str) -> bool:
    """Return whether the prior assistant turn offered an exercise AND the
    current message is a clean direct acceptance of that offer.

    Args:
        state: The current agent state, with conversation ``history``.
        message: The current user message.

    Returns:
        ``True`` when the most recent assistant turn contained an exercise
        offer and the current message matches an acceptance pattern. Returns
        ``False`` when the message is an acknowledgment plus a new question
        (which doesn't end-anchor to acceptance).
    """

    history = state.get("history", []) or []
    for turn in reversed(history):
        if turn.get("role") == "assistant":
            offered = _matches_any(
                turn.get("content", "").lower(), EXERCISE_OFFER_PATTERNS
            )
            return offered and _matches_any(message.lower(), ACCEPTANCE_PATTERNS)
    return False


def _is_bare_ack_to_open_question(state: AgentState, message: str) -> bool:
    """Return whether a bare acknowledgment fails to answer an open question.

    Args:
        state: The current agent state, with conversation ``history``.
        message: The current user message.

    Returns:
        ``True`` when the user says only "yes"/"ok"/similar after the
        assistant asked an open, non-exercise question.
    """

    lowered = message.lower()
    if not _matches_any(lowered, _BARE_ACKNOWLEDGMENT_PATTERNS):
        return False
    if _message_is_acceptance_of_offer(state, message):
        return False

    history = state.get("history", []) or []
    for turn in reversed(history):
        if turn.get("role") != "assistant":
            continue
        assistant_text = turn.get("content", "").lower()
        if _matches_any(assistant_text, EXERCISE_OFFER_PATTERNS):
            return False
        return _matches_any(assistant_text, _OPEN_QUESTION_PATTERNS)
    return False


def _is_active_exercise_clarification(message: str) -> bool:
    """Return whether an active-exercise turn asks about the instruction.

    Args:
        message: Current user message.

    Returns:
        ``True`` for narrow clarification questions about the current exercise
        instruction.
    """

    lowered = message.lower()
    return "?" in lowered and _matches_any(
        lowered, _ACTIVE_EXERCISE_CLARIFICATION_PATTERNS
    )


def _is_advice_request_without_exercise_consent(
    state: AgentState, message: str
) -> bool:
    """Return whether the message is a bare anaphoric advice request OR an
    informational walkthrough, with no consent signal.

    LOAD-BEARING ORDER: the consent check runs BEFORE the trigger checks.
    ``INFORMATIONAL_WALKTHROUGH_NOUN_PATTERN`` deliberately overlaps with
    ``WALKTHROUGH_CONSENT_PATTERN`` on edge cases like
    "walk me through grounding exercise" — disjointness is not enforced at
    the regex level. Routing correctness depends on consent winning when both
    patterns could match.

    Reordering these checks would change observed behavior. See Constraint 7
    in ``UNCONSENTED_EXERCISE_FIX_PLAN.md``.

    Args:
        state: The current agent state, with conversation ``history``.
        message: The current user message.

    Returns:
        ``True`` when an LLM ``guided_exercise`` pick should be rewritten to
        psychoeducation; ``False`` when the LLM's pick should stand.
    """

    lowered = message.lower()

    # Consent FIRST — load-bearing.
    if _matches_any(lowered, EXERCISE_CONSENT_PATTERNS):
        return False
    if _has_active_exercise(state):
        return False
    if _message_is_acceptance_of_offer(state, message):
        return False

    # Trigger conditions: anaphoric guidance OR informational walkthrough.
    if _matches_any(lowered, ANAPHORIC_GUIDANCE_PATTERNS):
        return True
    if _matches_any(
        lowered,
        (INFORMATIONAL_WALKTHROUGH_PATTERN, INFORMATIONAL_WALKTHROUGH_NOUN_PATTERN),
    ):
        return True
    return False
