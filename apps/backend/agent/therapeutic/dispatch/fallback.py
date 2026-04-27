"""Regex-only fallback routing for therapeutic dispatch."""

from __future__ import annotations

from typing import Literal

from agent.therapeutic.dispatch.guards import _matches_any, _word_count
from agent.therapeutic.dispatch.regex_catalog import (
    CLARIFYING_MAX_WORD_COUNT,
    CONFUSION_PATTERNS,
    REFLECTIVE_PATTERNS,
    SELF_REPORT_PATTERNS,
)


def pick_therapeutic_mode(
    message: str,
) -> Literal["supportive", "reflective", "clarifying"]:
    """Pick a fallback therapeutic mode from regex heuristics.

    Args:
        message: The current user message.

    Returns:
        The fallback mode name for regex-only dispatch.
    """

    lowered = message.lower()

    if _matches_any(lowered, REFLECTIVE_PATTERNS):
        return "reflective"

    if _matches_any(lowered, CONFUSION_PATTERNS):
        return "clarifying"

    is_short = _word_count(message) <= CLARIFYING_MAX_WORD_COUNT
    if is_short and not _matches_any(lowered, SELF_REPORT_PATTERNS):
        return "clarifying"

    return "supportive"
