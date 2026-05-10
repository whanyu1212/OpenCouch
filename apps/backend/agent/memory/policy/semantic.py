"""Shared semantic-memory heuristics for candidate promotion and hard guards.

This module centralizes the semantic marker vocabulary used by the
candidate builder and hard safety/storage guards so those paths do not
drift.
"""

from __future__ import annotations

from agent.memory.policy.constants import contains_any as _contains_any

SEMANTIC_STABLE_CATEGORIES: frozenset[str] = frozenset(
    {
        "relationship",
        "preference",
        "coping_strategy",
        "goal",
    }
)

SEMANTIC_SESSION_ONLY_CATEGORIES: frozenset[str] = frozenset(
    {
        "loss",
        "trigger",
    }
)

NEGATIVE_SELF_BELIEF_MARKERS: tuple[str, ...] = (
    "i always assume",
    "everyone will see i'm",
    "everyone will see im",
    "everyone will think i'm",
    "everyone will think im",
    "one mistake means",
    "i'm incompetent",
    "im incompetent",
    "i'm a failure",
    "im a failure",
    "i always fail",
    "i never get it right",
)

EMERGING_PATTERN_MARKERS: tuple[str, ...] = (
    "it keeps happening",
    "every new task makes me feel like",
    "every task makes me feel like",
    "i'm about to fail",
    "im about to fail",
    "every relationship ends",
    "this always happens",
)

DURABILITY_MARKERS: tuple[str, ...] = (
    "for years",
    "for a long time",
    "i always",
    "i usually",
    "every time",
    "whenever",
    "ever since",
)

TRANSIENT_MARKERS: tuple[str, ...] = (
    "today",
    "tonight",
    "right now",
    "this week",
    "this month",
    "this morning",
    "last night",
    "yesterday",
    "lately",
    "recently",
)


def contains_negative_self_belief(text: str) -> bool:
    """Return whether text contains negative self-belief language.

    Args:
        text (str): Text to inspect.

    Returns:
        bool: ``True`` when any negative self-belief marker is present.
    """

    return _contains_any(text.lower(), NEGATIVE_SELF_BELIEF_MARKERS)


def contains_emerging_pattern(text: str) -> bool:
    """Return whether text contains early emerging-pattern language.

    Args:
        text (str): Text to inspect.

    Returns:
        bool: ``True`` when any emerging-pattern marker is present.
    """

    return _contains_any(text.lower(), EMERGING_PATTERN_MARKERS)


def has_durability_marker(text: str) -> bool:
    """Return whether text contains durability language.

    Args:
        text (str): Text to inspect.

    Returns:
        bool: ``True`` when any durability marker is present.
    """

    return _contains_any(text.lower(), DURABILITY_MARKERS)


def looks_transient_context(text: str) -> bool:
    """Return whether a context statement looks recent or one-off.

    Args:
        text (str): Context text to inspect.

    Returns:
        bool: ``True`` when the text looks transient rather than durable.
    """

    lowered = text.lower()
    return _contains_any(lowered, TRANSIENT_MARKERS) and not _contains_any(
        lowered, DURABILITY_MARKERS
    )
