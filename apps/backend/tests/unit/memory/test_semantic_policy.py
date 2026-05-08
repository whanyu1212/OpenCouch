"""Unit tests for shared semantic-memory heuristics."""

from __future__ import annotations

from agent.memory.policy.semantic import (
    contains_emerging_pattern,
    contains_negative_self_belief,
    has_durability_marker,
    looks_transient_context,
)


def test_contains_negative_self_belief_detects_extended_marker() -> None:
    assert contains_negative_self_belief("I never get it right at work.") is True


def test_contains_emerging_pattern_detects_shared_pattern_marker() -> None:
    assert (
        contains_emerging_pattern("This always happens when I meet new people.") is True
    )


def test_has_durability_marker_detects_long_term_language() -> None:
    assert has_durability_marker("This has been true for years.") is True


def test_looks_transient_context_requires_recent_marker_without_durability() -> None:
    assert looks_transient_context("This week work has been rough.") is True
    assert looks_transient_context("For years, work has been rough.") is False
