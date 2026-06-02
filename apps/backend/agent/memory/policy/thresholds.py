"""Promotion threshold helpers for held memory candidates."""

from __future__ import annotations

from agent.memory.policy.candidates import ProceduralHoldAction, SemanticHoldAction


def should_commit_pattern(
    *,
    hold_action: SemanticHoldAction,
    evidence_count: int,
) -> bool:
    """Return whether a repetition-gated semantic candidate can promote.

    Args:
        hold_action (SemanticHoldAction): Policy action that held the candidate.
        evidence_count (int): Number of reinforcing occurrences seen so far.

    Returns:
        bool: ``True`` when the candidate is repetition-gated and has enough evidence.
    """

    if hold_action != "require_repetition":
        return False
    return evidence_count >= 2


def should_commit_implicit_procedural_preference(
    *,
    hold_action: ProceduralHoldAction,
    evidence_count: int,
) -> bool:
    """Return whether a buffered implicit procedural preference can promote.

    Args:
        hold_action (ProceduralHoldAction): Policy action that held the candidate.
        evidence_count (int): Number of reinforcing occurrences seen so far.

    Returns:
        bool: ``True`` when the held preference has enough evidence to promote.
    """

    if hold_action != "commit_at_session_end":
        return False
    return evidence_count >= 2


__all__ = [
    "should_commit_implicit_procedural_preference",
    "should_commit_pattern",
]
