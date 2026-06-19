"""Direct unit tests for session-end promotion thresholds.

These helpers (consumed by commit selection) previously had only indirect
coverage via the removed decide_*_candidate_llm_primary tests; this restores
direct coverage of their gate behavior.
"""

from __future__ import annotations

import pytest

from agent.memory.policy.thresholds import (
    should_commit_implicit_procedural_preference,
    should_commit_pattern,
)


@pytest.mark.parametrize(
    ("evidence_count", "expected"),
    [(0, False), (1, False), (2, True), (5, True)],
)
def test_should_commit_pattern_requires_two_evidence_when_repetition_gated(
    evidence_count: int, expected: bool
) -> None:
    assert (
        should_commit_pattern(
            hold_action="require_repetition", evidence_count=evidence_count
        )
        is expected
    )


def test_should_commit_pattern_false_for_non_repetition_hold() -> None:
    # A commit_at_session_end hold is not repetition-gated -> this gate is N/A.
    assert (
        should_commit_pattern(hold_action="commit_at_session_end", evidence_count=9)
        is False
    )


@pytest.mark.parametrize(
    ("evidence_count", "expected"),
    [(0, False), (1, False), (2, True), (4, True)],
)
def test_should_commit_implicit_procedural_requires_two_evidence(
    evidence_count: int, expected: bool
) -> None:
    assert (
        should_commit_implicit_procedural_preference(
            hold_action="commit_at_session_end", evidence_count=evidence_count
        )
        is expected
    )
