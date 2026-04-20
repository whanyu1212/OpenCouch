"""Deterministic write policy for hot-path memory persistence.

The LLM extractors still decide whether something is *candidate*
memory. This module decides whether that candidate is safe and durable
enough to commit immediately. Anything that is not ``commit_now`` is
buffered or dropped by the caller; session-end promotion logic lives in
``agent.nodes.commit_session_memory``.
"""

from __future__ import annotations

from agent.memory.candidates import (
    PolicyDecision,
    ProceduralCandidate,
    SemanticCandidate,
)
from agent.memory.constants import (
    PROCEDURAL_EXPLICIT_REQUEST_MARKERS as _PROCEDURAL_EXPLICIT_REQUEST_MARKERS,
    PROCEDURAL_SAFETY_CONFLICT_MARKERS as _PROCEDURAL_SAFETY_CONFLICT_MARKERS,
    PROCEDURAL_TURN_SCOPED_MARKERS as _PROCEDURAL_TURN_SCOPED_MARKERS,
    contains_any as _contains_any,
)

_SEMANTIC_STABLE_CATEGORIES = {
    "relationship",
    "preference",
    "coping_strategy",
    "goal",
}

_SEMANTIC_SESSION_ONLY_CATEGORIES = {
    "loss",
    "trigger",
}

_SEMANTIC_ONE_OFF_MARKERS = (
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

_NEGATIVE_SELF_BELIEF_MARKERS = (
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

_EMERGING_PATTERN_MARKERS = (
    "it keeps happening",
    "every new task makes me feel like",
    "every task makes me feel like",
    "i'm about to fail",
    "im about to fail",
    "every relationship ends",
    "this always happens",
)


def _lowered_texts(*values: str) -> str:
    return " ".join(v.lower() for v in values if v).strip()


def should_drop_candidate(action: str) -> bool:
    """Return whether a policy action means "do not persist in phase 1"."""

    return action != "commit_now"


def should_commit_pattern(candidate: SemanticCandidate, evidence_count: int) -> bool:
    """Return whether a repetition-gated semantic candidate can promote."""

    if candidate.policy_recommendation != "require_repetition":
        return False
    return evidence_count >= 2


def should_commit_implicit_procedural_preference(
    candidate: ProceduralCandidate,
    evidence_count: int,
) -> bool:
    """Return whether a buffered implicit procedural preference can promote."""

    if candidate.policy_recommendation != "commit_at_session_end":
        return False
    if candidate.explicitness == "explicit":
        return False
    return evidence_count >= 2


def decide_semantic_candidate(candidate: SemanticCandidate) -> PolicyDecision:
    """Return the deterministic write decision for a semantic candidate."""

    category = candidate.payload.category
    predicate = candidate.payload.predicate
    text = _lowered_texts(
        candidate.reason,
        candidate.payload.evidence_quote,
        candidate.payload.subject.identifier,
        candidate.payload.object.identifier,
    )

    if _contains_any(text, _NEGATIVE_SELF_BELIEF_MARKERS) or _contains_any(
        text, _EMERGING_PATTERN_MARKERS
    ):
        return PolicyDecision(
            action="require_repetition",
            reason="negative self-belief or emerging pattern requires repetition",
        )

    if candidate.sensitivity == "high" or category in _SEMANTIC_SESSION_ONLY_CATEGORIES:
        return PolicyDecision(
            action="commit_at_session_end",
            reason="high-sensitivity semantic candidate should not commit immediately",
        )

    if candidate.scope == "turn" or candidate.durability == "transient":
        return PolicyDecision(
            action="drop",
            reason="turn-scoped or transient semantic candidate should not persist",
        )

    if predicate == "MENTIONED_IN":
        return PolicyDecision(
            action="drop",
            reason="provenance predicates should not become durable semantic memory",
        )

    if category in _SEMANTIC_STABLE_CATEGORIES:
        return PolicyDecision(
            action="commit_now",
            reason="explicit stable semantic fact is safe for immediate commit",
        )

    if category == "context" and not _contains_any(text, _SEMANTIC_ONE_OFF_MARKERS):
        return PolicyDecision(
            action="commit_now",
            reason="stable context fact is safe for immediate commit",
        )

    return PolicyDecision(
        action="commit_at_session_end",
        reason="semantic candidate is plausible but should wait for session-level review",
    )


def decide_procedural_candidate(candidate: ProceduralCandidate) -> PolicyDecision:
    """Return the deterministic write decision for a procedural candidate."""

    text = _lowered_texts(
        candidate.reason,
        candidate.payload.rule,
        *candidate.payload.evidence,
    )

    if _contains_any(text, _PROCEDURAL_SAFETY_CONFLICT_MARKERS):
        return PolicyDecision(
            action="drop",
            reason="safety-conflicting procedural request cannot be persisted",
        )

    if candidate.scope == "turn" or _contains_any(
        text, _PROCEDURAL_TURN_SCOPED_MARKERS
    ):
        return PolicyDecision(
            action="drop",
            reason="turn-scoped procedural request should not become long-term memory",
        )

    if candidate.explicitness != "explicit" and not _contains_any(
        text, _PROCEDURAL_EXPLICIT_REQUEST_MARKERS
    ):
        return PolicyDecision(
            action="commit_at_session_end",
            reason="implicit procedural preference should wait for stronger evidence",
        )

    return PolicyDecision(
        action="commit_now",
        reason="explicit durable procedural request is safe for immediate commit",
    )
