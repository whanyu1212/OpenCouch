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
    classify_procedural_request,
)
from agent.memory.semantic_policy import (
    SEMANTIC_SESSION_ONLY_CATEGORIES,
    SEMANTIC_STABLE_CATEGORIES,
    contains_emerging_pattern,
    contains_negative_self_belief,
    looks_transient_context,
)


def _lowered_texts(*values: str) -> str:
    """Join non-empty values into one lowercase text blob.

    Args:
        values: Text fragments to normalize.

    Returns:
        Lowercase text joined with spaces.
    """

    return " ".join(v.lower() for v in values if v).strip()


def should_commit_pattern(candidate: SemanticCandidate, evidence_count: int) -> bool:
    """Return whether a repetition-gated semantic candidate can promote.

    Args:
        candidate (SemanticCandidate): Candidate being evaluated.
        evidence_count (int): Number of reinforcing occurrences seen so far.

    Returns:
        bool: ``True`` when the candidate is repetition-gated and has enough evidence.
    """

    if candidate.policy_recommendation != "require_repetition":
        return False
    return evidence_count >= 2


def should_commit_implicit_procedural_preference(
    candidate: ProceduralCandidate,
    evidence_count: int,
) -> bool:
    """Return whether a buffered implicit procedural preference can promote.

    Args:
        candidate (ProceduralCandidate): Candidate being evaluated.
        evidence_count (int): Number of reinforcing occurrences seen so far.

    Returns:
        bool: ``True`` when the implicit preference has enough evidence to promote.
    """

    if candidate.policy_recommendation != "commit_at_session_end":
        return False
    if candidate.explicitness == "explicit":
        return False
    return evidence_count >= 2


def decide_semantic_candidate(candidate: SemanticCandidate) -> PolicyDecision:
    """Return the deterministic write decision for a semantic candidate.

    Args:
        candidate (SemanticCandidate): Semantic candidate to evaluate.

    Returns:
        PolicyDecision: Deterministic commit, hold, repetition, or drop decision.
    """

    category = candidate.payload.category
    predicate = candidate.payload.predicate
    text = _lowered_texts(
        candidate.reason,
        candidate.payload.evidence_quote,
        candidate.payload.subject.identifier,
        candidate.payload.object.identifier,
    )

    if contains_negative_self_belief(text) or contains_emerging_pattern(text):
        return PolicyDecision(
            action="require_repetition",
            reason="negative self-belief or emerging pattern requires repetition",
        )

    if candidate.sensitivity == "high" or category in SEMANTIC_SESSION_ONLY_CATEGORIES:
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

    if category in SEMANTIC_STABLE_CATEGORIES:
        return PolicyDecision(
            action="commit_now",
            reason="explicit stable semantic fact is safe for immediate commit",
        )

    if category == "context" and not looks_transient_context(text):
        return PolicyDecision(
            action="commit_now",
            reason="stable context fact is safe for immediate commit",
        )

    return PolicyDecision(
        action="commit_at_session_end",
        reason="semantic candidate is plausible but should wait for session-level review",
    )


def decide_procedural_candidate(candidate: ProceduralCandidate) -> PolicyDecision:
    """Return the deterministic write decision for a procedural candidate.

    Args:
        candidate (ProceduralCandidate): Procedural candidate to evaluate.

    Returns:
        PolicyDecision: Deterministic commit, hold, or drop decision.
    """

    text = _lowered_texts(
        candidate.reason,
        candidate.payload.rule,
        *candidate.payload.evidence,
    )
    classification = classify_procedural_request(text)

    if classification.safety_conflict:
        return PolicyDecision(
            action="drop",
            reason="safety-conflicting procedural request cannot be persisted",
        )

    if candidate.scope == "turn" or classification.turn_scoped:
        return PolicyDecision(
            action="drop",
            reason="turn-scoped procedural request should not become long-term memory",
        )

    if candidate.explicitness != "explicit" and not classification.explicit:
        return PolicyDecision(
            action="commit_at_session_end",
            reason="implicit procedural preference should wait for stronger evidence",
        )

    return PolicyDecision(
        action="commit_now",
        reason="explicit durable procedural request is safe for immediate commit",
    )
