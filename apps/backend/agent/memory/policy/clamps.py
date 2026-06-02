"""Safety and product clamps for memory write-policy decisions."""

from __future__ import annotations

from agent.memory.policy.candidates import (
    PolicyDecision,
    ProceduralCandidate,
    SemanticCandidate,
)
from agent.memory.policy.markers import (
    procedural_candidate_is_fact_shaped_memory_request,
    procedural_request_is_turn_scoped,
    semantic_candidate_is_memory_control_request,
    semantic_candidate_is_turn_scoped,
    semantic_candidate_needs_repetition_guard,
    text_contains_memory_control_request,
)
from agent.memory.policy.prompts import (
    ProceduralWritePolicyDecision,
    SemanticWritePolicyDecision,
)
from agent.memory.policy.semantic import (
    SEMANTIC_SESSION_ONLY_CATEGORIES,
)

_POLICY_REASON_MAX_LENGTH = 240


def _prefixed_policy_reason(prefix: str, reason: str) -> str:
    """Return a schema-safe policy reason with a prefix.

    Args:
        prefix: Provenance prefix for the reason.
        reason: Classifier-provided reason text.

    Returns:
        Reason text capped to the PolicyDecision schema limit.
    """

    value = f"{prefix}: {reason}"
    if len(value) <= _POLICY_REASON_MAX_LENGTH:
        return value
    return value[: _POLICY_REASON_MAX_LENGTH - 3].rstrip() + "..."


def semantic_hard_policy_guard(
    candidate: SemanticCandidate,
) -> PolicyDecision | None:
    """Return a non-LLM semantic policy only for hard invariants.

    Args:
        candidate (SemanticCandidate): Semantic candidate to inspect.

    Returns:
        PolicyDecision | None: Hard policy decision, or ``None`` when the
        LLM policy classifier must decide.
    """

    if candidate.payload.predicate == "MENTIONED_IN":
        return PolicyDecision(
            action="drop",
            reason="provenance predicates should not become durable semantic memory",
        )

    if semantic_candidate_is_memory_control_request(candidate):
        return PolicyDecision(
            action="drop",
            reason="memory-control requests should not become semantic memory",
        )

    return None


def _semantic_candidate_can_commit_immediately(
    candidate: SemanticCandidate,
) -> bool:
    """Return whether a semantic candidate is eligible for immediate commit."""

    return False


def _procedural_candidate_can_commit_immediately(
    candidate: ProceduralCandidate,
) -> bool:
    """Return whether a procedural candidate is eligible for immediate commit."""

    text = " ".join(
        [candidate.payload.rule, *candidate.evidence_quotes],
    )
    return text_contains_memory_control_request(text)


def clamp_semantic_policy_decision(
    candidate: SemanticCandidate,
    decision: SemanticWritePolicyDecision,
) -> PolicyDecision:
    """Convert and safety-clamp an LLM semantic policy decision.

    Args:
        candidate (SemanticCandidate): Candidate being classified.
        decision (SemanticWritePolicyDecision): LLM policy decision.

    Returns:
        PolicyDecision: Final policy decision.
    """

    if decision.action != "drop" and semantic_candidate_is_turn_scoped(candidate):
        return PolicyDecision(
            action="drop",
            reason="turn-scoped semantic candidate cannot become durable memory",
        )

    if (
        decision.action == "commit_now"
        and candidate.payload.category in SEMANTIC_SESSION_ONLY_CATEGORIES
    ):
        return PolicyDecision(
            action="commit_at_session_end",
            reason="high-sensitivity semantic candidate should not commit immediately",
        )

    if decision.action in (
        "commit_now",
        "commit_at_session_end",
    ) and semantic_candidate_needs_repetition_guard(candidate):
        return PolicyDecision(
            action="require_repetition",
            reason="fragile negative self-belief requires repeated evidence",
        )

    if (
        decision.action == "commit_now"
        and not _semantic_candidate_can_commit_immediately(candidate)
    ):
        return PolicyDecision(
            action="commit_at_session_end",
            reason="semantic candidate should wait for session-end review",
            policy_version="phase1_v1",
        )

    return PolicyDecision(
        action=decision.action,
        reason=_prefixed_policy_reason("llm_policy", decision.reason),
        policy_version="phase1_llm_v1",
    )


def clamp_procedural_policy_decision(
    candidate: ProceduralCandidate,
    decision: ProceduralWritePolicyDecision,
) -> PolicyDecision:
    """Convert and safety-clamp an LLM procedural policy decision.

    Args:
        decision (ProceduralWritePolicyDecision): LLM policy decision.

    Returns:
        PolicyDecision: Final policy decision.
    """

    if decision.safety_conflict and decision.action != "drop":
        return PolicyDecision(
            action="drop",
            reason="safety-conflicting procedural candidate cannot be persisted",
        )

    if decision.action != "drop" and procedural_request_is_turn_scoped(candidate):
        return PolicyDecision(
            action="drop",
            reason="turn-scoped procedural request cannot become durable memory",
        )

    if decision.action != "drop" and procedural_candidate_is_fact_shaped_memory_request(
        candidate
    ):
        return PolicyDecision(
            action="drop",
            reason="fact-shaped memory request belongs in semantic memory",
        )

    if (
        decision.action == "commit_now"
        and not _procedural_candidate_can_commit_immediately(candidate)
    ):
        return PolicyDecision(
            action="commit_at_session_end",
            reason="procedural candidate should wait for session-end review",
            policy_version="phase1_v1",
        )

    return PolicyDecision(
        action=decision.action,
        reason=_prefixed_policy_reason("llm_policy", decision.reason),
        policy_version="phase1_llm_v1",
    )


__all__ = [
    "clamp_procedural_policy_decision",
    "clamp_semantic_policy_decision",
    "semantic_hard_policy_guard",
]
