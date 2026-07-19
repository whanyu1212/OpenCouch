"""Hard safety guard for semantic memory write decisions.

The LLM-policy clamps (clamp_*_policy_decision) and the write-decision schemas
were removed when session-end consolidation replaced the per-candidate
write-policy classifier. What remains is the non-LLM hard guard, still consumed
by commit selection via ``agent.memory.policy.write``.
"""

from __future__ import annotations

from agent.memory.policy.candidates import PolicyDecision, SemanticCandidate
from agent.memory.policy.markers import semantic_candidate_is_memory_control_request


def semantic_hard_policy_guard(
    candidate: SemanticCandidate,
) -> PolicyDecision | None:
    """Return a non-LLM semantic policy only for hard invariants.

    Args:
        candidate (SemanticCandidate): Semantic candidate to inspect.

    Returns:
        PolicyDecision | None: Hard policy decision, or ``None`` when no hard
        invariant applies.
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


__all__ = ["semantic_hard_policy_guard"]
