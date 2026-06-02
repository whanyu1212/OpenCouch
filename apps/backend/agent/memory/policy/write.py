"""Write policy for memory persistence.

Callers decide whether something is *candidate* memory. This module asks an
LLM-primary policy classifier whether that candidate should commit immediately,
wait for session-end review, require repeated evidence, or drop. Local code
only enforces hard safety/storage invariants; it does not provide
product-judgment fallback writes when the policy LLM is unavailable.
"""

from __future__ import annotations

import logging

from agent.memory.policy.candidates import (
    PolicyDecision,
    ProceduralCandidate,
    SemanticCandidate,
)
from agent.memory.policy.clamps import (
    clamp_procedural_policy_decision,
    clamp_semantic_policy_decision,
    semantic_hard_policy_guard,
)
from agent.memory.policy.markers import (
    semantic_candidate_is_turn_scoped,
    semantic_candidate_needs_repetition_guard,
    text_contains_memory_control_request,
)
from agent.memory.policy.prompts import (
    ProceduralWritePolicyDecision,
    SemanticWritePolicyDecision,
    procedural_policy_prompt,
    semantic_policy_prompt,
    write_policy_system_prompt,
)
from agent.memory.policy.thresholds import (
    should_commit_implicit_procedural_preference,
    should_commit_pattern,
)
from llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


async def decide_semantic_candidate_llm_primary(
    candidate: SemanticCandidate,
    *,
    llm_client: BaseLLMClient | None,
) -> PolicyDecision:
    """Return semantic write policy using an LLM primary path.

    Args:
        candidate (SemanticCandidate): Candidate to classify.
        llm_client (BaseLLMClient | None): Optional classifier client.

    Returns:
        PolicyDecision: Final write policy.
    """

    hard_guard = semantic_hard_policy_guard(candidate)
    if hard_guard is not None:
        return hard_guard
    if llm_client is None:
        raise RuntimeError("Semantic write-policy classification requires an LLM.")

    try:
        decision: SemanticWritePolicyDecision = await llm_client.generate_structured(
            prompt=semantic_policy_prompt(candidate),
            response_schema=SemanticWritePolicyDecision,
            system_instruction=write_policy_system_prompt(),
        )
    except Exception:
        logger.warning(
            "Semantic write-policy LLM classifier failed.",
            exc_info=True,
        )
        raise

    return clamp_semantic_policy_decision(candidate, decision)


async def decide_procedural_candidate_llm_primary(
    candidate: ProceduralCandidate,
    *,
    llm_client: BaseLLMClient | None,
) -> PolicyDecision:
    """Return procedural write policy using an LLM primary path.

    Args:
        candidate (ProceduralCandidate): Candidate to classify.
        llm_client (BaseLLMClient | None): Optional classifier client.

    Returns:
        PolicyDecision: Final write policy.
    """

    if llm_client is None:
        raise RuntimeError("Procedural write-policy classification requires an LLM.")

    try:
        decision: ProceduralWritePolicyDecision = await llm_client.generate_structured(
            prompt=procedural_policy_prompt(candidate),
            response_schema=ProceduralWritePolicyDecision,
            system_instruction=write_policy_system_prompt(),
        )
    except Exception:
        logger.warning(
            "Procedural write-policy LLM classifier failed.",
            exc_info=True,
        )
        raise

    return clamp_procedural_policy_decision(candidate, decision)


__all__ = [
    "decide_procedural_candidate_llm_primary",
    "decide_semantic_candidate_llm_primary",
    "semantic_candidate_is_turn_scoped",
    "semantic_candidate_needs_repetition_guard",
    "semantic_hard_policy_guard",
    "should_commit_implicit_procedural_preference",
    "should_commit_pattern",
    "text_contains_memory_control_request",
]
