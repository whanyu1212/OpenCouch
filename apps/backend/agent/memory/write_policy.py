"""Write policy for hot-path memory persistence.

The LLM extractors still decide whether something is *candidate*
memory. This module decides whether that candidate is safe and durable
enough to commit immediately. The async helpers use an LLM-primary
classifier with deterministic safety guards and fallback. Anything that
is not ``commit_now`` is buffered or dropped by the caller; session-end
promotion logic lives in ``agent.nodes.commit_session_memory``.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

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
from services.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


class SemanticWritePolicyDecision(BaseModel):
    """Structured output for semantic memory write-timing policy."""

    action: Literal["commit_now", "commit_at_session_end", "require_repetition", "drop"]
    reason: str = Field(min_length=1, max_length=240)
    confidence: Literal["low", "medium", "high"]


class ProceduralWritePolicyDecision(BaseModel):
    """Structured output for procedural memory write-timing policy."""

    action: Literal["commit_now", "commit_at_session_end", "drop"]
    reason: str = Field(min_length=1, max_length=240)
    confidence: Literal["low", "medium", "high"]


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


def _semantic_policy_prompt(candidate: SemanticCandidate) -> str:
    """Build the LLM prompt for semantic write-timing policy.

    Args:
        candidate (SemanticCandidate): Candidate memory write to classify.

    Returns:
        str: Prompt for a structured policy decision.
    """

    payload = candidate.payload
    return (
        "Decide the write timing for this candidate semantic memory. "
        "Use the most conservative safe action when uncertain.\n\n"
        "Actions:\n"
        "- commit_now: durable, low-sensitivity cross-session fact.\n"
        "- commit_at_session_end: plausible but sensitive or better reviewed "
        "with full-session context.\n"
        "- require_repetition: negative self-belief, fragile interpretation, "
        "or emerging pattern that should need repeated evidence.\n"
        "- drop: transient, turn-scoped, provenance-only, or not useful memory.\n\n"
        "Never commit_now for negative self-beliefs, fresh therapeutic "
        "interpretations, crisis/safety material, or high-sensitivity triggers.\n\n"
        f"Candidate metadata: durability={candidate.durability}, "
        f"sensitivity={candidate.sensitivity}, scope={candidate.scope}, "
        f"recommendation={candidate.policy_recommendation}, "
        f"reason={candidate.reason!r}\n"
        f"Category: {payload.category}\n"
        f"Predicate: {payload.predicate}\n"
        f"Object: {payload.object.type}:{payload.object.identifier}\n"
        f"Evidence: {payload.evidence_quote!r}"
    )


def _procedural_policy_prompt(candidate: ProceduralCandidate) -> str:
    """Build the LLM prompt for procedural write-timing policy.

    Args:
        candidate (ProceduralCandidate): Candidate procedural rule to classify.

    Returns:
        str: Prompt for a structured policy decision.
    """

    payload = candidate.payload
    return (
        "Decide the write timing for this candidate procedural memory. "
        "Procedural memory stores durable preferences about how the assistant "
        "should respond or use memory.\n\n"
        "Actions:\n"
        "- commit_now: explicit durable assistant-facing preference.\n"
        "- commit_at_session_end: implicit preference that needs stronger "
        "evidence before becoming durable.\n"
        "- drop: only applies to this turn/session, is not assistant-facing, "
        "or conflicts with safety behavior.\n\n"
        "Never commit a request to suppress crisis checks, safety questions, "
        "or crisis resources.\n\n"
        f"Candidate metadata: explicitness={candidate.explicitness}, "
        f"durability={candidate.durability}, sensitivity={candidate.sensitivity}, "
        f"scope={candidate.scope}, recommendation={candidate.policy_recommendation}, "
        f"reason={candidate.reason!r}\n"
        f"Rule: {payload.rule!r}\n"
        f"Evidence: {payload.evidence!r}"
    )


def _write_policy_system_prompt() -> str:
    """Return the common system prompt for memory policy classifiers.

    Returns:
        str: System instruction for structured policy classification.
    """

    return (
        "You are a strict memory write-policy classifier. Return only the "
        "structured decision. You do not write user-facing text."
    )


def _semantic_hard_guard(candidate: SemanticCandidate) -> PolicyDecision | None:
    """Return hard deterministic semantic policy when it must not be overridden.

    Args:
        candidate (SemanticCandidate): Candidate to inspect.

    Returns:
        PolicyDecision | None: Hard policy decision, or None when the LLM may
        classify the candidate.
    """

    deterministic = decide_semantic_candidate(candidate)
    text = _lowered_texts(
        candidate.reason,
        candidate.payload.evidence_quote,
        candidate.payload.subject.identifier,
        candidate.payload.object.identifier,
    )
    if candidate.policy_recommendation == "require_repetition":
        return PolicyDecision(
            action="require_repetition",
            reason=candidate.reason,
        )
    if contains_negative_self_belief(text) or contains_emerging_pattern(text):
        return deterministic
    if candidate.payload.predicate == "MENTIONED_IN":
        return deterministic
    if candidate.scope == "turn" or candidate.durability == "transient":
        return deterministic
    return None


def _clamp_semantic_policy_decision(
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

    deterministic = decide_semantic_candidate(candidate)
    if decision.confidence == "low":
        return deterministic

    if decision.action == "commit_now" and (
        candidate.sensitivity == "high"
        or candidate.payload.category in SEMANTIC_SESSION_ONLY_CATEGORIES
    ):
        return deterministic

    return PolicyDecision(
        action=decision.action,
        reason=f"llm_policy: {decision.reason}",
        policy_version="phase1_llm_v1",
    )


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
        PolicyDecision: Final write policy, falling back to deterministic policy
        when no classifier is available or the classifier is uncertain.
    """

    hard_guard = _semantic_hard_guard(candidate)
    if hard_guard is not None:
        return hard_guard
    if llm_client is None:
        return decide_semantic_candidate(candidate)

    try:
        decision: SemanticWritePolicyDecision = await llm_client.generate_structured(
            prompt=_semantic_policy_prompt(candidate),
            response_schema=SemanticWritePolicyDecision,
            system_instruction=_write_policy_system_prompt(),
        )
    except Exception:
        logger.warning(
            "Semantic write-policy LLM classifier failed; using deterministic fallback.",
            exc_info=True,
        )
        return decide_semantic_candidate(candidate)

    return _clamp_semantic_policy_decision(candidate, decision)


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


def _procedural_hard_guard(candidate: ProceduralCandidate) -> PolicyDecision | None:
    """Return hard deterministic procedural policy when it must not be overridden.

    Args:
        candidate (ProceduralCandidate): Candidate to inspect.

    Returns:
        PolicyDecision | None: Hard policy decision, or None when the LLM may
        classify the candidate.
    """

    text = _lowered_texts(
        candidate.reason,
        candidate.payload.rule,
        *candidate.payload.evidence,
    )
    classification = classify_procedural_request(text)
    if classification.safety_conflict or classification.turn_scoped:
        return decide_procedural_candidate(candidate)
    return None


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
        PolicyDecision: Final write policy, falling back to deterministic policy
        when no classifier is available or the classifier is uncertain.
    """

    hard_guard = _procedural_hard_guard(candidate)
    if hard_guard is not None:
        return hard_guard
    if llm_client is None:
        return decide_procedural_candidate(candidate)

    try:
        decision: ProceduralWritePolicyDecision = await llm_client.generate_structured(
            prompt=_procedural_policy_prompt(candidate),
            response_schema=ProceduralWritePolicyDecision,
            system_instruction=_write_policy_system_prompt(),
        )
    except Exception:
        logger.warning(
            "Procedural write-policy LLM classifier failed; using deterministic fallback.",
            exc_info=True,
        )
        return decide_procedural_candidate(candidate)

    if decision.confidence == "low":
        return decide_procedural_candidate(candidate)
    return PolicyDecision(
        action=decision.action,
        reason=f"llm_policy: {decision.reason}",
        policy_version="phase1_llm_v1",
    )
