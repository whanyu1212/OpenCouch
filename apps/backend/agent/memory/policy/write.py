"""Write policy for hot-path memory persistence.

The LLM extractors decide whether something is *candidate* memory.
This module asks an LLM-primary policy classifier whether that candidate
should commit immediately, wait for session-end review, require repeated
evidence, or drop. Local code only enforces hard safety/storage
invariants; it does not provide product-judgment fallback writes when the
policy LLM is unavailable.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from agent.memory.policy.candidates import (
    CandidateExplicitness,
    PolicyDecision,
    ProceduralCandidate,
    ProceduralHoldAction,
    SemanticCandidate,
    SemanticHoldAction,
)
from agent.memory.policy.constants import (
    classify_procedural_request,
)
from agent.memory.policy.semantic import (
    SEMANTIC_SESSION_ONLY_CATEGORIES,
    contains_emerging_pattern,
    contains_negative_self_belief,
)
from llm.base import BaseLLMClient

logger = logging.getLogger(__name__)

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
    explicitness: CandidateExplicitness,
    evidence_count: int,
) -> bool:
    """Return whether a buffered implicit procedural preference can promote.

    Args:
        hold_action (ProceduralHoldAction): Policy action that held the candidate.
        explicitness (CandidateExplicitness): Whether the preference was explicit.
        evidence_count (int): Number of reinforcing occurrences seen so far.

    Returns:
        bool: ``True`` when the implicit preference has enough evidence to promote.
    """

    if hold_action != "commit_at_session_end":
        return False
    if explicitness == "explicit":
        return False
    return evidence_count >= 2


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

    text = _lowered_texts(
        candidate.reason,
        candidate.payload.evidence_quote,
        candidate.payload.subject.identifier,
        candidate.payload.object.identifier,
    )

    if (
        candidate.policy_recommendation == "require_repetition"
        or contains_negative_self_belief(text)
        or contains_emerging_pattern(text)
    ):
        return PolicyDecision(
            action="require_repetition",
            reason="negative self-belief or emerging pattern requires repetition",
        )

    if candidate.payload.predicate == "MENTIONED_IN":
        return PolicyDecision(
            action="drop",
            reason="provenance predicates should not become durable semantic memory",
        )

    if candidate.scope == "turn" or candidate.durability == "transient":
        return PolicyDecision(
            action="drop",
            reason="turn-scoped or transient semantic candidate should not persist",
        )

    return None


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
        "Use commit_at_session_end, not drop, for clearly stated sensitive "
        "therapeutic context such as triggers or losses that may be useful "
        "after full-session review.\n"
        "Never commit_now for negative self-beliefs, fresh therapeutic "
        "interpretations, crisis/safety material, or high-sensitivity triggers.\n\n"
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
        "Treat direct future-facing requests as explicit durable preferences "
        "when they are assistant-facing and do not conflict with safety.\n\n"
        "Use commit_at_session_end for inferred preferences from statements "
        "about what is hard, helpful, or unpleasant unless the user directly "
        "asks for an ongoing assistant behavior change.\n\n"
        "Never commit a request to suppress crisis checks, safety questions, "
        "or crisis resources.\n\n"
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

    if decision.action == "commit_now" and (
        candidate.sensitivity == "high"
        or candidate.payload.category in SEMANTIC_SESSION_ONLY_CATEGORIES
    ):
        return PolicyDecision(
            action="commit_at_session_end",
            reason="high-sensitivity semantic candidate should not commit immediately",
        )

    return PolicyDecision(
        action=decision.action,
        reason=_prefixed_policy_reason("llm_policy", decision.reason),
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
        PolicyDecision: Final write policy.
    """

    hard_guard = semantic_hard_policy_guard(candidate)
    if hard_guard is not None:
        return hard_guard
    if llm_client is None:
        raise RuntimeError("Semantic write-policy classification requires an LLM.")

    try:
        decision: SemanticWritePolicyDecision = await llm_client.generate_structured(
            prompt=_semantic_policy_prompt(candidate),
            response_schema=SemanticWritePolicyDecision,
            system_instruction=_write_policy_system_prompt(),
        )
    except Exception:
        logger.warning(
            "Semantic write-policy LLM classifier failed.",
            exc_info=True,
        )
        raise

    return _clamp_semantic_policy_decision(candidate, decision)


def procedural_hard_policy_guard(
    candidate: ProceduralCandidate,
) -> PolicyDecision | None:
    """Return a non-LLM procedural policy only for hard invariants.

    Args:
        candidate (ProceduralCandidate): Procedural candidate to inspect.

    Returns:
        PolicyDecision | None: Hard policy decision, or ``None`` when the
        LLM policy classifier must decide.
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
        PolicyDecision: Final write policy.
    """

    hard_guard = procedural_hard_policy_guard(candidate)
    if hard_guard is not None:
        return hard_guard
    if llm_client is None:
        raise RuntimeError("Procedural write-policy classification requires an LLM.")

    try:
        decision: ProceduralWritePolicyDecision = await llm_client.generate_structured(
            prompt=_procedural_policy_prompt(candidate),
            response_schema=ProceduralWritePolicyDecision,
            system_instruction=_write_policy_system_prompt(),
        )
    except Exception:
        logger.warning(
            "Procedural write-policy LLM classifier failed.",
            exc_info=True,
        )
        raise

    return PolicyDecision(
        action=decision.action,
        reason=_prefixed_policy_reason("llm_policy", decision.reason),
        policy_version="phase1_llm_v1",
    )
