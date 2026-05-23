"""Write policy for memory persistence.

Callers decide whether something is *candidate* memory. This module asks an
LLM-primary policy classifier whether that candidate should commit immediately,
wait for session-end review, require repeated evidence, or drop. Local code
only enforces hard safety/storage invariants; it does not provide
product-judgment fallback writes when the policy LLM is unavailable.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from agent.memory.policy.candidates import (
    PolicyDecision,
    ProceduralCandidate,
    ProceduralHoldAction,
    SemanticCandidate,
    SemanticHoldAction,
)
from agent.memory.policy.semantic import (
    SEMANTIC_SESSION_ONLY_CATEGORIES,
)
from llm.base import BaseLLMClient

logger = logging.getLogger(__name__)

_POLICY_REASON_MAX_LENGTH = 240
_FRAGILE_SELF_BELIEF_NEGATIVE_MARKERS = (
    "incompetent",
    "incompetence",
    "failure",
    "worthless",
    "useless",
    "unlovable",
    "broken",
    "not good enough",
)
_FRAGILE_SELF_BELIEF_FRAME_MARKERS = (
    "i am",
    "i'm",
    "im ",
    "means i",
    "tell myself",
    "assume",
    "belief",
)
_TURN_SCOPED_MEMORY_CUES = (
    "for now",
    "for this reply",
    "this reply",
    "this time",
    "right now",
    "just today",
    "only today",
    "only this session",
)
_DURABLE_MEMORY_CUES = (
    "from now on",
    "going forward",
    "in the future",
    "next time",
    "always",
    "usually",
    "please keep",
    "remember",
    "for me when",
)
_ASSISTANT_FACING_PROCEDURAL_CUES = (
    "ask",
    "avoid",
    "be ",
    "check",
    "explain",
    "give",
    "guide",
    "help",
    "keep",
    "offer",
    "respond",
    "say",
    "support",
    "use ",
)
_FACT_SHAPED_PROCEDURAL_CUES = (
    "remember that ",
    "remember the user ",
)
_MEMORY_CONTROL_IMMEDIATE_CUES = (
    "don't save",
    "do not save",
    "dont save",
    "forget this",
    "forget that",
    "incognito",
    "private mode",
    "privacy mode",
    "do not remember",
    "don't remember",
)


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


def semantic_candidate_needs_repetition_guard(
    candidate: SemanticCandidate,
) -> bool:
    """Return whether a semantic candidate is a fragile negative self-belief.

    Args:
        candidate (SemanticCandidate): Candidate to inspect.

    Returns:
        bool: ``True`` when the candidate should require repeated evidence even
            if the LLM labels it as trigger/context.
    """

    text = " ".join(
        (
            candidate.payload.evidence_quote,
            candidate.payload.object.identifier,
        )
    ).lower()
    return any(
        marker in text for marker in _FRAGILE_SELF_BELIEF_NEGATIVE_MARKERS
    ) and any(marker in text for marker in _FRAGILE_SELF_BELIEF_FRAME_MARKERS)


def _turn_scoped_without_durable_cue(text: str) -> bool:
    return any(cue in text for cue in _TURN_SCOPED_MEMORY_CUES) and not any(
        cue in text for cue in _DURABLE_MEMORY_CUES
    )


def semantic_candidate_is_turn_scoped(candidate: SemanticCandidate) -> bool:
    """Return whether semantic evidence is scoped to the current turn only."""

    return _turn_scoped_without_durable_cue(candidate.payload.evidence_quote.lower())


def text_contains_memory_control_request(text: str) -> bool:
    """Return whether text contains an explicit memory-control request."""

    return any(cue in text.lower() for cue in _MEMORY_CONTROL_IMMEDIATE_CUES)


def semantic_candidate_is_memory_control_request(candidate: SemanticCandidate) -> bool:
    """Return whether semantic evidence is an explicit memory-control request."""

    text = " ".join(
        (
            candidate.payload.evidence_quote,
            candidate.payload.object.identifier,
        )
    )
    return text_contains_memory_control_request(text)


def _procedural_request_is_turn_scoped(candidate: ProceduralCandidate) -> bool:
    """Return whether procedural evidence is scoped to the current turn only."""

    evidence_text = " ".join(candidate.evidence_quotes).lower()
    return _turn_scoped_without_durable_cue(evidence_text)


def _procedural_candidate_is_fact_shaped_memory_request(
    candidate: ProceduralCandidate,
) -> bool:
    """Return whether a procedural candidate is just a semantic fact to remember."""

    rule_text = " ".join(candidate.payload.rule.lower().split())
    if not any(cue in rule_text for cue in _FACT_SHAPED_PROCEDURAL_CUES):
        return False
    return not any(cue in rule_text for cue in _ASSISTANT_FACING_PROCEDURAL_CUES)


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
    safety_conflict: bool = Field(
        default=False,
        description=(
            "True when the candidate asks the assistant to weaken, suppress, "
            "or bypass safety or crisis behavior."
        ),
    )


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
    ).lower()
    return any(cue in text for cue in _MEMORY_CONTROL_IMMEDIATE_CUES)


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
        "- commit_now: reserve for exceptional immediate-write cases only.\n"
        "- commit_at_session_end: default for durable facts or preferences that "
        "may be useful after full-session review.\n"
        "- require_repetition: negative self-belief, fragile interpretation, "
        "or emerging pattern that should need repeated evidence.\n"
        "- drop: transient, turn-scoped, provenance-only, or not useful memory.\n\n"
        "Prefer commit_at_session_end by default. Use it, not commit_now, for "
        "ordinary durable facts, coping context, support plans, triggers, and "
        "other memories that can wait for session-end review.\n"
        "Use commit_at_session_end, not drop, for clearly stated sensitive "
        "therapeutic context such as triggers or losses that may be useful "
        "after full-session review.\n"
        "Never commit_now for negative self-beliefs, fresh therapeutic "
        "interpretations, crisis/safety material, high-sensitivity triggers, "
        "or ordinary semantic facts.\n\n"
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
        "- commit_now: reserve for exceptional immediate-write cases only.\n"
        "- commit_at_session_end: default for durable assistant-facing "
        "preferences that should wait for session-end review.\n"
        "- drop: only applies to this turn/session, is not assistant-facing, "
        "or conflicts with safety behavior.\n\n"
        "Prefer commit_at_session_end by default, even for direct future-facing "
        "assistant preferences. Only use commit_now for exceptional immediate "
        "control cases.\n\n"
        "Set safety_conflict=true when the candidate asks the assistant to "
        "weaken, suppress, skip, hide, or bypass crisis checks, safety "
        "questions, emergency guidance, or crisis resources. Safety-conflict "
        "candidates must use action=drop.\n\n"
        "Do not use procedural memory for fact-shaped requests like "
        "'remember that the user has X' or 'remember that X happened'; those "
        "belong in semantic memory unless they also specify how the assistant "
        "should respond.\n\n"
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


def _clamp_procedural_policy_decision(
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

    if decision.action != "drop" and _procedural_request_is_turn_scoped(candidate):
        return PolicyDecision(
            action="drop",
            reason="turn-scoped procedural request cannot become durable memory",
        )

    if (
        decision.action != "drop"
        and _procedural_candidate_is_fact_shaped_memory_request(candidate)
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

    return _clamp_procedural_policy_decision(candidate, decision)
