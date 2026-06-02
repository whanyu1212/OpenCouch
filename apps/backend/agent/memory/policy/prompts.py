"""LLM prompt schemas and builders for memory write policy."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent.memory.policy.candidates import ProceduralCandidate, SemanticCandidate


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


def semantic_policy_prompt(candidate: SemanticCandidate) -> str:
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


def procedural_policy_prompt(candidate: ProceduralCandidate) -> str:
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


def write_policy_system_prompt() -> str:
    """Return the common system prompt for memory policy classifiers.

    Returns:
        str: System instruction for structured policy classification.
    """

    return (
        "You are a strict memory write-policy classifier. Return only the "
        "structured decision. You do not write user-facing text."
    )


__all__ = [
    "ProceduralWritePolicyDecision",
    "SemanticWritePolicyDecision",
    "procedural_policy_prompt",
    "semantic_policy_prompt",
    "write_policy_system_prompt",
]
