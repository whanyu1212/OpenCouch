"""Marker and text-cue helpers for memory write policy."""

from __future__ import annotations

import re

from agent.memory.policy.candidates import ProceduralCandidate, SemanticCandidate

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
_MEMORY_CONTROL_REQUEST_PATTERNS = (
    re.compile(
        r"^(?:actually,?\s+)?(?:please\s+)?(?:do\s+not|don't|dont)\s+"
        r"(?:save|remember|retain|keep)\b"
    ),
    re.compile(
        r"^(?:actually,?\s+)?(?:please\s+)?forget\s+"
        r"(?:this|that|it|everything|all\b)"
    ),
    re.compile(
        r"^(?:actually,?\s+)?(?:please\s+)?"
        r"(?:use|switch(?:\s+me)?\s+to|turn\s+on)\s+"
        r"(?:incognito|privacy|private)\s+mode\b"
    ),
)


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
    """Return whether text contains an explicit assistant-directed memory request."""

    normalized = " ".join(text.casefold().split())
    return any(
        pattern.search(normalized) for pattern in _MEMORY_CONTROL_REQUEST_PATTERNS
    )


def semantic_candidate_is_memory_control_request(candidate: SemanticCandidate) -> bool:
    """Return whether semantic evidence is an explicit memory-control request."""

    text = " ".join(
        (
            candidate.payload.evidence_quote,
            candidate.payload.object.identifier,
        )
    )
    return text_contains_memory_control_request(text)


def procedural_request_is_turn_scoped(candidate: ProceduralCandidate) -> bool:
    """Return whether procedural evidence is scoped to the current turn only."""

    evidence_text = " ".join(candidate.evidence_quotes).lower()
    return _turn_scoped_without_durable_cue(evidence_text)


def procedural_candidate_is_fact_shaped_memory_request(
    candidate: ProceduralCandidate,
) -> bool:
    """Return whether a procedural candidate is just a semantic fact to remember."""

    rule_text = " ".join(candidate.payload.rule.lower().split())
    if not any(cue in rule_text for cue in _FACT_SHAPED_PROCEDURAL_CUES):
        return False
    return not any(cue in rule_text for cue in _ASSISTANT_FACING_PROCEDURAL_CUES)


__all__ = [
    "procedural_candidate_is_fact_shaped_memory_request",
    "procedural_request_is_turn_scoped",
    "semantic_candidate_is_memory_control_request",
    "semantic_candidate_is_turn_scoped",
    "semantic_candidate_needs_repetition_guard",
    "text_contains_memory_control_request",
]
