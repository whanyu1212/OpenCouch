"""Candidate models for the memory write policy.

These types sit between the LLM extractor output and the persistent
memory store. The node layer first promotes extractor outputs into
``MemoryCandidate`` instances, then the write-policy layer decides
whether to commit immediately, defer, require repetition, or drop.

The extractor still owns *detection* of potentially memory-worthy
content. Candidate metadata is a hint for the LLM-primary policy layer,
not a fallback write decision.
"""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from agent.memory.models import MemoryWrite, ProceduralRuleDraft
from agent.memory.policy.constants import classify_procedural_request
from agent.memory.policy.semantic import (
    SEMANTIC_SESSION_ONLY_CATEGORIES,
    SEMANTIC_STABLE_CATEGORIES,
    contains_emerging_pattern,
    contains_negative_self_belief,
    looks_transient_context,
)

CandidateLayer = Literal["semantic", "procedural"]
CandidateExplicitness = Literal["explicit", "implied"]
CandidateDurability = Literal["transient", "possible", "stable"]
CandidateSensitivity = Literal["low", "medium", "high"]
CandidateScope = Literal["turn", "session", "cross_session"]
PolicyRecommendation = Literal[
    "commit_now",
    "commit_at_session_end",
    "require_repetition",
    "drop",
]


class MemoryCandidate(BaseModel):
    """Common candidate metadata shared across semantic/procedural writes."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    layer: CandidateLayer
    candidate_type: str = Field(min_length=1, max_length=80)
    evidence_quotes: list[str] = Field(default_factory=list)
    source_session_id: str
    source_turn_index: int = Field(ge=0)
    explicitness: CandidateExplicitness
    durability: CandidateDurability
    sensitivity: CandidateSensitivity
    scope: CandidateScope
    policy_recommendation: PolicyRecommendation
    reason: str = Field(min_length=1, max_length=240)
    payload: Any


class SemanticCandidate(MemoryCandidate):
    """Semantic candidate built from one ``MemoryWrite`` extractor item."""

    layer: Literal["semantic"] = "semantic"
    payload: MemoryWrite


class ProceduralCandidate(MemoryCandidate):
    """Procedural candidate built from one ``ProceduralRuleDraft`` item."""

    layer: Literal["procedural"] = "procedural"
    payload: ProceduralRuleDraft


class SessionMemoryBuffer(BaseModel):
    """Runtime-managed per-session buffer for held memory candidates."""

    session_id: str
    semantic_candidates: list[SemanticCandidate] = Field(default_factory=list)
    procedural_candidates: list[ProceduralCandidate] = Field(default_factory=list)

    # Per-turn therapeutic-approach accumulator. The runtime increments the
    # count for the dispatched approach after each turn. At session end, the
    # dominant approach is passed to the summarizer as a hint so it can extract
    # approach-specific structured context. Entries with key "none" or absent
    # approach are ignored when computing the dominant.
    approach_counts: dict[str, int] = Field(default_factory=dict)

    def record_approach(self, approach: str | None) -> None:
        """Record one occurrence of a dispatched therapeutic approach.

        Args:
            approach (str | None): Approach used for the completed turn.

        Returns:
            None: Updates ``approach_counts`` in place.
        """

        if approach and approach != "none":
            self.approach_counts[approach] = self.approach_counts.get(approach, 0) + 1

    def dominant_approach(self) -> str | None:
        """Return the most frequent recorded therapeutic approach.

        Returns:
            str | None: Most frequent non-``"none"`` approach, or ``None``.
        """

        if not self.approach_counts:
            return None
        return max(self.approach_counts, key=self.approach_counts.get)  # type: ignore[arg-type]


class PolicyDecision(BaseModel):
    """The final decision returned by the write policy."""

    action: PolicyRecommendation
    reason: str = Field(min_length=1, max_length=240)
    policy_version: str = "phase1_v1"


def build_semantic_candidate(
    write: MemoryWrite,
    *,
    message: str,
) -> SemanticCandidate:
    """Promote an extracted semantic fact into a memory candidate.

    Args:
        write (MemoryWrite): Extracted semantic fact.
        message (str): Current user message for durability/sensitivity heuristics.

    Returns:
        SemanticCandidate: Semantic candidate with policy metadata hints.
    """

    lowered = f"{message} {write.evidence_quote}".lower()
    category = write.category

    if contains_negative_self_belief(lowered) or contains_emerging_pattern(lowered):
        sensitivity: CandidateSensitivity = "high"
        durability: CandidateDurability = "possible"
        scope: CandidateScope = "session"
        recommendation: PolicyRecommendation = "require_repetition"
        reason = "negative self-belief or emerging pattern needs repetition"
    elif category in SEMANTIC_SESSION_ONLY_CATEGORIES:
        sensitivity = "high"
        durability = "possible"
        scope = "session"
        recommendation = "commit_at_session_end"
        reason = "high-sensitivity semantic content should wait for session review"
    elif category in SEMANTIC_STABLE_CATEGORIES:
        sensitivity = "low"
        durability = "stable"
        scope = "cross_session"
        recommendation = "commit_now"
        reason = "explicit stable semantic fact"
    elif category == "context":
        if looks_transient_context(lowered):
            sensitivity = "medium"
            durability = "possible"
            scope = "session"
            recommendation = "commit_at_session_end"
            reason = "context candidate looks recent or one-off"
        else:
            sensitivity = "low"
            durability = "stable"
            scope = "cross_session"
            recommendation = "commit_now"
            reason = "stable context fact"
    else:
        sensitivity = "medium"
        durability = "possible"
        scope = "session"
        recommendation = "commit_at_session_end"
        reason = "semantic candidate needs session-level confirmation"

    return SemanticCandidate(
        candidate_type=category,
        evidence_quotes=[write.evidence_quote],
        source_session_id=write.source_session_id,
        source_turn_index=write.source_turn_index,
        explicitness="explicit",
        durability=durability,
        sensitivity=sensitivity,
        scope=scope,
        policy_recommendation=recommendation,
        reason=reason,
        payload=write,
    )


def build_procedural_candidate(
    draft: ProceduralRuleDraft,
    *,
    message: str,
    session_id: str,
    turn_index: int,
) -> ProceduralCandidate:
    """Promote an extracted procedural draft into a memory candidate.

    Args:
        draft (ProceduralRuleDraft): Extracted procedural rule draft.
        message (str): Current user message for procedural heuristics.
        session_id (str): Session identifier for candidate provenance.
        turn_index (int): Turn index for candidate provenance.

    Returns:
        ProceduralCandidate: Procedural candidate with policy metadata hints.
    """

    lowered = f"{message} {draft.rule} {' '.join(draft.evidence)}".lower()

    classification = classify_procedural_request(lowered)
    explicit = classification.explicit
    turn_scoped = classification.turn_scoped
    safety_conflict = classification.safety_conflict

    if safety_conflict:
        recommendation: PolicyRecommendation = "drop"
        reason = "procedural request conflicts with safety behavior"
    elif turn_scoped:
        recommendation = "drop"
        reason = "procedural request only applies to the current turn"
    elif explicit:
        recommendation = "commit_now"
        reason = "explicit durable procedural request"
    else:
        recommendation = "commit_at_session_end"
        reason = "implicit procedural preference needs stronger evidence"

    return ProceduralCandidate(
        candidate_type="response_style",
        evidence_quotes=list(draft.evidence),
        source_session_id=session_id,
        source_turn_index=turn_index,
        explicitness="explicit" if explicit else "implied",
        durability="transient" if turn_scoped else "stable",
        sensitivity="high" if safety_conflict else "low",
        scope="turn" if turn_scoped else "cross_session",
        policy_recommendation=recommendation,
        reason=reason,
        payload=draft,
    )
