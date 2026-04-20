"""Candidate models for the phase-1 memory write policy split.

These types sit between the LLM extractor output and the persistent
memory store. Phase 1 does not add a real session buffer yet, but it
does stop treating "the extractor produced it" as identical to "write
it now". The node layer first promotes extractor outputs into
``MemoryCandidate`` instances, then the deterministic policy engine
decides whether to commit immediately, defer, require repetition, or
drop.

The extractor still owns *detection* of potentially memory-worthy
content. Code now owns the final write timing decision.
"""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from agent.memory.constants import (
    PROCEDURAL_EXPLICIT_REQUEST_MARKERS as _PROCEDURAL_EXPLICIT_REQUEST_MARKERS,
    PROCEDURAL_SAFETY_CONFLICT_MARKERS as _PROCEDURAL_SAFETY_CONFLICT_MARKERS,
    PROCEDURAL_TURN_SCOPED_MARKERS as _PROCEDURAL_TURN_SCOPED_MARKERS,
    contains_any as _contains_any,
)
from agent.memory.models import MemoryWrite, ProceduralRuleDraft

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


class PolicyDecision(BaseModel):
    """The final deterministic decision returned by the write policy."""

    action: PolicyRecommendation
    reason: str = Field(min_length=1, max_length=240)
    policy_version: str = "phase1_v1"


_NEGATIVE_SELF_BELIEF_MARKERS = (
    "i always assume",
    "everyone will see i'm",
    "everyone will see im",
    "everyone will think i'm",
    "everyone will think im",
    "one mistake means",
    "i'm incompetent",
    "im incompetent",
    "i'm a failure",
    "im a failure",
    "i always fail",
    "i never get it right",
)

_EMERGING_PATTERN_MARKERS = (
    "it keeps happening",
    "every new task makes me feel like",
    "every task makes me feel like",
    "i'm about to fail",
    "im about to fail",
    "every relationship ends",
    "this always happens",
)

_DURABILITY_MARKERS = (
    "for years",
    "for a long time",
    "i always",
    "i usually",
    "every time",
    "whenever",
    "ever since",
)

_TRANSIENT_MARKERS = (
    "today",
    "tonight",
    "right now",
    "this week",
    "this month",
    "this morning",
    "last night",
    "yesterday",
    "lately",
    "recently",
)


def build_semantic_candidate(
    write: MemoryWrite,
    *,
    message: str,
) -> SemanticCandidate:
    """Promote a ``MemoryWrite`` into a phase-1 semantic candidate."""

    lowered = f"{message} {write.evidence_quote}".lower()
    category = write.category

    if _contains_any(lowered, _NEGATIVE_SELF_BELIEF_MARKERS) or _contains_any(
        lowered, _EMERGING_PATTERN_MARKERS
    ):
        sensitivity: CandidateSensitivity = "high"
        durability: CandidateDurability = "possible"
        scope: CandidateScope = "session"
        recommendation: PolicyRecommendation = "require_repetition"
        reason = "negative self-belief or emerging pattern needs repetition"
    elif category in {"loss", "trigger"}:
        sensitivity = "high"
        durability = "possible"
        scope = "session"
        recommendation = "commit_at_session_end"
        reason = "high-sensitivity semantic content should wait for session review"
    elif category in {"relationship", "preference", "coping_strategy", "goal"}:
        sensitivity = "low"
        durability = "stable"
        scope = "cross_session"
        recommendation = "commit_now"
        reason = "explicit stable semantic fact"
    elif category == "context":
        if _contains_any(lowered, _TRANSIENT_MARKERS) and not _contains_any(
            lowered, _DURABILITY_MARKERS
        ):
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
    """Promote a ``ProceduralRuleDraft`` into a phase-1 procedural candidate."""

    lowered = f"{message} {draft.rule} {' '.join(draft.evidence)}".lower()

    explicit = _contains_any(lowered, _PROCEDURAL_EXPLICIT_REQUEST_MARKERS)
    turn_scoped = _contains_any(lowered, _PROCEDURAL_TURN_SCOPED_MARKERS)
    safety_conflict = _contains_any(lowered, _PROCEDURAL_SAFETY_CONFLICT_MARKERS)

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
