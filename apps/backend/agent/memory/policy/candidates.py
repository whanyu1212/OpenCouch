"""Candidate models for the memory write policy.

These types sit between proposed memory payloads and the persistent memory
store. Callers promote payloads into ``MemoryCandidate`` instances, then the
write-policy layer decides whether to commit immediately, defer, require
repetition, or drop. Candidates carry provenance and payload only; write timing
is owned by the policy decision.
"""

from __future__ import annotations

from typing import Any, Literal, cast
from uuid import uuid4

from pydantic import BaseModel, Field

from agent.memory.types import MemoryWrite, ProceduralRuleDraft

CandidateLayer = Literal["semantic", "procedural"]
PolicyAction = Literal[
    "commit_now",
    "commit_at_session_end",
    "require_repetition",
    "drop",
]


class MemoryCandidate(BaseModel):
    """Common candidate metadata shared across semantic/procedural writes."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    layer: CandidateLayer
    evidence_quotes: list[str] = Field(default_factory=list)
    source_session_id: str
    source_turn_index: int = Field(ge=0)
    payload: Any


class SemanticCandidate(MemoryCandidate):
    """Semantic candidate built from one proposed ``MemoryWrite`` item."""

    layer: Literal["semantic"] = "semantic"
    payload: MemoryWrite


class ProceduralCandidate(MemoryCandidate):
    """Procedural candidate built from one ``ProceduralRuleDraft`` item."""

    layer: Literal["procedural"] = "procedural"
    payload: ProceduralRuleDraft


class PolicyDecision(BaseModel):
    """The final decision returned by the write policy."""

    action: PolicyAction
    reason: str = Field(min_length=1, max_length=240)
    policy_version: str = "phase1_v1"


SemanticHoldAction = Literal["commit_at_session_end", "require_repetition"]
ProceduralHoldAction = Literal["commit_at_session_end"]


class BufferedSemanticCandidate(BaseModel):
    """Semantic candidate held with the policy decision that held it."""

    candidate: SemanticCandidate
    hold_action: SemanticHoldAction
    policy_reason: str = Field(min_length=1, max_length=240)
    policy_version: str = Field(min_length=1, max_length=80)


class BufferedProceduralCandidate(BaseModel):
    """Procedural candidate held with the policy decision that held it."""

    candidate: ProceduralCandidate
    hold_action: ProceduralHoldAction
    policy_reason: str = Field(min_length=1, max_length=240)
    policy_version: str = Field(min_length=1, max_length=80)


class SessionMemoryBuffer(BaseModel):
    """Runtime-managed per-session buffer for held memory candidates."""

    session_id: str
    held_semantic_candidates: list[BufferedSemanticCandidate] = Field(
        default_factory=list
    )
    held_procedural_candidates: list[BufferedProceduralCandidate] = Field(
        default_factory=list
    )

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

    def hold_semantic(
        self,
        candidate: SemanticCandidate,
        decision: PolicyDecision,
    ) -> None:
        """Hold a semantic candidate for session-end review or promotion.

        Args:
            candidate (SemanticCandidate): Candidate to buffer.
            decision (PolicyDecision): Policy decision that held the candidate.

        Returns:
            None: Mutates the buffer in place.
        """

        if decision.action not in ("commit_at_session_end", "require_repetition"):
            raise ValueError(f"Invalid semantic hold action: {decision.action!r}.")
        hold_action = cast(SemanticHoldAction, decision.action)
        self.held_semantic_candidates.append(
            BufferedSemanticCandidate(
                candidate=candidate,
                hold_action=hold_action,
                policy_reason=decision.reason,
                policy_version=decision.policy_version,
            )
        )

    def hold_procedural(
        self,
        candidate: ProceduralCandidate,
        decision: PolicyDecision,
    ) -> None:
        """Hold a procedural candidate for session-end review.

        Args:
            candidate (ProceduralCandidate): Candidate to buffer.
            decision (PolicyDecision): Policy decision that held the candidate.

        Returns:
            None: Mutates the buffer in place.
        """

        if decision.action != "commit_at_session_end":
            raise ValueError(f"Invalid procedural hold action: {decision.action!r}.")
        hold_action = cast(ProceduralHoldAction, decision.action)
        self.held_procedural_candidates.append(
            BufferedProceduralCandidate(
                candidate=candidate,
                hold_action=hold_action,
                policy_reason=decision.reason,
                policy_version=decision.policy_version,
            )
        )


def build_semantic_candidate(
    write: MemoryWrite,
    *,
    message: str,
) -> SemanticCandidate:
    """Promote a semantic fact into a memory candidate.

    Args:
        write (MemoryWrite): Semantic fact payload.
        message (str): Current user message.

    Returns:
        SemanticCandidate: Semantic candidate with provenance and payload.
    """

    _ = message

    return SemanticCandidate(
        evidence_quotes=[write.evidence_quote],
        source_session_id=write.source_session_id,
        source_turn_index=write.source_turn_index,
        payload=write,
    )


def build_procedural_candidate(
    draft: ProceduralRuleDraft,
    *,
    message: str,
    session_id: str,
    turn_index: int,
) -> ProceduralCandidate:
    """Promote a procedural draft into a memory candidate.

    Args:
        draft (ProceduralRuleDraft): Procedural rule draft.
        message (str): Current user message.
        session_id (str): Session identifier for candidate provenance.
        turn_index (int): Turn index for candidate provenance.

    Returns:
        ProceduralCandidate: Procedural candidate with provenance and payload.
    """

    _ = message

    return ProceduralCandidate(
        evidence_quotes=list(draft.evidence),
        source_session_id=session_id,
        source_turn_index=turn_index,
        payload=draft,
    )
