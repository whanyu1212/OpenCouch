"""Deferred consolidation models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent.memory.types.primitives import ConfidenceLevel

ConsolidationProposalType = Literal[
    "merge_facts",
    "mark_contradiction",
    "promote_to_procedural",
    "infer_graph_edge",
    "mark_dormant",
]


class ConsolidationProposal(BaseModel):
    """One proposal emitted by the phase-4 consolidation LLM pass."""

    proposal_type: ConsolidationProposalType
    confidence: ConfidenceLevel
    rationale: str = Field(min_length=1, max_length=500)
    evidence_fact_ids: list[str] = Field(min_length=1)


class MergeProposalDetail(BaseModel):
    """Detailed record of one merge proposal for audit/rollback."""

    source_fact_ids: list[str] = Field(min_length=2)
    merged_fact_id: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    rationale: str


class ConsolidationRunRecord(BaseModel):
    """One row in the ``consolidation_runs`` observability table."""

    run_id: str
    user_id: str
    started_at: str
    duration_seconds: float = Field(ge=0.0)
    proposals_total: int = Field(default=0, ge=0)
    proposals_applied: int = Field(default=0, ge=0)
    proposals_discarded: int = Field(default=0, ge=0)
    proposals_logged_for_review: int = Field(default=0, ge=0)
    merge_proposals: list[MergeProposalDetail] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


__all__ = [
    "ConsolidationProposalType",
    "ConsolidationProposal",
    "ConsolidationRunRecord",
    "MergeProposalDetail",
]
