"""Semantic-memory models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent.memory.types.primitives import (
    ConfidenceLevel,
    EntityRef,
    HotPathEdgeType,
    MemoryWriteTiming,
)

SemanticCategory = Literal[
    "loss",
    "preference",
    "coping_strategy",
    "relationship",
    "trigger",
    "goal",
    "context",
]


class MemoryWrite(BaseModel):
    """One fact extracted from a turn, structured as a graph triple."""

    category: SemanticCategory
    subject: EntityRef
    predicate: HotPathEdgeType
    object: EntityRef
    evidence_quote: str = Field(min_length=1, max_length=280)
    confidence: ConfidenceLevel
    source_session_id: str
    source_turn_index: int = Field(ge=0)


class SemanticFact(BaseModel):
    """A stored semantic fact record in the memory store."""

    id: str
    category: SemanticCategory
    subject: EntityRef
    predicate: HotPathEdgeType
    object: EntityRef
    evidence_quote: str
    confidence: ConfidenceLevel
    source_session_id: str
    source_turn_index: int
    created_at: str
    last_referenced_at: str
    dormant_at: str | None = None
    superseded_by: str | None = None
    user_visible: bool = True
    write_timing: MemoryWriteTiming = "immediate"
    write_reason: str = Field(default="", max_length=240)
    policy_version: str = Field(default="phase1_v1", min_length=1, max_length=40)


class ExtractionResult(BaseModel):
    """Structured-output shape returned by the semantic extractor LLM."""

    facts: list[MemoryWrite] = Field(default_factory=list)
    reason: str = Field(min_length=1, max_length=240)


__all__ = [
    "SemanticCategory",
    "MemoryWrite",
    "SemanticFact",
    "ExtractionResult",
]
