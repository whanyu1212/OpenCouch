"""Procedural-memory models."""

from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from agent.memory.types.primitives import ConfidenceLevel, MemoryWriteTiming

ProceduralRuleSource = Literal[
    "explicit_user",
    "consolidation",
    "manual",
]


class ProceduralRule(BaseModel):
    """One learned rule about how to talk to this specific user."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    rule: str = Field(min_length=1, max_length=280)
    evidence: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel
    added_at: str
    source: ProceduralRuleSource
    dormant_at: str | None = None
    superseded_by: str | None = None
    user_visible: bool = True
    write_timing: MemoryWriteTiming = "immediate"
    write_reason: str = Field(default="", max_length=240)
    policy_version: str = Field(default="phase1_v1", min_length=1, max_length=40)


class ProceduralProfile(BaseModel):
    """The single procedural-memory document for a user."""

    proactive_recall_enabled: bool = False
    rules: list[ProceduralRule] = Field(default_factory=list)
    archived_rules: list[ProceduralRule] = Field(default_factory=list)
    last_consolidated_at: str | None = None


class ProceduralRuleDraft(BaseModel):
    """LLM-output shape for a single procedural rule, pre-storage."""

    rule: str = Field(min_length=1, max_length=280)
    evidence: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel = "high"


class ProceduralExtractionResult(BaseModel):
    """Structured-output shape returned by the procedural writer LLM."""

    rules: list[ProceduralRuleDraft] = Field(default_factory=list)
    reason: str = Field(min_length=1, max_length=240)


__all__ = [
    "ProceduralRuleSource",
    "ProceduralRule",
    "ProceduralProfile",
    "ProceduralRuleDraft",
    "ProceduralExtractionResult",
]
