"""Episodic-memory models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent.memory.types.primitives import MemoryWriteTiming
from agent.memory.types.therapeutic import TherapeuticApproach


class CBTContext(BaseModel):
    """Structured context from a CBT-oriented session."""

    approach: Literal["cbt"] = "cbt"
    thought_examined: str | None = None
    action_step: str | None = None
    tool_used: str | None = None


class MIContext(BaseModel):
    """Structured context from a motivational interviewing session."""

    approach: Literal["motivational_interviewing"] = "motivational_interviewing"
    readiness_stage: str | None = None
    change_talk_themes: list[str] = Field(default_factory=list)
    sustain_talk_themes: list[str] = Field(default_factory=list)


class ACTContext(BaseModel):
    """Structured context from an ACT-oriented session."""

    approach: Literal["act"] = "act"
    values_identified: list[str] = Field(default_factory=list)
    fusion_patterns: list[str] = Field(default_factory=list)
    committed_action: str | None = None


class GriefContext(BaseModel):
    """Structured context from a grief support session."""

    approach: Literal["grief_support"] = "grief_support"
    person_lost: str | None = None
    relationship: str | None = None
    time_since_loss: str | None = None


class IPTContext(BaseModel):
    """Structured context from an interpersonal therapy session."""

    approach: Literal["interpersonal_therapy"] = "interpersonal_therapy"
    problem_area: str | None = None
    key_relationship: str | None = None
    communication_step_planned: str | None = None


class DBTContext(BaseModel):
    """Structured context from a DBT skills session."""

    approach: Literal["dbt_skills"] = "dbt_skills"
    skills_used: list[str] = Field(default_factory=list)
    primary_domain: str | None = None


class PFAContext(BaseModel):
    """Structured context from a psychological first aid session."""

    approach: Literal["pfa"] = "pfa"
    crisis_type: str | None = None
    support_connected: str | None = None


TherapeuticApproachContext = (
    CBTContext
    | MIContext
    | ACTContext
    | GriefContext
    | IPTContext
    | DBTContext
    | PFAContext
)


class MoodArc(BaseModel):
    """Session-level mood summary — how the user opened and closed."""

    opened: str = Field(min_length=1, max_length=40)
    closed: str = Field(min_length=1, max_length=40)


class SessionArc(BaseModel):
    """A completed session's structured summary, stored in episodic memory."""

    session_id: str
    started_at: str
    ended_at: str
    duration_seconds: int = Field(ge=0)
    turn_count: int = Field(ge=0)
    primary_themes: list[str] = Field(min_length=0, max_length=3)
    summary: str = Field(min_length=1, max_length=600)
    mood_arc: MoodArc
    open_loops: list[str] = Field(default_factory=list)
    resolved_threads: list[str] = Field(default_factory=list)
    approach_used: TherapeuticApproach | None = None
    approach_context: TherapeuticApproachContext | None = None


class StoredSessionArc(SessionArc):
    """Stored shape of a SessionArc with memory-layer metadata added."""

    id: str
    owner_id: str
    created_at: str
    last_referenced_at: str
    user_visible: bool = True
    write_timing: MemoryWriteTiming = "session_end"
    write_reason: str = Field(default="", max_length=240)
    policy_version: str = Field(default="phase5_v1", min_length=1, max_length=40)
    crisis_level_max: Literal[0, 1, 2, 3] = 0


class SummarizationResult(BaseModel):
    """Structured-output shape returned by the session summarizer LLM."""

    arc: SessionArc | None = None
    reason: str = Field(min_length=1, max_length=240)


__all__ = [
    "CBTContext",
    "MIContext",
    "ACTContext",
    "GriefContext",
    "IPTContext",
    "DBTContext",
    "PFAContext",
    "TherapeuticApproachContext",
    "MoodArc",
    "SessionArc",
    "StoredSessionArc",
    "SummarizationResult",
]
