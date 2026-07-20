"""Safety audit record models.

These models describe operational records written for safety and review.
They are not therapeutic memory and should not be loaded into prompt context
by ordinary memory recall paths.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CrisisOverrideOutcome = Literal["none"]
CrisisClassifierPath = Literal["llm_primary", "voice_concurrent", "voice_post_turn"]
SafetyAuditEventType = Literal["crisis_response", "voice_missed_crisis"]
CrisisResponsePath = Literal[
    "sdk",
    "sdk_tool_fallback",
    "response_llm_override",
    "safety_overlay",
    "not_routed",
    "unknown",
]
CrisisResourceLookupStatus = Literal[
    "not_attempted",
    "pending",
    "found",
    "no_location",
    "location_refused",
    "no_verified_results",
    "lookup_error",
]


class CrisisLogRecord(BaseModel):
    """One crisis event in the always-on safety log."""

    id: str
    event_type: SafetyAuditEventType = "crisis_response"
    session_id_opaque: str
    user_id_or_null: str | None = None
    detected_at: str
    level: Literal[0, 1, 2, 3]
    override_kind: CrisisOverrideOutcome
    classifier_path: CrisisClassifierPath
    reason: str = Field(max_length=500)
    response_node_completed: bool
    llm_failure_occurred: bool
    response_style: str = "crisis_response"
    resource_lookup_status: CrisisResourceLookupStatus = "not_attempted"
    resource_count: int = Field(default=0, ge=0)
    tool_calls: list[str] = Field(default_factory=list)
    response_path: CrisisResponsePath = "unknown"
    fallback_reason: str | None = Field(default=None, max_length=200)
    trace_id: str | None = None
    trace_session_id: str | None = None
    trace_turn_id: str | None = None
    trace_runtime_mode: Literal["text", "voice"] | None = None
    retention_extended_until: str | None = None
    retention_extended_reason: str | None = None


class CrisisLogLevelCounts(BaseModel):
    """Per-level event counts for a single day's crisis log aggregate."""

    level_0: int = Field(default=0, ge=0)
    level_1: int = Field(default=0, ge=0)
    level_2: int = Field(default=0, ge=0)
    level_3: int = Field(default=0, ge=0)


class CrisisLogPathCounts(BaseModel):
    """Per-classifier-path event counts for a single day's aggregate."""

    llm_primary: int = Field(default=0, ge=0)
    voice_concurrent: int = Field(default=0, ge=0)
    voice_post_turn: int = Field(default=0, ge=0)


class CrisisLogAggregate(BaseModel):
    """Daily rollup of crisis events with no per-user identifiers."""

    date: str
    events_total: int = Field(default=0, ge=0)
    events_by_level: CrisisLogLevelCounts
    events_by_classifier_path: CrisisLogPathCounts
    llm_failures_total: int = Field(default=0, ge=0)
    tool_fallbacks_total: int = Field(default=0, ge=0)
    response_llm_overrides_total: int = Field(default=0, ge=0)
    voice_missed_crises_total: int = Field(default=0, ge=0)
    response_node_completion_rate: float = Field(ge=0.0, le=1.0)


__all__ = [
    "CrisisOverrideOutcome",
    "CrisisClassifierPath",
    "SafetyAuditEventType",
    "CrisisResponsePath",
    "CrisisResourceLookupStatus",
    "CrisisLogRecord",
    "CrisisLogLevelCounts",
    "CrisisLogPathCounts",
    "CrisisLogAggregate",
]
