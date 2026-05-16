"""Audit and feedback record models.

These models describe operational records written for safety,
feedback, and review. They are not therapeutic memory and should not
be loaded into prompt context by ordinary memory recall paths.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CrisisOverrideOutcome = Literal["none"]
CrisisClassifierPath = Literal["llm_primary"]


class CrisisLogRecord(BaseModel):
    """One crisis event in the always-on safety log."""

    id: str
    session_id_opaque: str
    user_id_or_null: str | None = None
    detected_at: str
    level: Literal[0, 1, 2, 3]
    override_kind: CrisisOverrideOutcome
    classifier_path: CrisisClassifierPath
    reason: str = Field(max_length=500)
    response_node_completed: bool
    llm_failure_occurred: bool
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


class CrisisLogAggregate(BaseModel):
    """Daily rollup of crisis events with no per-user identifiers."""

    date: str
    events_total: int = Field(default=0, ge=0)
    events_by_level: CrisisLogLevelCounts
    events_by_classifier_path: CrisisLogPathCounts
    llm_failures_total: int = Field(default=0, ge=0)
    response_node_completion_rate: float = Field(ge=0.0, le=1.0)


FeedbackLabel = Literal["positive", "negative", "skip"]
FeedbackSource = Literal["cli_end", "cli_exit", "api_end"]


class SessionFeedbackRecord(BaseModel):
    """One end-of-session feedback record."""

    id: str
    session_id_opaque: str
    user_id_or_null: str | None = None
    recorded_at: str
    label: FeedbackLabel
    turn_count_at_end: int
    source: FeedbackSource
    schema_version: int = 1


__all__ = [
    "CrisisOverrideOutcome",
    "CrisisClassifierPath",
    "CrisisLogRecord",
    "CrisisLogLevelCounts",
    "CrisisLogPathCounts",
    "CrisisLogAggregate",
    "FeedbackLabel",
    "FeedbackSource",
    "SessionFeedbackRecord",
]
