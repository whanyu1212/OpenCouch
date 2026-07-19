"""Shared types for memory-control service modules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field


class PreferenceRuleDecision(BaseModel):
    """Structured rule generated from an explicit user preference."""

    rule_text: str = Field(
        min_length=1,
        max_length=280,
        description="One grammatical second-person procedural rule to persist.",
    )
    reasoning: str = Field(min_length=1, max_length=240)
    confidence: Literal["low", "medium", "high"]


@dataclass(frozen=True)
class MemoryControlServiceResult:
    """Result of executing one memory-management action."""

    response_text: str
    memory_control: dict[str, Any]
    procedural_profile: dict[str, Any] | None = None
    clear_session_buffer: bool = False


@dataclass(frozen=True)
class MemoryControlRequest:
    """Framework-neutral input for one memory-management action."""

    owner_id: str | None
    current_user_message: str
    action: Mapping[str, Any]
    pending_action: Mapping[str, Any] | None = None
    session_id: str | None = None
    turn_count: int = 0
