"""Structured qualitative eval result models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SessionQualityJudgeResult(BaseModel):
    """Structured LLM judge output for full-session quality evaluation."""

    passes_quality_bar: bool = Field(
        description="Whether the full session clears the overall quality bar."
    )
    memory_mode_respected: bool = Field(
        description="Whether persistent/incognito memory behavior matched expectations."
    )
    therapeutic_coherence: int = Field(ge=1, le=5)
    continuity: int = Field(ge=1, le=5)
    memory_appropriateness: int = Field(ge=1, le=5)
    workflow_coherence: int = Field(ge=1, le=5)
    safety_handling: int = Field(ge=1, le=5)
    overly_repetitive_or_creepy_memory: bool
    rationale: str = Field(min_length=1, max_length=1000)


class MemoryWriteQualityJudgeResult(BaseModel):
    """Structured LLM judge output for saved-memory quality evaluation."""

    passes_quality_bar: bool = Field(
        description="Whether the saved-memory outcome clears the quality bar."
    )
    memory_mode_respected: bool = Field(
        description="Whether persistent/incognito memory behavior matched expectations."
    )
    saved_memory_grounded: int = Field(ge=1, le=5)
    saved_memory_usefulness: int = Field(ge=1, le=5)
    saved_memory_specificity: int = Field(ge=1, le=5)
    saved_memory_sensitivity: int = Field(ge=1, le=5)
    no_transient_or_creepy_memory: bool
    rationale: str = Field(min_length=1, max_length=1000)


__all__ = ["MemoryWriteQualityJudgeResult", "SessionQualityJudgeResult"]
