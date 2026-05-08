"""Shared types for guided therapeutic exercises."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field


StepState = Literal["complete", "hold", "stuck", "exit"]


CompletionMode = Literal["items", "confirmation", "response", "llm_judged"]

ExerciseIntensity = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class ExerciseStep:
    """One step of a multi-turn exercise.

    Each step has:

    - ``instruction``: the canonical step instruction. The response LLM
      can rephrase it for the user.
    - ``id``: stable step identifier for durable exercise-state continuity.
      The numeric step index remains in state for backward compatibility.
    - ``completion_criteria``: optional natural-language guidance for what
      counts as enough to advance.
    - ``completion_mode``: how the LLM classifier should judge completion.
      ``"items"`` expects listed items, ``"confirmation"`` expects a private
      action confirmation, ``"response"`` treats a meaningful answer as
      enough, and ``"llm_judged"`` asks the classifier to apply the criteria
      more carefully.
    - ``target_items`` / ``min_items``: optional counts for ``"items"``
      steps. ``min_items`` is intentionally lenient so users can advance
      without matching the requested count perfectly.
    """

    instruction: str
    id: str = ""
    completion_mode: CompletionMode = "response"
    completion_criteria: str = ""
    target_items: int | None = None
    min_items: int | None = None


@dataclass(frozen=True)
class ExerciseDefinition:
    """Catalog entry for one guided exercise.

    Args:
        id: Stable exercise identifier stored in graph state.
        display_name: Human-readable name used in user-facing responses.
        selection_use_case: Compact description shown to the selector LLM.
        steps: Ordered exercise steps.
        version: Definition version stored in active exercise state.
        category: Broad exercise family used for candidate filtering and
            reporting.
        tags: Selection and filtering tags.
        duration_seconds: Approximate expected duration, when known.
        intensity: Expected user effort or emotional load.
        selection_aliases: Human-readable technique names and short phrases
            shown in selection prompts.
        approaches: Therapeutic approaches the exercise is specifically tied
            to. Empty means the exercise is generally available.
        channels: Delivery channels supported by this exercise. Text covers web,
            SMS, WhatsApp, Telegram, and test channels.
        required_skill: Optional capability key required before this exercise
            can be offered.
        voice_supported: Whether the exercise is suitable for voice mode.
    """

    id: str
    display_name: str
    selection_use_case: str
    steps: tuple[ExerciseStep, ...]
    version: int = 1
    category: str = ""
    tags: tuple[str, ...] = ()
    duration_seconds: int | None = None
    intensity: ExerciseIntensity = "medium"
    selection_aliases: tuple[str, ...] = ()
    approaches: tuple[str, ...] = ()
    channels: tuple[str, ...] = ("text",)
    required_skill: str | None = None
    voice_supported: bool = False


class ExerciseStepDecision(BaseModel):
    """Structured output for guided-exercise step classification."""

    step_state: StepState
    reasoning: str = Field(min_length=1, max_length=240)
    confidence: Literal["low", "medium", "high"]


class ExerciseSelectionDecision(BaseModel):
    """Structured output for guided-exercise selection."""

    exercise_type: str = Field(min_length=1)
    reasoning: str = Field(min_length=1, max_length=240)
    confidence: Literal["low", "medium", "high"]
