"""Shared types for guided therapeutic exercises."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field


StepState = Literal["complete", "hold", "stuck", "exit"]


CompletionMode = Literal["item_count", "user_confirmation"]


@dataclass(frozen=True)
class ExerciseSelectorGroup:
    """One ordered selector group for deterministic exercise fallback.

    Args:
        keywords: Regex or substring patterns used by the fallback selector.
        priority: Lower values are evaluated earlier.
    """

    keywords: tuple[str, ...]
    priority: int


@dataclass(frozen=True)
class ExerciseStep:
    """One step of a multi-turn exercise.

    Each step has:

    - ``prompt_fallback``: the deterministic response text used when
      no LLM client is available. The LLM path uses the same prompt
      but can vary wording turn-to-turn.
    - ``expected_count``: for counting-based steps (e.g., "name 5
      things you can see"), the number of items the user should name
      to count as "complete." The state classifier uses this to
      distinguish COMPLETE ("I see a lamp, a book, a plant, my
      coffee, and the window") from HOLD ("I see... a lamp?").
    - ``min_count_for_completion``: the minimum number of items that
      still counts as complete. Leniency matters — a user naming 4
      things on a "name 5" step should be allowed to advance rather
      than being held back on a technicality.
    - ``completion_mode``: how the classifier determines completion.
      ``"item_count"`` (default) counts listed items; used for steps
      that ask the user to name things. ``"user_confirmation"`` matches
      confirmation phrases ("ok", "done", "yes"); used for steps where
      the user performs an action (breathing, visualization) and
      confirms they did it.
    """

    prompt_fallback: str
    expected_count: int
    min_count_for_completion: int
    completion_mode: CompletionMode = "item_count"


@dataclass(frozen=True)
class ExerciseDefinition:
    """Catalog entry for one guided exercise.

    Args:
        id: Stable exercise identifier stored in graph state.
        display_name: Human-readable name used in user-facing responses.
        selection_use_case: Compact description shown to the selector LLM.
        steps: Ordered exercise steps.
        selector_groups: Deterministic fallback selectors with priorities.
        voice_supported: Whether the exercise is suitable for voice mode.
    """

    id: str
    display_name: str
    selection_use_case: str
    steps: tuple[ExerciseStep, ...]
    selector_groups: tuple[ExerciseSelectorGroup, ...] = ()
    voice_supported: bool = False


class ExerciseStepDecision(BaseModel):
    """Structured output for guided-exercise step classification."""

    step_state: StepState
    reasoning: str = Field(min_length=1, max_length=240)
    confidence: Literal["low", "medium", "high"]


@dataclass(frozen=True)
class ExerciseSelectionResult:
    """Internal result for guided-exercise selection."""

    exercise_type: str | None
    options: tuple[str, ...] = ()


class ExerciseSelectionDecision(BaseModel):
    """Structured output for guided-exercise selection."""

    selection_kind: Literal["selected", "ambiguous"]
    exercise_type: str | None = None
    option_types: list[str] = Field(default_factory=list)
    reasoning: str = Field(min_length=1, max_length=240)
    confidence: Literal["low", "medium", "high"]


class ExerciseOptionChoiceDecision(BaseModel):
    """Structured output for resolving pending exercise-option choices."""

    choice_kind: Literal["selected", "unclear"]
    exercise_type: str | None = None
    reasoning: str = Field(min_length=1, max_length=240)
    confidence: Literal["low", "medium", "high"]
