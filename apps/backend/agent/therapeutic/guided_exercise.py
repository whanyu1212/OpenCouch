"""Guided exercise response mode - public compatibility surface."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.therapeutic.exercises.node import (
    _handle_continue,
    _handle_start,
    run_guided_exercise_response_node as _run_guided_exercise_response_node,
)
from agent.therapeutic.exercises.registry import (
    ALL_EXERCISE_DEFINITIONS,
    EXERCISE_5_4_3_2_1,
    EXERCISE_BEHAVIORAL_EXPERIMENT,
    EXERCISE_BOX_BREATHING,
    EXERCISE_CONTINUUM,
    EXERCISE_GRATITUDE,
    EXERCISE_IMPROVE,
    EXERCISE_LEAVES_ON_STREAM,
    EXERCISE_MUSCLE_RELAXATION,
    EXERCISE_SELF_COMPASSION,
    EXERCISE_STOP_TECHNIQUE,
    EXERCISE_THOUGHT_RECORD,
    EXERCISE_TINY_ACTION,
    EXERCISE_VALUES_COMPASS,
    _DEFAULT_EXERCISE_OPTIONS,
    _EXERCISE_DISPLAY_NAMES,
    _EXERCISE_REGISTRY,
    _EXERCISE_SELECTION_USE_CASES,
    _VOICE_EXERCISE_IDS,
)
from agent.therapeutic.exercises.responses import (
    StreamWriterFactory,
    _FALLBACK_EXIT,
    _FALLBACK_HOLD,
    _FALLBACK_STUCK_REPHRASE,
    _build_advance_delta,
    _build_complete_delta,
    _build_exit_delta,
    _build_hold_delta,
    _build_selection_options_delta,
    _build_stuck_delta,
)
from agent.therapeutic.exercises.selection import (
    _EXERCISE_SELECTORS,
    _available_exercises_for_prompt,
    _build_exercise_selection_prompt,
    _build_pending_exercise_choice_prompt,
    _resolve_pending_exercise_choice,
    _resolve_pending_exercise_choice_llm_primary,
    _select_exercise,
    _select_exercise_llm_primary,
    _valid_exercise_options,
)
from agent.therapeutic.exercises.state import (
    _advance_step_delta,
    _clear_exercise_delta,
    _get_current_step,
    _is_last_step,
    _start_exercise_delta,
)
from agent.therapeutic.exercises.step_classifier import (
    _build_step_classifier_prompt,
    _classify_step_state,
    _classify_step_state_llm_primary,
    _count_listed_items,
    _matches_any,
)
from agent.therapeutic.exercises.types import (
    ExerciseOptionChoiceDecision,
    ExerciseSelectionDecision,
    ExerciseSelectionResult,
    ExerciseStep,
    ExerciseStepDecision,
)
from agent.therapeutic.exercises.memory import _write_exercise_completion_fact

logger = logging.getLogger(__name__)

__all__ = [
    "EXERCISE_5_4_3_2_1",
    "EXERCISE_BEHAVIORAL_EXPERIMENT",
    "EXERCISE_BOX_BREATHING",
    "EXERCISE_CONTINUUM",
    "EXERCISE_GRATITUDE",
    "EXERCISE_IMPROVE",
    "EXERCISE_LEAVES_ON_STREAM",
    "EXERCISE_MUSCLE_RELAXATION",
    "EXERCISE_SELF_COMPASSION",
    "EXERCISE_STOP_TECHNIQUE",
    "EXERCISE_THOUGHT_RECORD",
    "EXERCISE_TINY_ACTION",
    "EXERCISE_VALUES_COMPASS",
    "ALL_EXERCISE_DEFINITIONS",
    "ExerciseOptionChoiceDecision",
    "ExerciseSelectionDecision",
    "ExerciseSelectionResult",
    "ExerciseStep",
    "ExerciseStepDecision",
    "StreamWriterFactory",
    "_DEFAULT_EXERCISE_OPTIONS",
    "_EXERCISE_DISPLAY_NAMES",
    "_EXERCISE_REGISTRY",
    "_EXERCISE_SELECTION_USE_CASES",
    "_EXERCISE_SELECTORS",
    "_FALLBACK_EXIT",
    "_FALLBACK_HOLD",
    "_FALLBACK_STUCK_REPHRASE",
    "_VOICE_EXERCISE_IDS",
    "_advance_step_delta",
    "_available_exercises_for_prompt",
    "_build_advance_delta",
    "_build_complete_delta",
    "_build_exit_delta",
    "_build_exercise_selection_prompt",
    "_build_hold_delta",
    "_build_pending_exercise_choice_prompt",
    "_build_selection_options_delta",
    "_build_step_classifier_prompt",
    "_build_stuck_delta",
    "_classify_step_state",
    "_classify_step_state_llm_primary",
    "_clear_exercise_delta",
    "_count_listed_items",
    "_get_current_step",
    "_handle_continue",
    "_handle_start",
    "_is_last_step",
    "_matches_any",
    "_resolve_pending_exercise_choice",
    "_resolve_pending_exercise_choice_llm_primary",
    "_select_exercise",
    "_select_exercise_llm_primary",
    "_start_exercise_delta",
    "_valid_exercise_options",
    "_write_exercise_completion_fact",
    "get_stream_writer",
    "logger",
    "run_guided_exercise_response_node",
]


async def run_guided_exercise_response_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Drive a multi-turn guided exercise.

    Args:
        state: Current graph state.
        runtime: LangGraph runtime carrying configured dependencies.

    Returns:
        Response and state delta for the exercise turn.
    """

    return await _run_guided_exercise_response_node(
        state,
        runtime,
        stream_writer_factory=get_stream_writer,
    )
