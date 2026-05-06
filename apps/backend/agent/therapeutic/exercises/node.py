"""Guided therapeutic exercise node orchestration."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from agent.memory.modes import MemoryMode
from agent.memory.store import MemoryStore
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.therapeutic.exercises.registry import get_exercise_steps
from agent.therapeutic.exercises.responses import (
    StreamWriterFactory,
    _build_advance_delta,
    _build_complete_delta,
    _build_exit_delta,
    _build_hold_delta,
    _build_selection_options_delta,
    _build_stuck_delta,
)
from agent.therapeutic.exercises.selection import _select_exercise_llm_primary
from agent.therapeutic.exercises.state import (
    _clear_exercise_delta,
    _get_current_step,
    _is_last_step,
    _start_exercise_delta,
)
from agent.therapeutic.exercises.step_classifier import _classify_step_state_llm_primary
from agent.therapeutic.exercises.types import ExerciseStep
from agent.therapeutic.response_styles import therapeutic_response_delta
from services.base import BaseLLMClient

logger = logging.getLogger(__name__)


async def run_guided_exercise_response_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
    *,
    stream_writer_factory: StreamWriterFactory = get_stream_writer,
) -> dict[str, Any]:
    """Drive a multi-turn guided exercise.

    Two entry conditions:

    1. **Starting an exercise** — ``exercise_state.exercise_type`` is None
       (no exercise running). The dispatcher's LLM classifier picked
       this mode based on the user's current message. The node selects
       the exercise with an LLM-primary classifier; ambiguous selections
       produce a short option list instead of defaulting to grounding.

    2. **Continuing an exercise** — ``exercise_state.exercise_type`` is set
       from a prior turn. The dispatcher routed here with active exercise
       context. The node classifies the user's message as
       complete/hold/stuck/exit and acts accordingly.

    The node ALWAYS returns a response + routing delta, and may also
    return an exercise-state delta when the exercise state changes (start,
    advance, clear).

    Falls back to deterministic templates and regex exercise selection when no
    LLM client is available. The fallbacks are comprehensive enough to drive
    the full 5-4-3-2-1 exercise end-to-end with no LLM — not just start.

    Args:
        state: Current graph state.
        runtime: LangGraph runtime carrying configured dependencies.
        stream_writer_factory: Factory that returns the current LangGraph
            stream writer.

    Returns:
        Response and state delta for the exercise turn.
    """

    control_llm = runtime.context.llm_client
    response_llm = runtime.context.response_llm or control_llm
    memory_store = runtime.context.memory_store
    memory_mode = runtime.context.memory_mode
    exercise_state = state.get("exercise_state", {})
    exercise_type = exercise_state.get("exercise_type")
    step_index = exercise_state.get("exercise_step")

    # ── Entry condition 1: starting a new exercise ─────────────────
    if exercise_type is None or step_index is None:
        return await _handle_start(
            state,
            classifier_llm=control_llm,
        )

    # ── Entry condition 2: continuing an existing exercise ─────────
    current_step = _get_current_step(exercise_type, step_index)
    if current_step is None:
        # Invalid state (unknown exercise_type, or step_index out of
        # range). Clear and fall back to a fresh start — this is a
        # defensive path that should never fire in normal operation
        # but prevents lockup if the state gets corrupted.
        logger.warning(
            "run_guided_exercise_response_node: invalid exercise state "
            "exercise_type=%r step_index=%r; clearing and restarting",
            exercise_type,
            step_index,
        )
        cleared = _clear_exercise_delta(state)
        start_delta = await _handle_start(
            state,
            classifier_llm=control_llm,
        )
        # Merge: the start delta's exercise-state update wins over the clear
        return {**cleared, **start_delta}

    return await _handle_continue(
        state=state,
        classifier_llm=control_llm or response_llm,
        llm_client=response_llm,
        memory_store=memory_store,
        memory_mode=memory_mode,
        exercise_type=exercise_type,
        step_index=step_index,
        current_step=current_step,
        stream_writer_factory=stream_writer_factory,
    )


async def _handle_start(
    state: AgentState,
    *,
    classifier_llm: BaseLLMClient | None,
) -> dict[str, Any]:
    """Start a new exercise at step 0.

    Selects the exercise with an LLM-primary classifier. If the classifier is
    uncertain, the node offers a few choices instead of silently defaulting to
    grounding. Regex selection remains the no-LLM fallback.

    Args:
        state: Current graph state.
        classifier_llm: Control-plane LLM client, if configured.

    Returns:
        Response and exercise-state delta for the first step.
    """

    selection = await _select_exercise_llm_primary(
        state,
        classifier_llm=classifier_llm,
    )
    if selection.exercise_type is None:
        return _build_selection_options_delta(selection.options)

    selected = selection.exercise_type
    steps = get_exercise_steps(selected)
    if steps is None:
        raise KeyError(selected)
    response_text = steps[0].prompt_fallback

    start_exercise_delta = _start_exercise_delta(state, exercise_type=selected)
    return {
        **start_exercise_delta,
        **therapeutic_response_delta(
            response_style="guided_exercise",
            response_text=response_text,
        ),
    }


async def _handle_continue(
    *,
    state: AgentState,
    classifier_llm: BaseLLMClient | None,
    llm_client: BaseLLMClient | None,
    memory_store: MemoryStore | None,
    memory_mode: MemoryMode,
    exercise_type: str,
    step_index: int,
    current_step: ExerciseStep,
    stream_writer_factory: StreamWriterFactory = get_stream_writer,
) -> dict[str, Any]:
    """Continue an exercise based on the user's current message.

    Classifies the message as complete/hold/stuck/exit, builds the
    appropriate exercise-state + response delta, and returns it. The LLM
    (if available) generates the response text for the new state;
    the deterministic fallback is used when the LLM is unavailable
    or errors.

    Args:
        state: Current graph state.
        classifier_llm: Control-plane LLM client used for step-state
            classification, if configured.
        llm_client: Response LLM client, if configured.
        memory_store: Memory store used for completion facts, if configured.
        memory_mode: Current memory mode.
        exercise_type: Active exercise identifier.
        step_index: Current exercise step index.
        current_step: Exercise step the user is responding to.
        stream_writer_factory: Factory that returns the current LangGraph
            stream writer.

    Returns:
        Response and state delta for the continuation turn.
    """

    step_state = await _classify_step_state_llm_primary(
        state=state,
        classifier_llm=classifier_llm,
        exercise_type=exercise_type,
        step_index=step_index,
        current_step=current_step,
    )

    logger.debug(
        "guided_exercise continue: exercise_type=%s step_index=%d step_state=%s",
        exercise_type,
        step_index,
        step_state,
    )

    if step_state == "exit":
        return _build_exit_delta(state)

    if step_state == "stuck":
        return await _build_stuck_delta(
            state,
            llm_client=llm_client,
            stream_writer_factory=stream_writer_factory,
        )

    if step_state == "hold":
        return await _build_hold_delta(
            state,
            llm_client=llm_client,
            stream_writer_factory=stream_writer_factory,
        )

    # step_state == "complete" → advance or finish
    if _is_last_step(exercise_type, step_index):
        return await _build_complete_delta(
            state,
            llm_client=llm_client,
            memory_store=memory_store,
            memory_mode=memory_mode,
            stream_writer_factory=stream_writer_factory,
        )

    return await _build_advance_delta(
        state=state,
        llm_client=llm_client,
        exercise_type=exercise_type,
        next_step_index=step_index + 1,
        stream_writer_factory=stream_writer_factory,
    )
