"""Response-delta builders for guided therapeutic exercises."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.config import get_stream_writer

from agent.memory.modes import MemoryMode
from agent.memory.store import MemoryStore
from agent.state import AgentState
from agent.therapeutic.exercises.memory import _write_exercise_completion_fact
from agent.therapeutic.exercises.registry import (
    EXERCISE_5_4_3_2_1,
    fallback_suggestion_options,
    get_exercise_display_name,
    get_exercise_steps,
)
from agent.therapeutic.exercises.selection import _valid_exercise_options
from agent.therapeutic.exercises.state import (
    _advance_step_delta,
    _clear_exercise_delta,
    _get_current_step,
)
from agent.therapeutic.prompts import build_guided_exercise_system_prompt
from agent.therapeutic.response_styles import (
    StreamWriterFactory,
    generate_streamed_therapeutic_text,
    therapeutic_response_delta,
)
from services.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


# Deterministic fallback strings used when no LLM client is available.
_FALLBACK_HOLD = "Take your time — even one counts. There's no rush."
_FALLBACK_STUCK_REPHRASE = (
    "That's okay. Let's make it smaller — just one thing you can "
    "notice right now, whatever stands out."
)
_FALLBACK_EXIT = "Of course, let's stop. What would feel most helpful right now?"


def _build_selection_options_delta(options: tuple[str, ...]) -> dict[str, Any]:
    """Build a response that asks the user to choose an exercise.

    Args:
        options: Exercise identifiers to offer.

    Returns:
        Response and pending-selection state delta.
    """

    valid_options = _valid_exercise_options(options)
    if len(valid_options) < 2:
        valid_options = fallback_suggestion_options()
    option_lines = "\n".join(
        f"{index}. {get_exercise_display_name(exercise_type)}"
        for index, exercise_type in enumerate(valid_options, start=1)
    )
    response_text = (
        "A few options could fit here. Which would you like to try?\n"
        f"{option_lines}\n"
        "You can reply with a number or the exercise name."
    )
    return {
        "exercise_state": {
            "exercise_type": None,
            "exercise_step": None,
            "exercise_therapeutic_approach": None,
            "exercise_selection_options": list(valid_options),
        },
        **therapeutic_response_delta(
            response_style="guided_exercise",
            response_text=response_text,
        ),
    }


def _build_exit_delta(
    state: AgentState,
) -> dict[str, Any]:
    """Build the delta for an exit.

    Args:
        state: Current graph state.

    Returns:
        Response and state delta that clears the active exercise.
    """

    cleared = _clear_exercise_delta(state)
    return {
        **cleared,
        **therapeutic_response_delta(
            response_style="guided_exercise",
            response_text=_FALLBACK_EXIT,
        ),
    }


async def _build_stuck_delta(
    state: AgentState,
    *,
    llm_client: BaseLLMClient | None,
    stream_writer_factory: StreamWriterFactory = get_stream_writer,
) -> dict[str, Any]:
    """Build the delta for a stuck classification.

    Args:
        state: Current graph state.
        llm_client: Response LLM client, if configured.
        stream_writer_factory: Factory that returns the current LangGraph
            stream writer.

    Returns:
        Response delta that keeps the user on the current step.
    """

    exercise_state = state.get("exercise_state", {})
    step_index = exercise_state.get("exercise_step", 0)
    exercise_type = exercise_state.get("exercise_type") or EXERCISE_5_4_3_2_1
    current_step = _get_current_step(exercise_type, step_index)
    step_ref = current_step.prompt_fallback if current_step else ""

    directive = (
        f"The user is STUCK on step {step_index} of the exercise. "
        f'The step asked: "{step_ref}"\n'
        f"Offer a simpler version of the same step — make it smaller and "
        f"more concrete. Do NOT advance to the next step or repeat the "
        f"original instruction verbatim."
    )

    response_text = await generate_streamed_therapeutic_text(
        state=state,
        llm_client=llm_client,
        response_style="guided_exercise",
        system_prompt_builder=build_guided_exercise_system_prompt,
        fallback_text=_FALLBACK_STUCK_REPHRASE,
        logger=logger,
        failure_message=(
            "Guided exercise stuck-path LLM call failed; using deterministic fallback."
        ),
        step_directive=directive,
        stream_writer_factory=stream_writer_factory,
    )

    return therapeutic_response_delta(
        response_style="guided_exercise",
        response_text=response_text,
    )


async def _build_hold_delta(
    state: AgentState,
    *,
    llm_client: BaseLLMClient | None,
    stream_writer_factory: StreamWriterFactory = get_stream_writer,
) -> dict[str, Any]:
    """Build the delta for a hold classification.

    Args:
        state: Current graph state.
        llm_client: Response LLM client, if configured.
        stream_writer_factory: Factory that returns the current LangGraph
            stream writer.

    Returns:
        Response delta that preserves the current exercise step.
    """

    exercise_state = state.get("exercise_state", {})
    step_index = exercise_state.get("exercise_step", 0)
    exercise_type = exercise_state.get("exercise_type") or EXERCISE_5_4_3_2_1
    current_step = _get_current_step(exercise_type, step_index)
    step_ref = current_step.prompt_fallback if current_step else ""

    directive = (
        f"The user gave a tentative or partial response to step {step_index}. "
        f'The step asked: "{step_ref}"\n'
        f"Give brief encouragement, then restate this same step in short, "
        f"concrete language so the user knows exactly what to do next. "
        f"Preserve the core task wording from the step instead of drifting "
        f"into generic encouragement. Do NOT advance to the next step."
    )

    response_text = await generate_streamed_therapeutic_text(
        state=state,
        llm_client=llm_client,
        response_style="guided_exercise",
        system_prompt_builder=build_guided_exercise_system_prompt,
        fallback_text=_FALLBACK_HOLD,
        logger=logger,
        failure_message=(
            "Guided exercise hold-path LLM call failed; using deterministic fallback."
        ),
        step_directive=directive,
        stream_writer_factory=stream_writer_factory,
    )

    return therapeutic_response_delta(
        response_style="guided_exercise",
        response_text=response_text,
    )


async def _build_advance_delta(
    *,
    state: AgentState,
    llm_client: BaseLLMClient | None,
    exercise_type: str,
    next_step_index: int,
    stream_writer_factory: StreamWriterFactory = get_stream_writer,
) -> dict[str, Any]:
    """Build the delta for advancing to the next step.

    Updates ``exercise_state.exercise_step`` and returns the next step's
    prompt as the response text (via LLM or fallback).

    Args:
        state: Current graph state.
        llm_client: Response LLM client, if configured.
        exercise_type: Active exercise identifier.
        next_step_index: Step index to advance to.
        stream_writer_factory: Factory that returns the current LangGraph
            stream writer.

    Returns:
        Response and state delta for the next exercise step.
    """

    steps = get_exercise_steps(exercise_type)
    if steps is None:
        raise KeyError(exercise_type)
    next_step = steps[next_step_index]
    total_steps = len(steps)
    directive = (
        f"The user completed step {next_step_index - 1} of {total_steps - 1}. "
        f"Briefly acknowledge what they shared, then move to step "
        f"{next_step_index}.\n"
        f'Step {next_step_index} instruction: "{next_step.prompt_fallback}"\n'
        f"Rephrase naturally in your own words — do NOT repeat this "
        f"instruction verbatim. Do NOT repeat any earlier step."
    )

    response_text = await generate_streamed_therapeutic_text(
        state=state,
        llm_client=llm_client,
        response_style="guided_exercise",
        system_prompt_builder=build_guided_exercise_system_prompt,
        fallback_text=next_step.prompt_fallback,
        logger=logger,
        failure_message=(
            "Guided exercise advance-path LLM call failed; "
            "using deterministic fallback."
        ),
        step_directive=directive,
        stream_writer_factory=stream_writer_factory,
    )

    advance_exercise_state = _advance_step_delta(state)
    return {
        **advance_exercise_state,
        **therapeutic_response_delta(
            response_style="guided_exercise",
            response_text=response_text,
        ),
    }


async def _build_complete_delta(
    state: AgentState,
    *,
    llm_client: BaseLLMClient | None,
    memory_store: MemoryStore | None = None,
    memory_mode: MemoryMode | None = None,
    stream_writer_factory: StreamWriterFactory = get_stream_writer,
) -> dict[str, Any]:
    """Build the delta for natural completion of the exercise.

    Clears exercise state, writes a coping_strategy semantic fact
    (if memory is enabled), and returns a brief "you did it" response.

    Args:
        state: Current graph state.
        llm_client: Response LLM client, if configured.
        memory_store: Memory store used for completion facts, if configured.
        memory_mode: Current memory mode.
        stream_writer_factory: Factory that returns the current LangGraph
            stream writer.

    Returns:
        Response and state delta for natural exercise completion.
    """

    exercise_state = state.get("exercise_state", {})
    raw_exercise_type = exercise_state.get("exercise_type")
    exercise_type: str = (
        raw_exercise_type if raw_exercise_type is not None else EXERCISE_5_4_3_2_1
    )
    display_name = get_exercise_display_name(exercise_type, default="that exercise")

    directive = (
        f"The user just finished the LAST step of the exercise. "
        f"Briefly acknowledge what they shared, name what they just did "
        f"({display_name}), and invite them to notice how their body "
        f"feels now. End with a gentle, open check-in question about "
        f"how the exercise felt for them (e.g. 'How was that for you?'). "
        f"Do NOT start a new exercise."
    )

    fallback_complete = (
        f"You just walked yourself through {display_name}. "
        f"Notice how your body feels now compared to when we started. "
        f"How was that for you?"
    )
    response_text = await generate_streamed_therapeutic_text(
        state=state,
        llm_client=llm_client,
        response_style="guided_exercise",
        system_prompt_builder=build_guided_exercise_system_prompt,
        fallback_text=fallback_complete,
        logger=logger,
        failure_message=(
            "Guided exercise complete-path LLM call failed; "
            "using deterministic fallback."
        ),
        step_directive=directive,
        stream_writer_factory=stream_writer_factory,
    )

    # ── Write exercise completion as a coping_strategy fact ──────────
    # This runs BEFORE clearing exercise state, while exercise_type
    # is still available. Only writes in non-incognito mode with a
    # valid memory store.
    await _write_exercise_completion_fact(
        state=state,
        exercise_type=exercise_type,
        display_name=display_name,
        memory_store=memory_store,
        memory_mode=memory_mode,
    )

    cleared = _clear_exercise_delta(state)
    return {
        **cleared,
        "should_persist_memory": True,
        **therapeutic_response_delta(
            response_style="guided_exercise",
            response_text=response_text,
        ),
    }
