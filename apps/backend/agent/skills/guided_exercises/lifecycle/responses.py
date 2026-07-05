"""Response-delta builders for guided therapeutic exercises."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent.memory.modes import MemoryMode
from agent.memory.store import MemoryStore
from agent.specialists.guided_exercise import build_guided_exercise_system_prompt
from agent.specialists.therapeutic_response.prompts import (
    build_therapeutic_response_prompt,
)
from agent.state import AgentState
from agent.skills.guided_exercises.lifecycle.memory import (
    _write_exercise_completion_fact,
)
from agent.skills.guided_exercises.catalog.registry import (
    EXERCISE_5_4_3_2_1,
    get_exercise_display_name,
    get_exercise_steps,
)
from agent.skills.guided_exercises.rendering.directives import (
    GuidedExerciseDirective,
    render_full_guided_exercise_directive,
)
from agent.skills.guided_exercises.lifecycle.state import (
    _advance_step_delta,
    _get_current_step,
    _start_exercise_delta,
    clear_exercise_delta,
)
from llm.base import BaseLLMClient

StreamWriterFactory = Callable[[], Callable[[dict[str, str]], None]]


def _noop_stream_writer_factory() -> Callable[[dict[str, str]], None]:
    """Return a no-op stream writer for non-streaming runtimes."""

    return lambda _payload: None


def therapeutic_response_delta(
    *,
    response_style: str,
    response_text: str,
) -> dict[str, Any]:
    """Build the fixed response delta emitted by response-generation flows."""

    return {
        "response_text": response_text,
        "response_style": response_style,
    }


async def generate_streamed_therapeutic_text(
    *,
    state: AgentState,
    llm_client: Any,
    response_style: str,
    system_prompt_builder: Callable[[AgentState], str],
    step_directive: str | None = None,
    stream_writer_factory: StreamWriterFactory = _noop_stream_writer_factory,
) -> str:
    """Generate therapeutic response text with streaming."""

    if llm_client is None:
        raise RuntimeError(
            f"No LLM client available for {response_style} response generation."
        )

    writer = stream_writer_factory()
    chunks: list[str] = []
    async for chunk in llm_client.generate_text_stream(
        prompt=build_therapeutic_response_prompt(
            state,
            response_style=response_style,
            step_directive=step_directive,
        ),
        system_instruction=system_prompt_builder(state),
    ):
        chunks.append(chunk)
        writer({"type": "chunk", "text": chunk})
    return "".join(chunks)


def _render_response_directive(directive: GuidedExerciseDirective) -> str:
    """Render a guided-exercise directive for response generation."""

    return render_full_guided_exercise_directive(directive)


async def _build_start_delta(
    state: AgentState,
    *,
    llm_client: BaseLLMClient | None,
    exercise_type: str,
    stream_writer_factory: StreamWriterFactory = _noop_stream_writer_factory,
) -> dict[str, Any]:
    """Build the delta for starting a guided exercise.

    Args:
        state: Current runtime state.
        llm_client: Response LLM client.
        exercise_type: Exercise identifier to start.
        stream_writer_factory: Factory that returns the current stream writer.

    Returns:
        Response and state delta for the first exercise step.
    """

    steps = get_exercise_steps(exercise_type)
    if steps is None:
        raise KeyError(exercise_type)
    first_step = steps[0]
    display_name = get_exercise_display_name(exercise_type)
    directive = GuidedExerciseDirective(
        exercise_type=exercise_type,
        runtime_action="start",
        current_step_index=0,
        runtime_task=(
            f"Start the guided exercise {display_name}. "
            f"Briefly name the exercise and invite the user into step 0.\n"
            f'Step 0 instruction: "{first_step.instruction}"\n'
            f"Rephrase naturally in your own words. Do NOT present a menu or "
            f"ask whether they want a different exercise."
        ),
    )

    response_text = await generate_streamed_therapeutic_text(
        state=state,
        llm_client=llm_client,
        response_style="guided_exercise",
        system_prompt_builder=build_guided_exercise_system_prompt,
        step_directive=_render_response_directive(directive),
        stream_writer_factory=stream_writer_factory,
    )

    return {
        **_start_exercise_delta(state, exercise_type=exercise_type),
        **therapeutic_response_delta(
            response_style="guided_exercise",
            response_text=response_text,
        ),
    }


async def _build_exit_delta(
    state: AgentState,
    *,
    llm_client: BaseLLMClient | None,
    stream_writer_factory: StreamWriterFactory = _noop_stream_writer_factory,
) -> dict[str, Any]:
    """Build the delta for an exit.

    Args:
        state: Current runtime state.
        llm_client: Response LLM client.
        stream_writer_factory: Factory that returns the current stream writer.

    Returns:
        Response and state delta that clears the active exercise.
    """

    exercise_state = state.get("exercise_state", {})
    step_index = exercise_state.get("exercise_step", 0)
    exercise_type = exercise_state.get("exercise_type") or EXERCISE_5_4_3_2_1
    directive = GuidedExerciseDirective(
        exercise_type=exercise_type,
        runtime_action="exit",
        current_step_index=step_index,
        runtime_task=(
            "The user wants to stop or leave the current guided exercise. "
            "Briefly acknowledge that choice, do not continue the exercise, and "
            "ask what would feel most helpful now."
        ),
    )
    response_text = await generate_streamed_therapeutic_text(
        state=state,
        llm_client=llm_client,
        response_style="guided_exercise",
        system_prompt_builder=build_guided_exercise_system_prompt,
        step_directive=_render_response_directive(directive),
        stream_writer_factory=stream_writer_factory,
    )

    cleared = clear_exercise_delta(state)
    return {
        **cleared,
        **therapeutic_response_delta(
            response_style="guided_exercise",
            response_text=response_text,
        ),
    }


async def _build_resume_delta(
    state: AgentState,
    *,
    llm_client: BaseLLMClient | None,
    stream_writer_factory: StreamWriterFactory = _noop_stream_writer_factory,
) -> dict[str, Any]:
    """Build the delta for resuming an active exercise after a side turn.

    Args:
        state: Current runtime state.
        llm_client: Response LLM client, if configured.
        stream_writer_factory: Factory that returns the current stream writer.

    Returns:
        Response delta that preserves the current exercise step.
    """

    exercise_state = state.get("exercise_state", {})
    step_index = exercise_state.get("exercise_step", 0)
    exercise_type = exercise_state.get("exercise_type") or EXERCISE_5_4_3_2_1
    current_step = _get_current_step(exercise_type, step_index)
    step_ref = current_step.instruction if current_step else ""

    directive = GuidedExerciseDirective(
        exercise_type=exercise_type,
        runtime_action="resume",
        current_step_index=step_index,
        runtime_task=(
            f"The user is asking to return to the active guided exercise after a "
            f"side turn. Resume at step {step_index}; do not classify their message "
            f"as an answer to the step and do not exit the exercise.\n"
            f'Step {step_index} instruction: "{step_ref}"\n'
            f"Briefly re-orient them, then restate this same step in short, "
            f"concrete language."
        ),
    )

    response_text = await generate_streamed_therapeutic_text(
        state=state,
        llm_client=llm_client,
        response_style="guided_exercise",
        system_prompt_builder=build_guided_exercise_system_prompt,
        step_directive=_render_response_directive(directive),
        stream_writer_factory=stream_writer_factory,
    )

    return therapeutic_response_delta(
        response_style="guided_exercise",
        response_text=response_text,
    )


async def _build_stuck_delta(
    state: AgentState,
    *,
    llm_client: BaseLLMClient | None,
    stream_writer_factory: StreamWriterFactory = _noop_stream_writer_factory,
) -> dict[str, Any]:
    """Build the delta for a stuck classification.

    Args:
        state: Current runtime state.
        llm_client: Response LLM client, if configured.
        stream_writer_factory: Factory that returns the current stream writer.

    Returns:
        Response delta that keeps the user on the current step.
    """

    exercise_state = state.get("exercise_state", {})
    step_index = exercise_state.get("exercise_step", 0)
    exercise_type = exercise_state.get("exercise_type") or EXERCISE_5_4_3_2_1
    current_step = _get_current_step(exercise_type, step_index)
    step_ref = current_step.instruction if current_step else ""

    directive = GuidedExerciseDirective(
        exercise_type=exercise_type,
        runtime_action="stuck",
        current_step_index=step_index,
        runtime_task=(
            f"The user is STUCK on step {step_index} of the exercise. "
            f'The step asked: "{step_ref}"\n'
            f"Offer a simpler version of the same step — make it smaller and "
            f"more concrete. Do NOT advance to the next step or repeat the "
            f"original instruction verbatim."
        ),
    )

    response_text = await generate_streamed_therapeutic_text(
        state=state,
        llm_client=llm_client,
        response_style="guided_exercise",
        system_prompt_builder=build_guided_exercise_system_prompt,
        step_directive=_render_response_directive(directive),
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
    stream_writer_factory: StreamWriterFactory = _noop_stream_writer_factory,
) -> dict[str, Any]:
    """Build the delta for a hold classification.

    Args:
        state: Current runtime state.
        llm_client: Response LLM client, if configured.
        stream_writer_factory: Factory that returns the current stream writer.

    Returns:
        Response delta that preserves the current exercise step.
    """

    exercise_state = state.get("exercise_state", {})
    step_index = exercise_state.get("exercise_step", 0)
    exercise_type = exercise_state.get("exercise_type") or EXERCISE_5_4_3_2_1
    current_step = _get_current_step(exercise_type, step_index)
    step_ref = current_step.instruction if current_step else ""

    directive = GuidedExerciseDirective(
        exercise_type=exercise_type,
        runtime_action="hold",
        current_step_index=step_index,
        runtime_task=(
            f"The user gave a tentative or partial response to step {step_index}. "
            f'The step asked: "{step_ref}"\n'
            f"Give brief encouragement, then restate this same step in short, "
            f"concrete language so the user knows exactly what to do next. "
            f"Preserve the core task wording from the step instead of drifting "
            f"into generic encouragement. Do NOT advance to the next step."
        ),
    )

    response_text = await generate_streamed_therapeutic_text(
        state=state,
        llm_client=llm_client,
        response_style="guided_exercise",
        system_prompt_builder=build_guided_exercise_system_prompt,
        step_directive=_render_response_directive(directive),
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
    stream_writer_factory: StreamWriterFactory = _noop_stream_writer_factory,
) -> dict[str, Any]:
    """Build the delta for advancing to the next step.

    Updates ``exercise_state.exercise_step`` and returns the next step's
    instruction as the response text.

    Args:
        state: Current runtime state.
        llm_client: Response LLM client, if configured.
        exercise_type: Active exercise identifier.
        next_step_index: Step index to advance to.
        stream_writer_factory: Factory that returns the current stream writer.

    Returns:
        Response and state delta for the next exercise step.
    """

    steps = get_exercise_steps(exercise_type)
    if steps is None:
        raise KeyError(exercise_type)
    next_step = steps[next_step_index]
    total_steps = len(steps)
    directive = GuidedExerciseDirective(
        exercise_type=exercise_type,
        runtime_action="advance",
        current_step_index=next_step_index,
        runtime_task=(
            f"The user completed step {next_step_index - 1} of {total_steps - 1}. "
            f"Briefly acknowledge what they shared, then move to step "
            f"{next_step_index}.\n"
            f'Step {next_step_index} instruction: "{next_step.instruction}"\n'
            f"Rephrase naturally in your own words — do NOT repeat this "
            f"instruction verbatim. Do NOT repeat any earlier step."
        ),
    )

    response_text = await generate_streamed_therapeutic_text(
        state=state,
        llm_client=llm_client,
        response_style="guided_exercise",
        system_prompt_builder=build_guided_exercise_system_prompt,
        step_directive=_render_response_directive(directive),
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
    stream_writer_factory: StreamWriterFactory = _noop_stream_writer_factory,
) -> dict[str, Any]:
    """Build the delta for natural completion of the exercise.

    Clears exercise state, writes a coping_strategy semantic fact
    (if memory is enabled), and returns a brief "you did it" response.

    Args:
        state: Current runtime state.
        llm_client: Response LLM client, if configured.
        memory_store: Memory store used for completion facts, if configured.
        memory_mode: Current memory mode.
        stream_writer_factory: Factory that returns the current stream writer.

    Returns:
        Response and state delta for natural exercise completion.
    """

    exercise_state = state.get("exercise_state", {})
    raw_exercise_type = exercise_state.get("exercise_type")
    exercise_type: str = (
        raw_exercise_type if raw_exercise_type is not None else EXERCISE_5_4_3_2_1
    )
    display_name = get_exercise_display_name(exercise_type, default="that exercise")
    step_index = exercise_state.get("exercise_step", 0)

    directive = GuidedExerciseDirective(
        exercise_type=exercise_type,
        runtime_action="complete",
        current_step_index=step_index,
        runtime_task=(
            f"The user just finished the LAST step of the exercise. "
            f"Briefly acknowledge what they shared, name what they just did "
            f"({display_name}), and invite them to notice how their body "
            f"feels now. End with a gentle, open check-in question about "
            f"how the exercise felt for them (e.g. 'How was that for you?'). "
            f"Do NOT start a new exercise."
        ),
    )

    response_text = await generate_streamed_therapeutic_text(
        state=state,
        llm_client=llm_client,
        response_style="guided_exercise",
        system_prompt_builder=build_guided_exercise_system_prompt,
        step_directive=_render_response_directive(directive),
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

    cleared = clear_exercise_delta(state)
    return {
        **cleared,
        **therapeutic_response_delta(
            response_style="guided_exercise",
            response_text=response_text,
        ),
    }
