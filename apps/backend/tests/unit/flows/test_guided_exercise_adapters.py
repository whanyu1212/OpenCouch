"""Tests for guided-exercise response LLM adapters."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.flows.guided_exercise.adapters import OpenAIGuidedExerciseResponseLLM
from agent.memory.modes import MemoryMode
from agent.memory.store import OpenCouchMemoryStore
from agent.runtime.context import OpenAITextRunContext
from agent.runtime.workflow_context import WorkflowContext
from agent.skills.guided_exercises.catalog.registry import EXERCISE_BOX_BREATHING
from agent.skills.guided_exercises.rendering.directives import (
    GuidedExerciseDirective,
    render_full_guided_exercise_directive,
    render_tool_forced_guided_exercise_directive,
)
from agent.specialists.roster import build_openai_text_agent_roster
from agent.specialists.therapeutic_response.prompts import (
    build_therapeutic_response_prompt,
)
from agent.state import AgentState


def _sdk_delta(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="raw_response_event",
        data=SimpleNamespace(type="response.output_text.delta", delta=text),
    )


class _ControlledOpenAIStream:
    def __init__(
        self,
        *,
        call_skill_tool_before_first_chunk: bool = False,
        call_skill_tool_before_second_chunk: bool = False,
        call_skill_tool_after_last_chunk: bool = False,
    ) -> None:
        self.final_output = "first second"
        self.call_skill_tool_before_first_chunk = call_skill_tool_before_first_chunk
        self.call_skill_tool_before_second_chunk = call_skill_tool_before_second_chunk
        self.call_skill_tool_after_last_chunk = call_skill_tool_after_last_chunk
        self.context: OpenAITextRunContext | None = None
        self.first_event_seen = asyncio.Event()
        self.release_completion = asyncio.Event()
        self.completed = asyncio.Event()

    async def stream_events(self):  # noqa: ANN202
        self.first_event_seen.set()
        if self.call_skill_tool_before_first_chunk:
            self._record_skill_tool_result()
        yield _sdk_delta("first ")
        await self.release_completion.wait()
        if self.call_skill_tool_before_second_chunk:
            self._record_skill_tool_result()
        yield _sdk_delta("second")
        if self.call_skill_tool_after_last_chunk:
            self._record_skill_tool_result()
        self.completed.set()

    def _record_skill_tool_result(self) -> None:
        assert self.context is not None
        self.context.record_guided_exercise_skill_tool_result(
            exercise_type="grounding_box_breathing",
            current_step_index=0,
            runtime_action="start",
            skill_context="Box breathing context.",
        )


class _ControlledStreamingRunner:
    def __init__(
        self,
        stream: _ControlledOpenAIStream,
        *,
        fallback_output: str = "fallback guided reply",
    ) -> None:
        self.stream = stream
        self.fallback_output = fallback_output
        self.stream_calls: list[dict[str, Any]] = []
        self.run_calls: list[dict[str, Any]] = []

    def run_streamed(
        self,
        *,
        agent: Any,
        input_text: str,
        context: Any,
        session: Any | None = None,
    ) -> _ControlledOpenAIStream:
        self.stream.context = context
        self.stream_calls.append(
            {
                "agent": agent,
                "input_text": input_text,
                "context": context,
                "session": session,
            }
        )
        return self.stream

    async def run(
        self,
        *,
        agent: Any,
        input_text: str,
        context: Any,
        session: Any | None = None,
    ) -> SimpleNamespace:
        self.run_calls.append(
            {
                "agent": agent,
                "input_text": input_text,
                "context": context,
                "session": session,
            }
        )
        return SimpleNamespace(final_output=self.fallback_output)


def _run_context() -> OpenAITextRunContext:
    return OpenAITextRunContext(
        thread_id="thread-1",
        user_id="user-1",
        session_id="session-1",
        current_user_message="start a breathing exercise",
        workflow_context=WorkflowContext(
            llm_client=None,
            memory_store=OpenCouchMemoryStore(),
            crisis_log_backend=InMemoryCrisisLogBackend(),
            memory_mode=MemoryMode.LOCAL,
        ),
    )


def _state() -> AgentState:
    return cast(
        AgentState,
        {
            "message": "start a breathing exercise",
            "session_id": "session-1",
            "transcript": [],
            "working_memory": [],
            "exercise_state": {},
            "turn_lifecycle": {"active_flow": "guided_exercise", "action": "start"},
        },
    )


def _directive() -> GuidedExerciseDirective:
    return GuidedExerciseDirective(
        exercise_type=EXERCISE_BOX_BREATHING,
        runtime_action="start",
        current_step_index=0,
        runtime_task="Start the guided breathing exercise.",
    )


def _prompt_for_directive(
    state: AgentState,
    directive: GuidedExerciseDirective,
    *,
    tool_forced: bool,
) -> str:
    renderer = (
        render_tool_forced_guided_exercise_directive
        if tool_forced
        else render_full_guided_exercise_directive
    )
    return build_therapeutic_response_prompt(
        state,
        response_style="guided_exercise",
        step_directive=renderer(directive),
    )


@pytest.mark.asyncio
async def test_guided_exercise_openai_adapter_renders_tool_prompt_from_directive() -> (
    None
):
    stream = _ControlledOpenAIStream(call_skill_tool_before_first_chunk=True)
    runner = _ControlledStreamingRunner(stream)
    roster = build_openai_text_agent_roster(model="gpt-test")
    state = _state()
    directive = _directive()
    adapter = OpenAIGuidedExerciseResponseLLM(
        runner=runner,
        guided_exercise_agent=roster.guided_exercise_agent,
        run_context=_run_context(),
    )

    generator = adapter.generate_guided_exercise_text_stream(
        state=state,
        directive=directive,
        system_instruction="system",
    )

    first_task = asyncio.create_task(generator.__anext__())
    try:
        await asyncio.wait_for(stream.first_event_seen.wait(), timeout=1.0)
        first = await asyncio.wait_for(first_task, timeout=1.0)
        assert first == "first "
        assert runner.stream_calls[0]["input_text"] == _prompt_for_directive(
            state,
            directive,
            tool_forced=True,
        )
        assert (
            "Required tool: load_guided_exercise_skill"
            in runner.stream_calls[0]["input_text"]
        )
    finally:
        stream.release_completion.set()
        if not first_task.done():
            first_task.cancel()
            with suppress(asyncio.CancelledError):
                await first_task
        await generator.aclose()


@pytest.mark.asyncio
async def test_guided_exercise_openai_adapter_yields_tool_compliant_chunk_before_stream_done() -> (
    None
):
    stream = _ControlledOpenAIStream(call_skill_tool_before_first_chunk=True)
    runner = _ControlledStreamingRunner(stream)
    roster = build_openai_text_agent_roster(model="gpt-test")
    adapter = OpenAIGuidedExerciseResponseLLM(
        runner=runner,
        guided_exercise_agent=roster.guided_exercise_agent,
        run_context=_run_context(),
    )
    generator = adapter.generate_guided_exercise_text_stream(
        state=_state(),
        directive=_directive(),
    )

    first_task = asyncio.create_task(generator.__anext__())
    try:
        await asyncio.wait_for(stream.first_event_seen.wait(), timeout=1.0)
        try:
            first = await asyncio.wait_for(first_task, timeout=0.1)
        except TimeoutError:
            pytest.fail(
                "first guided-exercise stream chunk was buffered until completion"
            )

        assert first == "first "
        assert stream.completed.is_set() is False

        stream.release_completion.set()
        rest = [chunk async for chunk in generator]

        assert rest == ["second"]
        assert stream.completed.is_set() is True
        assert adapter.last_duration_ms is not None
        assert adapter.used_skill_tool_fallback is False
        assert runner.run_calls == []
        assert (
            "Required tool: load_guided_exercise_skill"
            in runner.stream_calls[0]["input_text"]
        )
    finally:
        stream.release_completion.set()
        if not first_task.done():
            first_task.cancel()
            with suppress(asyncio.CancelledError):
                await first_task
        await generator.aclose()


@pytest.mark.asyncio
async def test_guided_exercise_openai_adapter_preserves_order_when_tool_called_mid_stream() -> (
    None
):
    stream = _ControlledOpenAIStream(call_skill_tool_before_second_chunk=True)
    runner = _ControlledStreamingRunner(stream)
    roster = build_openai_text_agent_roster(model="gpt-test")
    adapter = OpenAIGuidedExerciseResponseLLM(
        runner=runner,
        guided_exercise_agent=roster.guided_exercise_agent,
        run_context=_run_context(),
    )
    generator = adapter.generate_guided_exercise_text_stream(
        state=_state(),
        directive=_directive(),
    )

    first_task = asyncio.create_task(generator.__anext__())
    try:
        await asyncio.wait_for(stream.first_event_seen.wait(), timeout=1.0)
        try:
            first = await asyncio.wait_for(asyncio.shield(first_task), timeout=0.1)
        except TimeoutError:
            first = None
        assert first is None, "forced-tool stream yielded before tool compliance"
        assert stream.completed.is_set() is False

        stream.release_completion.set()
        first = await asyncio.wait_for(first_task, timeout=1.0)
        rest = [chunk async for chunk in generator]

        assert first == "first "
        assert rest == ["second"]
        assert stream.completed.is_set() is True
        assert adapter.last_duration_ms is not None
        assert adapter.used_skill_tool_fallback is False
        assert runner.run_calls == []
    finally:
        stream.release_completion.set()
        if not first_task.done():
            first_task.cancel()
            with suppress(asyncio.CancelledError):
                await first_task
        await generator.aclose()


@pytest.mark.asyncio
async def test_guided_exercise_openai_adapter_flushes_buffer_when_tool_called_after_final_event() -> (
    None
):
    stream = _ControlledOpenAIStream(call_skill_tool_after_last_chunk=True)
    runner = _ControlledStreamingRunner(stream)
    roster = build_openai_text_agent_roster(model="gpt-test")
    adapter = OpenAIGuidedExerciseResponseLLM(
        runner=runner,
        guided_exercise_agent=roster.guided_exercise_agent,
        run_context=_run_context(),
    )
    generator = adapter.generate_guided_exercise_text_stream(
        state=_state(),
        directive=_directive(),
    )

    first_task = asyncio.create_task(generator.__anext__())
    try:
        await asyncio.wait_for(stream.first_event_seen.wait(), timeout=1.0)
        try:
            first = await asyncio.wait_for(asyncio.shield(first_task), timeout=0.1)
        except TimeoutError:
            first = None
        assert first is None, "forced-tool stream yielded before tool compliance"
        assert stream.completed.is_set() is False

        stream.release_completion.set()
        first = await asyncio.wait_for(first_task, timeout=1.0)
        rest = [chunk async for chunk in generator]

        assert first == "first "
        assert rest == ["second"]
        assert stream.completed.is_set() is True
        assert adapter.last_duration_ms is not None
        assert adapter.used_skill_tool_fallback is False
        assert runner.run_calls == []
        assert (
            "Required tool: load_guided_exercise_skill"
            in runner.stream_calls[0]["input_text"]
        )
    finally:
        stream.release_completion.set()
        if not first_task.done():
            first_task.cancel()
            with suppress(asyncio.CancelledError):
                await first_task
        await generator.aclose()


@pytest.mark.asyncio
async def test_guided_exercise_openai_adapter_falls_back_when_forced_tool_stream_skips_tool() -> (
    None
):
    stream = _ControlledOpenAIStream()
    runner = _ControlledStreamingRunner(stream)
    roster = build_openai_text_agent_roster(model="gpt-test")
    state = _state()
    directive = _directive()
    adapter = OpenAIGuidedExerciseResponseLLM(
        runner=runner,
        guided_exercise_agent=roster.guided_exercise_agent,
        run_context=_run_context(),
    )
    generator = adapter.generate_guided_exercise_text_stream(
        state=state,
        directive=directive,
    )

    first_task = asyncio.create_task(generator.__anext__())
    try:
        await asyncio.wait_for(stream.first_event_seen.wait(), timeout=1.0)
        try:
            first = await asyncio.wait_for(asyncio.shield(first_task), timeout=0.1)
        except TimeoutError:
            first = None
        assert first is None, "forced-tool stream yielded text before tool compliance"
        assert stream.completed.is_set() is False

        stream.release_completion.set()
        first = await asyncio.wait_for(first_task, timeout=1.0)
        rest = [chunk async for chunk in generator]

        assert first == "fallback guided reply"
        assert rest == []
        assert stream.completed.is_set() is True
        assert adapter.last_duration_ms is not None
        assert adapter.used_skill_tool_fallback is True
        assert len(runner.run_calls) == 1
        assert runner.run_calls[0]["input_text"] == _prompt_for_directive(
            state,
            directive,
            tool_forced=False,
        )
        assert (
            "Required tool: load_guided_exercise_skill"
            in runner.stream_calls[0]["input_text"]
        )
        assert (
            "Required tool: load_guided_exercise_skill"
            not in runner.run_calls[0]["input_text"]
        )
    finally:
        stream.release_completion.set()
        if not first_task.done():
            first_task.cancel()
            with suppress(asyncio.CancelledError):
                await first_task
        await generator.aclose()
