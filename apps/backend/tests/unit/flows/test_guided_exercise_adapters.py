"""Tests for guided-exercise response LLM adapters."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from types import SimpleNamespace
from typing import Any

import pytest

from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.flows.guided_exercise.adapters import OpenAIGuidedExerciseResponseLLM
from agent.memory.modes import MemoryMode
from agent.memory.store import OpenCouchMemoryStore
from agent.runtime.context import OpenAITextRunContext
from agent.runtime.workflow_context import WorkflowContext
from agent.skills.guided_exercises.registry import EXERCISE_BOX_BREATHING
from agent.skills.guided_exercises.rendering.skill_context import (
    render_exercise_skill_context,
)
from agent.specialists.roster import build_openai_text_agent_roster


def _sdk_delta(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="raw_response_event",
        data=SimpleNamespace(type="response.output_text.delta", delta=text),
    )


class _ControlledOpenAIStream:
    def __init__(self) -> None:
        self.final_output = "first second"
        self.first_event_seen = asyncio.Event()
        self.release_completion = asyncio.Event()
        self.completed = asyncio.Event()

    async def stream_events(self):  # noqa: ANN202
        self.first_event_seen.set()
        yield _sdk_delta("first ")
        await self.release_completion.wait()
        yield _sdk_delta("second")
        self.completed.set()


class _ControlledStreamingRunner:
    def __init__(self, stream: _ControlledOpenAIStream) -> None:
        self.stream = stream
        self.stream_calls: list[dict[str, Any]] = []

    def run_streamed(
        self,
        *,
        agent: Any,
        input_text: str,
        context: Any,
        session: Any | None = None,
    ) -> _ControlledOpenAIStream:
        self.stream_calls.append(
            {
                "agent": agent,
                "input_text": input_text,
                "context": context,
                "session": session,
            }
        )
        return self.stream


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


@pytest.mark.asyncio
async def test_guided_exercise_openai_adapter_yields_first_chunk_before_stream_done() -> (
    None
):
    stream = _ControlledOpenAIStream()
    runner = _ControlledStreamingRunner(stream)
    roster = build_openai_text_agent_roster(model="gpt-test")
    adapter = OpenAIGuidedExerciseResponseLLM(
        runner=runner,
        guided_exercise_agent=roster.guided_exercise_agent,
        run_context=_run_context(),
    )
    prompt = (
        f"{render_exercise_skill_context(EXERCISE_BOX_BREATHING, current_step_index=0, runtime_action='start')}"
        "\n\nRuntime task:\nStart the guided breathing exercise."
    )
    generator = adapter.generate_text_stream(prompt=prompt)

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
