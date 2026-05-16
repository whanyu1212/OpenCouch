"""Integration coverage for the hybrid OpenAI text runtime."""

from __future__ import annotations

import pytest

from agent.models import ChunkEvent, DoneEvent, ResponseReadyEvent
from agent.persistence import PersistentAgentRuntime
from agent.text_runtime import openai_adapter
from tests.support.openai_text import FakeOpenAISDKRunner
from tests.support.persistence import FakeCrossRestartLLM, runtime_paths


@pytest.mark.asyncio
async def test_persistent_runtime_openai_safe_turn_persists_transcript(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The OpenAI runtime should produce a normal persisted text turn."""

    runner = FakeOpenAISDKRunner("openai persistent reply")
    monkeypatch.setattr(openai_adapter, "_DEFAULT_OPENAI_RUNNER", runner)

    async with PersistentAgentRuntime(
        **runtime_paths(tmp_path),
        text_agent_runtime="openai",
        extract_in_foreground=True,
    ) as runtime:
        result = await runtime.run_turn(
            thread_id="thread-1",
            user_id="user-1",
            message="I feel tense before a presentation.",
            llm_client=FakeCrossRestartLLM(),
        )

        assert result.output.response_text == "openai persistent reply"
        assert result.output.response_style == "supportive"
        assert result.output.therapeutic_approach == "none"
        assert result.output.diagnostics["text_agent_runtime"] == "openai"
        assert [message.role.value for message in result.history] == [
            "user",
            "assistant",
        ]

        second = await runtime.run_turn(
            thread_id="thread-1",
            user_id="user-1",
            message="It is still bothering me.",
            llm_client=FakeCrossRestartLLM(),
        )

        assert second.output.response_text == "openai persistent reply"
        assert len(second.history) == 4
        assert "openai persistent reply" in runner.run_calls[1]["input_text"]


@pytest.mark.asyncio
async def test_persistent_runtime_openai_streaming_surface(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The OpenAI streaming path should keep the public event surface intact."""

    runner = FakeOpenAISDKRunner("openai streamed reply")
    monkeypatch.setattr(openai_adapter, "_DEFAULT_OPENAI_RUNNER", runner)

    async with PersistentAgentRuntime(
        **runtime_paths(tmp_path),
        text_agent_runtime="openai",
        extract_in_foreground=True,
    ) as runtime:
        events = [
            event
            async for event in runtime.run_turn_stream(
                thread_id="thread-stream",
                user_id="user-1",
                message="I need a steadying response.",
                llm_client=FakeCrossRestartLLM(),
            )
        ]

        assert any(
            isinstance(event, ChunkEvent) and event.text == "openai streamed reply"
            for event in events
        )
        assert any(isinstance(event, ResponseReadyEvent) for event in events)
        assert isinstance(events[-1], DoneEvent)
        assert events[-1].output.response_text == "openai streamed reply"
        state = await runtime.get_state("thread-stream")
        assert state is not None
        assert len(state["transcript"]) == 2
        assert runner.stream_calls
