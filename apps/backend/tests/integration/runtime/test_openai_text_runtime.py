"""Integration coverage for the hybrid OpenAI text runtime."""

from __future__ import annotations

import pytest

from agent.models import ChunkEvent, DoneEvent, ResponseReadyEvent
from agent.persistence import PersistentAgentRuntime
from agent.text_runtime import openai_adapter
from agent.text_runtime.openai_agents import CRISIS_AGENT_NAME, THERAPEUTIC_AGENT_NAME
from tests.support.openai_text import (
    FakeOpenAISDKRunner,
    ScriptedOpenAITextRouteLLM,
)
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
        assert "openai persistent reply" not in runner.run_calls[1]["input_text"]
        assert [call["session"] is not None for call in runner.run_calls[:2]] == [
            True,
            True,
        ]


@pytest.mark.asyncio
async def test_persistent_runtime_openai_memory_status_uses_sdk_tool(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read-only memory-control turns should run through the SDK memory tool."""

    runner = FakeOpenAISDKRunner(
        "unused sdk reply",
        tool_calls=[("show_memory_status", {})],
        tool_response_as_final=True,
    )
    monkeypatch.setattr(openai_adapter, "_DEFAULT_OPENAI_RUNNER", runner)

    async with PersistentAgentRuntime(
        **runtime_paths(tmp_path),
        text_agent_runtime="openai",
        extract_in_foreground=True,
    ) as runtime:
        result = await runtime.run_turn(
            thread_id="thread-memory-control",
            user_id="user-1",
            message="What is my memory status?",
            llm_client=ScriptedOpenAITextRouteLLM(route="memory_control"),
        )

        assert result.output.response_style == "memory_control"
        assert "Memory status:" in result.output.response_text
        assert result.output.diagnostics["openai_text_runtime_mode"] == "memory_control"
        assert (
            result.output.diagnostics["openai_selected_agent"] == THERAPEUTIC_AGENT_NAME
        )
        assert result.output.diagnostics["openai_memory_tool_calls"] == [
            "show_memory_status"
        ]
        state = await runtime.get_state("thread-memory-control")
        assert state is not None
        assert state["route"] == "memory_control"
        assert state["memory_control"]["pending_action"] is None
        assert runner.run_calls
        assert runner.stream_calls == []


@pytest.mark.asyncio
async def test_persistent_runtime_openai_grounded_lookup_uses_sdk_tool(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Grounded lookup should run through the SDK tool wrapper."""

    runner = FakeOpenAISDKRunner(
        "unused sdk reply",
        tool_calls=[("answer_grounded_lookup", {"query": "grounded query"})],
        tool_response_as_final=True,
    )
    monkeypatch.setattr(openai_adapter, "_DEFAULT_OPENAI_RUNNER", runner)

    async with PersistentAgentRuntime(
        **runtime_paths(tmp_path),
        text_agent_runtime="openai",
        extract_in_foreground=True,
    ) as runtime:
        result = await runtime.run_turn(
            thread_id="thread-grounded",
            user_id="user-1",
            message="Can you look up the current rule?",
            llm_client=ScriptedOpenAITextRouteLLM(route="grounded_lookup"),
        )

        assert result.output.response_style == "grounded_lookup"
        assert (
            result.output.response_text
            == "Official answer.\n\nSources:\n- Official source"
        )
        assert (
            result.output.diagnostics["openai_text_runtime_mode"] == "grounded_lookup"
        )
        assert (
            result.output.diagnostics["openai_selected_agent"] == THERAPEUTIC_AGENT_NAME
        )
        assert result.output.diagnostics["openai_grounded_tool_calls"] == [
            "answer_grounded_lookup"
        ]
        state = await runtime.get_state("thread-grounded")
        assert state is not None
        assert state["route"] == "grounded_lookup"
        assert state["grounded_lookup"] == {
            "query": "grounded query",
            "status": "answered",
        }
        assert runner.run_calls
        assert runner.stream_calls == []


@pytest.mark.asyncio
async def test_persistent_runtime_openai_crisis_response_uses_crisis_agent(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Level 2/3 crisis turns should be owned by the OpenAI crisis agent."""

    runner = FakeOpenAISDKRunner("Please contact local emergency services now.")
    monkeypatch.setattr(openai_adapter, "_DEFAULT_OPENAI_RUNNER", runner)

    async with PersistentAgentRuntime(
        **runtime_paths(tmp_path),
        text_agent_runtime="openai",
        extract_in_foreground=True,
    ) as runtime:
        result = await runtime.run_turn(
            thread_id="thread-crisis",
            user_id="user-1",
            message="I'm in Singapore and I will end my life tonight.",
            llm_client=ScriptedOpenAITextRouteLLM(
                route="therapeutic",
                crisis_level=3,
            ),
        )

        assert result.output.response_type.value == "crisis"
        assert result.output.response_style == "crisis_response"
        assert (
            result.output.response_text
            == "Please contact local emergency services now."
        )
        assert (
            result.output.diagnostics["openai_text_runtime_mode"] == "crisis_response"
        )
        assert result.output.diagnostics["openai_selected_agent"] == CRISIS_AGENT_NAME
        assert result.output.diagnostics["extract_facts_reason"] == (
            "skipped: crisis_path"
        )
        assert await runtime.crisis_log_backend.arecord_count() == 1
        state = await runtime.get_state("thread-crisis")
        assert state is not None
        assert state["route"] == "crisis"
        assert state["resource_lookup_status"] == "found"
        assert state["found_resources"][0]["phone"] == "1767"
        assert runner.run_calls
        assert runner.run_calls[0]["agent"].name == CRISIS_AGENT_NAME


@pytest.mark.asyncio
async def test_persistent_runtime_openai_level_one_uses_crisis_clarification(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ambiguous level-1 safety turns should not fall back to LangGraph."""

    runner = FakeOpenAISDKRunner("Are you in immediate danger right now?")
    monkeypatch.setattr(openai_adapter, "_DEFAULT_OPENAI_RUNNER", runner)

    async with PersistentAgentRuntime(
        **runtime_paths(tmp_path),
        text_agent_runtime="openai",
        extract_in_foreground=True,
    ) as runtime:
        result = await runtime.run_turn(
            thread_id="thread-crisis-check",
            user_id="user-1",
            message="I might hurt myself.",
            llm_client=ScriptedOpenAITextRouteLLM(
                route="therapeutic",
                crisis_level=1,
            ),
        )

        assert result.output.response_type.value == "therapeutic"
        assert result.output.response_style == "clarifying"
        assert result.output.response_text == "Are you in immediate danger right now?"
        assert (
            result.output.diagnostics["openai_text_runtime_mode"]
            == "crisis_clarification"
        )
        assert result.output.diagnostics["openai_selected_agent"] == CRISIS_AGENT_NAME
        assert await runtime.crisis_log_backend.arecord_count() == 0
        state = await runtime.get_state("thread-crisis-check")
        assert state is not None
        assert state["route"] == "therapeutic"
        assert state["crisis"].level == 1
        assert runner.run_calls
        assert runner.run_calls[0]["agent"].name == CRISIS_AGENT_NAME


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


@pytest.mark.asyncio
async def test_persistent_runtime_openai_memory_control_streaming_surface(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Memory-tool turns should still emit the public stream surface."""

    runner = FakeOpenAISDKRunner(
        "unused sdk reply",
        tool_calls=[("show_memory_status", {})],
        tool_response_as_final=True,
    )
    monkeypatch.setattr(openai_adapter, "_DEFAULT_OPENAI_RUNNER", runner)

    async with PersistentAgentRuntime(
        **runtime_paths(tmp_path),
        text_agent_runtime="openai",
        extract_in_foreground=True,
    ) as runtime:
        events = [
            event
            async for event in runtime.run_turn_stream(
                thread_id="thread-memory-stream",
                user_id="user-1",
                message="What is my memory status?",
                llm_client=ScriptedOpenAITextRouteLLM(route="memory_control"),
            )
        ]

        assert any(
            isinstance(event, ChunkEvent) and "Memory status:" in event.text
            for event in events
        )
        ready = [event for event in events if isinstance(event, ResponseReadyEvent)]
        assert len(ready) == 1
        assert ready[0].output.response_style == "memory_control"
        assert ready[0].output.diagnostics["openai_memory_tool_calls"] == [
            "show_memory_status"
        ]
        assert isinstance(events[-1], DoneEvent)
        assert events[-1].output.response_style == "memory_control"
        assert runner.run_calls == []
        assert runner.stream_calls


@pytest.mark.asyncio
async def test_persistent_runtime_openai_shadow_does_not_mutate_state(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shadow comparison should not write checkpoints or transcript turns."""

    runner = FakeOpenAISDKRunner("shadow-only reply")
    monkeypatch.setattr(openai_adapter, "_DEFAULT_OPENAI_RUNNER", runner)

    async with PersistentAgentRuntime(
        **runtime_paths(tmp_path),
        extract_in_foreground=True,
    ) as runtime:
        result = await runtime.run_openai_text_shadow_turn(
            thread_id="thread-shadow",
            user_id="user-1",
            message="I need help settling down before work.",
            llm_client=FakeCrossRestartLLM(),
        )

        assert result.status == "eligible"
        assert result.eligible is True
        assert result.response_text_length == len("shadow-only reply")
        assert runner.run_calls
        assert (await runtime.session_status("thread-shadow")).value == "absent"
        assert await runtime.get_state("thread-shadow") is None
        assert await runtime.get_history("thread-shadow") == []
