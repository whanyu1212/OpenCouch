"""Integration coverage for the OpenAI text runtime."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from agent.models import ChunkEvent, DoneEvent, ResponseReadyEvent
import agent.runtime.openai_text_runtime as openai_runtime
from agent.runtime import PersistentAgentRuntime
from agent.specialists.crisis import CRISIS_AGENT_NAME
from agent.specialists.therapeutic import THERAPEUTIC_AGENT_NAME
from agent.runtime.session_store import messages_from_sdk_session_items
from tests.support.openai_text import (
    FakeOpenAISDKRunner,
    ScriptedOpenAITextRouteLLM,
)
from tests.support.persistence import FakeCrossRestartLLM, runtime_paths


class _SessionInspectingOpenAIRunner(FakeOpenAISDKRunner):
    """Fake SDK runner that records model-visible SDK history at handoff."""

    def __init__(self, final_output: str = "openai reply") -> None:
        super().__init__(final_output)
        self.run_session_history: list[list[tuple[str, str]]] = []
        self.stream_session_history: list[list[tuple[str, str]]] = []

    async def run(
        self,
        *,
        agent: Any,
        input_text: str,
        context: Any,
        session: Any | None = None,
    ) -> Any:
        self.run_session_history.append(await _session_history(session))
        return await super().run(
            agent=agent,
            input_text=input_text,
            context=context,
            session=session,
        )

    def run_streamed(
        self,
        *,
        agent: Any,
        input_text: str,
        context: Any,
        session: Any | None = None,
    ) -> Any:
        stream = super().run_streamed(
            agent=agent,
            input_text=input_text,
            context=context,
            session=session,
        )
        return _SessionInspectingStream(
            stream,
            session=session,
            observed_history=self.stream_session_history,
        )


class _SessionInspectingStream:
    """Stream wrapper that snapshots SDK history before yielding events."""

    def __init__(
        self,
        stream: Any,
        *,
        session: Any | None,
        observed_history: list[list[tuple[str, str]]],
    ) -> None:
        self._stream = stream
        self._session = session
        self._observed_history = observed_history

    @property
    def final_output(self) -> Any:
        return getattr(self._stream, "final_output", None)

    async def stream_events(self) -> Any:
        self._observed_history.append(await _session_history(self._session))
        async for event in self._stream.stream_events():
            yield event


class _RecordingTextLLM(FakeCrossRestartLLM):
    """Fake response LLM that records prompt text for boundary assertions."""

    def __init__(self, text: str = "recorded response") -> None:
        super().__init__()
        self.text = text
        self.prompts: list[str] = []
        self.system_instructions: list[str | None] = []

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        del use_search
        self.prompts.append(prompt)
        self.system_instructions.append(system_instruction)
        return self.text

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        self.prompts.append(prompt)
        self.system_instructions.append(system_instruction)
        yield self.text


async def _session_history(session: Any | None) -> list[tuple[str, str]]:
    if session is None:
        return []
    messages = messages_from_sdk_session_items(await session.get_items())
    return [(message.role.value, message.content) for message in messages]


@pytest.mark.asyncio
async def test_persistent_runtime_openai_safe_turn_persists_transcript(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The OpenAI runtime should produce a normal persisted text turn."""

    runner = FakeOpenAISDKRunner("openai persistent reply")
    monkeypatch.setattr(openai_runtime, "_DEFAULT_OPENAI_RUNNER", runner)

    async with PersistentAgentRuntime(
        **runtime_paths(tmp_path),
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
        second_input = runner.run_calls[1]["input_text"]
        assert "I feel tense before a presentation." not in second_input
        assert "openai persistent reply" not in second_input
        assert "It is still bothering me." in second_input
        assert [call["session"] is not None for call in runner.run_calls[:2]] == [
            True,
            True,
        ]


@pytest.mark.asyncio
async def test_persistent_runtime_disabled_sdk_session_keeps_legacy_prompt_history(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without SDK sessions, prompts should still carry recent transcript."""

    runner = FakeOpenAISDKRunner("legacy first reply")
    monkeypatch.setattr(openai_runtime, "_DEFAULT_OPENAI_RUNNER", runner)

    async with PersistentAgentRuntime(
        **runtime_paths(tmp_path),
        text_session_backend="disabled",
    ) as runtime:
        await runtime.run_turn(
            thread_id="thread-legacy-history",
            user_id="user-1",
            message="I get tense before presentations.",
            llm_client=FakeCrossRestartLLM(),
        )

        runner.final_output = "legacy second reply"
        await runtime.run_turn(
            thread_id="thread-legacy-history",
            user_id="user-1",
            message="Can you remind me what I said?",
            llm_client=FakeCrossRestartLLM(),
        )

        second_input = runner.run_calls[1]["input_text"]
        assert runner.run_calls[1]["session"] is None
        assert "I get tense before presentations." in second_input
        assert "legacy first reply" in second_input
        assert "Can you remind me what I said?" in second_input


@pytest.mark.asyncio
async def test_persistent_runtime_seeds_empty_openai_sdk_session_from_state(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persisted runtime transcript should backfill an empty SDK session."""

    runner = _SessionInspectingOpenAIRunner("first seeded reply")
    monkeypatch.setattr(openai_runtime, "_DEFAULT_OPENAI_RUNNER", runner)

    async with PersistentAgentRuntime(
        **runtime_paths(tmp_path),
    ) as runtime:
        await runtime.run_turn(
            thread_id="thread-seed-sdk",
            user_id="user-1",
            message="My presentation is on Tuesday.",
            llm_client=FakeCrossRestartLLM(),
        )

        assert runtime._text_session_store is not None  # noqa: SLF001
        await runtime._text_session_store.clear_thread("thread-seed-sdk")  # noqa: SLF001
        runner.final_output = "second seeded reply"

        await runtime.run_turn(
            thread_id="thread-seed-sdk",
            user_id="user-1",
            message="What did I say was coming up?",
            llm_client=FakeCrossRestartLLM(),
        )

        assert runner.run_session_history[1] == [
            ("user", "My presentation is on Tuesday."),
            ("assistant", "first seeded reply"),
        ]
        history = await runtime._text_session_store.get_history(  # noqa: SLF001
            "thread-seed-sdk"
        )
        assert [(message.role.value, message.content) for message in history] == [
            ("user", "My presentation is on Tuesday."),
            ("assistant", "first seeded reply"),
            ("user", "What did I say was coming up?"),
            ("assistant", "second seeded reply"),
        ]


@pytest.mark.asyncio
async def test_persistent_runtime_stream_seeds_empty_openai_sdk_session_from_state(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming turns should also seed SDK history before hiding prior state."""

    runner = _SessionInspectingOpenAIRunner("first streamed-seed reply")
    monkeypatch.setattr(openai_runtime, "_DEFAULT_OPENAI_RUNNER", runner)

    async with PersistentAgentRuntime(
        **runtime_paths(tmp_path),
    ) as runtime:
        await runtime.run_turn(
            thread_id="thread-stream-seed-sdk",
            user_id="user-1",
            message="The review meeting is on Friday.",
            llm_client=FakeCrossRestartLLM(),
        )

        assert runtime._text_session_store is not None  # noqa: SLF001
        await runtime._text_session_store.clear_thread(  # noqa: SLF001
            "thread-stream-seed-sdk"
        )
        runner.final_output = "second streamed-seed reply"

        events = [
            event
            async for event in runtime.run_turn_stream(
                thread_id="thread-stream-seed-sdk",
                user_id="user-1",
                message="What timing did I mention?",
                llm_client=FakeCrossRestartLLM(),
            )
        ]

        assert runner.stream_session_history[0] == [
            ("user", "The review meeting is on Friday."),
            ("assistant", "first streamed-seed reply"),
        ]
        streamed_input = runner.stream_calls[0]["input_text"]
        assert "The review meeting is on Friday." not in streamed_input
        assert "first streamed-seed reply" not in streamed_input
        assert "What timing did I mention?" in streamed_input
        assert isinstance(events[-1], DoneEvent)
        history = await runtime._text_session_store.get_history(  # noqa: SLF001
            "thread-stream-seed-sdk"
        )
        assert [(message.role.value, message.content) for message in history] == [
            ("user", "The review meeting is on Friday."),
            ("assistant", "first streamed-seed reply"),
            ("user", "What timing did I mention?"),
            ("assistant", "second streamed-seed reply"),
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
    monkeypatch.setattr(openai_runtime, "_DEFAULT_OPENAI_RUNNER", runner)

    async with PersistentAgentRuntime(
        **runtime_paths(tmp_path),
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
    monkeypatch.setattr(openai_runtime, "_DEFAULT_OPENAI_RUNNER", runner)

    async with PersistentAgentRuntime(
        **runtime_paths(tmp_path),
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

    runner = FakeOpenAISDKRunner(
        "Please contact local emergency services now.",
        tool_calls=[("lookup_crisis_resources", {})],
    )
    monkeypatch.setattr(openai_runtime, "_DEFAULT_OPENAI_RUNNER", runner)

    async with PersistentAgentRuntime(
        **runtime_paths(tmp_path),
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
        assert result.output.diagnostics["openai_crisis_tool_calls"] == [
            "lookup_crisis_resources"
        ]
        assert result.output.diagnostics["openai_crisis_tool_fallback"] is False
        assert "extract_facts_reason" not in result.output.diagnostics
        assert await runtime.crisis_log_backend.arecord_count() == 1
        state = await runtime.get_state("thread-crisis")
        assert state is not None
        assert state["route"] == "crisis"
        assert state["resource_lookup_status"] == "found"
        assert state["found_resources"][0]["phone"] == "1767"
        assert runner.run_calls
        assert runner.run_calls[0]["agent"].name == CRISIS_AGENT_NAME
        assert [tool.name for tool in runner.run_calls[0]["agent"].tools] == [
            "lookup_crisis_resources"
        ]


@pytest.mark.asyncio
async def test_persistent_runtime_openai_crisis_uses_sdk_session_not_prompt_history(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Crisis prompts should not replay transcript when SDK session is active."""

    runner = FakeOpenAISDKRunner("prior assistant reply")
    monkeypatch.setattr(openai_runtime, "_DEFAULT_OPENAI_RUNNER", runner)

    async with PersistentAgentRuntime(
        **runtime_paths(tmp_path),
    ) as runtime:
        await runtime.run_turn(
            thread_id="thread-crisis-boundary",
            user_id="user-1",
            message="I feel tense before presentations.",
            llm_client=FakeCrossRestartLLM(),
        )

        runner.final_output = "Please contact local emergency services now."
        runner.tool_calls = [("lookup_crisis_resources", {})]
        await runtime.run_turn(
            thread_id="thread-crisis-boundary",
            user_id="user-1",
            message="I'm in Singapore and I will end my life tonight.",
            llm_client=ScriptedOpenAITextRouteLLM(
                route="therapeutic",
                crisis_level=3,
            ),
        )

        crisis_input = runner.run_calls[1]["input_text"]
        assert runner.run_calls[1]["session"] is not None
        assert "I feel tense before presentations." not in crisis_input
        assert "prior assistant reply" not in crisis_input
        assert "I will end my life tonight." in crisis_input


@pytest.mark.asyncio
async def test_persistent_runtime_crisis_response_llm_omits_sdk_session_history(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct crisis response LLM overrides must follow the SDK boundary."""

    runner = FakeOpenAISDKRunner("prior assistant reply")
    response_llm = _RecordingTextLLM("override crisis reply")
    monkeypatch.setattr(openai_runtime, "_DEFAULT_OPENAI_RUNNER", runner)

    async with PersistentAgentRuntime(
        **runtime_paths(tmp_path),
    ) as runtime:
        await runtime.run_turn(
            thread_id="thread-crisis-response-llm-boundary",
            user_id="user-1",
            message="I freeze before presentations.",
            llm_client=FakeCrossRestartLLM(),
        )

        result = await runtime.run_turn(
            thread_id="thread-crisis-response-llm-boundary",
            user_id="user-1",
            message="I'm in Singapore and I will end my life tonight.",
            llm_client=ScriptedOpenAITextRouteLLM(
                route="therapeutic",
                crisis_level=3,
            ),
            response_llm_client=response_llm,
        )

        assert result.output.response_text == "override crisis reply"
        prompt = response_llm.prompts[-1]
        assert "I freeze before presentations." not in prompt
        assert "prior assistant reply" not in prompt
        assert "I will end my life tonight." in prompt


@pytest.mark.asyncio
async def test_persistent_runtime_openai_level_one_uses_crisis_clarification(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ambiguous level-1 safety turns should not fall back to runtime."""

    runner = FakeOpenAISDKRunner("Are you in immediate danger right now?")
    monkeypatch.setattr(openai_runtime, "_DEFAULT_OPENAI_RUNNER", runner)

    async with PersistentAgentRuntime(
        **runtime_paths(tmp_path),
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
        assert runner.run_calls[0]["agent"].tools == []


@pytest.mark.asyncio
async def test_persistent_runtime_openai_streaming_surface(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The OpenAI streaming path should keep the public event surface intact."""

    runner = FakeOpenAISDKRunner("openai streamed reply")
    monkeypatch.setattr(openai_runtime, "_DEFAULT_OPENAI_RUNNER", runner)

    async with PersistentAgentRuntime(
        **runtime_paths(tmp_path),
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
    monkeypatch.setattr(openai_runtime, "_DEFAULT_OPENAI_RUNNER", runner)

    async with PersistentAgentRuntime(
        **runtime_paths(tmp_path),
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
    """Shadow comparison should not write state snapshots or transcript turns."""

    runner = FakeOpenAISDKRunner("shadow-only reply")
    monkeypatch.setattr(openai_runtime, "_DEFAULT_OPENAI_RUNNER", runner)

    async with PersistentAgentRuntime(
        **runtime_paths(tmp_path),
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


@pytest.mark.asyncio
async def test_persistent_runtime_guided_response_llm_omits_sdk_session_history(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guided-exercise response LLM overrides must strip transcript history."""

    runner = FakeOpenAISDKRunner("prior assistant reply")
    response_llm = _RecordingTextLLM("guided fallback reply")
    monkeypatch.setattr(openai_runtime, "_DEFAULT_OPENAI_RUNNER", runner)

    async with PersistentAgentRuntime(
        **runtime_paths(tmp_path),
    ) as runtime:
        await runtime.run_turn(
            thread_id="thread-guided-response-llm-boundary",
            user_id="user-1",
            message="I freeze before presentations.",
            llm_client=FakeCrossRestartLLM(),
        )

        result = await runtime.run_turn(
            thread_id="thread-guided-response-llm-boundary",
            user_id="user-1",
            message="Can we do box breathing?",
            llm_client=ScriptedOpenAITextRouteLLM(
                route="therapeutic",
                therapeutic_response_style="guided_exercise",
                exercise_start_basis="explicit_user_request",
                exercise_type="grounding_box_breathing",
            ),
            response_llm_client=response_llm,
        )

        assert result.output.response_style == "guided_exercise"
        prompt = response_llm.prompts[-1]
        assert "I freeze before presentations." not in prompt
        assert "prior assistant reply" not in prompt
        assert "Can we do box breathing?" in prompt
        assert "(conversation history is provided by the SDK session)" in prompt
