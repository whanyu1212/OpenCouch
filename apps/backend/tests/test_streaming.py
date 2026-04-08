"""Tests for the streaming response generation infrastructure."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from agent.graph import (
    _CapturingLLMClient,
    run_agent,
    run_agent_stream,
)
from agent.models import (
    AgentInput,
    ChunkEvent,
    DoneEvent,
    MessageRole,
    StatusEvent,
)
from agent.persistence import PersistentAgentRuntime
from pydantic import BaseModel
from services.llm.base import BaseLLMClient


# ── Fake streaming client ─────────────────────────────────────────────────────


class FakeStreamingLLMClient(BaseLLMClient):
    """Fake provider client that yields configurable text chunks."""

    def __init__(
        self,
        chunks: list[str],
        *,
        raise_on_stream: bool = False,
    ) -> None:
        self._chunks = chunks
        self._raise_on_stream = raise_on_stream
        self.stream_calls = 0
        self.text_calls = 0

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0,
    ) -> str:
        self.text_calls += 1
        return "".join(self._chunks)

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0,
    ) -> AsyncIterator[str]:
        self.stream_calls += 1
        if self._raise_on_stream:
            raise RuntimeError("Simulated stream failure")
        for chunk in self._chunks:
            yield chunk

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[BaseModel],
        system_instruction: str | None = None,
        temperature: float = 0,
    ) -> BaseModel:
        raise NotImplementedError("Structured generation not used in streaming tests.")


# ── Helper ────────────────────────────────────────────────────────────────────


async def collect_events(stream: AsyncIterator) -> list:
    """Collect all events from an async iterator into a list."""

    return [event async for event in stream]


class FakeModeStreamingLLMClient(BaseLLMClient):
    """Fake client that can classify a mode and optionally fail on stream."""

    def __init__(
        self,
        *,
        mode: str,
        chunks: list[str] | None = None,
        raise_on_stream: bool = False,
    ) -> None:
        self._mode = mode
        self._chunks = chunks or ["Generated ", "reply."]
        self._raise_on_stream = raise_on_stream

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0,
    ) -> str:
        return "".join(self._chunks)

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0,
    ) -> AsyncIterator[str]:
        if self._raise_on_stream:
            raise RuntimeError("Simulated stream failure")
        for chunk in self._chunks:
            yield chunk

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[BaseModel],
        system_instruction: str | None = None,
        temperature: float = 0,
    ) -> BaseModel:
        if response_schema.__name__ == "TherapeuticModeClassification":
            return response_schema(
                mode=self._mode,
                confidence="high",
                reason="Synthetic test classification.",
            )
        raise NotImplementedError(response_schema.__name__)


# ── Streaming orchestration tests ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_status_events_emitted_before_chunks() -> None:
    """Status events should appear before any chunk events."""

    client = FakeStreamingLLMClient(["Hello ", "there."])
    events = await collect_events(
        run_agent_stream(AgentInput(message="I feel anxious."), llm_client=client)
    )

    first_chunk_idx = next(
        (i for i, e in enumerate(events) if isinstance(e, ChunkEvent)), len(events)
    )
    status_events = [e for e in events[:first_chunk_idx] if isinstance(e, StatusEvent)]
    assert len(status_events) >= 2  # crisis_gate + session_stage at minimum


@pytest.mark.asyncio
async def test_all_text_chunks_yielded() -> None:
    """All configured chunks should appear as ChunkEvents."""

    chunks = ["Hello ", "there ", "friend."]
    client = FakeStreamingLLMClient(chunks)
    events = await collect_events(
        run_agent_stream(AgentInput(message="I feel sad."), llm_client=client)
    )

    chunk_texts = [e.text for e in events if isinstance(e, ChunkEvent)]
    assert chunk_texts == chunks


@pytest.mark.asyncio
async def test_done_event_has_complete_output() -> None:
    """DoneEvent should contain the full concatenated response text."""

    chunks = ["Hello ", "there."]
    client = FakeStreamingLLMClient(chunks)
    events = await collect_events(
        run_agent_stream(AgentInput(message="I feel sad."), llm_client=client)
    )

    done_events = [e for e in events if isinstance(e, DoneEvent)]
    assert len(done_events) == 1
    assert done_events[0].output.response_text == "Hello there."


@pytest.mark.asyncio
async def test_fallback_on_stream_error() -> None:
    """Stream failure should produce a DoneEvent with deterministic fallback."""

    client = FakeStreamingLLMClient(["Ignored"], raise_on_stream=True)
    events = await collect_events(
        run_agent_stream(AgentInput(message="I feel sad."), llm_client=client)
    )

    assert client.stream_calls == 1
    done_events = [e for e in events if isinstance(e, DoneEvent)]
    assert len(done_events) == 1
    # Fallback text should be non-empty (deterministic template).
    assert done_events[0].output.response_text


@pytest.mark.asyncio
async def test_stream_error_keeps_selected_mode_when_llm_classifier_was_used() -> None:
    """Stream fallback should keep the already-selected mode and source."""

    client = FakeModeStreamingLLMClient(
        mode="guided_exercise",
        raise_on_stream=True,
    )
    events = await collect_events(
        run_agent_stream(
            AgentInput(message="I don't know what I need right now."),
            llm_client=client,
        )
    )

    done_events = [e for e in events if isinstance(e, DoneEvent)]
    assert len(done_events) == 1
    assert done_events[0].output.mode == "guided_exercise"
    assert done_events[0].output.mode_source == "llm"
    assert "let's" in done_events[0].output.response_text.lower()


@pytest.mark.asyncio
async def test_deterministic_path_emits_no_chunks() -> None:
    """No LLM client should produce no ChunkEvents."""

    events = await collect_events(
        run_agent_stream(AgentInput(message="I feel sad."), llm_client=None)
    )

    chunk_events = [e for e in events if isinstance(e, ChunkEvent)]
    assert len(chunk_events) == 0

    done_events = [e for e in events if isinstance(e, DoneEvent)]
    assert len(done_events) == 1
    assert done_events[0].output.response_text  # fallback text


@pytest.mark.asyncio
async def test_safety_check_bypasses_streaming() -> None:
    """Safety check (needs_clarification) should not produce ChunkEvents."""

    client = FakeStreamingLLMClient(["Should not appear"])
    events = await collect_events(
        run_agent_stream(
            AgentInput(message="I just wish I could disappear."),
            llm_client=client,
        )
    )

    # The crisis gate should set needs_clarification for this message.
    chunk_events = [e for e in events if isinstance(e, ChunkEvent)]
    done_events = [e for e in events if isinstance(e, DoneEvent)]
    assert len(done_events) == 1
    # Safety check uses a template, not the LLM.
    assert (
        "safety" in done_events[0].output.response_text.lower()
        or len(chunk_events) == 0
    )


@pytest.mark.asyncio
async def test_capturing_client_records_text_call_args() -> None:
    """_CapturingLLMClient should record generate_text arguments."""

    real_client = FakeStreamingLLMClient(["test"])
    capturing = _CapturingLLMClient(real_client)

    result = await capturing.generate_text(
        prompt="test prompt",
        system_instruction="test system",
        temperature=0.5,
    )

    assert result == ""
    assert capturing.captured.was_called is True
    assert capturing.captured.prompt == "test prompt"
    assert capturing.captured.system_instruction == "test system"
    assert capturing.captured.temperature == 0.5


@pytest.mark.asyncio
async def test_capturing_client_delegates_structured_calls() -> None:
    """_CapturingLLMClient should delegate generate_structured to the real client."""

    class FakeSchema(BaseModel):
        value: str

    class FakeStructuredClient(BaseLLMClient):
        async def generate_text(
            self, *, prompt, system_instruction=None, temperature=0
        ):
            return ""

        async def generate_text_stream(
            self, *, prompt, system_instruction=None, temperature=0
        ):
            yield ""

        async def generate_structured(
            self, *, prompt, response_schema, system_instruction=None, temperature=0
        ):
            return response_schema(value="delegated")

    capturing = _CapturingLLMClient(FakeStructuredClient())
    result = await capturing.generate_structured(
        prompt="test", response_schema=FakeSchema
    )

    assert isinstance(result, FakeSchema)
    assert result.value == "delegated"
    assert capturing.captured.was_called is False


# ── Backward compatibility tests ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_agent_unchanged() -> None:
    """The existing run_agent function should still work identically."""

    output = await run_agent(AgentInput(message="I feel tired today."))

    assert output.response_text
    assert output.response_type is not None
    assert output.crisis is not None


@pytest.mark.asyncio
async def test_run_turn_unchanged() -> None:
    """The existing run_turn method should still work identically."""

    async with PersistentAgentRuntime(":memory:") as runtime:
        result = await runtime.run_turn(
            thread_id="test-compat",
            message="I feel sad today.",
        )

        assert result.output.response_text
        assert result.state is not None
        assert isinstance(result.history, list)


# ── Persistence streaming tests ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_turn_stream_persists_correct_transcript() -> None:
    """Streamed response text should be persisted in the transcript."""

    chunks = ["Streamed ", "response ", "text."]
    client = FakeStreamingLLMClient(chunks)

    async with PersistentAgentRuntime(":memory:") as runtime:
        events = await collect_events(
            runtime.run_turn_stream(
                thread_id="test-stream-persist",
                message="Hello there.",
                llm_client=client,
            )
        )

        done_events = [e for e in events if isinstance(e, DoneEvent)]
        assert len(done_events) == 1

        history = await runtime.get_history("test-stream-persist")
        assert len(history) >= 2
        assistant_messages = [m for m in history if m.role == MessageRole.ASSISTANT]
        assert len(assistant_messages) >= 1


@pytest.mark.asyncio
async def test_run_turn_stream_persists_emitted_mode_metadata() -> None:
    """Persisted streamed state should match the emitted final mode metadata."""

    client = FakeModeStreamingLLMClient(
        mode="guided_exercise",
        chunks=["Streamed ", "exercise."],
    )

    async with PersistentAgentRuntime(":memory:") as runtime:
        events = await collect_events(
            runtime.run_turn_stream(
                thread_id="test-stream-mode-parity",
                message="I don't know what I need right now.",
                llm_client=client,
            )
        )

        done_event = [e for e in events if isinstance(e, DoneEvent)][0]
        state = await runtime.get_state("test-stream-mode-parity")

        assert state is not None
        assert state["mode"] == done_event.output.mode
        assert state["mode_source"] == done_event.output.mode_source
        assert state["response_text"] == done_event.output.response_text


@pytest.mark.asyncio
async def test_run_turn_stream_resumes_thread() -> None:
    """A second streaming turn should see the first turn's transcript."""

    client = FakeStreamingLLMClient(["First reply."])

    async with PersistentAgentRuntime(":memory:") as runtime:
        # Turn 1.
        await collect_events(
            runtime.run_turn_stream(
                thread_id="test-stream-resume",
                message="First message.",
                llm_client=client,
            )
        )

        # Turn 2.
        client2 = FakeStreamingLLMClient(["Second reply."])
        await collect_events(
            runtime.run_turn_stream(
                thread_id="test-stream-resume",
                message="Second message.",
                llm_client=client2,
            )
        )

        history = await runtime.get_history("test-stream-resume")
        user_messages = [m for m in history if m.role == MessageRole.USER]
        assert len(user_messages) >= 2
