"""Tests for LangGraph-backed persistent thread state."""

import pytest

from agent.models import Channel
from agent.persistence import PersistentAgentRuntime
from services.llm.base import BaseLLMClient


class FakeContextLLMClient(BaseLLMClient):
    """Fake provider client used to verify runtime-context injection."""

    def __init__(self) -> None:
        self.text_calls = 0

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0,
    ) -> str:
        """Return a fixed text response for runtime-context tests.

        Args:
            prompt: User/task prompt sent to the fake provider.
            system_instruction: Optional system prompt sent to the fake provider.
            temperature: Sampling temperature for generation.

        Returns:
            A fixed provider response string.
        """

        self.text_calls += 1
        return "Context-injected reply"

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema,
        system_instruction: str | None = None,
        temperature: float = 0,
    ):
        """Raise because structured generation is not used in this test.

        Args:
            prompt: User/task prompt sent to the fake provider.
            response_schema: Structured schema requested by the caller.
            system_instruction: Optional system prompt sent to the fake provider.
            temperature: Sampling temperature for generation.

        Returns:
            This function does not return a value.

        Raises:
            NotImplementedError: Always raised for this fake client path.
        """

        raise NotImplementedError


@pytest.mark.asyncio
async def test_persistent_runtime_resumes_thread_state(tmp_path) -> None:
    """Persisted threads should resume transcript and context across turns.

    Args:
        tmp_path: Pytest-provided temporary directory for the SQLite database.

    Returns:
        None.
    """

    sqlite_path = tmp_path / "threads.sqlite3"

    async with PersistentAgentRuntime(sqlite_path) as runtime:
        first_turn = await runtime.run_turn(
            thread_id="thread-a",
            message="Hi, I'm new here. How does this work?",
            channel=Channel.TEST,
        )
        second_turn = await runtime.run_turn(
            thread_id="thread-a",
            message="Can you help me understand why I keep ending up in the same pattern with people?",
            channel=Channel.TEST,
        )

        assert first_turn.output.mode == "orientation"
        assert second_turn.output.mode == "reflection"
        assert len(second_turn.history) == 4
        assert second_turn.state["turn_count"] == 2
        assert len(second_turn.state["transcript"]) == 4

        persisted_state = await runtime.get_state("thread-a")

    assert persisted_state is not None
    assert len(persisted_state["transcript"]) == 4
    assert persisted_state["turn_count"] == 2
    assert persisted_state["history"] == []


@pytest.mark.asyncio
async def test_persistent_runtime_can_reset_thread(tmp_path) -> None:
    """Resetting a thread should remove persisted checkpoints.

    Args:
        tmp_path: Pytest-provided temporary directory for the SQLite database.

    Returns:
        None.
    """

    sqlite_path = tmp_path / "threads.sqlite3"

    async with PersistentAgentRuntime(sqlite_path) as runtime:
        await runtime.run_turn(
            thread_id="thread-reset",
            message="I feel overwhelmed lately.",
            channel=Channel.TEST,
        )
        await runtime.reset_thread("thread-reset")

        persisted_state = await runtime.get_state("thread-reset")
        persisted_history = await runtime.get_history("thread-reset")

    assert persisted_state is None
    assert persisted_history == []


@pytest.mark.asyncio
async def test_persistent_runtime_uses_runtime_context_for_llm_client(tmp_path) -> None:
    """Workflow nodes should receive the provider client through runtime context.

    Args:
        tmp_path: Pytest-provided temporary directory for the SQLite database.

    Returns:
        None.
    """

    sqlite_path = tmp_path / "threads.sqlite3"
    llm_client = FakeContextLLMClient()

    async with PersistentAgentRuntime(sqlite_path) as runtime:
        result = await runtime.run_turn(
            thread_id="thread-context",
            message="I had a rough day and feel drained.",
            channel=Channel.TEST,
            llm_client=llm_client,
        )

    assert result.output.response_text == "Context-injected reply"
    assert llm_client.text_calls == 1
