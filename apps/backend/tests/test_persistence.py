"""Tests for LangGraph-backed persistent thread state."""

import pytest

from agent.models import Channel, ModeType
from agent.persistence import PersistentAgentRuntime
from agent.state import AgentState
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

    async def generate_text_stream(
        self, *, prompt, system_instruction=None, temperature=0
    ):
        yield await self.generate_text(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
        )

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


class FakeGraphMemoryStore:
    """Fake graph-memory adapter for persistence integration tests."""

    def __init__(self) -> None:
        self.retrieve_calls = 0
        self.record_calls = 0
        self.last_query: str | None = None

    async def retrieve(
        self,
        *,
        owner_id: str,
        query: str,
        limit: int = 4,
    ) -> list[str]:
        self.retrieve_calls += 1
        self.last_query = query
        return [f"graph memory for {owner_id}: anxiety gets worse at night"]

    async def record_episode(
        self,
        *,
        owner_id: str,
        state: AgentState,
    ) -> bool:
        self.record_calls += 1
        return True


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
        assert first_turn.output.mode_type == ModeType.OPERATIONAL
        assert second_turn.output.mode == "pattern_reflection"
        assert second_turn.output.mode_type == ModeType.THERAPEUTIC
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


@pytest.mark.asyncio
async def test_persistent_runtime_retrieves_profile_memory_between_turns(
    tmp_path,
) -> None:
    """Typed profile memory should be written after one turn and injected on the next."""

    sqlite_path = tmp_path / "threads.sqlite3"

    async with PersistentAgentRuntime(sqlite_path) as runtime:
        first_turn = await runtime.run_turn(
            thread_id="thread-memory",
            message="I just want to vent. I don't want advice right now.",
            channel=Channel.TEST,
        )
        second_turn = await runtime.run_turn(
            thread_id="thread-memory",
            message="Can you just stay with me for a moment?",
            channel=Channel.TEST,
        )

    assert first_turn.output.should_persist_memory is True
    assert any(
        "Support preference: Sometimes wants space before advice." in entry
        for entry in second_turn.state["working_memory"]
    )


@pytest.mark.asyncio
async def test_persistent_runtime_retrieves_graph_memory_between_turns(
    tmp_path,
) -> None:
    """Injected graph-memory adapters should feed working_memory on each turn."""

    sqlite_path = tmp_path / "threads.sqlite3"
    graph_memory = FakeGraphMemoryStore()

    async with PersistentAgentRuntime(
        sqlite_path,
        graph_memory_store=graph_memory,
    ) as runtime:
        result = await runtime.run_turn(
            thread_id="thread-graph-memory",
            message="I'm feeling anxious again tonight.",
            channel=Channel.TEST,
        )

    assert any(
        "Related history: graph memory for thread-graph-memory: anxiety gets worse at night"
        in entry
        for entry in result.state["working_memory"]
    )
    assert graph_memory.retrieve_calls == 1
    assert graph_memory.record_calls == 1
    assert result.output.should_persist_memory is True


@pytest.mark.asyncio
async def test_persistent_runtime_skips_graph_memory_for_orientation_turn(
    tmp_path,
) -> None:
    """Orientation turns should not trigger graph-memory retrieval or writes."""

    sqlite_path = tmp_path / "threads.sqlite3"
    graph_memory = FakeGraphMemoryStore()

    async with PersistentAgentRuntime(
        sqlite_path,
        graph_memory_store=graph_memory,
    ) as runtime:
        result = await runtime.run_turn(
            thread_id="thread-orientation-memory",
            message="Hi, what can you do for me?",
            channel=Channel.TEST,
        )

    assert result.output.mode == "orientation"
    assert graph_memory.retrieve_calls == 0
    assert graph_memory.record_calls == 0


@pytest.mark.asyncio
async def test_persistent_runtime_builds_curated_graph_query_when_recall_is_needed(
    tmp_path,
) -> None:
    """Recall-oriented turns should build a richer Graphiti query from prior state."""

    sqlite_path = tmp_path / "threads.sqlite3"
    graph_memory = FakeGraphMemoryStore()

    async with PersistentAgentRuntime(
        sqlite_path,
        graph_memory_store=graph_memory,
    ) as runtime:
        await runtime.run_turn(
            thread_id="thread-pattern-memory",
            message="I've been anxious around my partner lately.",
            channel=Channel.TEST,
        )
        result = await runtime.run_turn(
            thread_id="thread-pattern-memory",
            message="Why does this keep happening with us?",
            channel=Channel.TEST,
        )

    assert any(
        "Related history: graph memory for thread-pattern-memory: anxiety gets worse at night"
        in entry
        for entry in result.state["working_memory"]
    )
    assert graph_memory.retrieve_calls == 1
    assert graph_memory.last_query is not None
    assert (
        "Current user message: Why does this keep happening with us?"
        in graph_memory.last_query
    )
    assert "Active concerns:" in graph_memory.last_query
