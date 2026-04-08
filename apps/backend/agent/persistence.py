"""Persistent thread runtime built on top of LangGraph checkpoints."""

from __future__ import annotations

from collections.abc import AsyncIterator
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from agent.graph import run_agent_stream, state_to_output
from agent.memory_graph import (
    GraphMemoryStore,
    build_graph_memory_query,
    create_graph_memory_store_from_env,
    should_record_graph_episode,
    should_retrieve_graph_memory,
)
from agent.memory_profile import (
    SqliteProfileMemoryStore,
    compile_working_memory,
    extract_profile_memory_writes,
)
from agent.models import (
    AgentInput,
    AgentOutput,
    Channel,
    DoneEvent,
    Message,
    MessageRole,
    StreamEvent,
)
from agent.state import AgentState
from agent.workflow import build_agent_workflow
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph
from services.llm.base import BaseLLMClient

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THREAD_DB_PATH = BACKEND_ROOT / ".opencouch_threads.sqlite3"
ALLOWED_MSGPACK_MODULES = [
    ("agent.models", "Channel"),
    ("agent.models", "CrisisAssessment"),
    ("agent.models", "ModeType"),
    ("agent.models", "ResponseKind"),
]


@dataclass(slots=True)
class PersistentTurnResult:
    """Return value for one persisted conversation turn.

    Attributes:
        output: Public response payload for the completed turn.
        state: Final internal state snapshot stored for the thread.
        history: Full persisted transcript materialized as validated messages.
    """

    output: AgentOutput
    state: AgentState
    history: list[Message]


class PersistentAgentRuntime:
    """Thread-backed runtime that persists graph state in SQLite."""

    def __init__(
        self,
        sqlite_path: str | Path = DEFAULT_THREAD_DB_PATH,
        *,
        graph_memory_store: GraphMemoryStore | None = None,
    ) -> None:
        """Initialize the persistent runtime.

        Args:
            sqlite_path: SQLite database path for LangGraph checkpoints.
            graph_memory_store: Optional graph-memory adapter for episodic retrieval.
        """

        self.sqlite_path = (
            Path(sqlite_path) if sqlite_path != ":memory:" else Path(":memory:")
        )
        self._saver_cm = None
        self._checkpointer: AsyncSqliteSaver | None = None
        self._graph: CompiledStateGraph | None = None
        self._profile_memory = SqliteProfileMemoryStore(self.sqlite_path)
        self._graph_memory = graph_memory_store or create_graph_memory_store_from_env()

    async def __aenter__(self) -> PersistentAgentRuntime:
        """Open the SQLite-backed checkpointer for the runtime.

        Returns:
            The initialized runtime.
        """

        if self.sqlite_path != Path(":memory:"):
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)

        serde = JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSGPACK_MODULES)
        self._saver_cm = AsyncSqliteSaver.from_conn_string(str(self.sqlite_path))
        self._checkpointer = await self._saver_cm.__aenter__()
        self._checkpointer.serde = serde
        await self._profile_memory.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Close the SQLite-backed checkpointer context."""

        await self._profile_memory.close()
        close_graph_memory = getattr(self._graph_memory, "close", None)
        if callable(close_graph_memory):
            await close_graph_memory()
        if self._saver_cm is not None:
            await self._saver_cm.__aexit__(exc_type, exc, tb)

    def _ensure_open(self) -> None:
        """Raise if the runtime has not entered its async context yet.

        Raises:
            RuntimeError: If the runtime is used before entering the context manager.
        """

        if self._checkpointer is None:
            raise RuntimeError(
                "PersistentAgentRuntime must be used inside 'async with'."
            )

    def _config_for_thread(self, thread_id: str) -> dict[str, dict[str, str]]:
        """Build the LangGraph config payload for one thread.

        Args:
            thread_id: Stable thread identifier for the conversation.

        Returns:
            Runnable config payload with the configured thread id.
        """

        return {"configurable": {"thread_id": thread_id}}

    @staticmethod
    def _memory_owner_id(thread_id: str, user_id: str | None) -> str:
        """Resolve the durable owner key for long-term memory storage."""

        return user_id or thread_id

    async def _retrieve_working_memory(
        self,
        *,
        thread_id: str,
        user_id: str | None,
        message: str,
        prior_state: AgentState | None = None,
    ) -> list[str]:
        """Retrieve typed profile and graph memory for the current turn."""

        owner_id = self._memory_owner_id(thread_id, user_id)
        profile_memories = await self._profile_memory.list_memories(owner_id)
        graph_memories: list[str] = []
        if should_retrieve_graph_memory(message=message, prior_state=prior_state):
            graph_memories = await self._graph_memory.retrieve(
                owner_id=owner_id,
                query=build_graph_memory_query(
                    message=message,
                    prior_state=prior_state,
                ),
                limit=3,
            )
        return compile_working_memory(profile_memories, graph_memories)

    async def _persist_memory(
        self,
        *,
        thread_id: str,
        user_id: str | None,
        state: AgentState,
    ) -> bool:
        """Persist post-turn profile memory and hand off episodic writes."""

        owner_id = self._memory_owner_id(thread_id, user_id)
        writes = extract_profile_memory_writes(state)
        await self._profile_memory.upsert_memories(owner_id, writes)
        did_persist_graph_memory = False
        if should_record_graph_episode(state):
            did_persist_graph_memory = await self._graph_memory.record_episode(
                owner_id=owner_id,
                state=state,
            )
        did_persist_memory = bool(writes) or did_persist_graph_memory
        state["should_persist_memory"] = did_persist_memory
        return did_persist_memory

    def _get_graph(self):
        """Return the compiled LangGraph workflow for this runtime.

        Returns:
            A compiled LangGraph workflow.
        """

        self._ensure_open()
        if self._graph is None:
            self._graph = build_agent_workflow(checkpointer=self._checkpointer)
        return self._graph

    @staticmethod
    def _messages_from_transcript(transcript: list[dict[str, str]]) -> list[Message]:
        """Materialize validated messages from a serialized transcript.

        Args:
            transcript: Serialized transcript entries from persisted state.

        Returns:
            Validated transcript entries that the CLI and tests can consume.
        """

        messages: list[Message] = []
        for turn in transcript:
            role = turn.get("role")
            content = turn.get("content", "").strip()
            if role not in {"system", "user", "assistant"} or not content:
                continue
            messages.append(Message(role=MessageRole(role), content=content))
        return messages

    async def get_state(self, thread_id: str) -> AgentState | None:
        """Load the latest persisted state snapshot for a thread.

        Args:
            thread_id: Stable thread identifier to load.

        Returns:
            The latest state snapshot, or `None` if the thread has no checkpoints.
        """

        graph = self._get_graph()
        snapshot = await graph.aget_state(self._config_for_thread(thread_id))
        values = snapshot.values or {}
        return values or None

    async def get_history(self, thread_id: str) -> list[Message]:
        """Load the full persisted transcript for a thread.

        Args:
            thread_id: Stable thread identifier to load.

        Returns:
            The persisted transcript for the thread.
        """

        state = await self.get_state(thread_id)
        if state is None:
            return []
        return self._messages_from_transcript(state.get("transcript", []))

    async def reset_thread(self, thread_id: str) -> None:
        """Delete all persisted checkpoints for a thread.

        Args:
            thread_id: Stable thread identifier to delete.

        Returns:
            None.
        """

        self._ensure_open()
        await self._checkpointer.adelete_thread(thread_id)

    async def run_turn(
        self,
        *,
        thread_id: str,
        message: str,
        channel: Channel = Channel.TEST,
        user_id: str | None = None,
        installed_skills: list[str] | None = None,
        llm_client: BaseLLMClient | None = None,
    ) -> PersistentTurnResult:
        """Run one persisted conversation turn through the LangGraph workflow.

        Args:
            thread_id: Stable thread identifier for the conversation.
            message: Current inbound user message.
            channel: Input channel for the current turn.
            user_id: Optional user identifier for future ownership checks.
            installed_skills: Optional skill names to pass into the turn.
            llm_client: Optional provider-backed client for classification/generation.

        Returns:
            The completed turn result with output, stored state, and transcript.
        """

        graph = self._get_graph()
        existing_state = await self.get_state(thread_id)
        working_memory = await self._retrieve_working_memory(
            thread_id=thread_id,
            user_id=user_id,
            message=message,
            prior_state=existing_state,
        )
        final_state = await graph.ainvoke(
            {
                "message": message,
                "channel": channel,
                "user_id": user_id,
                "session_id": thread_id,
                "working_memory": working_memory,
                "installed_skills": list(installed_skills or []),
            },
            config=self._config_for_thread(thread_id),
            context={"llm_client": llm_client},
        )
        await self._persist_memory(
            thread_id=thread_id,
            user_id=user_id,
            state=final_state,
        )

        return PersistentTurnResult(
            output=state_to_output(final_state),
            state=final_state,
            history=self._messages_from_transcript(final_state.get("transcript", [])),
        )

    async def run_turn_stream(
        self,
        *,
        thread_id: str,
        message: str,
        channel: Channel = Channel.TEST,
        user_id: str | None = None,
        installed_skills: list[str] | None = None,
        llm_client: BaseLLMClient | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Run one persisted conversation turn, yielding stream events.

        Yields StatusEvent and ChunkEvent during execution, then a final
        DoneEvent. After all events are yielded, the finalized streamed state
        is checkpointed directly so persistence matches the emitted result.

        Args:
            thread_id: Stable thread identifier for the conversation.
            message: Current inbound user message.
            channel: Input channel for the current turn.
            user_id: Optional user identifier for future ownership checks.
            installed_skills: Optional skill names to pass into the turn.
            llm_client: Optional provider-backed client for classification/generation.

        Yields:
            StreamEvent instances for pipeline progress, text chunks, and completion.
        """

        existing_state = await self.get_state(thread_id)
        history: list[Message] = []
        if existing_state is not None:
            history = self._messages_from_transcript(
                existing_state.get("transcript", [])
            )
        working_memory = await self._retrieve_working_memory(
            thread_id=thread_id,
            user_id=user_id,
            message=message,
            prior_state=existing_state,
        )

        agent_input = AgentInput(
            message=message,
            channel=channel,
            user_id=user_id,
            session_id=thread_id,
            history=history,
            working_memory=working_memory,
            installed_skills=list(installed_skills or []),
        )

        final_states: list[AgentState] = []
        final_done_event: DoneEvent | None = None
        async for event in run_agent_stream(
            agent_input,
            llm_client=llm_client,
            state_sink=final_states,
        ):
            if isinstance(event, DoneEvent):
                final_done_event = event
                continue
            yield event

        if not final_states:
            raise RuntimeError("Streaming turn completed without a final agent state.")

        persisted_state = deepcopy(final_states[-1])
        await self._persist_memory(
            thread_id=thread_id,
            user_id=user_id,
            state=persisted_state,
        )
        persisted_state["history"] = []
        graph = self._get_graph()
        await graph.aupdate_state(
            self._config_for_thread(thread_id),
            persisted_state,
            as_node="compact_persisted_state",
        )

        if final_done_event is None:
            raise RuntimeError("Streaming turn completed without a DoneEvent.")
        yield DoneEvent(output=state_to_output(persisted_state))
