"""Persistent thread runtime for the fresh START -> load_memory -> END graph."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from agent.graph import build_agent_workflow, build_initial_state, state_to_output
from agent.memory.crisis_log import CrisisLogBackend, InMemoryCrisisLogBackend
from agent.memory.modes import MemoryMode
from agent.memory.store import OpenCouchMemoryStore
from agent.models import (
    AgentInput,
    AgentOutput,
    Channel,
    DoneEvent,
    Message,
    MessageRole,
    StatusEvent,
    StreamEvent,
)
from agent.state import AgentState
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
    """Return value for one persisted conversation turn."""

    output: AgentOutput
    state: AgentState
    history: list[Message]


@dataclass(slots=True)
class ThreadSummary:
    """Compact persisted-thread summary for CLI thread management."""

    thread_id: str
    turn_count: int
    message_count: int
    has_context: bool


class PersistentAgentRuntime:
    """Thread-backed runtime with incognito, local, or synced memory mode."""

    def __init__(
        self,
        sqlite_path: str | Path = DEFAULT_THREAD_DB_PATH,
        *,
        memory_store: OpenCouchMemoryStore | None = None,
        crisis_log_backend: CrisisLogBackend | None = None,
        memory_mode: MemoryMode = MemoryMode.LOCAL,
    ) -> None:
        """Initialize the runtime.

        Args:
            sqlite_path: SQLite database path for LangGraph checkpoints.
            memory_store: Optional unified memory store. Defaults to a
                fresh in-memory :class:`OpenCouchMemoryStore`; phase 1
                v0.1 only ships the in-memory backing, SQLite/Postgres
                backings land in later phases.
            crisis_log_backend: Optional crisis log backend. Defaults to
                :class:`InMemoryCrisisLogBackend`. The crisis log is
                always-on regardless of memory_mode (see schema.yaml §2
                namespaces.crisis_log for the privacy asymmetry).
            memory_mode: Persistence tier for the runtime. ``INCOGNITO``
                uses ephemeral in-memory stores only; ``LOCAL`` persists
                to the configured SQLite path; ``SYNCED`` is reserved
                for a future remote backend.
        """

        self.memory_mode = memory_mode
        is_incognito = memory_mode == MemoryMode.INCOGNITO
        resolved_sqlite = ":memory:" if is_incognito else sqlite_path
        self.sqlite_path = (
            Path(resolved_sqlite) if resolved_sqlite != ":memory:" else Path(":memory:")
        )

        self._saver_cm = None
        self._checkpointer: AsyncSqliteSaver | None = None
        self._graph: CompiledStateGraph | None = None

        # v0.1 uses in-memory backings regardless of mode. The mode flag
        # currently affects only the LangGraph checkpointer sqlite path;
        # SQLite-backed memory store + crisis log land in v0.8.
        self._memory_store = memory_store or OpenCouchMemoryStore()
        self._crisis_log_backend = crisis_log_backend or InMemoryCrisisLogBackend()

    async def __aenter__(self) -> PersistentAgentRuntime:
        """Open runtime resources."""

        if self.sqlite_path != Path(":memory:"):
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)

        serde = JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSGPACK_MODULES)
        self._saver_cm = AsyncSqliteSaver.from_conn_string(str(self.sqlite_path))
        self._checkpointer = await self._saver_cm.__aenter__()
        self._checkpointer.serde = serde
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Close runtime resources."""

        await self._memory_store.aclose()
        await self._crisis_log_backend.aclose()
        if self._saver_cm is not None:
            await self._saver_cm.__aexit__(exc_type, exc, tb)

    def _ensure_open(self) -> None:
        """Raise if runtime is used outside its async context."""

        if self._checkpointer is None:
            raise RuntimeError(
                "PersistentAgentRuntime must be used inside 'async with'."
            )

    @property
    def memory_store(self) -> OpenCouchMemoryStore:
        """Return the runtime's unified memory store.

        Exposed for CLI and debug-tooling use (e.g. ``/memory status``).
        The store is the same instance passed into node runtime contexts,
        so reads reflect the current live state.
        """

        return self._memory_store

    @property
    def crisis_log_backend(self) -> CrisisLogBackend:
        """Return the runtime's crisis log backend.

        Exposed for CLI and debug-tooling use. The backend is the same
        instance passed into node runtime contexts and is always-on
        regardless of memory mode.
        """

        return self._crisis_log_backend

    def _config_for_thread(self, thread_id: str) -> dict[str, dict[str, str]]:
        """Build LangGraph config payload for one thread."""

        return {"configurable": {"thread_id": thread_id}}

    def _context_for_turn(
        self,
        *,
        llm_client: BaseLLMClient | None,
    ) -> dict[str, object]:
        """Build LangGraph runtime context for one turn."""

        return {
            "llm_client": llm_client,
            "memory_store": self._memory_store,
            "crisis_log_backend": self._crisis_log_backend,
            "memory_mode": self.memory_mode,
        }

    @staticmethod
    def _messages_from_transcript(transcript: list[dict[str, str]]) -> list[Message]:
        """Materialize validated messages from a serialized transcript."""

        messages: list[Message] = []
        for turn in transcript:
            role = turn.get("role")
            content = turn.get("content", "").strip()
            if role not in {"system", "user", "assistant"} or not content:
                continue
            messages.append(Message(role=MessageRole(role), content=content))
        return messages

    def _get_graph(self) -> CompiledStateGraph:
        """Return the compiled LangGraph workflow for this runtime."""

        self._ensure_open()
        if self._graph is None:
            self._graph = build_agent_workflow(checkpointer=self._checkpointer)
        return self._graph

    async def get_state(self, thread_id: str) -> AgentState | None:
        """Load the latest persisted state snapshot for a thread."""

        graph = self._get_graph()
        snapshot = await graph.aget_state(self._config_for_thread(thread_id))
        values = snapshot.values or {}
        return values or None

    async def get_history(self, thread_id: str) -> list[Message]:
        """Load the full persisted transcript for a thread."""

        state = await self.get_state(thread_id)
        if state is None:
            return []
        return self._messages_from_transcript(state.get("transcript", []))

    async def reset_thread(self, thread_id: str) -> None:
        """Delete all persisted checkpoints for a thread."""

        self._ensure_open()
        await self._checkpointer.adelete_thread(thread_id)

    async def list_threads(self, *, limit: int = 20) -> list[ThreadSummary]:
        """List the most recent persisted threads in SQLite."""

        self._ensure_open()
        await self._checkpointer.setup()

        thread_ids: list[str] = []
        async with (
            self._checkpointer.lock,
            self._checkpointer.conn.execute(
                """
            SELECT thread_id
            FROM checkpoints
            GROUP BY thread_id
            ORDER BY MAX(rowid) DESC
            LIMIT ?
            """,
                (limit,),
            ) as cursor,
        ):
            async for row in cursor:
                thread_ids.append(row[0])

        summaries: list[ThreadSummary] = []
        for thread_id in thread_ids:
            state = await self.get_state(thread_id)
            history = self._messages_from_transcript(
                state.get("transcript", []) if state is not None else []
            )
            progress = state.get("progress", {}) if state is not None else {}
            turn_count = progress.get("turn_count", 0)
            summaries.append(
                ThreadSummary(
                    thread_id=thread_id,
                    turn_count=int(turn_count),
                    message_count=len(history),
                    has_context=state is not None,
                )
            )
        return summaries

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
        """Run one conversation turn through the minimal workflow."""

        graph = self._get_graph()
        history = await self.get_history(thread_id)

        agent_input = AgentInput(
            message=message,
            channel=channel,
            user_id=user_id,
            session_id=thread_id,
            history=history,
            working_memory=[],
            installed_skills=list(installed_skills or []),
        )
        initial_state = build_initial_state(agent_input)

        final_state = await graph.ainvoke(
            initial_state,
            config=self._config_for_thread(thread_id),
            context=self._context_for_turn(llm_client=llm_client),
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
        """Run one turn and yield minimal status + done events."""

        yield StatusEvent(stage="load_memory")
        result = await self.run_turn(
            thread_id=thread_id,
            message=message,
            channel=channel,
            user_id=user_id,
            installed_skills=installed_skills,
            llm_client=llm_client,
        )
        yield DoneEvent(output=result.output)
