"""Persistent thread runtime built on top of LangGraph checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent.graph import state_to_output
from agent.models import AgentOutput, Channel, Message, MessageRole
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

    def __init__(self, sqlite_path: str | Path = DEFAULT_THREAD_DB_PATH) -> None:
        """Initialize the persistent runtime.

        Args:
            sqlite_path: SQLite database path for LangGraph checkpoints.
        """

        self.sqlite_path = (
            Path(sqlite_path) if sqlite_path != ":memory:" else Path(":memory:")
        )
        self._saver_cm = None
        self._checkpointer: AsyncSqliteSaver | None = None
        self._graph: CompiledStateGraph | None = None

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
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Close the SQLite-backed checkpointer context."""

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
        final_state = await graph.ainvoke(
            {
                "message": message,
                "channel": channel,
                "user_id": user_id,
                "session_id": thread_id,
                "installed_skills": list(installed_skills or []),
            },
            config=self._config_for_thread(thread_id),
            context={"llm_client": llm_client},
        )

        return PersistentTurnResult(
            output=state_to_output(final_state),
            state=final_state,
            history=self._messages_from_transcript(final_state.get("transcript", [])),
        )
