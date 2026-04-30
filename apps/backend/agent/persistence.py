"""Persistent runtime for thread-backed OpenCouch sessions."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

from langchain_core.runnables import RunnableConfig

from agent.graph import build_agent_workflow, build_initial_state, state_to_output
from agent.memory.candidates import SessionMemoryBuffer
from agent.audit.crisis_log import CrisisLogBackend, InMemoryCrisisLogBackend
from agent.audit.session_feedback import (
    InMemorySessionFeedbackBackend,
    SessionFeedbackBackend,
)
from agent.audit.sqlite_crisis_log import SqliteCrisisLogBackend
from agent.audit.sqlite_session_feedback import SqliteSessionFeedbackBackend
from agent.memory.hashing import hash_session_id, iso_now
from agent.memory.hashing import iso_now as _iso_now
from agent.memory.embeddings import (
    EmbeddingProvider,
    NullEmbeddingProvider,
    create_configured_embedding_provider,
)
from agent.audit.models import FeedbackLabel, FeedbackSource, SessionFeedbackRecord
from agent.memory.models import StoredSessionArc
from agent.memory.modes import MemoryMode
from agent.memory.sqlite_store import SqliteMemoryStore
from agent.memory.store import MemoryStore, OpenCouchMemoryStore
from agent.models import (
    AgentInput,
    AgentOutput,
    Channel,
    ChunkEvent,
    CrisisAssessment,
    DoneEvent,
    Message,
    MessageRole,
    ResponseReadyEvent,
    StatusEvent,
    StreamEvent,
)
from agent.nodes.commit_session_memory import run_commit_session_memory
from agent.nodes.extract_facts import run_extract_semantic_facts_node
from agent.nodes.extract_procedural_rules import run_extract_procedural_rules_node
from agent.nodes.summarize_session import run_summarize_session
from agent.runtime_context import WorkflowContext
from agent.state import AgentGraphInputState, AgentGraphOutputState, AgentState
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph
from services.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)

AgentWorkflow = CompiledStateGraph[
    AgentState,
    WorkflowContext,
    AgentGraphInputState,
    AgentGraphOutputState,
]

# Keep graph node names and CLI/API stage labels aligned in one place.
_NODE_TO_STAGE = {
    "load_memory_node": "load_memory",
    "crisis_gate_node": "crisis_gate",
    "crisis_resource_lookup_node": "crisis_resource_lookup",
    "crisis_response_node": "crisis_response",
    "crisis_log_node": "crisis_log",
    "memory_control_gate_node": "memory_control_gate",
    "memory_control_node": "memory_control",
    "grounded_lookup_gate_node": "grounded_lookup_gate",
    "grounded_answer_node": "grounded_lookup",
    "therapeutic_subgraph": "therapeutic",
    "extract_semantic_facts_node": "extract_facts",
    "extract_procedural_rules_node": "extract_procedural",
    "finalize_turn_node": "finalize",
}

BACKEND_ROOT = Path(__file__).resolve().parents[1]
_STORE_DIR = BACKEND_ROOT / ".store"
DEFAULT_THREAD_DB_PATH = _STORE_DIR / "threads.sqlite3"
# Keep runtime-owned stores separate from the LangGraph checkpoint DB.
DEFAULT_MEMORY_DB_PATH = _STORE_DIR / "memory.sqlite3"
DEFAULT_CRISIS_LOG_DB_PATH = _STORE_DIR / "crisis.sqlite3"
DEFAULT_FEEDBACK_DB_PATH = _STORE_DIR / "session_feedback.sqlite3"
ALLOWED_MSGPACK_MODULES = [
    ("agent.models", "Channel"),
    ("agent.models", "CrisisAssessment"),
    ("agent.models", "ResponseCategory"),
    ("agent.models", "ResponseStyleType"),
]
SESSION_TIMEOUT = timedelta(minutes=20)
ACTIVE_SESSION_STATE_DDL = """
CREATE TABLE IF NOT EXISTS opencouch_active_sessions (
    thread_id TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    mutation_token TEXT,
    mutation_kind TEXT,
    rotate_after_this_turn INTEGER NOT NULL DEFAULT 0,
    finalize_required_reason TEXT
);
"""
ACTIVE_SESSION_EXTRA_COLUMNS = {
    "mutation_token": "TEXT",
    "mutation_kind": "TEXT",
    "rotate_after_this_turn": "INTEGER NOT NULL DEFAULT 0",
    "finalize_required_reason": "TEXT",
}
_EXERCISE_STATE_FIELDS = (
    "exercise_type",
    "exercise_step",
    "exercise_therapeutic_approach",
    "exercise_selection_options",
)
ExpectedSessionLiveness = Literal["active", "absent"]


class SessionStatus(StrEnum):
    """Runtime liveness states for active-session coordination."""

    ABSENT = "absent"
    ACTIVE = "active"
    EXPIRED_UNFINALIZED = "expired_unfinalized"
    INTERRUPTED = "interrupted"
    ROTATION_REQUIRED = "rotation_required"


class SessionLeaseExpired(RuntimeError):
    """Raised when a turn was submitted against a non-active session lease."""

    def __init__(self, thread_id: str, status: SessionStatus) -> None:
        """Initialize the liveness mismatch error.

        Args:
            thread_id: Thread whose lease check failed.
            status: Observed session status.
        """

        self.thread_id = thread_id
        self.status = status
        super().__init__(
            f"thread {thread_id!r} is not active; observed status={status.value}"
        )


class ActiveSessionExists(RuntimeError):
    """Raised when a caller expected no active session but one exists."""

    def __init__(self, thread_id: str, status: SessionStatus) -> None:
        """Initialize the active-session conflict.

        Args:
            thread_id: Thread whose absence check failed.
            status: Observed session status.
        """

        self.thread_id = thread_id
        self.status = status
        super().__init__(
            f"thread {thread_id!r} already has session status={status.value}"
        )


class SessionInterrupted(RuntimeError):
    """Raised when a persisted session needs explicit recovery finalization."""

    def __init__(self, thread_id: str) -> None:
        """Initialize the interrupted-session error.

        Args:
            thread_id: Thread whose session is interrupted.
        """

        self.thread_id = thread_id
        super().__init__(f"thread {thread_id!r} has an interrupted session")


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


@dataclass(slots=True)
class _RuntimeShim:
    """Minimal runtime shim for direct node reuse outside LangGraph."""

    context: WorkflowContext


@dataclass(slots=True)
class PersistedActiveSessionState:
    """Durable runtime-owned session state for active-session recovery."""

    thread_id: str
    started_at: str
    last_active_at: str
    transcript_start_index: int
    max_crisis_level: int
    session_buffer: SessionMemoryBuffer

    def to_json(self) -> str:
        """Serialize the session record to JSON.

        Returns:
            The JSON-encoded session payload.
        """

        return json.dumps(
            {
                "thread_id": self.thread_id,
                "started_at": self.started_at,
                "last_active_at": self.last_active_at,
                "transcript_start_index": self.transcript_start_index,
                "max_crisis_level": self.max_crisis_level,
                "session_buffer": self.session_buffer.model_dump(mode="json"),
            }
        )

    @classmethod
    def from_json(cls, payload_json: str) -> PersistedActiveSessionState:
        """Deserialize a persisted session record.

        Args:
            payload_json: The JSON payload to decode.

        Returns:
            The decoded ``PersistedActiveSessionState`` instance.
        """

        payload = json.loads(payload_json)
        return cls(
            thread_id=str(payload["thread_id"]),
            started_at=str(payload["started_at"]),
            last_active_at=str(payload["last_active_at"]),
            transcript_start_index=max(
                0, int(payload.get("transcript_start_index", 0) or 0)
            ),
            max_crisis_level=max(0, int(payload.get("max_crisis_level", 0) or 0)),
            session_buffer=SessionMemoryBuffer.model_validate(
                payload.get("session_buffer")
                or {"session_id": str(payload["thread_id"])}
            ),
        )


@dataclass(slots=True)
class _PersistedActiveSessionRow:
    """Raw active-session row including coordination metadata."""

    payload_json: str
    mutation_token: str | None
    mutation_kind: str | None
    rotate_after_this_turn: bool
    finalize_required_reason: str | None


def _parse_iso_timestamp(value: str | None) -> datetime | None:
    """Parse a stored ISO timestamp.

    Args:
        value: The timestamp string to parse.

    Returns:
        A parsed ``datetime`` or ``None`` when parsing fails.
    """

    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class PersistentAgentRuntime:
    """Thread-backed runtime with mode-aware persistence backends."""

    def __init__(
        self,
        sqlite_path: str | Path = DEFAULT_THREAD_DB_PATH,
        *,
        memory_store: MemoryStore | None = None,
        crisis_log_backend: CrisisLogBackend | None = None,
        session_feedback_backend: SessionFeedbackBackend | None = None,
        memory_mode: MemoryMode = MemoryMode.LOCAL,
        memory_sqlite_path: str | Path = DEFAULT_MEMORY_DB_PATH,
        crisis_log_sqlite_path: str | Path = DEFAULT_CRISIS_LOG_DB_PATH,
        feedback_sqlite_path: str | Path = DEFAULT_FEEDBACK_DB_PATH,
        embedding_provider: "EmbeddingProvider | None" = None,
        default_llm_client: BaseLLMClient | None = None,
        session_timeout: timedelta = SESSION_TIMEOUT,
        session_sweep_interval_seconds: float = 30.0,
        finalize_active_sessions_on_close: bool = True,
        auto_finalize_excluded: Callable[[str], bool] | None = None,
    ) -> None:
        """Initialize the runtime.

        Args:
            sqlite_path: SQLite database path for LangGraph checkpoints.
                Forced to ``:memory:`` in incognito mode.
            memory_store: Optional explicit memory-store override.
            crisis_log_backend: Optional explicit crisis-log override.
            session_feedback_backend: Optional explicit feedback-backend override.
            memory_mode: Persistence tier for the runtime.
            memory_sqlite_path: SQLite path for the default memory store.
            crisis_log_sqlite_path: SQLite path for the default crisis log.
            feedback_sqlite_path: SQLite path for the default feedback store.
            embedding_provider: Optional explicit embedding provider override.
            default_llm_client: Optional fallback LLM client for shutdown and
                timeout-driven finalization.
            session_timeout: Inactivity window before an active session expires.
            session_sweep_interval_seconds: How often the sweeper checks for
                expired sessions.
            finalize_active_sessions_on_close: Whether ``__aexit__`` should
                best-effort finalize unresolved sessions.
            auto_finalize_excluded: Optional predicate for thread ids that
                external channel registries own and should finalize explicitly.
        """

        self.memory_mode = memory_mode
        is_incognito = memory_mode == MemoryMode.INCOGNITO

        resolved_sqlite = ":memory:" if is_incognito else sqlite_path
        self.sqlite_path = (
            Path(resolved_sqlite) if resolved_sqlite != ":memory:" else Path(":memory:")
        )

        self._saver_cm: AbstractAsyncContextManager[AsyncSqliteSaver] | None = None
        self._checkpointer: AsyncSqliteSaver | None = None
        self._graph: AgentWorkflow | None = None
        self._default_llm_client = default_llm_client
        self._session_timeout = session_timeout
        self._session_sweep_interval_seconds = max(
            1.0, float(session_sweep_interval_seconds)
        )
        self._finalize_active_sessions_on_close = finalize_active_sessions_on_close
        self._auto_finalize_excluded = auto_finalize_excluded
        self._session_sweeper_task: asyncio.Task[None] | None = None
        self._thread_llm_clients: dict[str, BaseLLMClient | None] = {}
        self._thread_locks: dict[str, asyncio.Lock] = {}
        self._active_mutation_tokens: set[str] = set()
        self._runtime_instance_id = uuid4().hex

        if memory_store is not None:
            self._memory_store = memory_store
        elif is_incognito:
            self._memory_store = OpenCouchMemoryStore()
        else:
            self._memory_store = SqliteMemoryStore(memory_sqlite_path)

        if crisis_log_backend is not None:
            self._crisis_log_backend = crisis_log_backend
        elif is_incognito:
            self._crisis_log_backend = InMemoryCrisisLogBackend()
        else:
            self._crisis_log_backend = SqliteCrisisLogBackend(crisis_log_sqlite_path)

        if session_feedback_backend is not None:
            self._session_feedback_backend = session_feedback_backend
        elif is_incognito:
            self._session_feedback_backend = InMemorySessionFeedbackBackend()
        else:
            self._session_feedback_backend = SqliteSessionFeedbackBackend(
                feedback_sqlite_path,
            )

        if embedding_provider is not None:
            self._embedding_provider: EmbeddingProvider = embedding_provider
        elif is_incognito:
            self._embedding_provider = NullEmbeddingProvider()
        else:
            self._embedding_provider = create_configured_embedding_provider()

        # Runtime-managed per-thread session trackers.
        self._session_starts: dict[str, str] = {}
        self._max_crisis_levels: dict[str, int] = {}
        self._session_memory_buffers: dict[str, SessionMemoryBuffer] = {}
        self._session_transcript_starts: dict[str, int] = {}

    async def __aenter__(self) -> PersistentAgentRuntime:
        """Open runtime resources.

        Returns:
            The initialized runtime instance.
        """

        if self.sqlite_path != Path(":memory:"):
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)

        serde = JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSGPACK_MODULES)
        self._saver_cm = AsyncSqliteSaver.from_conn_string(str(self.sqlite_path))
        self._checkpointer = await self._saver_cm.__aenter__()
        self._checkpointer.serde = serde
        await self._ensure_runtime_schema()
        await self._prewarm()
        self._session_sweeper_task = asyncio.create_task(self._session_sweeper_loop())
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Close runtime resources.

        Args:
            exc_type: The active exception type, if any.
            exc: The active exception instance, if any.
            tb: The active traceback, if any.
        """

        if self._session_sweeper_task is not None:
            self._session_sweeper_task.cancel()
            try:
                await self._session_sweeper_task
            except asyncio.CancelledError:
                pass
            self._session_sweeper_task = None
        if self._finalize_active_sessions_on_close:
            await self.finalize_active_sessions(llm_client=self._default_llm_client)
        await self._memory_store.aclose()
        await self._crisis_log_backend.aclose()
        await self._session_feedback_backend.aclose()
        if self._saver_cm is not None:
            await self._saver_cm.__aexit__(exc_type, exc, tb)

    def _ensure_open(self) -> AsyncSqliteSaver:
        """Raise when the runtime is used outside its async context.

        Returns:
            The active SQLite checkpointer.

        Raises:
            RuntimeError: If the runtime has not been entered yet.
        """

        if self._checkpointer is None:
            raise RuntimeError(
                "PersistentAgentRuntime must be used inside 'async with'."
            )
        return self._checkpointer

    async def _ensure_runtime_schema(self) -> None:
        """Create runtime-owned tables.

        Returns:
            None.
        """

        checkpointer = self._ensure_open()
        await checkpointer.setup()

    async def _prewarm(self) -> None:
        """Warm runtime resources before the first user turn.

        Returns:
            None.
        """

        self._get_graph()

        try:
            await asyncio.wait_for(self._embedding_provider.awarmup(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(
                "PersistentAgentRuntime prewarm: embedding warmup timed out; "
                "continuing with cold provider."
            )
        except Exception:
            logger.warning(
                "PersistentAgentRuntime prewarm: embedding warmup failed; "
                "continuing with cold provider.",
                exc_info=True,
            )

        checkpointer = self._ensure_open()
        async with checkpointer.lock:
            await checkpointer.conn.execute(ACTIVE_SESSION_STATE_DDL)
            async with checkpointer.conn.execute(
                "PRAGMA table_info(opencouch_active_sessions)"
            ) as cursor:
                rows = await cursor.fetchall()
            present_columns = {str(row[1]) for row in rows}
            for column_name, column_ddl in ACTIVE_SESSION_EXTRA_COLUMNS.items():
                if column_name in present_columns:
                    continue
                await checkpointer.conn.execute(
                    "ALTER TABLE opencouch_active_sessions "
                    f"ADD COLUMN {column_name} {column_ddl}"
                )
            await checkpointer.conn.commit()

    def _thread_lock(self, thread_id: str) -> asyncio.Lock:
        """Return the in-process lock for one thread.

        Args:
            thread_id: Thread identifier.

        Returns:
            The per-thread asyncio lock.
        """

        lock = self._thread_locks.get(thread_id)
        if lock is None:
            lock = asyncio.Lock()
            self._thread_locks[thread_id] = lock
        return lock

    def _auto_finalization_excluded(self, thread_id: str) -> bool:
        """Return whether runtime background finalization should skip a thread.

        Args:
            thread_id: Thread identifier.

        Returns:
            True when an external registry owns session finalization.
        """

        if self._auto_finalize_excluded is None:
            return False
        try:
            return bool(self._auto_finalize_excluded(thread_id))
        except Exception:
            logger.warning(
                "auto-finalize exclusion predicate failed for thread %s",
                thread_id,
                exc_info=True,
            )
            return False

    def _new_mutation_token(self) -> str:
        """Return a process-scoped mutation token.

        Returns:
            A mutation token that identifies this runtime instance.
        """

        return f"{uuid4().hex}:{self._runtime_instance_id}:{os.getpid()}"

    def _is_mutation_in_flight(self, token: str | None) -> bool:
        """Return whether a mutation token is actively running in this process.

        Args:
            token: Persisted mutation token.

        Returns:
            True when this runtime currently owns an in-flight mutation.
        """

        return token in self._active_mutation_tokens

    @asynccontextmanager
    async def _active_session_mutation(
        self,
        thread_id: str,
        *,
        mutation_kind: str,
        finalize_required_reason: str | None = None,
    ) -> AsyncIterator[str]:
        """Track an in-flight active-session mutation for recovery.

        Args:
            thread_id: Thread identifier.
            mutation_kind: Mutation kind for diagnostics.
            finalize_required_reason: Optional durable recovery reason.

        Yields:
            The process-scoped mutation token.
        """

        mutation_token = self._new_mutation_token()
        self._active_mutation_tokens.add(mutation_token)
        await self._set_active_session_mutation(
            thread_id,
            mutation_token=mutation_token,
            mutation_kind=mutation_kind,
            finalize_required_reason=finalize_required_reason,
        )
        try:
            yield mutation_token
        finally:
            self._active_mutation_tokens.discard(mutation_token)

    async def _load_persisted_active_session_row(
        self,
        thread_id: str,
    ) -> _PersistedActiveSessionRow | None:
        """Load a raw active-session row.

        Args:
            thread_id: Thread identifier.

        Returns:
            The persisted row, or ``None`` when absent.
        """

        if self.memory_mode == MemoryMode.INCOGNITO:
            return None

        checkpointer = self._ensure_open()
        async with checkpointer.lock:
            async with checkpointer.conn.execute(
                """
                SELECT
                    payload_json,
                    mutation_token,
                    mutation_kind,
                    rotate_after_this_turn,
                    finalize_required_reason
                FROM opencouch_active_sessions
                WHERE thread_id = ?
                """,
                (thread_id,),
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return _PersistedActiveSessionRow(
            payload_json=str(row[0]),
            mutation_token=str(row[1]) if row[1] is not None else None,
            mutation_kind=str(row[2]) if row[2] is not None else None,
            rotate_after_this_turn=bool(row[3]),
            finalize_required_reason=str(row[4]) if row[4] is not None else None,
        )

    async def _load_persisted_active_session(
        self,
        thread_id: str,
    ) -> PersistedActiveSessionState | None:
        """Load the persisted active-session record for a thread.

        Args:
            thread_id: The thread identifier to read.

        Returns:
            The persisted session record, or ``None`` when absent.
        """

        if self.memory_mode == MemoryMode.INCOGNITO:
            return None

        row = await self._load_persisted_active_session_row(thread_id)
        if row is None:
            return None
        return PersistedActiveSessionState.from_json(row.payload_json)

    async def _list_persisted_active_session_ids(self) -> list[str]:
        """List thread ids with unresolved active sessions.

        Returns:
            The unresolved active-session thread ids.
        """

        if self.memory_mode == MemoryMode.INCOGNITO:
            return list(self._session_starts)

        checkpointer = self._ensure_open()
        async with checkpointer.lock:
            async with checkpointer.conn.execute(
                """
                SELECT thread_id
                FROM opencouch_active_sessions
                ORDER BY thread_id
                """
            ) as cursor:
                rows = await cursor.fetchall()
        return [str(row[0]) for row in rows]

    async def _save_persisted_active_session(
        self,
        session: PersistedActiveSessionState,
    ) -> None:
        """Persist one active-session record.

        Args:
            session: The session record to upsert.

        Returns:
            None.
        """

        if self.memory_mode == MemoryMode.INCOGNITO:
            return

        checkpointer = self._ensure_open()
        payload_json = session.to_json()
        async with checkpointer.lock:
            await checkpointer.conn.execute(
                """
                INSERT INTO opencouch_active_sessions(thread_id, payload_json)
                VALUES(?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    payload_json = excluded.payload_json
                """,
                (session.thread_id, payload_json),
            )
            await checkpointer.conn.commit()

    async def _set_active_session_mutation(
        self,
        thread_id: str,
        *,
        mutation_token: str,
        mutation_kind: str,
        finalize_required_reason: str | None = None,
    ) -> None:
        """Persist a best-effort marker for an in-flight session mutation.

        Args:
            thread_id: Thread identifier.
            mutation_token: Process-scoped mutation token.
            mutation_kind: Mutation kind for diagnostics.
            finalize_required_reason: Optional durable recovery reason.

        Returns:
            None.
        """

        if self.memory_mode == MemoryMode.INCOGNITO:
            return

        checkpointer = self._ensure_open()
        if finalize_required_reason is None:
            sql = """
                UPDATE opencouch_active_sessions
                SET mutation_token = ?, mutation_kind = ?
                WHERE thread_id = ?
                """
            params: tuple[Any, ...] = (mutation_token, mutation_kind, thread_id)
        else:
            sql = """
                UPDATE opencouch_active_sessions
                SET
                    mutation_token = ?,
                    mutation_kind = ?,
                    finalize_required_reason = ?
                WHERE thread_id = ?
                """
            params = (
                mutation_token,
                mutation_kind,
                finalize_required_reason,
                thread_id,
            )
        async with checkpointer.lock:
            await checkpointer.conn.execute(sql, params)
            await checkpointer.conn.commit()

    async def _clear_active_session_mutation(
        self,
        thread_id: str,
        mutation_token: str,
    ) -> None:
        """Clear a mutation marker when the current process owns it.

        Args:
            thread_id: Thread identifier.
            mutation_token: Token to clear.

        Returns:
            None.
        """

        if self.memory_mode == MemoryMode.INCOGNITO:
            return

        checkpointer = self._ensure_open()
        async with checkpointer.lock:
            await checkpointer.conn.execute(
                """
                UPDATE opencouch_active_sessions
                SET mutation_token = NULL, mutation_kind = NULL
                WHERE thread_id = ? AND mutation_token = ?
                """,
                (thread_id, mutation_token),
            )
            await checkpointer.conn.commit()

    async def _set_active_session_rotation_required(self, thread_id: str) -> None:
        """Mark a persisted active session for channel-level rotation.

        Args:
            thread_id: Thread identifier.

        Returns:
            None.
        """

        if self.memory_mode == MemoryMode.INCOGNITO:
            return

        checkpointer = self._ensure_open()
        async with checkpointer.lock:
            await checkpointer.conn.execute(
                """
                UPDATE opencouch_active_sessions
                SET rotate_after_this_turn = 1
                WHERE thread_id = ?
                """,
                (thread_id,),
            )
            await checkpointer.conn.commit()

    async def _delete_persisted_active_session(self, thread_id: str) -> None:
        """Delete the persisted active-session record for a thread.

        Args:
            thread_id: The thread identifier to delete.

        Returns:
            None.
        """

        if self.memory_mode == MemoryMode.INCOGNITO:
            return

        checkpointer = self._ensure_open()
        async with checkpointer.lock:
            await checkpointer.conn.execute(
                """
                DELETE FROM opencouch_active_sessions
                WHERE thread_id = ?
                """,
                (thread_id,),
            )
            await checkpointer.conn.commit()

    @staticmethod
    def _transcript_length(state: AgentState | None) -> int:
        """Return the durable transcript length for a thread state.

        Args:
            state: The thread state snapshot.

        Returns:
            The transcript length, or ``0`` when state is absent.
        """

        if state is None:
            return 0
        return len(state.get("transcript", []) or [])

    @staticmethod
    def _slice_state_to_active_session(
        state: AgentState,
        *,
        transcript_start_index: int,
    ) -> AgentState:
        """Slice a state snapshot to the active-session transcript window.

        Args:
            state: The full thread state.
            transcript_start_index: The transcript index where the active session begins.

        Returns:
            A shallow state copy limited to the active session window.
        """

        transcript = list(state.get("transcript", []) or [])
        start = min(max(transcript_start_index, 0), len(transcript))
        windowed = cast(AgentState, dict(state))
        windowed["transcript"] = transcript[start:]

        if "history" in state:
            history = list(state.get("history", []) or [])
            history_start = min(start, len(history))
            windowed["history"] = history[history_start:]

        return windowed

    @staticmethod
    def _session_continuity_clear_delta(state: AgentState | None) -> dict[str, Any]:
        """Build a delta that clears session-scoped continuity fields.

        Args:
            state: The current checkpointed state, if any.

        Returns:
            A partial state update that clears stale session continuity.
        """

        if state is None:
            return {}

        delta: dict[str, Any] = {}
        exercise_state = state.get("exercise_state", {}) or {}
        if any(
            exercise_state.get(field) is not None for field in _EXERCISE_STATE_FIELDS
        ):
            delta["exercise_state"] = {
                "exercise_type": None,
                "exercise_step": None,
                "exercise_therapeutic_approach": None,
                "exercise_selection_options": None,
            }

        if state.get("therapeutic_approach") is not None:
            delta["therapeutic_approach"] = None

        return delta

    async def _clear_session_continuity_in_checkpoint(
        self,
        thread_id: str,
        state: AgentState | None,
        *,
        suppress_errors: bool = False,
    ) -> None:
        """Clear session-scoped continuity fields from a persisted checkpoint.

        Args:
            thread_id: The thread identifier to update.
            state: The current checkpointed state, if any.
            suppress_errors: Whether checkpoint update failures should be logged.

        Returns:
            None.

        Raises:
            Exception: Propagates checkpoint update failures when
                ``suppress_errors`` is ``False``.
        """

        delta = self._session_continuity_clear_delta(state)
        if not delta:
            return

        try:
            graph = self._get_graph()
            await graph.aupdate_state(
                self._config_for_thread(thread_id),
                delta,
                as_node="finalize_turn_node",
            )
        except Exception:
            if suppress_errors:
                logger.warning(
                    "failed to clear session continuity for thread %s",
                    thread_id,
                    exc_info=True,
                )
                return
            raise

    def _session_has_expired(self, session: PersistedActiveSessionState) -> bool:
        """Return whether an active session crossed the inactivity timeout.

        Args:
            session: The persisted active-session record.

        Returns:
            ``True`` when the session is expired.
        """

        last_active = _parse_iso_timestamp(session.last_active_at)
        if last_active is None:
            return True
        return (
            datetime.now(tz=last_active.tzinfo) - last_active >= self._session_timeout
        )

    def _clear_runtime_session_tracking(self, thread_id: str) -> None:
        """Drop all in-process session trackers for one thread.

        Args:
            thread_id: The thread identifier to clear.

        Returns:
            None.
        """

        self._session_starts.pop(thread_id, None)
        self._max_crisis_levels.pop(thread_id, None)
        self._session_memory_buffers.pop(thread_id, None)
        self._session_transcript_starts.pop(thread_id, None)
        self._thread_llm_clients.pop(thread_id, None)

    def _has_runtime_session_tracking(self, thread_id: str) -> bool:
        """Return whether a thread has in-process session trackers.

        Args:
            thread_id: The thread identifier to check.

        Returns:
            ``True`` when any runtime session tracker exists for the thread.
        """

        return (
            thread_id in self._session_starts
            or thread_id in self._session_transcript_starts
            or thread_id in self._session_memory_buffers
        )

    def _hydrate_runtime_session_tracking(
        self,
        session: PersistedActiveSessionState,
    ) -> None:
        """Restore in-process trackers from a persisted session record.

        Args:
            session: The persisted session record to hydrate from.

        Returns:
            None.
        """

        self._session_starts[session.thread_id] = session.started_at
        self._max_crisis_levels[session.thread_id] = session.max_crisis_level
        self._session_transcript_starts[session.thread_id] = (
            session.transcript_start_index
        )
        self._session_memory_buffers[session.thread_id] = (
            session.session_buffer.model_copy(deep=True)
        )

    def _remember_llm_client(
        self,
        thread_id: str,
        llm_client: BaseLLMClient | None,
    ) -> None:
        """Remember the latest LLM client for a thread.

        Args:
            thread_id: The thread identifier.
            llm_client: The client to remember.

        Returns:
            None.
        """

        if llm_client is not None:
            self._thread_llm_clients[thread_id] = llm_client

    def _effective_llm_client(
        self,
        thread_id: str,
        llm_client: BaseLLMClient | None = None,
    ) -> BaseLLMClient | None:
        """Resolve the effective LLM client for a thread.

        Args:
            thread_id: The thread identifier.
            llm_client: An explicit per-call override.

        Returns:
            The resolved client, or ``None`` when unavailable.
        """

        return (
            llm_client
            or self._thread_llm_clients.get(thread_id)
            or self._default_llm_client
        )

    async def _persist_runtime_session_tracking(
        self,
        thread_id: str,
        *,
        last_active_at: str | None = None,
    ) -> None:
        """Persist in-process session trackers for one thread.

        Args:
            thread_id: The thread identifier to persist.
            last_active_at: Optional explicit last-active timestamp.

        Returns:
            None.
        """

        started_at = self._session_starts.get(thread_id)
        transcript_start_index = self._session_transcript_starts.get(thread_id)
        if started_at is None or transcript_start_index is None:
            return

        session = PersistedActiveSessionState(
            thread_id=thread_id,
            started_at=started_at,
            last_active_at=last_active_at or _iso_now(),
            transcript_start_index=transcript_start_index,
            max_crisis_level=self._max_crisis_levels.get(thread_id, 0),
            session_buffer=self._session_memory_buffer_for_thread(thread_id).model_copy(
                deep=True
            ),
        )
        await self._save_persisted_active_session(session)

    async def _finalize_expired_sessions_once(self) -> None:
        """Finalize any sessions that crossed the inactivity timeout.

        Returns:
            None.
        """

        try:
            active_thread_ids = await self._list_persisted_active_session_ids()
        except Exception:
            logger.warning(
                "finalize_expired_sessions_once: failed to list active sessions",
                exc_info=True,
            )
            return

        for active_thread_id in active_thread_ids:
            try:
                if self._auto_finalization_excluded(active_thread_id):
                    continue
                persisted = await self._load_persisted_active_session(active_thread_id)
                if persisted is None or not self._session_has_expired(persisted):
                    continue
                logger.info(
                    "session timeout reached for thread %s; auto-finalizing expired session",
                    active_thread_id,
                )
                await self.end_session(
                    active_thread_id,
                    llm_client=self._effective_llm_client(active_thread_id),
                )
            except Exception:
                logger.warning(
                    "finalize_expired_sessions_once: failed to end expired session for thread %s",
                    active_thread_id,
                    exc_info=True,
                )

    async def _session_sweeper_loop(self) -> None:
        """Run the background session-timeout sweeper loop.

        Returns:
            None.
        """

        try:
            while True:
                await asyncio.sleep(self._session_sweep_interval_seconds)
                await self._finalize_expired_sessions_once()
        except asyncio.CancelledError:
            raise

    async def _prepare_session_for_turn(
        self,
        *,
        thread_id: str,
        prior_state: AgentState | None,
        llm_client: BaseLLMClient | None,
        expected_liveness: ExpectedSessionLiveness | None = None,
    ) -> None:
        """Restore or create the active session before a new turn.

        Args:
            thread_id: The thread identifier being prepared.
            prior_state: The last checkpointed state for the thread.
            llm_client: The LLM client for any timeout-driven finalization.
            expected_liveness: Optional caller-owned liveness expectation.

        Returns:
            None.
        """

        status = await self._session_status_unlocked(thread_id)
        if expected_liveness == "active" and status != SessionStatus.ACTIVE:
            if status == SessionStatus.INTERRUPTED:
                raise SessionInterrupted(thread_id)
            raise SessionLeaseExpired(thread_id, status)
        if expected_liveness == "absent" and status != SessionStatus.ABSENT:
            raise ActiveSessionExists(thread_id, status)
        if expected_liveness is None:
            if status == SessionStatus.INTERRUPTED:
                raise SessionInterrupted(thread_id)
            if status == SessionStatus.ROTATION_REQUIRED:
                raise SessionLeaseExpired(thread_id, status)

        persisted = await self._load_persisted_active_session(thread_id)
        if persisted is not None:
            self._hydrate_runtime_session_tracking(persisted)
            if self._session_has_expired(persisted):
                logger.info(
                    "session timeout reached for thread %s; ending prior session before new turn",
                    thread_id,
                )
                await self._end_session_unlocked(thread_id, llm_client=llm_client)
                persisted = None

        if persisted is None and self._has_runtime_session_tracking(thread_id):
            return

        if persisted is None:
            await self._clear_session_continuity_in_checkpoint(thread_id, prior_state)
            now = _iso_now()
            self._session_starts[thread_id] = now
            self._max_crisis_levels[thread_id] = 0
            self._session_transcript_starts[thread_id] = self._transcript_length(
                prior_state
            )
            self._session_memory_buffers[thread_id] = SessionMemoryBuffer(
                session_id=thread_id
            )
            await self._persist_runtime_session_tracking(
                thread_id,
                last_active_at=now,
            )

    async def _record_successful_turn_tracking(
        self,
        thread_id: str,
        final_state: AgentState,
        *,
        session_transcript_soft_limit: int | None,
    ) -> None:
        """Persist runtime-owned tracking after a successful turn.

        Args:
            thread_id: The thread identifier.
            final_state: The post-turn state.
            session_transcript_soft_limit: Optional active-session transcript
                message limit that triggers channel rotation.

        Returns:
            None.
        """

        turn_crisis = final_state.get("crisis")
        turn_level = 0
        if isinstance(turn_crisis, CrisisAssessment):
            turn_level = turn_crisis.level
        elif isinstance(turn_crisis, Mapping):
            turn_level = int(turn_crisis.get("level", 0) or 0)
        prior_max = self._max_crisis_levels.get(thread_id, 0)
        self._max_crisis_levels[thread_id] = max(prior_max, turn_level)

        turn_approach = final_state.get("therapeutic_approach")
        self._session_memory_buffer_for_thread(thread_id).record_approach(turn_approach)

        await self._persist_runtime_session_tracking(thread_id)

        if session_transcript_soft_limit is None:
            return
        start = self._session_transcript_starts.get(thread_id, 0)
        active_transcript_len = max(0, self._transcript_length(final_state) - start)
        if active_transcript_len >= session_transcript_soft_limit:
            await self._set_active_session_rotation_required(thread_id)

    async def _finalize_session_window(
        self,
        *,
        thread_id: str,
        state: AgentState,
        started_at: str,
        ended_at: str,
        crisis_level_max: int,
        session_buffer: SessionMemoryBuffer | None,
        llm_client: BaseLLMClient | None,
    ) -> StoredSessionArc | None:
        """Run the shared session-end summarization and memory commit path.

        Args:
            thread_id: The thread identifier being finalized.
            state: The state window to summarize.
            started_at: The session start timestamp.
            ended_at: The session end timestamp.
            crisis_level_max: The max crisis level observed in the session.
            session_buffer: The buffered session memory candidates.
            llm_client: The LLM client used by the summarizer.

        Returns:
            The stored session arc, or ``None`` when summarization is skipped.
        """

        approach_hint = session_buffer.dominant_approach() if session_buffer else None

        stored_arc = await run_summarize_session(
            state,
            llm_client=llm_client,
            memory_store=self._memory_store,
            memory_mode=self.memory_mode,
            session_id=thread_id,
            started_at=started_at,
            ended_at=ended_at,
            crisis_level_max=crisis_level_max,
            embedding_provider=self._embedding_provider,
            approach_hint=approach_hint,
        )

        commit_result = await run_commit_session_memory(
            state,
            memory_store=self._memory_store,
            session_buffer=session_buffer,
            stored_arc=stored_arc,
            embedding_provider=self._embedding_provider,
            llm_client=llm_client,
        )
        if commit_result is not None:
            logger.info(
                "end_session: committed %d semantic facts, %d procedural rules "
                "(%d semantic bumps, %d semantic skipped, %d procedural skipped)",
                commit_result.semantic_writes,
                commit_result.procedural_writes,
                commit_result.semantic_bumps,
                commit_result.semantic_skips,
                commit_result.procedural_skips,
            )
        return stored_arc

    async def _extract_memory_from_transcript(
        self,
        *,
        thread_id: str,
        user_id: str | None,
        transcript: list[dict[str, Any]],
        llm_client: BaseLLMClient | None,
        session_buffer: SessionMemoryBuffer,
    ) -> None:
        """Replay transcript user turns through the extractor nodes.

        Args:
            thread_id: The thread identifier for provenance.
            user_id: The resolved user identifier, if any.
            transcript: The serialized transcript to replay.
            llm_client: The LLM client used by the extractors.
            session_buffer: The session buffer to populate during replay.

        Returns:
            None.
        """

        runtime = _RuntimeShim(
            context=WorkflowContext(
                llm_client=llm_client,
                memory_store=self._memory_store,
                crisis_log_backend=self._crisis_log_backend,
                memory_mode=self.memory_mode,
                embedding_provider=self._embedding_provider,
                session_memory_buffer=session_buffer,
            )
        )

        user_turn_count = 0
        for transcript_index, turn in enumerate(transcript):
            if turn.get("role") != "user":
                continue

            message = (turn.get("content") or "").strip()
            if not message:
                continue

            user_turn_count += 1
            state = cast(
                AgentState,
                {
                    "message": message,
                    "user_id": user_id,
                    "session_id": thread_id,
                    "transcript": list(transcript[: transcript_index + 1]),
                    "session_progress": {"turn_count": user_turn_count},
                    "route": "therapeutic",
                },
            )
            await run_extract_semantic_facts_node(state, cast(Any, runtime))
            await run_extract_procedural_rules_node(state, cast(Any, runtime))

    @property
    def memory_store(self) -> MemoryStore:
        """Return the runtime's unified memory store.

        Returns:
            The configured memory store.
        """

        return self._memory_store

    @property
    def crisis_log_backend(self) -> CrisisLogBackend:
        """Return the runtime's crisis log backend.

        Returns:
            The configured crisis log backend.
        """

        return self._crisis_log_backend

    @property
    def session_feedback_backend(self) -> SessionFeedbackBackend:
        """Return the runtime's session-feedback backend.

        Returns:
            The configured session-feedback backend.
        """

        return self._session_feedback_backend

    def _config_for_thread(
        self,
        thread_id: str,
        *,
        channel: Channel | None = None,
        user_id: str | None = None,
        streaming: bool = False,
    ) -> RunnableConfig:
        """Build LangGraph config for one thread.

        Args:
            thread_id: The thread identifier.
            channel: The current channel, if known.
            user_id: The user identifier, if known.
            streaming: Whether the graph run is streaming.

        Returns:
            The LangGraph config payload.
        """

        metadata = {
            "thread_id": thread_id,
            "therapeutic_approach": "text",
            "streaming": streaming,
            "channel": channel.value if channel is not None else None,
            "user_scope": "persistent" if user_id else "guest",
            "memory_mode": self.memory_mode.value,
        }
        return {
            "configurable": {"thread_id": thread_id},
            "metadata": metadata,
        }

    def _session_memory_buffer_for_thread(self, thread_id: str) -> SessionMemoryBuffer:
        """Return the runtime-managed session buffer for a thread.

        Args:
            thread_id: The thread identifier.

        Returns:
            The per-thread session memory buffer.
        """

        if thread_id not in self._session_memory_buffers:
            self._session_memory_buffers[thread_id] = SessionMemoryBuffer(
                session_id=thread_id
            )
        return self._session_memory_buffers[thread_id]

    def _context_for_turn(
        self,
        *,
        thread_id: str,
        llm_client: BaseLLMClient | None,
        response_llm_client: BaseLLMClient | None = None,
    ) -> WorkflowContext:
        """Build the LangGraph runtime context for one turn.

        Args:
            thread_id: The thread identifier.
            llm_client: The control-plane LLM client.
            response_llm_client: Optional response-writer override.

        Returns:
            The runtime context for the turn.
        """

        return WorkflowContext(
            llm_client=llm_client,
            response_llm=response_llm_client or llm_client,
            memory_store=self._memory_store,
            crisis_log_backend=self._crisis_log_backend,
            memory_mode=self.memory_mode,
            embedding_provider=self._embedding_provider,
            session_memory_buffer=self._session_memory_buffer_for_thread(thread_id),
        )

    @staticmethod
    def _messages_from_transcript(
        transcript: list[dict[str, Any]],
    ) -> list[Message]:
        """Materialize validated messages from a serialized transcript.

        Args:
            transcript: The serialized transcript entries.

        Returns:
            The validated ``Message`` objects.
        """

        messages: list[Message] = []
        for turn in transcript:
            role = turn.get("role")
            content = (turn.get("content") or "").strip()
            if role not in {"system", "user", "assistant"} or not content:
                continue
            style = turn.get("response_style") if role == "assistant" else None
            messages.append(
                Message(role=MessageRole(role), content=content, response_style=style)
            )
        return messages

    def _get_graph(self) -> AgentWorkflow:
        """Return the compiled LangGraph workflow for this runtime.

        Returns:
            The compiled workflow instance.
        """

        checkpointer = self._ensure_open()
        if self._graph is None:
            self._graph = build_agent_workflow(checkpointer=checkpointer)
        return self._graph

    async def get_state(self, thread_id: str) -> AgentState | None:
        """Load the latest persisted state snapshot for a thread.

        Args:
            thread_id: The thread identifier.

        Returns:
            The latest checkpointed state, if any.
        """

        graph = self._get_graph()
        snapshot = await graph.aget_state(self._config_for_thread(thread_id))
        values = snapshot.values or None
        if values is None:
            return None
        return cast(AgentState, dict(values))

    async def get_history(self, thread_id: str) -> list[Message]:
        """Load the full persisted transcript for a thread.

        Args:
            thread_id: The thread identifier.

        Returns:
            The materialized transcript messages for the thread.
        """

        state = await self.get_state(thread_id)
        if state is None:
            return []
        return self._messages_from_transcript(state.get("transcript", []))

    async def session_status(self, thread_id: str) -> SessionStatus:
        """Return the active-session liveness status for a thread.

        Args:
            thread_id: Thread identifier.

        Returns:
            The current session status.
        """

        return await self._session_status_unlocked(thread_id)

    async def _session_status_unlocked(self, thread_id: str) -> SessionStatus:
        """Return session status without acquiring the per-thread lock.

        Args:
            thread_id: Thread identifier.

        Returns:
            The current session status.
        """

        if self.memory_mode == MemoryMode.INCOGNITO:
            if self._has_runtime_session_tracking(thread_id):
                return SessionStatus.ACTIVE
            return SessionStatus.ABSENT

        row = await self._load_persisted_active_session_row(thread_id)
        if row is None:
            if self._has_runtime_session_tracking(thread_id):
                return SessionStatus.ACTIVE
            return SessionStatus.ABSENT

        if row.finalize_required_reason == "interrupted":
            return SessionStatus.INTERRUPTED

        if row.mutation_token is not None:
            if not self._is_mutation_in_flight(row.mutation_token):
                return SessionStatus.INTERRUPTED

        if row.rotate_after_this_turn:
            return SessionStatus.ROTATION_REQUIRED

        try:
            session = PersistedActiveSessionState.from_json(row.payload_json)
        except Exception:
            logger.warning(
                "active session payload could not be decoded for thread %s",
                thread_id,
                exc_info=True,
            )
            return SessionStatus.INTERRUPTED

        if self._session_has_expired(session):
            return SessionStatus.EXPIRED_UNFINALIZED

        return SessionStatus.ACTIVE

    async def has_active_session(self, thread_id: str) -> bool:
        """Return whether a thread currently has an unresolved session.

        Args:
            thread_id: The thread identifier.

        Returns:
            ``True`` when the thread still has active session tracking.
        """

        return await self.session_status(thread_id) == SessionStatus.ACTIVE

    async def reset_thread(self, thread_id: str) -> None:
        """Delete all persisted checkpoints and session state for a thread.

        Args:
            thread_id: The thread identifier.

        Returns:
            None.
        """

        async with self._thread_lock(thread_id):
            status = await self._session_status_unlocked(thread_id)
            if status != SessionStatus.ABSENT:
                raise ActiveSessionExists(thread_id, status)

            checkpointer = self._ensure_open()
            await checkpointer.adelete_thread(thread_id)
            await self._delete_persisted_active_session(thread_id)
            self._clear_runtime_session_tracking(thread_id)

    async def list_threads(self, *, limit: int = 20) -> list[ThreadSummary]:
        """List the most recent persisted threads in SQLite.

        Args:
            limit: The maximum number of threads to return.

        Returns:
            The most recent persisted thread summaries.
        """

        checkpointer = self._ensure_open()
        await checkpointer.setup()

        thread_ids: list[str] = []
        async with (
            checkpointer.lock,
            checkpointer.conn.execute(
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
            session_progress: Mapping[str, Any] = (
                state.get("session_progress", {}) if state is not None else {}
            )
            turn_count = session_progress.get("turn_count", 0)
            summaries.append(
                ThreadSummary(
                    thread_id=thread_id,
                    turn_count=int(turn_count),
                    message_count=len(history),
                    has_context=state is not None,
                )
            )
        return summaries

    @staticmethod
    def _turn_count_from_state(state: AgentState | None) -> int:
        """Extract the persisted turn count from a checkpoint snapshot.

        Args:
            state: The checkpointed state snapshot, if any.

        Returns:
            The persisted turn count.
        """

        if state is None:
            return 0
        session_progress = state.get("session_progress", {}) or {}
        return int(session_progress.get("turn_count", 0) or 0)

    @staticmethod
    def _build_turn_initial_state(
        *,
        thread_id: str,
        message: str,
        channel: Channel,
        user_id: str | None,
        installed_skills: list[str] | None,
        prior_turn_count: int,
    ) -> AgentGraphInputState:
        """Build the graph input state for one user turn.

        Args:
            thread_id: Thread identifier used as the session id.
            message: Current user message.
            channel: Channel metadata for the turn.
            user_id: Optional user identifier.
            installed_skills: Optional installed skill names.
            prior_turn_count: Persisted user-turn count before this turn.

        Returns:
            Initial graph state for the turn.
        """

        return build_initial_state(
            AgentInput(
                message=message,
                channel=channel,
                user_id=user_id,
                session_id=thread_id,
                history=[],
                working_memory=[],
                installed_skills=list(installed_skills or []),
            ),
            prior_turn_count=prior_turn_count,
        )

    @staticmethod
    def _stamp_turn_total_ms(
        state: AgentState,
        *,
        started_at: float,
    ) -> None:
        """Record total turn latency in state diagnostics.

        Args:
            state: Final graph state for the turn.
            started_at: Monotonic timestamp captured before graph execution.
        """

        if "diagnostics" not in state or state["diagnostics"] is None:
            state["diagnostics"] = {}
        state["diagnostics"]["turn_total_ms"] = round(
            (time.monotonic() - started_at) * 1000,
            2,
        )

    @staticmethod
    def _response_ready_output(
        state: Mapping[str, Any] | None,
        *,
        finalize_seen: bool,
        response_ready_emitted: bool,
    ) -> AgentOutput | None:
        """Return the durable output once finalize has completed.

        Args:
            state: Latest streamed state snapshot.
            finalize_seen: Whether the finalize node has emitted an update.
            response_ready_emitted: Whether a ready event was already emitted.

        Returns:
            The finalized output, if it is ready to surface.
        """

        if state is None or not finalize_seen or response_ready_emitted:
            return None
        response_text = str(state.get("response_text", "") or "").strip()
        if not response_text:
            return None
        return state_to_output(state)

    async def run_turn(
        self,
        *,
        thread_id: str,
        message: str,
        channel: Channel = Channel.TEST,
        user_id: str | None = None,
        installed_skills: list[str] | None = None,
        llm_client: BaseLLMClient | None = None,
        response_llm_client: BaseLLMClient | None = None,
        expected_liveness: ExpectedSessionLiveness | None = None,
        session_transcript_soft_limit: int | None = None,
    ) -> PersistentTurnResult:
        """Run one conversation turn through the runtime workflow.

        Args:
            thread_id: The thread identifier.
            message: The user message to process.
            channel: The channel metadata for the turn.
            user_id: The optional user identifier.
            installed_skills: Optional installed skill names.
            llm_client: The control-plane LLM client.
            response_llm_client: Optional response-writer override.
            expected_liveness: Optional active-session liveness expectation.
            session_transcript_soft_limit: Optional active-session transcript
                message limit that marks the session for rotation after success.

        Returns:
            The persisted turn result, including output, state, and history.
        """

        async with self._thread_lock(thread_id):
            graph = self._get_graph()
            self._remember_llm_client(thread_id, llm_client)

            # Reducers restore transcript; only turn_count is needed here.
            prior_state = await self.get_state(thread_id)
            await self._prepare_session_for_turn(
                thread_id=thread_id,
                prior_state=prior_state,
                llm_client=llm_client,
                expected_liveness=expected_liveness,
            )
            prior_turn_count = self._turn_count_from_state(prior_state)

            initial_state = self._build_turn_initial_state(
                thread_id=thread_id,
                message=message,
                channel=channel,
                user_id=user_id,
                installed_skills=installed_skills,
                prior_turn_count=prior_turn_count,
            )

            async with self._active_session_mutation(
                thread_id,
                mutation_kind="turn",
            ) as mutation_token:
                turn_start = time.monotonic()
                graph_output = await graph.ainvoke(
                    initial_state,
                    config=self._config_for_thread(
                        thread_id,
                        channel=channel,
                        user_id=user_id,
                        streaming=False,
                    ),
                    context=self._context_for_turn(
                        thread_id=thread_id,
                        llm_client=llm_client,
                        response_llm_client=response_llm_client,
                    ),
                )
                final_state = await self.get_state(thread_id)
                if final_state is None:
                    final_state = cast(
                        AgentState, {**dict(initial_state), **dict(graph_output)}
                    )

                self._stamp_turn_total_ms(final_state, started_at=turn_start)

                await self._record_successful_turn_tracking(
                    thread_id,
                    final_state,
                    session_transcript_soft_limit=session_transcript_soft_limit,
                )

                result = PersistentTurnResult(
                    output=state_to_output(final_state),
                    state=final_state,
                    history=self._messages_from_transcript(
                        final_state.get("transcript", [])
                    ),
                )

                await self._clear_active_session_mutation(thread_id, mutation_token)
                return result

    async def end_session(
        self,
        thread_id: str,
        *,
        llm_client: BaseLLMClient | None = None,
    ) -> StoredSessionArc | None:
        """Summarize the active session for a thread and write it to memory.

        Args:
            thread_id: The thread whose active session should be summarized.
            llm_client: The optional LLM client for session summarization.

        Returns:
            The written session arc, or ``None`` when summarization is skipped.
        """

        async with self._thread_lock(thread_id):
            return await self._end_session_unlocked(thread_id, llm_client=llm_client)

    async def _end_session_unlocked(
        self,
        thread_id: str,
        *,
        llm_client: BaseLLMClient | None = None,
    ) -> StoredSessionArc | None:
        """Summarize an active session while the caller owns the thread lock.

        Args:
            thread_id: The thread whose active session should be summarized.
            llm_client: The optional LLM client for session summarization.

        Returns:
            The written session arc, or ``None`` when summarization is skipped.
        """

        effective_llm_client = self._effective_llm_client(thread_id, llm_client)
        status = await self._session_status_unlocked(thread_id)
        persisted = await self._load_persisted_active_session(thread_id)
        if persisted is not None:
            self._hydrate_runtime_session_tracking(persisted)
        has_active_session = (
            persisted is not None
            or thread_id in self._session_starts
            or thread_id in self._session_transcript_starts
            or thread_id in self._session_memory_buffers
        )

        if not has_active_session:
            return None

        mutation_token = self._new_mutation_token()
        if persisted is not None:
            self._active_mutation_tokens.add(mutation_token)
            await self._set_active_session_mutation(
                thread_id,
                mutation_token=mutation_token,
                mutation_kind="finalize",
                finalize_required_reason=(
                    "interrupted" if status == SessionStatus.INTERRUPTED else None
                ),
            )

        state = await self.get_state(thread_id)

        if state is None:
            await self._delete_persisted_active_session(thread_id)
            self._clear_runtime_session_tracking(thread_id)
            self._active_mutation_tokens.discard(mutation_token)
            return None

        try:
            transcript_start_index = self._session_transcript_starts.get(thread_id, 0)
            session_state = self._slice_state_to_active_session(
                state,
                transcript_start_index=transcript_start_index,
            )
            started_at = self._session_starts.get(thread_id, _iso_now())
            ended_at = _iso_now()
            crisis_level_max = self._max_crisis_levels.get(thread_id, 0)
            session_buffer = self._session_memory_buffers.get(thread_id)
            stored_arc = await self._finalize_session_window(
                thread_id=thread_id,
                state=session_state,
                started_at=started_at,
                ended_at=ended_at,
                crisis_level_max=crisis_level_max,
                session_buffer=session_buffer,
                llm_client=effective_llm_client,
            )
            await self._clear_session_continuity_in_checkpoint(
                thread_id,
                state,
                suppress_errors=True,
            )
            await self._delete_persisted_active_session(thread_id)
            self._clear_runtime_session_tracking(thread_id)
            return stored_arc
        except Exception:
            await self._clear_active_session_mutation(thread_id, mutation_token)
            raise
        finally:
            self._active_mutation_tokens.discard(mutation_token)

    async def end_transcript_session(
        self,
        *,
        thread_id: str,
        user_id: str | None,
        transcript: list[dict[str, Any]],
        llm_client: BaseLLMClient | None = None,
        started_at: str | None = None,
        ended_at: str | None = None,
        crisis_level_max: int = 0,
    ) -> StoredSessionArc | None:
        """Finalize a session represented only by a transcript.

        Args:
            thread_id: The thread identifier.
            user_id: The optional user identifier.
            transcript: The serialized transcript entries for the session.
            llm_client: The optional LLM client for memory extraction.
            started_at: Optional session start timestamp.
            ended_at: Optional session end timestamp.
            crisis_level_max: The highest crisis level seen in the session.

        Returns:
            The written session arc, or ``None`` when summarization is skipped.
        """

        if not transcript:
            return None

        self._remember_llm_client(thread_id, llm_client)
        session_buffer = SessionMemoryBuffer(session_id=thread_id)
        await self._extract_memory_from_transcript(
            thread_id=thread_id,
            user_id=user_id,
            transcript=transcript,
            llm_client=self._effective_llm_client(thread_id, llm_client),
            session_buffer=session_buffer,
        )
        session_state = cast(
            AgentState,
            {
                "user_id": user_id,
                "session_id": thread_id,
                "transcript": list(transcript),
            },
        )
        return await self._finalize_session_window(
            thread_id=thread_id,
            state=session_state,
            started_at=started_at or _iso_now(),
            ended_at=ended_at or _iso_now(),
            crisis_level_max=crisis_level_max,
            session_buffer=session_buffer,
            llm_client=self._effective_llm_client(thread_id, llm_client),
        )

    async def finalize_active_sessions(
        self,
        *,
        llm_client: BaseLLMClient | None = None,
    ) -> None:
        """Finalize any unresolved active sessions.

        Args:
            llm_client: The fallback LLM client to use for summarization.

        Returns:
            None.
        """

        try:
            active_thread_ids = await self._list_persisted_active_session_ids()
        except Exception:
            logger.warning(
                "finalize_active_sessions: failed to list active sessions",
                exc_info=True,
            )
            return

        for active_thread_id in active_thread_ids:
            try:
                if self._auto_finalization_excluded(active_thread_id):
                    continue
                await self.end_session(
                    active_thread_id,
                    llm_client=self._effective_llm_client(active_thread_id, llm_client),
                )
            except Exception:
                logger.warning(
                    "finalize_active_sessions: failed to end session for thread %s",
                    active_thread_id,
                    exc_info=True,
                )

    async def record_session_feedback(
        self,
        thread_id: str,
        *,
        label: FeedbackLabel,
        source: FeedbackSource,
    ) -> SessionFeedbackRecord | None:
        """Record an explicit end-of-session feedback label.

        Args:
            thread_id: The thread whose session is ending.
            label: The explicit feedback label the user provided.
            source: Which end-session surface produced this feedback.

        Returns:
            The written feedback record, or ``None`` on failure.
        """

        try:
            state = await self.get_state(thread_id)
            turn_count = self._turn_count_from_state(state)

            # Mirror the crisis-log privacy contract in incognito mode.
            if self.memory_mode == MemoryMode.INCOGNITO:
                user_id: str | None = None
            elif state is not None:
                user_id = state.get("user_id")
            else:
                user_id = None

            record = SessionFeedbackRecord(
                id=str(uuid4()),
                session_id_opaque=hash_session_id(thread_id),
                user_id_or_null=user_id,
                recorded_at=iso_now(),
                label=label,
                turn_count_at_end=turn_count,
                source=source,
                schema_version=1,
            )

            await self._session_feedback_backend.aappend(record)
            return record
        except Exception:
            logger.warning(
                "session feedback write failed for thread %s",
                thread_id,
                exc_info=True,
            )
            return None

    async def run_turn_stream(
        self,
        *,
        thread_id: str,
        message: str,
        channel: Channel = Channel.TEST,
        user_id: str | None = None,
        installed_skills: list[str] | None = None,
        llm_client: BaseLLMClient | None = None,
        response_llm_client: BaseLLMClient | None = None,
        expected_liveness: ExpectedSessionLiveness | None = None,
        session_transcript_soft_limit: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Run one turn and stream status and response events.

        Args:
            thread_id: The thread identifier.
            message: The user message to process.
            channel: The channel metadata for the turn.
            user_id: The optional user identifier.
            installed_skills: Optional installed skill names.
            llm_client: The control-plane LLM client.
            response_llm_client: Optional response-writer override.
            expected_liveness: Optional active-session liveness expectation.
            session_transcript_soft_limit: Optional active-session transcript
                message limit that marks the session for rotation after success.

        Yields:
            Stream events for status updates, response readiness, and completion.
        """

        async with self._thread_lock(thread_id):
            graph = self._get_graph()
            self._remember_llm_client(thread_id, llm_client)

            # Reducers restore transcript; only turn_count is needed here.
            prior_state = await self.get_state(thread_id)
            await self._prepare_session_for_turn(
                thread_id=thread_id,
                prior_state=prior_state,
                llm_client=llm_client,
                expected_liveness=expected_liveness,
            )
            prior_turn_count = self._turn_count_from_state(prior_state)

            initial_state = self._build_turn_initial_state(
                thread_id=thread_id,
                message=message,
                channel=channel,
                user_id=user_id,
                installed_skills=installed_skills,
                prior_turn_count=prior_turn_count,
            )

            turn_start = time.monotonic()
            final_state: AgentState | None = None
            chunks_emitted = False
            finalize_seen = False
            response_ready_emitted = False

            async with self._active_session_mutation(
                thread_id,
                mutation_kind="turn",
            ) as mutation_token:
                async for chunk in graph.astream(
                    initial_state,
                    config=self._config_for_thread(
                        thread_id,
                        channel=channel,
                        user_id=user_id,
                        streaming=True,
                    ),
                    context=self._context_for_turn(
                        thread_id=thread_id,
                        llm_client=llm_client,
                        response_llm_client=response_llm_client,
                    ),
                    stream_mode=("custom", "updates", "values"),
                    subgraphs=True,
                    version="v2",
                ):
                    if chunk["type"] == "custom":
                        # Forward token chunks from any namespace.
                        payload = chunk["data"]
                        if isinstance(payload, dict) and payload.get("type") == "chunk":
                            yield ChunkEvent(text=payload["text"])
                            chunks_emitted = True
                    elif chunk["type"] == "updates" and chunk["ns"] == ():
                        # Skip subgraph internals to avoid duplicate status events.
                        for node_name in chunk["data"]:
                            stage = _NODE_TO_STAGE.get(node_name, node_name)
                            yield StatusEvent(stage=stage)
                            if node_name == "finalize_turn_node":
                                finalize_seen = True
                                ready_output = self._response_ready_output(
                                    final_state,
                                    finalize_seen=finalize_seen,
                                    response_ready_emitted=response_ready_emitted,
                                )
                                if ready_output is not None:
                                    if not chunks_emitted:
                                        yield ChunkEvent(
                                            text=ready_output.response_text
                                        )
                                        chunks_emitted = True
                                    yield ResponseReadyEvent(output=ready_output)
                                    response_ready_emitted = True
                    elif chunk["type"] == "values" and chunk["ns"] == ():
                        final_state = cast(AgentState, chunk["data"])
                        ready_output = self._response_ready_output(
                            final_state,
                            finalize_seen=finalize_seen,
                            response_ready_emitted=response_ready_emitted,
                        )
                        if ready_output is not None:
                            if not chunks_emitted:
                                yield ChunkEvent(text=ready_output.response_text)
                                chunks_emitted = True
                            yield ResponseReadyEvent(output=ready_output)
                            response_ready_emitted = True

                # Fall back to the checkpoint if the stream never yielded state values.
                if final_state is None:
                    fallback = await self.get_state(thread_id)
                    if fallback is None:
                        raise RuntimeError(
                            "run_turn_stream: graph stream yielded no values chunks "
                            "and no checkpoint was found for this thread."
                        )
                    final_state = fallback

                self._stamp_turn_total_ms(final_state, started_at=turn_start)

                await self._record_successful_turn_tracking(
                    thread_id,
                    final_state,
                    session_transcript_soft_limit=session_transcript_soft_limit,
                )

                await self._clear_active_session_mutation(thread_id, mutation_token)
                yield DoneEvent(output=state_to_output(final_state))
