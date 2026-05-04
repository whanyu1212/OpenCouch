"""Persistent runtime for thread-backed OpenCouch sessions."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

from langchain_core.runnables import RunnableConfig

from agent.active_session_manager import (
    ActiveSessionManager,
    PersistedActiveSessionRow,
    PersistedActiveSessionState,
)
from agent.active_session_store import PostgresActiveSessionStore
from agent.legacy.active_session_store_sqlite import SqliteActiveSessionStore
from agent.graph import build_agent_workflow, build_initial_state, state_to_output
from agent.graph_constants import FINALIZE_TURN_NODE
from agent.memory.candidates import SessionMemoryBuffer
from agent.audit.crisis_log import CrisisLogBackend
from agent.audit.session_feedback import SessionFeedbackBackend
from agent.memory.hashing import hash_session_id, iso_now
from agent.memory.hashing import iso_now as _iso_now
from agent.memory.embeddings import EmbeddingProvider
from agent.audit.models import FeedbackLabel, FeedbackSource, SessionFeedbackRecord
from agent.memory.models import StoredSessionArc
from agent.runtime.session_finalization import (
    extract_memory_from_transcript,
    finalize_session_window,
)
from agent.runtime.streaming import (
    chunk_event_from_custom_payload,
    messages_from_transcript,
    response_ready_output,
    stamp_turn_total_ms,
    status_stage_for_node,
)
from agent.memory.modes import MemoryMode
from agent.memory.store import MemoryStore
from agent.runtime.backends import (
    create_crisis_log_backend,
    create_embedding_provider,
    create_memory_store,
    create_session_feedback_backend,
    effective_thread_persistence_backend,
)
from agent.runtime.checkpointer import (
    ALLOWED_MSGPACK_MODULES as CHECKPOINT_ALLOWED_MSGPACK_MODULES,
    open_checkpointer,
    validate_thread_checkpointer_config,
)
from agent.runtime.session_state import (
    active_transcript_length,
    crisis_level_from_state,
    has_runtime_session_tracking,
    session_continuity_clear_delta,
    slice_state_to_active_session,
    transcript_length,
    turn_count_from_state,
)
from agent.models import (
    AgentInput,
    Channel,
    DoneEvent,
    Message,
    ResponseReadyEvent,
    StatusEvent,
    StreamEvent,
)
from agent.runtime.types import (
    ActiveSessionExists,
    ExpectedSessionLiveness,
    PersistentTurnResult,
    SessionInterrupted,
    SessionLeaseExpired,
    SessionStatus,
    ThreadSummary,
)
from agent.runtime_context import WorkflowContext
from agent.state import AgentGraphInputState, AgentGraphOutputState, AgentState
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
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

BACKEND_ROOT = Path(__file__).resolve().parents[1]
_STORE_DIR = BACKEND_ROOT / ".store"
DEFAULT_THREAD_DB_PATH = _STORE_DIR / "threads.sqlite3"
# Keep runtime-owned stores separate from the LangGraph checkpoint DB.
DEFAULT_MEMORY_DB_PATH = _STORE_DIR / "memory.sqlite3"
DEFAULT_CRISIS_LOG_DB_PATH = _STORE_DIR / "crisis.sqlite3"
DEFAULT_FEEDBACK_DB_PATH = _STORE_DIR / "session_feedback.sqlite3"
ALLOWED_MSGPACK_MODULES = CHECKPOINT_ALLOWED_MSGPACK_MODULES
SESSION_TIMEOUT = timedelta(minutes=20)


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
        memory_backend: Literal["sqlite", "postgres"] = "sqlite",
        memory_database_url: str | None = None,
        thread_persistence_backend: Literal["sqlite", "postgres"] = "sqlite",
        thread_database_url: str | None = None,
        crisis_log_persistence_backend: Literal["sqlite", "postgres"] = "sqlite",
        crisis_log_database_url: str | None = None,
        session_feedback_persistence_backend: Literal["sqlite", "postgres"] = "sqlite",
        session_feedback_database_url: str | None = None,
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
            memory_backend: Memory-store backend to use for persistent modes.
            memory_database_url: PostgreSQL connection string used when
                ``memory_backend`` is ``"postgres"``.
            thread_persistence_backend: Checkpointer backend to use for
                persistent modes.
            thread_database_url: PostgreSQL connection string used when
                ``thread_persistence_backend`` is ``"postgres"``.
            crisis_log_persistence_backend: Crisis-log backend to use for
                persistent modes.
            crisis_log_database_url: PostgreSQL connection string used when
                ``crisis_log_persistence_backend`` is ``"postgres"``.
            session_feedback_persistence_backend: Session-feedback backend to use
                for persistent modes.
            session_feedback_database_url: PostgreSQL connection string used when
                ``session_feedback_persistence_backend`` is ``"postgres"``.
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

        self._thread_persistence_backend = effective_thread_persistence_backend(
            memory_mode=memory_mode,
            thread_persistence_backend=thread_persistence_backend,
        )
        self._thread_database_url = thread_database_url
        validate_thread_checkpointer_config(
            thread_persistence_backend=self._thread_persistence_backend,
            thread_database_url=thread_database_url,
        )

        self._saver_cm: AbstractAsyncContextManager[Any] | None = None
        self._checkpointer: AsyncSqliteSaver | AsyncPostgresSaver | None = None
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

        self._memory_store = create_memory_store(
            memory_mode=memory_mode,
            memory_store=memory_store,
            memory_backend=memory_backend,
            memory_database_url=memory_database_url,
            memory_sqlite_path=memory_sqlite_path,
        )
        self._crisis_log_backend = create_crisis_log_backend(
            memory_mode=memory_mode,
            crisis_log_backend=crisis_log_backend,
            crisis_log_persistence_backend=crisis_log_persistence_backend,
            crisis_log_database_url=crisis_log_database_url,
            crisis_log_sqlite_path=crisis_log_sqlite_path,
        )
        self._session_feedback_backend = create_session_feedback_backend(
            memory_mode=memory_mode,
            session_feedback_backend=session_feedback_backend,
            session_feedback_persistence_backend=session_feedback_persistence_backend,
            session_feedback_database_url=session_feedback_database_url,
            feedback_sqlite_path=feedback_sqlite_path,
        )
        self._embedding_provider: EmbeddingProvider = create_embedding_provider(
            memory_mode=memory_mode,
            embedding_provider=embedding_provider,
        )

        # Runtime-managed per-thread session trackers.
        self._session_starts: dict[str, str] = {}
        self._max_crisis_levels: dict[str, int] = {}
        self._session_memory_buffers: dict[str, SessionMemoryBuffer] = {}
        self._session_transcript_starts: dict[str, int] = {}
        if self._thread_persistence_backend == "postgres":
            self._active_session_store = PostgresActiveSessionStore(
                checkpointer_getter=self._ensure_postgres_open
            )
        else:
            self._active_session_store = SqliteActiveSessionStore(
                checkpointer_getter=self._ensure_sqlite_open
            )
        self._active_session_manager = ActiveSessionManager(
            store=self._active_session_store,
            memory_mode=self.memory_mode,
            session_timeout=self._session_timeout,
        )

    async def __aenter__(self) -> PersistentAgentRuntime:
        """Open runtime resources.

        Returns:
            The initialized runtime instance.
        """

        self._saver_cm = open_checkpointer(
            thread_persistence_backend=self._thread_persistence_backend,
            sqlite_path=self.sqlite_path,
            thread_database_url=self._thread_database_url,
        )
        self._checkpointer = await self._saver_cm.__aenter__()
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

    def _ensure_open(self) -> AsyncSqliteSaver | AsyncPostgresSaver:
        """Raise when the runtime is used outside its async context.

        Returns:
            The active runtime checkpointer.

        Raises:
            RuntimeError: If the runtime has not been entered yet.
        """

        if self._checkpointer is None:
            raise RuntimeError(
                "PersistentAgentRuntime must be used inside 'async with'."
            )
        return self._checkpointer

    def _ensure_sqlite_open(self) -> AsyncSqliteSaver:
        """Return the active SQLite checkpointer.

        Returns:
            AsyncSqliteSaver: The active SQLite checkpointer.

        Raises:
            RuntimeError: If the runtime is not using the SQLite checkpointer.
        """

        checkpointer = self._ensure_open()
        if not isinstance(checkpointer, AsyncSqliteSaver):
            raise RuntimeError("PersistentAgentRuntime is not using SQLite threads.")
        return checkpointer

    def _ensure_postgres_open(self) -> AsyncPostgresSaver:
        """Return the active Postgres checkpointer.

        Returns:
            AsyncPostgresSaver: The active Postgres checkpointer.

        Raises:
            RuntimeError: If the runtime is not using the Postgres checkpointer.
        """

        checkpointer = self._ensure_open()
        if not isinstance(checkpointer, AsyncPostgresSaver):
            raise RuntimeError("PersistentAgentRuntime is not using Postgres threads.")
        return checkpointer

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

        await self._active_session_manager.ensure_schema()

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

        return self._active_session_manager.new_mutation_token()

    def _is_mutation_in_flight(self, token: str | None) -> bool:
        """Return whether a mutation token is actively running in this process.

        Args:
            token: Persisted mutation token.

        Returns:
            True when this runtime currently owns an in-flight mutation.
        """

        return self._active_session_manager.is_mutation_in_flight(token)

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

        async with self._active_session_manager.active_session_mutation(
            thread_id,
            mutation_kind=mutation_kind,
            finalize_required_reason=finalize_required_reason,
        ) as mutation_token:
            yield mutation_token

    async def _load_persisted_active_session_row(
        self,
        thread_id: str,
    ) -> PersistedActiveSessionRow | None:
        """Load a raw active-session row.

        Args:
            thread_id: Thread identifier.

        Returns:
            The persisted row, or ``None`` when absent.
        """

        return await self._active_session_manager.load_persisted_active_session_row(
            thread_id
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

        return await self._active_session_manager.load_persisted_active_session(
            thread_id
        )

    async def _list_persisted_active_session_ids(self) -> list[str]:
        """List thread ids with unresolved active sessions.

        Returns:
            The unresolved active-session thread ids.
        """

        if self.memory_mode == MemoryMode.INCOGNITO:
            return list(self._session_starts)
        return await self._active_session_manager.list_persisted_active_session_ids()

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

        await self._active_session_manager.save_persisted_active_session(session)

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

        await self._active_session_manager.set_active_session_mutation(
            thread_id,
            mutation_token=mutation_token,
            mutation_kind=mutation_kind,
            finalize_required_reason=finalize_required_reason,
        )

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

        await self._active_session_manager.clear_active_session_mutation(
            thread_id,
            mutation_token,
        )

    async def _set_active_session_rotation_required(self, thread_id: str) -> None:
        """Mark a persisted active session for channel-level rotation.

        Args:
            thread_id: Thread identifier.

        Returns:
            None.
        """

        await self._active_session_manager.set_active_session_rotation_required(
            thread_id
        )

    async def _delete_persisted_active_session(self, thread_id: str) -> None:
        """Delete the persisted active-session record for a thread.

        Args:
            thread_id: The thread identifier to delete.

        Returns:
            None.
        """

        await self._active_session_manager.delete_persisted_active_session(thread_id)

    @staticmethod
    def _transcript_length(state: AgentState | None) -> int:
        """Return the durable transcript length for a thread state.

        Args:
            state: The thread state snapshot.

        Returns:
            The transcript length, or ``0`` when state is absent.
        """

        return transcript_length(state)

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

        return slice_state_to_active_session(
            state,
            transcript_start_index=transcript_start_index,
        )

    @staticmethod
    def _session_continuity_clear_delta(state: AgentState | None) -> dict[str, Any]:
        """Build a delta that clears session-scoped continuity fields.

        Args:
            state: The current checkpointed state, if any.

        Returns:
            A partial state update that clears stale session continuity.
        """

        return session_continuity_clear_delta(state)

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
                as_node=FINALIZE_TURN_NODE,
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

        return self._active_session_manager.session_has_expired(session)

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

        return has_runtime_session_tracking(
            thread_id,
            session_starts=self._session_starts,
            session_transcript_starts=self._session_transcript_starts,
            session_memory_buffers=self._session_memory_buffers,
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

        turn_level = crisis_level_from_state(final_state)
        prior_max = self._max_crisis_levels.get(thread_id, 0)
        self._max_crisis_levels[thread_id] = max(prior_max, turn_level)

        turn_approach = final_state.get("therapeutic_approach")
        self._session_memory_buffer_for_thread(thread_id).record_approach(turn_approach)

        await self._persist_runtime_session_tracking(thread_id)

        if session_transcript_soft_limit is None:
            return
        transcript_start_index = self._session_transcript_starts.get(thread_id, 0)
        active_transcript_len = active_transcript_length(
            final_state,
            transcript_start_index=transcript_start_index,
        )
        if active_transcript_len >= session_transcript_soft_limit:
            await self._set_active_session_rotation_required(thread_id)

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
        return messages_from_transcript(state.get("transcript", []))

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
        """List the most recent persisted threads.

        Args:
            limit: The maximum number of threads to return.

        Returns:
            The most recent persisted thread summaries.
        """

        checkpointer = self._ensure_open()
        await checkpointer.setup()

        thread_ids: list[str] = []
        seen_thread_ids: set[str] = set()
        async for checkpoint_tuple in checkpointer.alist(
            None, limit=max(limit * 5, limit)
        ):
            thread_id = str(checkpoint_tuple.config["configurable"]["thread_id"])
            if thread_id in seen_thread_ids:
                continue
            seen_thread_ids.add(thread_id)
            thread_ids.append(thread_id)
            if len(thread_ids) >= limit:
                break

        summaries: list[ThreadSummary] = []
        for thread_id in thread_ids:
            state = await self.get_state(thread_id)
            history = messages_from_transcript(
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

        return turn_count_from_state(state)

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

                stamp_turn_total_ms(final_state, started_at=turn_start)

                await self._record_successful_turn_tracking(
                    thread_id,
                    final_state,
                    session_transcript_soft_limit=session_transcript_soft_limit,
                )

                result = PersistentTurnResult(
                    output=state_to_output(final_state),
                    state=final_state,
                    history=messages_from_transcript(final_state.get("transcript", [])),
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
            persisted is not None or self._has_runtime_session_tracking(thread_id)
        )

        if not has_active_session:
            return None

        @asynccontextmanager
        async def _finalize_mutation_scope() -> AsyncIterator[str | None]:
            if persisted is None:
                yield None
                return
            async with self._active_session_mutation(
                thread_id,
                mutation_kind="finalize",
                finalize_required_reason=(
                    "interrupted" if status == SessionStatus.INTERRUPTED else None
                ),
            ) as mutation_token:
                yield mutation_token

        async with _finalize_mutation_scope() as mutation_token:
            state = await self.get_state(thread_id)

            if state is None:
                await self._delete_persisted_active_session(thread_id)
                self._clear_runtime_session_tracking(thread_id)
                return None

            try:
                transcript_start_index = self._session_transcript_starts.get(
                    thread_id, 0
                )
                session_state = self._slice_state_to_active_session(
                    state,
                    transcript_start_index=transcript_start_index,
                )
                started_at = self._session_starts.get(thread_id, _iso_now())
                ended_at = _iso_now()
                crisis_level_max = self._max_crisis_levels.get(thread_id, 0)
                session_buffer = self._session_memory_buffers.get(thread_id)
                stored_arc = await finalize_session_window(
                    session_state,
                    thread_id=thread_id,
                    started_at=started_at,
                    ended_at=ended_at,
                    crisis_level_max=crisis_level_max,
                    session_buffer=session_buffer,
                    llm_client=effective_llm_client,
                    memory_store=self._memory_store,
                    memory_mode=self.memory_mode,
                    embedding_provider=self._embedding_provider,
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
                if mutation_token is not None:
                    await self._clear_active_session_mutation(
                        thread_id,
                        mutation_token,
                    )
                raise

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
        await extract_memory_from_transcript(
            thread_id=thread_id,
            user_id=user_id,
            transcript=transcript,
            llm_client=self._effective_llm_client(thread_id, llm_client),
            session_buffer=session_buffer,
            memory_store=self._memory_store,
            memory_mode=self.memory_mode,
            crisis_log_backend=self._crisis_log_backend,
            embedding_provider=self._embedding_provider,
        )
        session_state = cast(
            AgentState,
            {
                "user_id": user_id,
                "session_id": thread_id,
                "transcript": list(transcript),
            },
        )
        return await finalize_session_window(
            session_state,
            thread_id=thread_id,
            started_at=started_at or _iso_now(),
            ended_at=ended_at or _iso_now(),
            crisis_level_max=crisis_level_max,
            session_buffer=session_buffer,
            llm_client=self._effective_llm_client(thread_id, llm_client),
            memory_store=self._memory_store,
            memory_mode=self.memory_mode,
            embedding_provider=self._embedding_provider,
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
                        event = chunk_event_from_custom_payload(chunk["data"])
                        if event is not None:
                            yield event
                            chunks_emitted = True
                    elif chunk["type"] == "updates" and chunk["ns"] == ():
                        # Skip subgraph internals to avoid duplicate status events.
                        for node_name in chunk["data"]:
                            yield StatusEvent(stage=status_stage_for_node(node_name))
                            if node_name == FINALIZE_TURN_NODE:
                                finalize_seen = True
                                ready_output = response_ready_output(
                                    final_state,
                                    finalize_seen=finalize_seen,
                                    response_ready_emitted=response_ready_emitted,
                                )
                                if ready_output is not None:
                                    if not chunks_emitted:
                                        yield chunk_event_from_custom_payload(
                                            {
                                                "type": "chunk",
                                                "text": ready_output.response_text,
                                            }
                                        )
                                        chunks_emitted = True
                                    yield ResponseReadyEvent(output=ready_output)
                                    response_ready_emitted = True
                    elif chunk["type"] == "values" and chunk["ns"] == ():
                        final_state = cast(AgentState, chunk["data"])
                        ready_output = response_ready_output(
                            final_state,
                            finalize_seen=finalize_seen,
                            response_ready_emitted=response_ready_emitted,
                        )
                        if ready_output is not None:
                            if not chunks_emitted:
                                yield chunk_event_from_custom_payload(
                                    {
                                        "type": "chunk",
                                        "text": ready_output.response_text,
                                    }
                                )
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

                stamp_turn_total_ms(final_state, started_at=turn_start)

                await self._record_successful_turn_tracking(
                    thread_id,
                    final_state,
                    session_transcript_soft_limit=session_transcript_soft_limit,
                )

                await self._clear_active_session_mutation(thread_id, mutation_token)
                yield DoneEvent(output=state_to_output(final_state))
