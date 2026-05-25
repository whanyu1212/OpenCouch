"""Persistent runtime for session-persisted OpenCouch interactions."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, cast

from agent.runtime.session.active_session import (
    ActiveSessionManager,
    PersistedActiveSessionState,
    PostgresActiveSessionStore,
    SqliteActiveSessionStore,
)
from agent.memory.policy.candidates import SessionMemoryBuffer
from agent.audit.crisis_log import CrisisLogBackend
from agent.feedback.session_feedback import SessionFeedbackBackend
from agent.memory.hashing import iso_now as _iso_now
from agent.memory.embeddings import EmbeddingProvider
from agent.memory.policy.write import text_contains_memory_control_request
from agent.memory.procedural_profile import aget_procedural_profile
from agent.memory.recall import load_memory_for_turn
from agent.feedback.models import FeedbackLabel, FeedbackSource, SessionFeedbackRecord
from agent.memory.types import StoredSessionArc
from agent.runtime.session import (
    RuntimeSessionTracker,
    active_transcript_length,
    crisis_level_from_state,
    finalize_session_window,
    session_continuity_clear_delta,
    slice_state_to_active_session,
    transcript_length,
    turn_count_from_state,
)
from agent.runtime.session.history import messages_from_transcript
from agent.runtime.session_feedback import (
    record_session_feedback as record_runtime_session_feedback,
)
from agent.runtime.streaming import (
    response_ready_output,
    stamp_turn_total_ms,
)
from agent.runtime.session_store import (
    TextSessionBackend,
    TextSessionStore,
    create_text_session_store,
)
from agent.runtime.openai_text_runtime import OpenAITextRuntime
from agent.runtime.context import OpenAITextRunContext
from agent.voice.transcript import voice_turn_to_transcript_entries
from agent.voice.turn_policy import VoiceTurnPolicy, build_voice_turn_policy
from agent.memory.modes import MemoryMode
from agent.memory.store import MemoryStore
from agent.runtime.backends import (
    create_crisis_log_backend,
    create_embedding_provider,
    create_memory_store,
    create_session_feedback_backend,
    effective_thread_persistence_backend,
)
from agent.runtime.state_store import (
    RuntimeStateStore,
    create_runtime_state_store,
)
from agent.models import (
    AgentInput,
    Channel,
    ChunkEvent,
    DoneEvent,
    Message,
    MessageRole,
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
    TextRuntimeChunkEvent,
    TextRuntimeConfig,
    TextRuntimeShadowResult,
    TextRuntimeStateEvent,
    TextRuntimeStatusEvent,
    ThreadSummary,
)
from agent.runtime_context import PrefetchedTurnMemory
from agent.runtime_context import WorkflowContext
from agent.state import AgentState, AgentTurnInputState, resolve_owner_id
from llm.base import BaseLLMClient

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[2]
_STORE_DIR = BACKEND_ROOT / ".store"
DEFAULT_THREAD_DB_PATH = _STORE_DIR / "threads.sqlite3"
DEFAULT_MEMORY_DB_PATH = _STORE_DIR / "memory.sqlite3"
DEFAULT_TEXT_SESSION_DB_PATH = _STORE_DIR / "text_sessions.sqlite3"
DEFAULT_CRISIS_LOG_DB_PATH = _STORE_DIR / "crisis.sqlite3"
DEFAULT_FEEDBACK_DB_PATH = _STORE_DIR / "session_feedback.sqlite3"
ALLOWED_MSGPACK_MODULES: tuple[str, ...] = ()
SESSION_TIMEOUT = timedelta(minutes=20)


@dataclass(slots=True)
class SessionSweepResult:
    """Summary of one expired-session sweep pass."""

    checked: int = 0
    finalized: int = 0
    skipped_excluded: int = 0
    skipped_missing: int = 0
    skipped_not_expired: int = 0
    failed_to_list: bool = False
    failed_thread_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RuntimeStoragePaths:
    """Grouped SQLite path overrides for runtime-owned storage."""

    sqlite_path: str | Path = DEFAULT_THREAD_DB_PATH
    memory_sqlite_path: str | Path = DEFAULT_MEMORY_DB_PATH
    crisis_log_sqlite_path: str | Path = DEFAULT_CRISIS_LOG_DB_PATH
    feedback_sqlite_path: str | Path = DEFAULT_FEEDBACK_DB_PATH
    text_session_sqlite_path: str | Path | None = None


@dataclass(slots=True)
class RuntimePersistenceConfig:
    """Grouped backend and database URL configuration for the runtime."""

    memory_mode: MemoryMode = MemoryMode.LOCAL
    memory_backend: Literal["sqlite", "postgres"] = "sqlite"
    memory_database_url: str | None = None
    thread_persistence_backend: Literal["sqlite", "postgres"] = "sqlite"
    thread_database_url: str | None = None
    crisis_log_persistence_backend: Literal["sqlite", "postgres"] = "sqlite"
    crisis_log_database_url: str | None = None
    session_feedback_persistence_backend: Literal["sqlite", "postgres"] = "sqlite"
    session_feedback_database_url: str | None = None
    text_session_backend: TextSessionBackend = "auto"
    text_session_database_url: str | None = None


@dataclass(slots=True)
class RuntimeDependencies:
    """Grouped dependency injection hooks for runtime construction."""

    memory_store: MemoryStore | None = None
    crisis_log_backend: CrisisLogBackend | None = None
    session_feedback_backend: SessionFeedbackBackend | None = None
    embedding_provider: "EmbeddingProvider | None" = None
    default_llm_client: BaseLLMClient | None = None
    auto_finalize_excluded: Callable[[str], bool] | None = None


@dataclass(slots=True)
class RuntimeBehaviorConfig:
    """Grouped operational behavior settings for the runtime."""

    text_session_create_tables: bool = True
    text_session_history_limit: int | None = None
    session_timeout: timedelta = SESSION_TIMEOUT
    session_sweep_interval_seconds: float = 30.0
    finalize_active_sessions_on_close: bool = True
    speculative_memory_prefetch: bool = True


class PersistentAgentRuntime:
    """Session-persisted runtime with mode-aware persistence backends."""

    def __init__(
        self,
        sqlite_path: str | Path = DEFAULT_THREAD_DB_PATH,
        *,
        storage_paths: RuntimeStoragePaths | None = None,
        persistence_config: RuntimePersistenceConfig | None = None,
        dependencies: RuntimeDependencies | None = None,
        behavior_config: RuntimeBehaviorConfig | None = None,
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
        text_session_backend: TextSessionBackend = "auto",
        text_session_database_url: str | None = None,
        text_session_sqlite_path: str | Path | None = None,
        text_session_create_tables: bool = True,
        text_session_history_limit: int | None = None,
        crisis_log_sqlite_path: str | Path = DEFAULT_CRISIS_LOG_DB_PATH,
        feedback_sqlite_path: str | Path = DEFAULT_FEEDBACK_DB_PATH,
        embedding_provider: "EmbeddingProvider | None" = None,
        default_llm_client: BaseLLMClient | None = None,
        session_timeout: timedelta = SESSION_TIMEOUT,
        session_sweep_interval_seconds: float = 30.0,
        finalize_active_sessions_on_close: bool = True,
        auto_finalize_excluded: Callable[[str], bool] | None = None,
        speculative_memory_prefetch: bool = True,
    ) -> None:
        """Initialize the runtime.

        Args:
            sqlite_path: SQLite database path for runtime thread state.
                Forced to ``:memory:`` in incognito mode.
            storage_paths: Optional grouped SQLite path overrides. When provided,
                these values take precedence over the legacy path arguments.
            persistence_config: Optional grouped backend and database URL
                settings. When provided, these values take precedence over the
                legacy persistence arguments.
            dependencies: Optional grouped dependency overrides. When provided,
                these values take precedence over the legacy dependency args.
            behavior_config: Optional grouped runtime behavior settings. When
                provided, these values take precedence over the legacy behavior
                arguments.
            memory_store: Optional explicit memory-store override.
            crisis_log_backend: Optional explicit crisis-log override.
            session_feedback_backend: Optional explicit feedback-backend override.
            memory_mode: Persistence tier for the runtime.
            memory_backend: Memory-store backend to use for persistent modes.
            memory_database_url: PostgreSQL connection string used when
                ``memory_backend`` is ``"postgres"``.
            thread_persistence_backend: Runtime thread-state backend to use for
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
            text_session_backend: Optional OpenAI Agents SDK session backend
                used for model-visible short-term conversation memory.
            text_session_database_url: SQLAlchemy async-capable database URL
                used when ``text_session_backend`` is ``"sqlalchemy"``.
            text_session_sqlite_path: SQLite path for the SDK session store.
                Defaults to a ``text_sessions.sqlite3`` sibling of the runtime
                state database, and to ``:memory:`` for in-memory threads.
            text_session_create_tables: Whether SQLAlchemy SDK sessions may
                create their own tables when first used.
            text_session_history_limit: Optional SDK session item limit.
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
            speculative_memory_prefetch: When ``True`` (default), schedule a
                turn-memory load at turn start so it overlaps with the
                crisis/control/grounded gates. The wasted work on non-load
                paths is bounded; set to ``False`` to revert to the strictly
                sequential load.
        """

        if storage_paths is not None:
            sqlite_path = storage_paths.sqlite_path
            memory_sqlite_path = storage_paths.memory_sqlite_path
            crisis_log_sqlite_path = storage_paths.crisis_log_sqlite_path
            feedback_sqlite_path = storage_paths.feedback_sqlite_path
            text_session_sqlite_path = storage_paths.text_session_sqlite_path

        if persistence_config is not None:
            memory_mode = persistence_config.memory_mode
            memory_backend = persistence_config.memory_backend
            memory_database_url = persistence_config.memory_database_url
            thread_persistence_backend = persistence_config.thread_persistence_backend
            thread_database_url = persistence_config.thread_database_url
            crisis_log_persistence_backend = (
                persistence_config.crisis_log_persistence_backend
            )
            crisis_log_database_url = persistence_config.crisis_log_database_url
            session_feedback_persistence_backend = (
                persistence_config.session_feedback_persistence_backend
            )
            session_feedback_database_url = (
                persistence_config.session_feedback_database_url
            )
            text_session_backend = persistence_config.text_session_backend
            text_session_database_url = persistence_config.text_session_database_url

        if dependencies is not None:
            memory_store = dependencies.memory_store
            crisis_log_backend = dependencies.crisis_log_backend
            session_feedback_backend = dependencies.session_feedback_backend
            embedding_provider = dependencies.embedding_provider
            default_llm_client = dependencies.default_llm_client
            auto_finalize_excluded = dependencies.auto_finalize_excluded

        if behavior_config is not None:
            text_session_create_tables = behavior_config.text_session_create_tables
            text_session_history_limit = behavior_config.text_session_history_limit
            session_timeout = behavior_config.session_timeout
            session_sweep_interval_seconds = (
                behavior_config.session_sweep_interval_seconds
            )
            finalize_active_sessions_on_close = (
                behavior_config.finalize_active_sessions_on_close
            )
            speculative_memory_prefetch = behavior_config.speculative_memory_prefetch

        self.memory_mode = memory_mode
        is_incognito = memory_mode == MemoryMode.INCOGNITO

        resolved_sqlite = ":memory:" if is_incognito else sqlite_path
        self.sqlite_path = (
            Path(resolved_sqlite) if resolved_sqlite != ":memory:" else Path(":memory:")
        )
        if text_session_sqlite_path is not None:
            resolved_text_session_sqlite_path = text_session_sqlite_path
        elif resolved_sqlite == ":memory:":
            resolved_text_session_sqlite_path = ":memory:"
        else:
            resolved_text_session_sqlite_path = Path(resolved_sqlite).with_name(
                DEFAULT_TEXT_SESSION_DB_PATH.name
            )

        self._thread_persistence_backend = effective_thread_persistence_backend(
            memory_mode=memory_mode,
            thread_persistence_backend=thread_persistence_backend,
        )
        self._thread_database_url = thread_database_url
        self._openai_text_runtime: OpenAITextRuntime | None = None
        self._openai_shadow_runtime: OpenAITextRuntime | None = None
        self._state_store: RuntimeStateStore = create_runtime_state_store(
            backend=self._thread_persistence_backend,
            sqlite_path=self.sqlite_path,
            database_url=self._thread_database_url,
        )
        self._text_session_store: TextSessionStore | None = create_text_session_store(
            memory_mode=memory_mode,
            backend=text_session_backend,
            sqlite_path=resolved_text_session_sqlite_path,
            database_url=text_session_database_url,
            create_tables=text_session_create_tables,
            history_limit=text_session_history_limit,
        )
        self._default_llm_client = default_llm_client
        self._session_timeout = session_timeout
        self._session_sweep_interval_seconds = max(
            1.0, float(session_sweep_interval_seconds)
        )
        self._finalize_active_sessions_on_close = finalize_active_sessions_on_close
        self._auto_finalize_excluded = auto_finalize_excluded
        self._speculative_memory_prefetch = speculative_memory_prefetch
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

        self._session_tracker = RuntimeSessionTracker()
        if self._thread_persistence_backend == "postgres":
            if not self._thread_database_url:
                raise ValueError(
                    "thread_database_url is required when "
                    "thread_persistence_backend='postgres'"
                )
            self._active_session_store = PostgresActiveSessionStore(
                dsn=self._thread_database_url
            )
        else:
            self._active_session_store = SqliteActiveSessionStore(
                sqlite_path=self.sqlite_path
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
        if self._text_session_store is not None:
            await self._text_session_store.aclose()
        await self._state_store.aclose()
        await self._active_session_store.aclose()

    async def _ensure_runtime_schema(self) -> None:
        """Create runtime-owned tables.

        Returns:
            None.
        """

        await self._state_store.ensure_schema()

    async def _prewarm(self) -> None:
        """Warm runtime resources before the first user turn.

        Returns:
            None.
        """

        self._get_openai_text_runtime()

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

    async def _list_active_thread_ids(self) -> list[str]:
        """List thread ids with unresolved active sessions.

        Incognito runtimes have no persisted store, so they return the
        in-process tracker's known threads. Persistent runtimes delegate
        to the active-session manager.

        Returns:
            The unresolved active-session thread ids.
        """

        if self.memory_mode == MemoryMode.INCOGNITO:
            return self._session_tracker.thread_ids()
        return await self._active_session_manager.list_persisted_active_session_ids()

    async def _clear_session_continuity_in_state(
        self,
        thread_id: str,
        state: AgentState | None,
        *,
        suppress_errors: bool = False,
    ) -> None:
        """Clear session-scoped continuity fields from persisted runtime state.

        Args:
            thread_id: The thread identifier to update.
            state: The current persisted runtime state, if any.
            suppress_errors: Whether state update failures should be logged.

        Returns:
            None.

        Raises:
            Exception: Propagates state update failures when
                ``suppress_errors`` is ``False``.
        """

        delta = session_continuity_clear_delta(state)
        if not delta:
            return

        try:
            updated = cast(AgentState, dict(state))
            for key, value in delta.items():
                if isinstance(value, Mapping) and isinstance(updated.get(key), Mapping):
                    updated[key] = cast(Any, {**dict(updated.get(key, {})), **value})
                else:
                    updated[key] = cast(Any, value)
            await self._state_store.save_state(thread_id, updated)
        except Exception:
            if suppress_errors:
                logger.warning(
                    "failed to clear session continuity for thread %s",
                    thread_id,
                    exc_info=True,
                )
                return
            raise

    def _clear_thread_state(self, thread_id: str) -> None:
        """Drop all in-process state for one thread.

        Clears the session tracker (active-session metadata, transcript
        start index, max crisis level, session memory buffer) and pops
        the cached per-thread LLM client. Composite operation — both
        pieces are runtime-owned and always cleared together.

        Args:
            thread_id: The thread identifier to clear.
        """

        self._session_tracker.clear(thread_id)
        self._thread_llm_clients.pop(thread_id, None)

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
        session_buffer: SessionMemoryBuffer | None = None,
        *,
        last_active_at: str | None = None,
    ) -> None:
        """Persist in-process session trackers for one thread.

        Args:
            thread_id: The thread identifier to persist.
            session_buffer: Optional session buffer override. When in-process
                tracking has been cleared but a persisted active-session row
                still exists, this lets callers preserve held candidates.
            last_active_at: Optional explicit last-active timestamp.

        Returns:
            None.
        """

        session = self._session_tracker.to_persisted_session(
            thread_id,
            last_active_at=last_active_at or _iso_now(),
        )
        if session is None and session_buffer is not None:
            persisted = (
                await self._active_session_manager.load_persisted_active_session(
                    thread_id
                )
            )
            if persisted is None:
                return
            session = PersistedActiveSessionState(
                thread_id=persisted.thread_id,
                started_at=persisted.started_at,
                last_active_at=last_active_at or persisted.last_active_at,
                transcript_start_index=persisted.transcript_start_index,
                max_crisis_level=persisted.max_crisis_level,
                session_buffer=session_buffer.model_copy(deep=True),
            )
        elif session is not None and session_buffer is not None:
            session = PersistedActiveSessionState(
                thread_id=session.thread_id,
                started_at=session.started_at,
                last_active_at=session.last_active_at,
                transcript_start_index=session.transcript_start_index,
                max_crisis_level=session.max_crisis_level,
                session_buffer=session_buffer.model_copy(deep=True),
            )
        if session is None:
            return
        await self._active_session_manager.save_persisted_active_session(session)

    async def _finalize_expired_sessions_once(self) -> SessionSweepResult:
        """Finalize any sessions that crossed the inactivity timeout.

        Returns:
            Summary of the sweep pass.
        """

        result = SessionSweepResult()

        try:
            active_thread_ids = await self._list_active_thread_ids()
        except Exception:
            result.failed_to_list = True
            logger.warning(
                "finalize_expired_sessions_once: failed to list active sessions",
                exc_info=True,
            )
            return result

        result.checked = len(active_thread_ids)

        for active_thread_id in active_thread_ids:
            try:
                if self._auto_finalization_excluded(active_thread_id):
                    result.skipped_excluded += 1
                    continue
                persisted = (
                    await self._active_session_manager.load_persisted_active_session(
                        active_thread_id
                    )
                )
                if persisted is None:
                    result.skipped_missing += 1
                    continue
                if not self._active_session_manager.session_has_expired(persisted):
                    result.skipped_not_expired += 1
                    continue
                logger.info(
                    "session timeout reached for thread %s; auto-finalizing expired session",
                    active_thread_id,
                )
                await self.end_session(
                    active_thread_id,
                    llm_client=self._effective_llm_client(active_thread_id),
                )
                result.finalized += 1
            except Exception:
                result.failed_thread_ids.append(active_thread_id)
                logger.warning(
                    "finalize_expired_sessions_once: failed to end expired session for thread %s",
                    active_thread_id,
                    exc_info=True,
                )

        return result

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
            prior_state: The last persisted runtime state for the thread.
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

        persisted = await self._active_session_manager.load_persisted_active_session(
            thread_id
        )
        if persisted is not None:
            self._session_tracker.hydrate(persisted)
            if self._active_session_manager.session_has_expired(persisted):
                logger.info(
                    "session timeout reached for thread %s; ending prior session before new turn",
                    thread_id,
                )
                await self._end_session_unlocked(thread_id, llm_client=llm_client)
                persisted = None

        if persisted is None and self._session_tracker.has_tracking(thread_id):
            return

        if persisted is None:
            await self._clear_session_continuity_in_state(thread_id, prior_state)
            now = _iso_now()
            self._session_tracker.start_session(
                thread_id,
                started_at=now,
                transcript_start_index=transcript_length(prior_state),
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
        self._session_tracker.record_crisis_level(thread_id, turn_level)

        turn_approach = final_state.get("therapeutic_approach")
        session_buffer = self._session_memory_buffer_for_thread(thread_id)
        session_buffer.record_approach(turn_approach)
        diagnostics = final_state.get("diagnostics", {}) or {}
        transcript = final_state.get("transcript", []) or []
        latest_user_text = next(
            (
                str(message.get("content") or "")
                for message in reversed(transcript)
                if isinstance(message, Mapping) and message.get("role") == "user"
            ),
            "",
        )
        if diagnostics.get("openai_triage_no_clarification_reason") == (
            "explicit_privacy_control"
        ) or text_contains_memory_control_request(latest_user_text):
            session_buffer.held_semantic_candidates.clear()
            session_buffer.held_procedural_candidates.clear()

        await self._persist_runtime_session_tracking(thread_id)

        if session_transcript_soft_limit is None:
            return
        transcript_start_index = self._session_tracker.transcript_start_index(thread_id)
        active_transcript_len = active_transcript_length(
            final_state,
            transcript_start_index=transcript_start_index,
        )
        if active_transcript_len >= session_transcript_soft_limit:
            await self._active_session_manager.set_active_session_rotation_required(
                thread_id
            )

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
    ) -> TextRuntimeConfig:
        """Build text-runtime config for one thread.

        Args:
            thread_id: The thread identifier.
            channel: The current channel, if known.
            user_id: The user identifier, if known.
            streaming: Whether the runtime turn is streaming.

        Returns:
            The text-runtime config payload.
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

        return self._session_tracker.session_memory_buffer_for_thread(thread_id)

    def _context_for_turn(
        self,
        *,
        thread_id: str,
        message: str,
        prior_state: AgentState | None,
        user_id: str | None,
        llm_client: BaseLLMClient | None,
        response_llm_client: BaseLLMClient | None = None,
        track_session: bool = True,
    ) -> WorkflowContext:
        """Build the agent workflow runtime context for one turn.

        Args:
            thread_id: The thread identifier.
            message: The user message for this turn. Used to seed the
                speculative memory pre-fetch with the current user text.
            prior_state: The last persisted runtime state for this thread, used
                to compute ``is_first_turn`` for the pre-fetch.
            user_id: The optional user identifier. Together with ``thread_id``
                it determines the memory owner via
                :func:`agent.state.resolve_owner_id`.
            llm_client: The control-plane LLM client.
            response_llm_client: Optional response-writer override.
            track_session: Whether the context should create runtime-local
                session tracking helpers. Shadow runs must keep this disabled
                so they do not affect liveness or recovery state.

        Returns:
            The runtime context for the turn.
        """

        return WorkflowContext(
            llm_client=llm_client,
            response_llm=response_llm_client,
            memory_store=self._memory_store,
            crisis_log_backend=self._crisis_log_backend,
            memory_mode=self.memory_mode,
            embedding_provider=self._embedding_provider,
            session_memory_buffer=(
                self._session_memory_buffer_for_thread(thread_id)
                if track_session
                else None
            ),
            pre_fetched_memory=(
                self._schedule_memory_prefetch(
                    thread_id=thread_id,
                    user_id=user_id,
                    message=message,
                    prior_state=prior_state,
                )
                if track_session
                else None
            ),
        )

    def _schedule_memory_prefetch(
        self,
        *,
        thread_id: str,
        user_id: str | None,
        message: str,
        prior_state: AgentState | None,
    ) -> PrefetchedTurnMemory | None:
        """Schedule a speculative turn-memory load when applicable.

        The fetch overlaps with the crisis/control/grounded gates so that the
        therapeutic path can ``await`` an already-resolved result. The
        crisis/control/grounded paths discard the result; the wasted work is
        bounded to one DB query batch plus the embedding compute.

        Args:
            thread_id: The thread identifier; used as the memory owner when
                no ``user_id`` is set.
            user_id: Optional user identifier; takes precedence over
                ``thread_id`` for owner resolution to mirror
                :func:`agent.state.resolve_owner_id`.
            message: User message text used as the retrieval query.
            prior_state: Last persisted runtime state for the thread; used to
                compute ``is_first_turn``.

        Returns:
            The scheduled prefetch wrapper when speculation is active; ``None``
            when speculation is disabled, the runtime is incognito, or the
            owner could not be resolved (defensive — should not occur for
            normal turn inputs).
        """

        if not self._speculative_memory_prefetch:
            return None
        if self.memory_mode == MemoryMode.INCOGNITO:
            return None

        owner_id = user_id or thread_id
        if not owner_id:
            return None

        is_first_turn = transcript_length(prior_state) == 0
        return PrefetchedTurnMemory(
            task=asyncio.create_task(
                load_memory_for_turn(
                    memory_store=self._memory_store,
                    embedding_provider=self._embedding_provider,
                    owner_id=owner_id,
                    query=message,
                    is_first_turn=is_first_turn,
                ),
                name=f"memory-prefetch:{thread_id}",
            ),
            owner_id=owner_id,
            query=message,
            is_first_turn=is_first_turn,
        )

    def _get_openai_text_runtime(self) -> OpenAITextRuntime:
        """Return the serving OpenAI Agents SDK text runtime."""

        if self._openai_text_runtime is None:
            self._openai_text_runtime = OpenAITextRuntime()
        return self._openai_text_runtime

    def _get_openai_shadow_runtime(self) -> OpenAITextRuntime:
        """Return a non-serving OpenAI runtime for shadow comparisons."""

        if self._openai_shadow_runtime is None:
            self._openai_shadow_runtime = OpenAITextRuntime()
        return self._openai_shadow_runtime

    async def _openai_sdk_session_for_thread(
        self,
        thread_id: str,
        *,
        current_user_message: str,
        prior_state: AgentState | None,
    ) -> Any | None:
        """Return the SDK session for OpenAI serving turns when enabled."""

        if self._text_session_store is None:
            return None
        await self._recover_empty_openai_sdk_session_from_state(thread_id, prior_state)
        return self._text_session_store.turn_session_for_thread(
            thread_id,
            current_user_message=current_user_message,
        )

    async def _recover_empty_openai_sdk_session_from_state(
        self,
        thread_id: str,
        prior_state: AgentState | None,
    ) -> bool:
        """Recover an empty SDK session from app-visible transcript state.

        The SDK session owns model-visible episodic conversation history.
        ``AgentState.transcript`` is app-visible state for UI/API/audit use, and
        only crosses back into the SDK session when that session is empty.
        """

        if self._text_session_store is None or prior_state is None:
            return False
        messages = messages_from_transcript(prior_state.get("transcript", []))
        if not messages:
            return False
        return await self._text_session_store.seed_thread_from_messages(
            thread_id,
            messages,
        )

    async def _ensure_openai_sdk_turn_recorded(
        self,
        thread_id: str,
        *,
        user_message: str,
        final_state: AgentState,
    ) -> None:
        """Ensure SDK history contains the finalized OpenAI user/assistant turn."""

        if self._text_session_store is None:
            return
        response_text = str(final_state.get("response_text") or "").strip()
        await self._text_session_store.ensure_turn_recorded(
            thread_id,
            user_message=user_message,
            assistant_message=response_text,
        )

    async def _history_for_final_state(
        self,
        thread_id: str,
        final_state: AgentState,
    ) -> list[Message]:
        """Return public history without calling the public get_history method."""

        if self._text_session_store is not None:
            history = await self._text_session_store.get_history(thread_id)
            if history:
                return _merge_history_response_styles(history, final_state)
        return messages_from_transcript(final_state.get("transcript", []))

    async def get_state(self, thread_id: str) -> AgentState | None:
        """Load the latest persisted state snapshot for a thread.

        Args:
            thread_id: The thread identifier.

        Returns:
            The latest persisted runtime state, if any.
        """

        return await self._state_store.load_state(thread_id)

    async def build_voice_tool_context(
        self,
        *,
        thread_id: str,
        user_id: str | None,
        current_user_message: str,
        transcript: list[dict[str, object]],
        llm_client: BaseLLMClient | None = None,
    ) -> OpenAITextRunContext:
        """Build the app-owned context used by voice function tools."""

        effective_user_message = current_user_message.strip() or _latest_user_text(
            transcript
        )
        if not effective_user_message:
            effective_user_message = "voice tool call"
        prior_state = await self.get_state(thread_id)
        prior_turn_count = turn_count_from_state(prior_state)
        initial_state = self._build_turn_initial_state(
            thread_id=thread_id,
            message=effective_user_message,
            channel=Channel.VOICE,
            user_id=user_id,
            installed_skills=None,
            prior_turn_count=prior_turn_count,
        )
        state = cast(AgentState, {**dict(prior_state or {}), **dict(initial_state)})
        if transcript:
            state["transcript"] = cast(Any, [dict(turn) for turn in transcript])
        elif prior_state is not None:
            state["transcript"] = list(prior_state.get("transcript", []) or [])

        memory_control = state.get("memory_control", {}) or {}
        pending_memory_action = (
            memory_control.get("pending_action")
            if isinstance(memory_control, Mapping)
            else None
        )
        workflow_context = self._context_for_turn(
            thread_id=thread_id,
            message=effective_user_message,
            prior_state=prior_state,
            user_id=user_id,
            llm_client=llm_client,
            response_llm_client=llm_client,
            track_session=False,
        )
        return OpenAITextRunContext(
            thread_id=thread_id,
            workflow_context=workflow_context,
            current_user_message=effective_user_message,
            user_id=user_id,
            session_id=thread_id,
            channel=Channel.VOICE,
            pending_memory_action=(
                dict(pending_memory_action)
                if isinstance(pending_memory_action, Mapping)
                else None
            ),
            agent_state=state,
            installed_skills=list(state.get("installed_skills", []) or []),
            transcript=cast(list[dict[str, Any]], list(state.get("transcript", []))),
            turn_count=turn_count_from_state(state),
        )

    async def voice_session_memory_context(
        self,
        *,
        thread_id: str,
        user_id: str | None,
        memory_mode: str | None = None,
    ) -> str:
        """Return compact saved-memory context for a Realtime voice session.

        Voice sessions are created before the user has said anything, so a
        semantic recall here has no real query to match on. Bootstrap context
        is therefore limited to the user's standing procedural rules and the
        proactive-recall toggle; topic-specific recall happens mid-session via
        the ``recall_saved_memory`` tool when the user introduces a topic.
        """

        if memory_mode == "incognito" or self.memory_mode == MemoryMode.INCOGNITO:
            return ""

        initial_state = self._build_turn_initial_state(
            thread_id=thread_id,
            message="",
            channel=Channel.VOICE,
            user_id=user_id,
            installed_skills=None,
            prior_turn_count=0,
        )
        owner_id = resolve_owner_id(cast(AgentState, initial_state))
        profile = await aget_procedural_profile(self._memory_store, user_id=owner_id)
        delta: dict[str, Any] = {
            "working_memory": [],
            "session_memory": {"summary": ""},
            "procedural_profile": {
                "procedural_rules": [{"rule": rule.rule} for rule in profile.rules],
                "proactive_recall_enabled": profile.proactive_recall_enabled,
            },
        }
        return _compact_voice_memory_context(delta)

    async def record_voice_turn(
        self,
        *,
        thread_id: str,
        user_id: str | None,
        user_text: str,
        assistant_text: str,
        route: str | None = None,
        response_style: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        llm_client: BaseLLMClient | None = None,
    ) -> AgentState:
        """Persist a finalized voice turn without running the text agent."""

        entries = voice_turn_to_transcript_entries(
            user_text=user_text,
            assistant_text=assistant_text,
            response_style=response_style,
        )
        if not entries:
            raise ValueError("record_voice_turn requires user_text or assistant_text.")

        async with self._thread_lock(thread_id):
            self._remember_llm_client(thread_id, llm_client)
            prior_state = await self.get_state(thread_id)
            await self._prepare_session_for_turn(
                thread_id=thread_id,
                prior_state=prior_state,
                llm_client=llm_client,
            )
            prior_state = await self.get_state(thread_id)
            prior_turn_count = turn_count_from_state(prior_state)
            seed_message = user_text.strip() or assistant_text.strip()
            initial_state = self._build_turn_initial_state(
                thread_id=thread_id,
                message=seed_message,
                channel=Channel.VOICE,
                user_id=user_id,
                installed_skills=None,
                prior_turn_count=prior_turn_count,
            )

            state = cast(
                AgentState,
                {**dict(initial_state), **dict(prior_state or {})},
            )
            prior_transcript = (
                list(prior_state.get("transcript", []) or [])
                if prior_state is not None
                else []
            )
            voice_tool_calls = list(tool_calls or [])
            grounded_lookup: dict[str, Any] = {}
            for call in voice_tool_calls:
                if not isinstance(call, Mapping):
                    continue
                output = call.get("output")
                if not isinstance(output, Mapping):
                    continue
                grounded_output = output.get("grounded_lookup")
                if isinstance(grounded_output, Mapping):
                    grounded_lookup = dict(grounded_output)

            state.update(
                {
                    "message": user_text.strip(),
                    "channel": Channel.VOICE,
                    "user_id": user_id,
                    "session_id": thread_id,
                    "response_text": assistant_text.strip(),
                    "response_style": response_style
                    or ("grounded_lookup" if grounded_lookup else "voice"),
                    "route": route or "therapeutic",
                    "session_action": "none",
                    "should_persist_memory": False,
                    "transcript": [*prior_transcript, *entries],
                    "grounded_lookup": {
                        **dict(state.get("grounded_lookup", {}) or {}),
                        **grounded_lookup,
                    },
                    "diagnostics": {
                        **dict(state.get("diagnostics", {}) or {}),
                        "voice_runtime": "openai_realtime",
                        "voice_tool_calls": [
                            str(call.get("tool_name"))
                            for call in voice_tool_calls
                            if isinstance(call, Mapping) and call.get("tool_name")
                        ],
                    },
                    "session_progress": {
                        **dict(state.get("session_progress", {}) or {}),
                        "turn_count": prior_turn_count + 1,
                        "is_guest": user_id is None,
                    },
                }
            )

            async with self._active_session_manager.active_session_mutation(
                thread_id,
                mutation_kind="voice_turn",
            ) as mutation_token:
                await self._record_successful_turn_tracking(
                    thread_id,
                    state,
                    session_transcript_soft_limit=None,
                )
                await self._state_store.save_state(thread_id, state)
                await self._ensure_openai_sdk_turn_recorded(
                    thread_id,
                    user_message=user_text,
                    final_state=state,
                )
                await self._active_session_manager.clear_active_session_mutation(
                    thread_id,
                    mutation_token,
                )
                return state

    async def prepare_voice_turn_policy(
        self,
        *,
        thread_id: str,
        user_id: str | None,
        user_text: str,
        memory_mode: str,
    ) -> VoiceTurnPolicy:
        """Return observe-only voice turn policy for one finalized transcript."""

        del user_id
        prior_state = await self.get_state(thread_id)
        exercise_state = (
            prior_state.get("exercise_state", {}) if prior_state is not None else {}
        )
        memory_control = (
            prior_state.get("memory_control", {}) if prior_state is not None else {}
        )
        has_active_guided_exercise = bool(
            isinstance(exercise_state, Mapping)
            and exercise_state.get("exercise_type") is not None
            and exercise_state.get("exercise_step") is not None
        )
        pending_memory_action = bool(
            isinstance(memory_control, Mapping)
            and memory_control.get("pending_action") is not None
        )
        return build_voice_turn_policy(
            user_text=user_text,
            memory_mode=memory_mode,
            has_active_guided_exercise=has_active_guided_exercise,
            pending_memory_action=pending_memory_action,
        )

    async def get_history(self, thread_id: str) -> list[Message]:
        """Load the full persisted transcript for a thread.

        Args:
            thread_id: The thread identifier.

        Returns:
            The materialized transcript messages for the thread.
        """

        state = await self.get_state(thread_id)
        if self._text_session_store is not None:
            history = await self._text_session_store.get_history(thread_id)
            if history:
                return _merge_history_response_styles(history, state)
            if state is None:
                return history
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
            if self._session_tracker.has_tracking(thread_id):
                return SessionStatus.ACTIVE
            return SessionStatus.ABSENT

        row = await self._active_session_manager.load_persisted_active_session_row(
            thread_id
        )
        if row is None:
            if self._session_tracker.has_tracking(thread_id):
                return SessionStatus.ACTIVE
            return SessionStatus.ABSENT

        if row.finalize_required_reason == "interrupted":
            return SessionStatus.INTERRUPTED

        if row.mutation_token is not None:
            if not self._active_session_manager.is_mutation_in_flight(
                row.mutation_token
            ):
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

        if self._active_session_manager.session_has_expired(session):
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
        """Delete all persisted runtime, SDK-session, and active-session state.

        Args:
            thread_id: The thread identifier.

        Returns:
            None.
        """

        async with self._thread_lock(thread_id):
            status = await self._session_status_unlocked(thread_id)
            if status != SessionStatus.ABSENT:
                raise ActiveSessionExists(thread_id, status)

            await self._state_store.delete_thread(thread_id)
            if self._text_session_store is not None:
                await self._text_session_store.clear_thread(thread_id)
            await self._active_session_manager.delete_persisted_active_session(
                thread_id
            )
            self._clear_thread_state(thread_id)

    async def list_threads(self, *, limit: int = 20) -> list[ThreadSummary]:
        """List the most recent persisted threads.

        Args:
            limit: The maximum number of threads to return.

        Returns:
            The most recent persisted thread summaries.
        """

        thread_ids = await self._state_store.list_thread_ids(limit=limit)

        summaries: list[ThreadSummary] = []
        for thread_id in thread_ids:
            state = await self.get_state(thread_id)
            history = await self.get_history(thread_id)
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
    def _build_turn_initial_state(
        *,
        thread_id: str,
        message: str,
        channel: Channel,
        user_id: str | None,
        installed_skills: list[str] | None,
        prior_turn_count: int,
    ) -> AgentTurnInputState:
        """Build the runtime input state for one user turn.

        Args:
            thread_id: Thread identifier used as the session id.
            message: Current user message.
            channel: Channel metadata for the turn.
            user_id: Optional user identifier.
            installed_skills: Optional installed skill names.
            prior_turn_count: Persisted user-turn count before this turn.

        Returns:
            Initial runtime state for the turn.
        """

        from agent.runtime.turn import build_initial_state

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

    async def run_openai_text_shadow_turn(
        self,
        *,
        thread_id: str,
        message: str,
        channel: Channel = Channel.TEST,
        user_id: str | None = None,
        installed_skills: list[str] | None = None,
        llm_client: BaseLLMClient | None = None,
        response_llm_client: BaseLLMClient | None = None,
    ) -> TextRuntimeShadowResult:
        """Evaluate the OpenAI text runtime without mutating served state.

        The shadow path is for evals and dogfood observability. It uses the
        same initial-turn construction and app-owned context as a normal turn,
        but it does not prepare active sessions, write runtime state, append
        transcript entries, schedule extraction, or return output to users.
        """

        async with self._thread_lock(thread_id):
            prior_state = await self.get_state(thread_id)
            prior_turn_count = turn_count_from_state(prior_state)
            initial_state = self._build_turn_initial_state(
                thread_id=thread_id,
                message=message,
                channel=channel,
                user_id=user_id,
                installed_skills=installed_skills,
                prior_turn_count=prior_turn_count,
            )
            return await self._get_openai_shadow_runtime().run_shadow_turn(
                initial_state,
                config=self._config_for_thread(
                    thread_id,
                    channel=channel,
                    user_id=user_id,
                    streaming=False,
                ),
                context=self._context_for_turn(
                    thread_id=thread_id,
                    message=message,
                    prior_state=prior_state,
                    user_id=user_id,
                    llm_client=llm_client,
                    response_llm_client=response_llm_client,
                    track_session=False,
                ),
                prior_state=prior_state,
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
            runtime = self._get_openai_text_runtime()
            self._remember_llm_client(thread_id, llm_client)

            # Runtime state restores transcript and can bootstrap an empty
            # OpenAI SDK session during migration or local session-db loss.
            prior_state = await self.get_state(thread_id)
            await self._prepare_session_for_turn(
                thread_id=thread_id,
                prior_state=prior_state,
                llm_client=llm_client,
                expected_liveness=expected_liveness,
            )
            prior_state = await self.get_state(thread_id)
            prior_turn_count = turn_count_from_state(prior_state)

            initial_state = self._build_turn_initial_state(
                thread_id=thread_id,
                message=message,
                channel=channel,
                user_id=user_id,
                installed_skills=installed_skills,
                prior_turn_count=prior_turn_count,
            )
            sdk_session = await self._openai_sdk_session_for_thread(
                thread_id,
                current_user_message=message,
                prior_state=prior_state,
            )

            async with self._active_session_manager.active_session_mutation(
                thread_id,
                mutation_kind="turn",
            ) as mutation_token:
                turn_start = time.monotonic()
                turn_output = await runtime.run_turn(
                    initial_state,
                    config=self._config_for_thread(
                        thread_id,
                        channel=channel,
                        user_id=user_id,
                        streaming=False,
                    ),
                    context=self._context_for_turn(
                        thread_id=thread_id,
                        message=message,
                        prior_state=prior_state,
                        user_id=user_id,
                        llm_client=llm_client,
                        response_llm_client=response_llm_client,
                    ),
                    session=sdk_session,
                    prior_state=prior_state,
                )
                final_state = cast(AgentState, dict(turn_output))

                stamp_turn_total_ms(final_state, started_at=turn_start)

                await self._record_successful_turn_tracking(
                    thread_id,
                    final_state,
                    session_transcript_soft_limit=session_transcript_soft_limit,
                )

                await self._state_store.save_state(thread_id, final_state)
                await self._ensure_openai_sdk_turn_recorded(
                    thread_id,
                    user_message=message,
                    final_state=final_state,
                )

                from agent.runtime.turn import state_to_output

                result = PersistentTurnResult(
                    output=state_to_output(final_state),
                    state=final_state,
                    history=await self._history_for_final_state(thread_id, final_state),
                )

                await self._active_session_manager.clear_active_session_mutation(
                    thread_id, mutation_token
                )
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
        persisted = await self._active_session_manager.load_persisted_active_session(
            thread_id
        )
        if persisted is not None:
            self._session_tracker.hydrate(persisted)
        has_active_session = (
            persisted is not None or self._session_tracker.has_tracking(thread_id)
        )

        if not has_active_session:
            return None

        @asynccontextmanager
        async def _finalize_mutation_scope() -> AsyncIterator[str | None]:
            if persisted is None:
                yield None
                return
            async with self._active_session_manager.active_session_mutation(
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
                await self._active_session_manager.delete_persisted_active_session(
                    thread_id
                )
                self._clear_thread_state(thread_id)
                return None

            try:
                transcript_start_index = self._session_tracker.transcript_start_index(
                    thread_id
                )
                session_state = slice_state_to_active_session(
                    state,
                    transcript_start_index=transcript_start_index,
                )
                started_at = self._session_tracker.started_at(
                    thread_id,
                    default=_iso_now(),
                )
                ended_at = _iso_now()
                crisis_level_max = self._session_tracker.max_crisis_level(thread_id)
                session_buffer = self._session_tracker.session_memory_buffer_or_none(
                    thread_id
                )
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
                await self._clear_session_continuity_in_state(
                    thread_id,
                    state,
                    suppress_errors=True,
                )
                await self._active_session_manager.delete_persisted_active_session(
                    thread_id
                )
                self._clear_thread_state(thread_id)
                return stored_arc
            except Exception:
                if mutation_token is not None:
                    await self._active_session_manager.clear_active_session_mutation(
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
            llm_client: The optional LLM client for session summarization.
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
            active_thread_ids = await self._list_active_thread_ids()
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
        except Exception:
            logger.warning(
                "session feedback write failed for thread %s",
                thread_id,
                exc_info=True,
            )
            return None

        return await record_runtime_session_feedback(
            backend=self._session_feedback_backend,
            thread_id=thread_id,
            state=state,
            memory_mode=self.memory_mode,
            label=label,
            source=source,
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
            runtime = self._get_openai_text_runtime()
            self._remember_llm_client(thread_id, llm_client)

            # Runtime state restores transcript and can bootstrap an empty
            # OpenAI SDK session during migration or local session-db loss.
            prior_state = await self.get_state(thread_id)
            await self._prepare_session_for_turn(
                thread_id=thread_id,
                prior_state=prior_state,
                llm_client=llm_client,
                expected_liveness=expected_liveness,
            )
            prior_state = await self.get_state(thread_id)
            prior_turn_count = turn_count_from_state(prior_state)

            initial_state = self._build_turn_initial_state(
                thread_id=thread_id,
                message=message,
                channel=channel,
                user_id=user_id,
                installed_skills=installed_skills,
                prior_turn_count=prior_turn_count,
            )
            sdk_session = await self._openai_sdk_session_for_thread(
                thread_id,
                current_user_message=message,
                prior_state=prior_state,
            )

            turn_start = time.monotonic()
            final_state: AgentState | None = None
            chunks_emitted = False
            finalize_seen = False
            response_ready_emitted = False

            async with self._active_session_manager.active_session_mutation(
                thread_id,
                mutation_kind="turn",
            ) as mutation_token:
                async for event in runtime.run_turn_stream(
                    initial_state,
                    config=self._config_for_thread(
                        thread_id,
                        channel=channel,
                        user_id=user_id,
                        streaming=True,
                    ),
                    context=self._context_for_turn(
                        thread_id=thread_id,
                        message=message,
                        prior_state=prior_state,
                        user_id=user_id,
                        llm_client=llm_client,
                        response_llm_client=response_llm_client,
                    ),
                    session=sdk_session,
                    prior_state=prior_state,
                ):
                    if isinstance(event, TextRuntimeChunkEvent):
                        yield ChunkEvent(text=event.text)
                        chunks_emitted = True
                    elif isinstance(event, TextRuntimeStatusEvent):
                        yield StatusEvent(stage=event.stage)
                        if event.turn_finalized:
                            finalize_seen = True
                            ready_output = response_ready_output(
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
                    elif isinstance(event, TextRuntimeStateEvent):
                        final_state = event.state
                        ready_output = response_ready_output(
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

                if final_state is None:
                    raise RuntimeError(
                        "run_turn_stream: text runtime stream yielded no final state."
                    )

                stamp_turn_total_ms(final_state, started_at=turn_start)

                await self._record_successful_turn_tracking(
                    thread_id,
                    final_state,
                    session_transcript_soft_limit=session_transcript_soft_limit,
                )

                await self._state_store.save_state(thread_id, final_state)
                await self._ensure_openai_sdk_turn_recorded(
                    thread_id,
                    user_message=message,
                    final_state=final_state,
                )

                await self._active_session_manager.clear_active_session_mutation(
                    thread_id, mutation_token
                )
                from agent.runtime.turn import state_to_output

                yield DoneEvent(output=state_to_output(final_state))


def _merge_history_response_styles(
    history: list[Message],
    state: AgentState | None,
) -> list[Message]:
    """Overlay assistant response styles from runtime transcript onto history."""

    if state is None:
        return history
    transcript_messages = messages_from_transcript(state.get("transcript", []))
    transcript_assistants = [
        message
        for message in transcript_messages
        if message.role == MessageRole.ASSISTANT
    ]
    if not transcript_assistants:
        return history

    enriched: list[Message] = []
    assistant_index = 0
    for message in history:
        if message.role != MessageRole.ASSISTANT or message.response_style is not None:
            enriched.append(message)
            continue

        response_style = None
        while assistant_index < len(transcript_assistants):
            candidate = transcript_assistants[assistant_index]
            assistant_index += 1
            if candidate.content == message.content:
                response_style = candidate.response_style
                break
        enriched.append(
            Message(
                role=message.role,
                content=message.content,
                response_style=response_style,
            )
        )
    return enriched


def _latest_user_text(transcript: list[dict[str, object]]) -> str:
    for turn in reversed(transcript):
        if turn.get("role") == "user":
            return str(turn.get("content") or "").strip()
    return ""


def _compact_voice_memory_context(delta: Mapping[str, Any]) -> str:
    blocks: list[str] = []
    procedural_profile = delta.get("procedural_profile") or {}
    proactive_recall_enabled = False
    if isinstance(procedural_profile, Mapping):
        proactive_recall_enabled = bool(
            procedural_profile.get("proactive_recall_enabled", False)
        )

    if proactive_recall_enabled:
        working_memory = delta.get("working_memory") or []
        if working_memory:
            rendered = [
                _compact_memory_value(item) for item in list(working_memory)[:5]
            ]
            rendered = [item for item in rendered if item]
            if rendered:
                blocks.append(
                    "Relevant saved facts:\n"
                    + "\n".join(f"- {item}" for item in rendered)
                )

        session_memory = delta.get("session_memory") or {}
        if isinstance(session_memory, Mapping):
            summary = str(session_memory.get("summary") or "").strip()
            if summary and summary != "Guest session without long-term memory.":
                blocks.append(f"Recent session summary: {summary}")

    if isinstance(procedural_profile, Mapping):
        rules = procedural_profile.get("procedural_rules") or []
        rendered_rules = [_compact_memory_value(rule) for rule in list(rules)[:5]]
        rendered_rules = [rule for rule in rendered_rules if rule]
        if rendered_rules:
            blocks.append(
                "Saved response preferences:\n"
                + "\n".join(f"- {rule}" for rule in rendered_rules)
            )
    if proactive_recall_enabled:
        blocks.append("Proactive memory recall is enabled.")
    else:
        blocks.append("Proactive saved-memory recall is disabled.")

    return "\n\n".join(blocks)[:2000]


def _compact_memory_value(value: object) -> str:
    if isinstance(value, Mapping):
        for key in (
            "evidence_quote",
            "rule",
            "summary",
            "preference",
            "text",
            "content",
        ):
            text = str(value.get(key) or "").strip()
            if text:
                return text
        return json.dumps(dict(value), sort_keys=True, default=str)[:300]
    return str(value).strip()[:300]
