"""Persistent runtime for session-persisted OpenCouch interactions."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
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
from agent.memory.providers.embeddings import EmbeddingProvider
from agent.memory.policy.write import text_contains_memory_control_request
from agent.memory.retrieval.service import load_memory_for_turn
from agent.feedback.models import (
    FeedbackLabel,
    FeedbackModality,
    FeedbackSource,
    SessionFeedbackRecord,
)
from agent.memory.types import StoredSessionArc
from agent.runtime.session import (
    RuntimeSessionTracker,
    active_transcript_length,
    crisis_level_from_state,
    finalize_session_window,
    transcript_length,
    turn_count_from_state,
)
from agent.runtime.session.history import messages_from_transcript
from agent.runtime.session.service import (
    SessionLifecycleService,
    SessionSweepResult,
)
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
from agent.state import AgentState, AgentTurnInputState
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
_UNSET = object()


@dataclass(slots=True)
class RuntimeStoragePaths:
    """Grouped SQLite path overrides for runtime-owned storage."""

    sqlite_path: str | Path | object = _UNSET
    memory_sqlite_path: str | Path | object = _UNSET
    crisis_log_sqlite_path: str | Path | object = _UNSET
    feedback_sqlite_path: str | Path | object = _UNSET
    text_session_sqlite_path: str | Path | None | object = _UNSET


@dataclass(slots=True)
class RuntimePersistenceConfig:
    """Grouped backend and database URL configuration for the runtime."""

    memory_mode: MemoryMode | object = _UNSET
    memory_backend: Literal["sqlite", "postgres"] | object = _UNSET
    memory_database_url: str | None | object = _UNSET
    thread_persistence_backend: Literal["sqlite", "postgres"] | object = _UNSET
    thread_database_url: str | None | object = _UNSET
    crisis_log_persistence_backend: Literal["sqlite", "postgres"] | object = _UNSET
    crisis_log_database_url: str | None | object = _UNSET
    session_feedback_persistence_backend: Literal["sqlite", "postgres"] | object = (
        _UNSET
    )
    session_feedback_database_url: str | None | object = _UNSET
    text_session_backend: TextSessionBackend | object = _UNSET
    text_session_database_url: str | None | object = _UNSET


@dataclass(slots=True)
class RuntimeDependencies:
    """Grouped dependency injection hooks for runtime construction."""

    memory_store: MemoryStore | None | object = _UNSET
    crisis_log_backend: CrisisLogBackend | None | object = _UNSET
    session_feedback_backend: SessionFeedbackBackend | None | object = _UNSET
    embedding_provider: "EmbeddingProvider | None | object" = _UNSET
    default_llm_client: BaseLLMClient | None | object = _UNSET
    auto_finalize_excluded: Callable[[str], bool] | None | object = _UNSET


@dataclass(slots=True)
class RuntimeBehaviorConfig:
    """Grouped operational behavior settings for the runtime."""

    text_session_create_tables: bool | object = _UNSET
    text_session_history_limit: int | None | object = _UNSET
    session_timeout: timedelta | object = _UNSET
    session_sweep_interval_seconds: float | object = _UNSET
    finalize_active_sessions_on_close: bool | object = _UNSET
    speculative_memory_prefetch: bool | object = _UNSET


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
            if storage_paths.sqlite_path is not _UNSET:
                sqlite_path = cast(str | Path, storage_paths.sqlite_path)
            if storage_paths.memory_sqlite_path is not _UNSET:
                memory_sqlite_path = cast(
                    str | Path,
                    storage_paths.memory_sqlite_path,
                )
            if storage_paths.crisis_log_sqlite_path is not _UNSET:
                crisis_log_sqlite_path = cast(
                    str | Path,
                    storage_paths.crisis_log_sqlite_path,
                )
            if storage_paths.feedback_sqlite_path is not _UNSET:
                feedback_sqlite_path = cast(
                    str | Path,
                    storage_paths.feedback_sqlite_path,
                )
            if storage_paths.text_session_sqlite_path is not _UNSET:
                text_session_sqlite_path = cast(
                    str | Path | None,
                    storage_paths.text_session_sqlite_path,
                )

        if persistence_config is not None:
            if persistence_config.memory_mode is not _UNSET:
                memory_mode = cast(MemoryMode, persistence_config.memory_mode)
            if persistence_config.memory_backend is not _UNSET:
                memory_backend = cast(
                    Literal["sqlite", "postgres"],
                    persistence_config.memory_backend,
                )
            if persistence_config.memory_database_url is not _UNSET:
                memory_database_url = cast(
                    str | None,
                    persistence_config.memory_database_url,
                )
            if persistence_config.thread_persistence_backend is not _UNSET:
                thread_persistence_backend = cast(
                    Literal["sqlite", "postgres"],
                    persistence_config.thread_persistence_backend,
                )
            if persistence_config.thread_database_url is not _UNSET:
                thread_database_url = cast(
                    str | None,
                    persistence_config.thread_database_url,
                )
            if persistence_config.crisis_log_persistence_backend is not _UNSET:
                crisis_log_persistence_backend = cast(
                    Literal["sqlite", "postgres"],
                    persistence_config.crisis_log_persistence_backend,
                )
            if persistence_config.crisis_log_database_url is not _UNSET:
                crisis_log_database_url = cast(
                    str | None,
                    persistence_config.crisis_log_database_url,
                )
            if persistence_config.session_feedback_persistence_backend is not _UNSET:
                session_feedback_persistence_backend = cast(
                    Literal["sqlite", "postgres"],
                    persistence_config.session_feedback_persistence_backend,
                )
            if persistence_config.session_feedback_database_url is not _UNSET:
                session_feedback_database_url = cast(
                    str | None,
                    persistence_config.session_feedback_database_url,
                )
            if persistence_config.text_session_backend is not _UNSET:
                text_session_backend = cast(
                    TextSessionBackend,
                    persistence_config.text_session_backend,
                )
            if persistence_config.text_session_database_url is not _UNSET:
                text_session_database_url = cast(
                    str | None,
                    persistence_config.text_session_database_url,
                )

        if dependencies is not None:
            if dependencies.memory_store is not _UNSET:
                memory_store = cast(MemoryStore | None, dependencies.memory_store)
            if dependencies.crisis_log_backend is not _UNSET:
                crisis_log_backend = cast(
                    CrisisLogBackend | None,
                    dependencies.crisis_log_backend,
                )
            if dependencies.session_feedback_backend is not _UNSET:
                session_feedback_backend = cast(
                    SessionFeedbackBackend | None,
                    dependencies.session_feedback_backend,
                )
            if dependencies.embedding_provider is not _UNSET:
                embedding_provider = cast(
                    EmbeddingProvider | None,
                    dependencies.embedding_provider,
                )
            if dependencies.default_llm_client is not _UNSET:
                default_llm_client = cast(
                    BaseLLMClient | None,
                    dependencies.default_llm_client,
                )
            if dependencies.auto_finalize_excluded is not _UNSET:
                auto_finalize_excluded = cast(
                    Callable[[str], bool] | None,
                    dependencies.auto_finalize_excluded,
                )

        if behavior_config is not None:
            if behavior_config.text_session_create_tables is not _UNSET:
                text_session_create_tables = cast(
                    bool,
                    behavior_config.text_session_create_tables,
                )
            if behavior_config.text_session_history_limit is not _UNSET:
                text_session_history_limit = cast(
                    int | None,
                    behavior_config.text_session_history_limit,
                )
            if behavior_config.session_timeout is not _UNSET:
                session_timeout = cast(timedelta, behavior_config.session_timeout)
            if behavior_config.session_sweep_interval_seconds is not _UNSET:
                session_sweep_interval_seconds = cast(
                    float,
                    behavior_config.session_sweep_interval_seconds,
                )
            if behavior_config.finalize_active_sessions_on_close is not _UNSET:
                finalize_active_sessions_on_close = cast(
                    bool,
                    behavior_config.finalize_active_sessions_on_close,
                )
            if behavior_config.speculative_memory_prefetch is not _UNSET:
                speculative_memory_prefetch = cast(
                    bool,
                    behavior_config.speculative_memory_prefetch,
                )

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
        self._thread_llm_clients: dict[str, BaseLLMClient | None] = {}

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
        self._session_lifecycle = SessionLifecycleService(
            memory_mode=self.memory_mode,
            session_tracker=self._session_tracker,
            active_session_manager=self._active_session_manager,
            state_store=self._state_store,
            memory_store=self._memory_store,
            embedding_provider=self._embedding_provider,
            thread_llm_clients=self._thread_llm_clients,
            session_sweep_interval_seconds=self._session_sweep_interval_seconds,
            auto_finalize_excluded=self._auto_finalize_excluded,
        )

        from agent.voice.runtime_facade import VoiceRuntimeFacade

        self.voice = VoiceRuntimeFacade(
            runtime=self,
            state_store=self._state_store,
            memory_store=self._memory_store,
            active_session_manager=self._active_session_manager,
            lock_for=self._session_lifecycle.thread_lock,
            memory_mode=self.memory_mode,
        )

    async def __aenter__(self) -> PersistentAgentRuntime:
        """Open runtime resources.

        Returns:
            The initialized runtime instance.
        """

        await self._ensure_runtime_schema()
        await self._prewarm()
        self._session_lifecycle.start_background_tasks(
            finalize_expired_sessions_once=self._finalize_expired_sessions_once
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Close runtime resources.

        Args:
            exc_type: The active exception type, if any.
            exc: The active exception instance, if any.
            tb: The active traceback, if any.
        """

        await self._session_lifecycle.stop_background_tasks()
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

        return self._session_lifecycle.thread_lock(thread_id)

    def _auto_finalization_excluded(self, thread_id: str) -> bool:
        """Return whether runtime background finalization should skip a thread."""

        return self._session_lifecycle.auto_finalization_excluded(thread_id)

    async def _list_active_thread_ids(self) -> list[str]:
        """List thread ids with unresolved active sessions."""

        return await self._session_lifecycle.list_active_thread_ids()

    async def _clear_session_continuity_in_state(
        self,
        thread_id: str,
        state: AgentState | None,
        *,
        suppress_errors: bool = False,
    ) -> None:
        """Clear session-scoped continuity fields from persisted runtime state."""

        await self._session_lifecycle.clear_session_continuity_in_state(
            thread_id,
            state,
            suppress_errors=suppress_errors,
        )

    def _clear_thread_state(self, thread_id: str) -> None:
        """Drop all in-process state for one thread."""

        self._session_lifecycle.clear_thread_state(thread_id)

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
        """Persist in-process session trackers for one thread."""

        await self._session_lifecycle.persist_runtime_session_tracking(
            thread_id,
            session_buffer,
            last_active_at=last_active_at,
        )

    async def _finalize_expired_sessions_once(self) -> SessionSweepResult:
        """Finalize any sessions that crossed the inactivity timeout."""

        return await self._session_lifecycle.finalize_expired_sessions_once(
            end_session=self.end_session,
            effective_llm_client=self._effective_llm_client,
            list_active_thread_ids=self._list_active_thread_ids,
            is_auto_finalization_excluded=self._auto_finalization_excluded,
        )

    async def _prepare_session_for_turn(
        self,
        *,
        thread_id: str,
        prior_state: AgentState | None,
        llm_client: BaseLLMClient | None,
        expected_liveness: ExpectedSessionLiveness | None = None,
    ) -> None:
        """Restore or create the active session before a new turn."""

        await self._session_lifecycle.prepare_session_for_turn(
            thread_id=thread_id,
            prior_state=prior_state,
            llm_client=llm_client,
            expected_liveness=expected_liveness,
            session_status_unlocked=self._session_status_unlocked,
            end_session_unlocked=self._end_session_unlocked,
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
        """Summarize an active session while the caller owns the thread lock."""

        return await self._session_lifecycle.end_session_unlocked(
            thread_id,
            llm_client=llm_client,
            effective_llm_client=self._effective_llm_client,
            session_status_unlocked=self._session_status_unlocked,
            get_state=self.get_state,
        )

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
        """Finalize any unresolved active sessions."""

        await self._session_lifecycle.finalize_active_sessions(
            llm_client=llm_client,
            end_session=self.end_session,
            effective_llm_client=self._effective_llm_client,
            list_active_thread_ids=self._list_active_thread_ids,
            is_auto_finalization_excluded=self._auto_finalization_excluded,
        )

    async def record_session_feedback(
        self,
        thread_id: str,
        *,
        label: FeedbackLabel,
        source: FeedbackSource,
        modality: FeedbackModality = "text",
    ) -> SessionFeedbackRecord | None:
        """Record an explicit end-of-session feedback label.

        Args:
            thread_id: The thread whose session is ending.
            label: The explicit feedback label the user provided.
            source: Which end-session surface produced this feedback.
            modality: Which interaction channel the user is rating.

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
            modality=modality,
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
