"""Persistent runtime for session-persisted OpenCouch interactions."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, cast

from agent.memory.policy.candidates import SessionMemoryBuffer
from agent.audit.crisis_log import CrisisLogBackend
from agent.feedback.session_feedback import SessionFeedbackBackend
from agent.memory.hashing import iso_now as _iso_now
from agent.memory.providers.embeddings import EmbeddingProvider
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
    ThreadLockManager,
    finalize_session_window,
    transcript_length,
    turn_count_from_state,
)
from agent.runtime.session.service import (
    SessionLifecycleService,
    SessionSweepResult,
)
from agent.runtime.thread_state_reader import ThreadStateReader
from agent.runtime.session_feedback import (
    record_session_feedback as record_runtime_session_feedback,
)
from agent.runtime.streaming import (
    response_ready_output,
    stamp_turn_total_ms,
)
from agent.runtime.session_store import TextSessionBackend
from agent.runtime.openai_text_runtime import OpenAITextRuntime
from agent.runtime.sdk_session_bridge import SdkSessionBridge
from agent.memory.modes import MemoryMode
from agent.memory.store import MemoryStore
from agent.runtime import configuration as _runtime_configuration
from agent.runtime.configuration import (
    SESSION_TIMEOUT,
    RuntimeBehaviorConfig,
    RuntimeDependencies,
    RuntimePersistenceConfig,
    RuntimeStoragePaths,
    _UNSET,
    _resolve_runtime_behavior_config,
    _resolve_runtime_dependencies,
    _resolve_runtime_persistence_config,
    _resolve_runtime_storage_paths,
)
from agent.runtime.resources import RuntimeResources, build_runtime_resources
from agent.models import (
    AgentInput,
    Channel,
    ChunkEvent,
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
    SessionStatus,
    TextRuntimeChunkEvent,
    TextRuntimeConfig,
    TextRuntimeStateEvent,
    TextRuntimeStatusEvent,
    ThreadSummary,
)
from agent.runtime.workflow_context import PrefetchedTurnMemory
from agent.runtime.workflow_context import WorkflowContext
from agent.state import AgentState, AgentTurnInputState
from llm.base import BaseLLMClient

logger = logging.getLogger(__name__)

# Preserve direct imports from ``agent.runtime.runtime`` for external callers.
DEFAULT_THREAD_DB_PATH = _runtime_configuration.DEFAULT_THREAD_DB_PATH
DEFAULT_MEMORY_DB_PATH = _runtime_configuration.DEFAULT_MEMORY_DB_PATH
DEFAULT_TEXT_SESSION_DB_PATH = _runtime_configuration.DEFAULT_TEXT_SESSION_DB_PATH
DEFAULT_CRISIS_LOG_DB_PATH = _runtime_configuration.DEFAULT_CRISIS_LOG_DB_PATH
DEFAULT_FEEDBACK_DB_PATH = _runtime_configuration.DEFAULT_FEEDBACK_DB_PATH


@dataclass(slots=True)
class PreparedTextTurn:
    """Shared persistent text-turn inputs prepared before route execution."""

    text_runtime: OpenAITextRuntime
    prior_state: AgentState | None
    initial_state: AgentTurnInputState
    sdk_session: Any | None


@dataclass(slots=True)
class TextTurnExecutionContext:
    """Per-turn context/config built after active-session mutation setup."""

    workflow_context: WorkflowContext
    config: TextRuntimeConfig


class PersistentAgentRuntime:
    """Session-persisted runtime with mode-aware persistence backends."""

    def __init__(
        self,
        sqlite_path: str | Path | object = _UNSET,
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
        memory_sqlite_path: str | Path | object = _UNSET,
        text_session_backend: TextSessionBackend = "auto",
        text_session_database_url: str | None = None,
        text_session_sqlite_path: str | Path | None | object = _UNSET,
        text_session_create_tables: bool = True,
        text_session_history_limit: int | None = None,
        crisis_log_sqlite_path: str | Path | object = _UNSET,
        feedback_sqlite_path: str | Path | object = _UNSET,
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
            sqlite_path: Deprecated direct SQLite database path for runtime
                thread state. Use ``storage_paths`` instead. Forced to
                ``:memory:`` in incognito mode.
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
            memory_sqlite_path: Deprecated direct SQLite path for the default
                memory store. Use ``storage_paths`` instead.
            text_session_backend: Optional OpenAI Agents SDK session backend
                used for model-visible short-term conversation memory.
            text_session_database_url: SQLAlchemy async-capable database URL
                used when ``text_session_backend`` is ``"sqlalchemy"``.
            text_session_sqlite_path: Deprecated direct SQLite path for the SDK
                session store. Use ``storage_paths`` instead. Defaults to a
                ``text_sessions.sqlite3`` sibling of the runtime state database,
                and to ``:memory:`` for in-memory threads.
            text_session_create_tables: Whether SQLAlchemy SDK sessions may
                create their own tables when first used.
            text_session_history_limit: Optional SDK session item limit.
            crisis_log_sqlite_path: Deprecated direct SQLite path for the
                default crisis log. Use ``storage_paths`` instead.
            feedback_sqlite_path: Deprecated direct SQLite path for the default
                feedback store. Use ``storage_paths`` instead.
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

        resolved_storage_paths = _resolve_runtime_storage_paths(
            sqlite_path=sqlite_path,
            storage_paths=storage_paths,
            memory_sqlite_path=memory_sqlite_path,
            crisis_log_sqlite_path=crisis_log_sqlite_path,
            feedback_sqlite_path=feedback_sqlite_path,
            text_session_sqlite_path=text_session_sqlite_path,
        )
        sqlite_path = resolved_storage_paths.sqlite_path
        memory_sqlite_path = resolved_storage_paths.memory_sqlite_path
        crisis_log_sqlite_path = resolved_storage_paths.crisis_log_sqlite_path
        feedback_sqlite_path = resolved_storage_paths.feedback_sqlite_path
        text_session_sqlite_path = resolved_storage_paths.text_session_sqlite_path

        resolved_dependencies = _resolve_runtime_dependencies(
            dependencies=dependencies,
            memory_store=memory_store,
            crisis_log_backend=crisis_log_backend,
            session_feedback_backend=session_feedback_backend,
            embedding_provider=embedding_provider,
            default_llm_client=default_llm_client,
            auto_finalize_excluded=auto_finalize_excluded,
        )
        memory_store = resolved_dependencies.memory_store
        crisis_log_backend = resolved_dependencies.crisis_log_backend
        session_feedback_backend = resolved_dependencies.session_feedback_backend
        embedding_provider = resolved_dependencies.embedding_provider
        default_llm_client = resolved_dependencies.default_llm_client
        auto_finalize_excluded = resolved_dependencies.auto_finalize_excluded

        resolved_persistence_config = _resolve_runtime_persistence_config(
            persistence_config=persistence_config,
            memory_mode=memory_mode,
            memory_backend=memory_backend,
            memory_database_url=memory_database_url,
            memory_sqlite_path=memory_sqlite_path,
            memory_sqlite_path_configured=(
                resolved_storage_paths.memory_sqlite_path_configured
            ),
            memory_store=memory_store,
            thread_persistence_backend=thread_persistence_backend,
            thread_database_url=thread_database_url,
            sqlite_path=sqlite_path,
            sqlite_path_configured=resolved_storage_paths.sqlite_path_configured,
            crisis_log_persistence_backend=crisis_log_persistence_backend,
            crisis_log_database_url=crisis_log_database_url,
            crisis_log_sqlite_path=crisis_log_sqlite_path,
            crisis_log_sqlite_path_configured=(
                resolved_storage_paths.crisis_log_sqlite_path_configured
            ),
            crisis_log_backend=crisis_log_backend,
            session_feedback_persistence_backend=session_feedback_persistence_backend,
            session_feedback_database_url=session_feedback_database_url,
            feedback_sqlite_path=feedback_sqlite_path,
            feedback_sqlite_path_configured=(
                resolved_storage_paths.feedback_sqlite_path_configured
            ),
            session_feedback_backend=session_feedback_backend,
            text_session_backend=text_session_backend,
            text_session_database_url=text_session_database_url,
            text_session_sqlite_path=text_session_sqlite_path,
            text_session_sqlite_path_configured=(
                resolved_storage_paths.text_session_sqlite_path_configured
            ),
        )
        memory_mode = resolved_persistence_config.memory_mode
        memory_backend = resolved_persistence_config.memory_backend
        memory_database_url = resolved_persistence_config.memory_database_url
        thread_persistence_backend = (
            resolved_persistence_config.thread_persistence_backend
        )
        thread_database_url = resolved_persistence_config.thread_database_url
        crisis_log_persistence_backend = (
            resolved_persistence_config.crisis_log_persistence_backend
        )
        crisis_log_database_url = resolved_persistence_config.crisis_log_database_url
        session_feedback_persistence_backend = (
            resolved_persistence_config.session_feedback_persistence_backend
        )
        session_feedback_database_url = (
            resolved_persistence_config.session_feedback_database_url
        )
        text_session_backend = resolved_persistence_config.text_session_backend
        text_session_database_url = (
            resolved_persistence_config.text_session_database_url
        )

        resolved_behavior_config = _resolve_runtime_behavior_config(
            behavior_config=behavior_config,
            text_session_create_tables=text_session_create_tables,
            text_session_history_limit=text_session_history_limit,
            session_timeout=session_timeout,
            session_sweep_interval_seconds=session_sweep_interval_seconds,
            finalize_active_sessions_on_close=finalize_active_sessions_on_close,
            speculative_memory_prefetch=speculative_memory_prefetch,
        )
        text_session_create_tables = resolved_behavior_config.text_session_create_tables
        text_session_history_limit = resolved_behavior_config.text_session_history_limit
        session_timeout = resolved_behavior_config.session_timeout
        session_sweep_interval_seconds = (
            resolved_behavior_config.session_sweep_interval_seconds
        )
        finalize_active_sessions_on_close = (
            resolved_behavior_config.finalize_active_sessions_on_close
        )
        speculative_memory_prefetch = (
            resolved_behavior_config.speculative_memory_prefetch
        )

        runtime_resources = build_runtime_resources(
            memory_mode=memory_mode,
            sqlite_path=sqlite_path,
            text_session_sqlite_path=text_session_sqlite_path,
            thread_persistence_backend=thread_persistence_backend,
            thread_database_url=thread_database_url,
            text_session_backend=text_session_backend,
            text_session_database_url=text_session_database_url,
            text_session_create_tables=text_session_create_tables,
            text_session_history_limit=text_session_history_limit,
            memory_store=memory_store,
            memory_backend=memory_backend,
            memory_database_url=memory_database_url,
            memory_sqlite_path=memory_sqlite_path,
            crisis_log_backend=crisis_log_backend,
            crisis_log_persistence_backend=crisis_log_persistence_backend,
            crisis_log_database_url=crisis_log_database_url,
            crisis_log_sqlite_path=crisis_log_sqlite_path,
            session_feedback_backend=session_feedback_backend,
            session_feedback_persistence_backend=session_feedback_persistence_backend,
            session_feedback_database_url=session_feedback_database_url,
            feedback_sqlite_path=feedback_sqlite_path,
            embedding_provider=embedding_provider,
        )
        self._wire_runtime_resources(
            resources=runtime_resources,
            memory_mode=memory_mode,
            default_llm_client=default_llm_client,
            session_timeout=session_timeout,
            session_sweep_interval_seconds=session_sweep_interval_seconds,
            finalize_active_sessions_on_close=finalize_active_sessions_on_close,
            auto_finalize_excluded=auto_finalize_excluded,
            speculative_memory_prefetch=speculative_memory_prefetch,
        )

    def _wire_runtime_resources(
        self,
        *,
        resources: RuntimeResources,
        memory_mode: MemoryMode,
        default_llm_client: BaseLLMClient | None,
        session_timeout: timedelta,
        session_sweep_interval_seconds: float,
        finalize_active_sessions_on_close: bool,
        auto_finalize_excluded: Callable[[str], bool] | None,
        speculative_memory_prefetch: bool,
    ) -> None:
        """Attach runtime resources and dependent services."""

        self.memory_mode = memory_mode
        self._default_llm_client = default_llm_client
        self._session_timeout = session_timeout
        self._session_sweep_interval_seconds = max(
            1.0, float(session_sweep_interval_seconds)
        )
        self._finalize_active_sessions_on_close = finalize_active_sessions_on_close
        self._auto_finalize_excluded = auto_finalize_excluded
        self._speculative_memory_prefetch = speculative_memory_prefetch
        self._thread_llm_clients: dict[str, BaseLLMClient | None] = {}
        self._session_tracker = RuntimeSessionTracker()
        self._thread_lock_manager = ThreadLockManager()

        self._resources: RuntimeResources = resources
        self.sqlite_path = resources.sqlite_path
        self._thread_persistence_backend = resources.thread_persistence_backend
        self._thread_database_url = resources.thread_database_url
        self._state_store = resources.state_store
        self._text_session_store = resources.text_session_store
        self._memory_store = resources.memory_store
        self._crisis_log_backend = resources.crisis_log_backend
        self._session_feedback_backend = resources.session_feedback_backend
        self._embedding_provider = resources.embedding_provider
        self._active_session_store = resources.active_session_store
        self._active_session_manager = resources.active_session_manager
        self._thread_state_reader = ThreadStateReader(
            state_store=self._state_store,
            text_session_store=self._text_session_store,
            active_session_manager=self._active_session_manager,
            session_tracker=self._session_tracker,
            memory_mode=self.memory_mode,
            session_timeout=self._session_timeout,
        )
        self._sdk_bridge = SdkSessionBridge(
            text_session_store=self._text_session_store,
        )
        self._session_lifecycle = SessionLifecycleService(
            memory_mode=self.memory_mode,
            session_tracker=self._session_tracker,
            thread_lock_manager=self._thread_lock_manager,
            active_session_manager=self._active_session_manager,
            state_store=self._state_store,
            memory_store=self._memory_store,
            embedding_provider=self._embedding_provider,
            thread_llm_clients=self._thread_llm_clients,
            session_timeout=self._session_timeout,
            session_sweep_interval_seconds=self._session_sweep_interval_seconds,
            auto_finalize_excluded=self._auto_finalize_excluded,
        )

        from agent.voice.runtime_facade import VoiceRuntimeFacade

        self.voice = VoiceRuntimeFacade(
            runtime=self,
            state_store=self._state_store,
            memory_store=self._memory_store,
            active_session_manager=self._active_session_manager,
            session_lifecycle=self._session_lifecycle,
            lock_for=self._thread_lock_manager.get_lock,
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
        await self.voice.aclose()
        await self._resources.aclose()

    async def _ensure_runtime_schema(self) -> None:
        """Create runtime-owned tables.

        Returns:
            None.
        """

        await self._resources.ensure_schema()

    async def _prewarm(self) -> None:
        """Warm runtime resources before the first user turn.

        Returns:
            None.
        """

        await self._resources.prewarm(get_text_runtime=self._get_openai_text_runtime)

    def _thread_lock(self, thread_id: str) -> asyncio.Lock:
        """Return the in-process lock for one thread.

        Args:
            thread_id: Thread identifier.

        Returns:
            The per-thread asyncio lock.
        """

        return self._thread_lock_manager.get_lock(thread_id)

    async def _list_active_thread_ids(self) -> list[str]:
        """List thread ids with unresolved active sessions.

        Kept as a runtime-level shim because tests monkeypatch it on the runtime
        and ``_finalize_expired_sessions_once`` passes it through as a callback.
        """

        return await self._session_lifecycle.list_active_thread_ids()

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
        """Persist session tracking through the lifecycle owner.

        Retained as a runtime compatibility shim for integration and recovery
        callers; lifecycle policy remains in ``SessionLifecycleService``.
        """
        await self._session_lifecycle.persist_runtime_session_tracking(
            thread_id,
            session_buffer,
            last_active_at=last_active_at,
        )

    async def _finalize_expired_sessions_once(self) -> SessionSweepResult:
        """Finalize any sessions that crossed the inactivity timeout.

        Kept as a runtime-level shim because the persistence sweeper tests call
        it directly (and monkeypatch ``_list_active_thread_ids`` on the runtime).
        """

        return await self._session_lifecycle.finalize_expired_sessions_once(
            end_session=self.end_session,
            effective_llm_client=self._effective_llm_client,
            list_active_thread_ids=self._list_active_thread_ids,
            is_auto_finalization_excluded=(
                self._session_lifecycle.auto_finalization_excluded
            ),
        )

    async def _prepare_session_for_turn(
        self,
        *,
        thread_id: str,
        prior_state: AgentState | None,
        llm_client: BaseLLMClient | None,
        expected_liveness: ExpectedSessionLiveness | None = None,
    ) -> None:
        """Restore or create the active session before a new turn.

        Kept as a runtime-level shim because the voice runtime facade reaches it.
        """

        await self._session_lifecycle.prepare_session_for_turn(
            thread_id=thread_id,
            prior_state=prior_state,
            llm_client=llm_client,
            expected_liveness=expected_liveness,
            session_status_unlocked=self._session_status_unlocked,
            end_session_unlocked=self._end_session_unlocked,
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
                session tracking helpers. Non-serving callers (e.g. the voice
                runtime facade) keep this disabled so they do not affect
                liveness or recovery state.

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

        return self._sdk_bridge.get_text_runtime()

    async def _ensure_openai_sdk_turn_recorded(
        self,
        thread_id: str,
        *,
        user_message: str,
        final_state: AgentState,
    ) -> None:
        """Ensure SDK history contains the finalized OpenAI user/assistant turn."""

        await self._sdk_bridge.ensure_turn_recorded(
            thread_id,
            user_message=user_message,
            final_state=final_state,
        )

    async def get_state(self, thread_id: str) -> AgentState | None:
        """Load the latest persisted state snapshot for a thread."""

        return await self._thread_state_reader.get_state(thread_id)

    async def get_history(self, thread_id: str) -> list[Message]:
        """Load the full persisted transcript for a thread."""

        return await self._thread_state_reader.get_history(thread_id)

    async def session_status(self, thread_id: str) -> SessionStatus:
        """Return the active-session liveness status for a thread."""

        return await self._thread_state_reader.session_status(thread_id)

    async def _session_status_unlocked(self, thread_id: str) -> SessionStatus:
        """Return session status without acquiring the per-thread lock.

        Retained as a runtime shim because ``reset_thread`` and the
        session-lifecycle paths depend on the unlocked liveness check.
        """

        return await self._thread_state_reader.session_status_unlocked(thread_id)

    async def has_active_session(self, thread_id: str) -> bool:
        """Return whether a thread currently has an unresolved session."""

        return await self._thread_state_reader.has_active_session(thread_id)

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
        """List the most recent persisted threads."""

        return await self._thread_state_reader.list_threads(limit=limit)

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
            installed_skills: Optional installed capability keys.
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

    async def _prepare_text_turn(
        self,
        *,
        thread_id: str,
        message: str,
        channel: Channel,
        user_id: str | None,
        installed_skills: list[str] | None,
        llm_client: BaseLLMClient | None,
        expected_liveness: ExpectedSessionLiveness | None,
    ) -> PreparedTextTurn:
        """Prepare shared persistent state and SDK session inputs for a text turn."""

        text_runtime = self._get_openai_text_runtime()
        self._remember_llm_client(thread_id, llm_client)

        # Runtime state restores transcript and can bootstrap an empty OpenAI SDK
        # session during migration or local session-db loss.
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
        sdk_session = await self._sdk_bridge.session_for_thread(
            thread_id,
            current_user_message=message,
            prior_state=prior_state,
        )
        return PreparedTextTurn(
            text_runtime=text_runtime,
            prior_state=prior_state,
            initial_state=initial_state,
            sdk_session=sdk_session,
        )

    def _text_turn_execution_context(
        self,
        *,
        thread_id: str,
        message: str,
        channel: Channel,
        user_id: str | None,
        llm_client: BaseLLMClient | None,
        response_llm_client: BaseLLMClient | None,
        prior_state: AgentState | None,
        streaming: bool,
    ) -> TextTurnExecutionContext:
        """Build context/config once active-session mutation setup succeeds."""

        return TextTurnExecutionContext(
            workflow_context=self._context_for_turn(
                thread_id=thread_id,
                message=message,
                prior_state=prior_state,
                user_id=user_id,
                llm_client=llm_client,
                response_llm_client=response_llm_client,
            ),
            config=self._config_for_thread(
                thread_id,
                channel=channel,
                user_id=user_id,
                streaming=streaming,
            ),
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
            installed_skills: Optional installed capability keys.
            llm_client: The control-plane LLM client.
            response_llm_client: Optional response-writer override.
            expected_liveness: Optional active-session liveness expectation.
            session_transcript_soft_limit: Optional active-session transcript
                message limit that marks the session for rotation after success.

        Returns:
            The persisted turn result, including output, state, and history.
        """

        async with self._thread_lock(thread_id):
            prepared = await self._prepare_text_turn(
                thread_id=thread_id,
                message=message,
                channel=channel,
                user_id=user_id,
                installed_skills=installed_skills,
                llm_client=llm_client,
                expected_liveness=expected_liveness,
            )

            async with self._active_session_manager.active_session_mutation(
                thread_id,
                mutation_kind="turn",
            ) as mutation_token:
                turn_start = time.monotonic()
                execution = self._text_turn_execution_context(
                    thread_id=thread_id,
                    message=message,
                    channel=channel,
                    user_id=user_id,
                    llm_client=llm_client,
                    response_llm_client=response_llm_client,
                    prior_state=prepared.prior_state,
                    streaming=False,
                )
                turn_output = await prepared.text_runtime.run_turn(
                    prepared.initial_state,
                    config=execution.config,
                    context=execution.workflow_context,
                    session=prepared.sdk_session,
                    prior_state=prepared.prior_state,
                )
                final_state = cast(AgentState, dict(turn_output))

                stamp_turn_total_ms(final_state, started_at=turn_start)

                await self._session_lifecycle.complete_successful_turn(
                    thread_id=thread_id,
                    user_message=message,
                    final_state=final_state,
                    workflow_context=execution.workflow_context,
                    mutation_token=mutation_token,
                    ensure_sdk_turn_recorded=self._ensure_openai_sdk_turn_recorded,
                    session_transcript_soft_limit=session_transcript_soft_limit,
                )

                from agent.runtime.turn import state_to_output

                result = PersistentTurnResult(
                    output=state_to_output(final_state),
                    state=final_state,
                    history=await self._sdk_bridge.history_for_final_state(
                        thread_id, final_state
                    ),
                )

                return result

    async def end_session(
        self,
        thread_id: str,
        *,
        llm_client: BaseLLMClient | None = None,
        finalize_only_if_expired: bool = False,
    ) -> StoredSessionArc | None:
        """Summarize the active session for a thread and write it to memory.

        Args:
            thread_id: The thread whose active session should be summarized.
            llm_client: The optional LLM client for session summarization.
            finalize_only_if_expired: When ``True`` (background sweeper), re-check
                expiry under the lock and skip if the session was renewed. Left
                ``False`` for explicit/shutdown callers, which finalize
                unconditionally.

        Returns:
            The written session arc, or ``None`` when summarization is skipped.
        """

        async with self._thread_lock(thread_id):
            return await self._end_session_unlocked(
                thread_id,
                llm_client=llm_client,
                finalize_only_if_expired=finalize_only_if_expired,
            )

    async def _end_session_unlocked(
        self,
        thread_id: str,
        *,
        llm_client: BaseLLMClient | None = None,
        finalize_only_if_expired: bool = False,
    ) -> StoredSessionArc | None:
        """Summarize an active session while the caller owns the thread lock."""

        return await self._session_lifecycle.end_session_unlocked(
            thread_id,
            llm_client=llm_client,
            effective_llm_client=self._effective_llm_client,
            session_status_unlocked=self._session_status_unlocked,
            get_state=self.get_state,
            finalize_only_if_expired=finalize_only_if_expired,
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
            is_auto_finalization_excluded=(
                self._session_lifecycle.auto_finalization_excluded
            ),
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
            installed_skills: Optional installed capability keys.
            llm_client: The control-plane LLM client.
            response_llm_client: Optional response-writer override.
            expected_liveness: Optional active-session liveness expectation.
            session_transcript_soft_limit: Optional active-session transcript
                message limit that marks the session for rotation after success.

        Yields:
            Stream events for status updates, response readiness, and completion.
        """

        async with self._thread_lock(thread_id):
            prepared = await self._prepare_text_turn(
                thread_id=thread_id,
                message=message,
                channel=channel,
                user_id=user_id,
                installed_skills=installed_skills,
                llm_client=llm_client,
                expected_liveness=expected_liveness,
            )

            turn_start = time.monotonic()
            final_state: AgentState | None = None
            chunks_emitted = False
            finalize_seen = False

            async with self._active_session_manager.active_session_mutation(
                thread_id,
                mutation_kind="turn",
            ) as mutation_token:
                execution = self._text_turn_execution_context(
                    thread_id=thread_id,
                    message=message,
                    channel=channel,
                    user_id=user_id,
                    llm_client=llm_client,
                    response_llm_client=response_llm_client,
                    prior_state=prepared.prior_state,
                    streaming=True,
                )
                async for event in prepared.text_runtime.run_turn_stream(
                    prepared.initial_state,
                    config=execution.config,
                    context=execution.workflow_context,
                    session=prepared.sdk_session,
                    prior_state=prepared.prior_state,
                ):
                    if isinstance(event, TextRuntimeChunkEvent):
                        yield ChunkEvent(text=event.text)
                        chunks_emitted = True
                    elif isinstance(event, TextRuntimeStatusEvent):
                        yield StatusEvent(stage=event.stage)
                        if event.turn_finalized:
                            finalize_seen = True
                    elif isinstance(event, TextRuntimeStateEvent):
                        final_state = event.state

                if final_state is None:
                    raise RuntimeError(
                        "run_turn_stream: text runtime stream yielded no final state."
                    )

                stamp_turn_total_ms(final_state, started_at=turn_start)

                await self._session_lifecycle.complete_successful_turn(
                    thread_id=thread_id,
                    user_message=message,
                    final_state=final_state,
                    workflow_context=execution.workflow_context,
                    mutation_token=mutation_token,
                    ensure_sdk_turn_recorded=self._ensure_openai_sdk_turn_recorded,
                    session_transcript_soft_limit=session_transcript_soft_limit,
                )

                ready_output = response_ready_output(
                    final_state,
                    finalize_seen=finalize_seen,
                    response_ready_emitted=False,
                )
                if ready_output is not None:
                    if not chunks_emitted:
                        yield ChunkEvent(text=ready_output.response_text)
                        chunks_emitted = True
                    yield ResponseReadyEvent(output=ready_output)

                from agent.runtime.turn import state_to_output

                yield DoneEvent(output=state_to_output(final_state))
