"""High-level session lifecycle orchestration for the persistent runtime."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any, cast

from agent.memory.providers.embeddings import EmbeddingProvider
from agent.memory.hashing import iso_now as _iso_now
from agent.memory.modes import MemoryMode
from agent.memory.policy.candidates import SessionMemoryBuffer
from agent.memory.policy.write import text_contains_memory_control_request
from agent.memory.store import MemoryStore
from agent.memory.types import StoredSessionArc
from agent.runtime.session.active_session import (
    ActiveSessionManager,
    PersistedActiveSessionState,
)
from agent.runtime.session.finalization import finalize_session_window
from agent.runtime.session.lock import ThreadLockManager
from agent.runtime.session.state import (
    active_transcript_length,
    crisis_level_from_state,
    session_continuity_clear_delta,
    session_has_expired,
    slice_state_to_active_session,
    transcript_length,
)
from agent.runtime.session.tracking import RuntimeSessionTracker
from agent.runtime.state_store import RuntimeStateStore
from agent.runtime.types import (
    ActiveSessionExists,
    ExpectedSessionLiveness,
    SessionInterrupted,
    SessionLeaseExpired,
    SessionStatus,
)
from agent.runtime.workflow_context import WorkflowContext
from agent.state import AgentState
from llm.base import BaseLLMClient

if TYPE_CHECKING:
    from agent.audit.capture import SafetyEventCaptureResult
    from agent.runtime.finalization import EnsureSdkTurnRecorded

logger = logging.getLogger(__name__)

EffectiveLLMResolver = Callable[
    [str, BaseLLMClient | None],
    BaseLLMClient | None,
]
EndSessionCallback = Callable[..., Awaitable[StoredSessionArc | None]]
ListActiveThreadIdsCallback = Callable[[], Awaitable[list[str]]]
SessionStatusResolver = Callable[[str], Awaitable[SessionStatus]]
StateLoader = Callable[[str], Awaitable[AgentState | None]]


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


class SessionLifecycleService:
    """Own high-level active-session lifecycle orchestration."""

    def __init__(
        self,
        *,
        memory_mode: MemoryMode,
        session_tracker: RuntimeSessionTracker,
        thread_lock_manager: ThreadLockManager,
        active_session_manager: ActiveSessionManager,
        state_store: RuntimeStateStore,
        memory_store: MemoryStore,
        embedding_provider: EmbeddingProvider,
        thread_llm_clients: dict[str, BaseLLMClient | None],
        session_timeout: timedelta,
        session_sweep_interval_seconds: float,
        auto_finalize_excluded: Callable[[str], bool] | None = None,
    ) -> None:
        """Initialize the session lifecycle service."""
        self._memory_mode = memory_mode
        self._session_tracker = session_tracker
        self._thread_lock_manager = thread_lock_manager
        self._active_session_manager = active_session_manager
        self._state_store = state_store
        self._memory_store = memory_store
        self._embedding_provider = embedding_provider
        self._thread_llm_clients = thread_llm_clients
        self._session_timeout = session_timeout
        self._session_sweep_interval_seconds = session_sweep_interval_seconds
        self._auto_finalize_excluded = auto_finalize_excluded
        self._session_sweeper_task: asyncio.Task[None] | None = None

    def thread_lock(self, thread_id: str) -> asyncio.Lock:
        """Return the in-process lock for one thread."""
        return self._thread_lock_manager.get_lock(thread_id)

    def prune_idle_thread_locks(self) -> int:
        """Drop in-process locks for threads with no live or tracked work."""
        return self._thread_lock_manager.prune_idle_locks(
            is_tracked=self._session_tracker.has_tracking
        )

    def start_background_tasks(
        self,
        *,
        finalize_expired_sessions_once: Callable[[], Awaitable[SessionSweepResult]],
    ) -> None:
        """Start lifecycle-owned background tasks."""
        if (
            self._session_sweeper_task is not None
            and not self._session_sweeper_task.done()
        ):
            return
        self._session_sweeper_task = asyncio.create_task(
            self.session_sweeper_loop(
                finalize_expired_sessions_once=finalize_expired_sessions_once
            )
        )

    async def stop_background_tasks(self) -> None:
        """Stop lifecycle-owned background tasks."""
        if self._session_sweeper_task is None:
            return
        self._session_sweeper_task.cancel()
        try:
            await self._session_sweeper_task
        except asyncio.CancelledError:
            pass
        self._session_sweeper_task = None

    def auto_finalization_excluded(self, thread_id: str) -> bool:
        """Return whether runtime background finalization should skip a thread."""
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

    async def list_active_thread_ids(self) -> list[str]:
        """List thread ids with unresolved active sessions."""
        if self._memory_mode == MemoryMode.INCOGNITO:
            return self._session_tracker.thread_ids()
        return await self._active_session_manager.list_persisted_active_session_ids()

    async def clear_session_continuity_in_state(
        self,
        thread_id: str,
        state: AgentState | None,
        *,
        suppress_errors: bool = False,
    ) -> None:
        """Clear session-scoped continuity fields from persisted runtime state."""
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

    def clear_thread_state(self, thread_id: str) -> None:
        """Drop all in-process state for one thread."""
        self._session_tracker.clear(thread_id)
        self._thread_llm_clients.pop(thread_id, None)

    async def persist_runtime_session_tracking(
        self,
        thread_id: str,
        session_buffer: SessionMemoryBuffer | None = None,
        *,
        last_active_at: str | None = None,
    ) -> None:
        """Persist in-process session trackers for one thread."""
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

    async def _record_successful_turn(
        self,
        thread_id: str,
        final_state: AgentState,
        *,
        session_transcript_soft_limit: int | None,
    ) -> None:
        """Persist lifecycle-owned tracking after a successful locked turn."""
        turn_level = crisis_level_from_state(final_state)
        self._session_tracker.record_crisis_level(thread_id, turn_level)

        session_buffer = self._session_tracker.session_memory_buffer_for_thread(
            thread_id
        )
        session_buffer.record_approach(final_state.get("therapeutic_approach"))

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

        await self.persist_runtime_session_tracking(thread_id)

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

    async def complete_successful_turn(
        self,
        *,
        thread_id: str,
        user_message: str,
        final_state: AgentState,
        workflow_context: WorkflowContext,
        mutation_token: str,
        ensure_sdk_turn_recorded: EnsureSdkTurnRecorded,
        session_transcript_soft_limit: int | None,
        capture_safety_event: bool = True,
    ) -> SafetyEventCaptureResult:
        """Complete one successful turn under its thread lock and mutation scope."""
        from agent.runtime.finalization import finalize_successful_turn

        await self._record_successful_turn(
            thread_id,
            final_state,
            session_transcript_soft_limit=session_transcript_soft_limit,
        )
        return await finalize_successful_turn(
            thread_id=thread_id,
            user_message=user_message,
            final_state=final_state,
            workflow_context=workflow_context,
            state_store=self._state_store,
            active_session_manager=self._active_session_manager,
            mutation_token=mutation_token,
            ensure_sdk_turn_recorded=ensure_sdk_turn_recorded,
            capture_safety_event=capture_safety_event,
        )

    async def finalize_expired_sessions_once(
        self,
        *,
        end_session: EndSessionCallback,
        effective_llm_client: EffectiveLLMResolver,
        list_active_thread_ids: ListActiveThreadIdsCallback | None = None,
        is_auto_finalization_excluded: Callable[[str], bool] | None = None,
    ) -> SessionSweepResult:
        """Finalize any sessions that crossed the inactivity timeout."""
        result = SessionSweepResult()
        list_active_thread_ids = list_active_thread_ids or self.list_active_thread_ids
        is_auto_finalization_excluded = (
            is_auto_finalization_excluded or self.auto_finalization_excluded
        )

        try:
            active_thread_ids = await list_active_thread_ids()
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
                if is_auto_finalization_excluded(active_thread_id):
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
                if not session_has_expired(
                    persisted.last_active_at,
                    session_timeout=self._session_timeout,
                ):
                    result.skipped_not_expired += 1
                    continue
                logger.info(
                    "session timeout reached for thread %s; auto-finalizing expired session",
                    active_thread_id,
                )
                # finalize_only_if_expired re-checks expiry under the thread lock;
                # if a concurrent turn renewed the session since the unlocked check
                # above, end_session skips finalization (logged at debug) and leaves
                # it active. ``finalized`` counts sweep finalize attempts that did
                # not raise, consistent with its existing meaning.
                await end_session(
                    active_thread_id,
                    llm_client=effective_llm_client(active_thread_id, None),
                    finalize_only_if_expired=True,
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

    async def session_sweeper_loop(
        self,
        *,
        finalize_expired_sessions_once: Callable[[], Awaitable[SessionSweepResult]],
    ) -> None:
        """Run the background session-timeout sweeper loop."""
        try:
            while True:
                await asyncio.sleep(self._session_sweep_interval_seconds)
                await finalize_expired_sessions_once()
                self.prune_idle_thread_locks()
        except asyncio.CancelledError:
            raise

    async def prepare_session_for_turn(
        self,
        *,
        thread_id: str,
        prior_state: AgentState | None,
        llm_client: BaseLLMClient | None,
        expected_liveness: ExpectedSessionLiveness | None = None,
        session_status_unlocked: SessionStatusResolver,
        end_session_unlocked: EndSessionCallback,
    ) -> None:
        """Restore or create the active session before a new turn."""
        status = await session_status_unlocked(thread_id)
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
            if session_has_expired(
                persisted.last_active_at,
                session_timeout=self._session_timeout,
            ):
                logger.info(
                    "session timeout reached for thread %s; ending prior session before new turn",
                    thread_id,
                )
                await end_session_unlocked(thread_id, llm_client=llm_client)
                persisted = None

        if persisted is None and self._session_tracker.has_tracking(thread_id):
            return

        if persisted is None:
            await self.clear_session_continuity_in_state(thread_id, prior_state)
            now = _iso_now()
            self._session_tracker.start_session(
                thread_id,
                started_at=now,
                transcript_start_index=transcript_length(prior_state),
            )
            await self.persist_runtime_session_tracking(
                thread_id,
                last_active_at=now,
            )

    async def end_session_unlocked(
        self,
        thread_id: str,
        *,
        llm_client: BaseLLMClient | None = None,
        effective_llm_client: EffectiveLLMResolver,
        session_status_unlocked: SessionStatusResolver,
        get_state: StateLoader,
        finalize_only_if_expired: bool = False,
    ) -> StoredSessionArc | None:
        """Summarize an active session while the caller owns the thread lock.

        Args:
            finalize_only_if_expired: When ``True`` (the background sweeper
                path), re-check expiry against the freshly-loaded persisted row
                under the lock and skip finalization if the session is no longer
                expired. This closes the TOCTOU where a concurrent turn renews
                ``last_active_at`` between the sweeper's unlocked expiry check and
                lock acquisition. Explicit and shutdown callers leave this
                ``False`` so they always finalize unconditionally.
        """
        resolved_llm_client = effective_llm_client(thread_id, llm_client)
        status = await session_status_unlocked(thread_id)
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

        # Sweeper-only: the expiry check that selected this thread ran without the
        # lock. Now that we hold it, re-evaluate against the fresh persisted row;
        # a concurrent turn may have renewed the session in the meantime. Leave it
        # active and let a later sweep re-evaluate.
        if (
            finalize_only_if_expired
            and persisted is not None
            and not session_has_expired(
                persisted.last_active_at,
                session_timeout=self._session_timeout,
            )
        ):
            logger.debug(
                "session for thread %s was renewed since the sweep check; "
                "skipping auto-finalize",
                thread_id,
            )
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
            state = await get_state(thread_id)

            if state is None:
                await self._active_session_manager.delete_persisted_active_session(
                    thread_id
                )
                self.clear_thread_state(thread_id)
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
                    llm_client=resolved_llm_client,
                    memory_store=self._memory_store,
                    memory_mode=self._memory_mode,
                    embedding_provider=self._embedding_provider,
                    # Normal conversational session end extracts memory candidates
                    # from the whole transcript; the transcript-finalize path does
                    # not (summarize-only contract).
                    extract_candidates=True,
                )
                await self.clear_session_continuity_in_state(
                    thread_id,
                    state,
                    suppress_errors=True,
                )
                await self._active_session_manager.delete_persisted_active_session(
                    thread_id
                )
                self.clear_thread_state(thread_id)
                return stored_arc
            except Exception:
                if mutation_token is not None:
                    await self._active_session_manager.clear_active_session_mutation(
                        thread_id,
                        mutation_token,
                    )
                raise

    async def finalize_active_sessions(
        self,
        *,
        llm_client: BaseLLMClient | None = None,
        end_session: EndSessionCallback,
        effective_llm_client: EffectiveLLMResolver,
        list_active_thread_ids: ListActiveThreadIdsCallback | None = None,
        is_auto_finalization_excluded: Callable[[str], bool] | None = None,
    ) -> None:
        """Finalize any unresolved active sessions."""
        list_active_thread_ids = list_active_thread_ids or self.list_active_thread_ids
        is_auto_finalization_excluded = (
            is_auto_finalization_excluded or self.auto_finalization_excluded
        )

        try:
            active_thread_ids = await list_active_thread_ids()
        except Exception:
            logger.warning(
                "finalize_active_sessions: failed to list active sessions",
                exc_info=True,
            )
            return

        for active_thread_id in active_thread_ids:
            try:
                if is_auto_finalization_excluded(active_thread_id):
                    continue
                await end_session(
                    active_thread_id,
                    llm_client=effective_llm_client(active_thread_id, llm_client),
                )
            except Exception:
                logger.warning(
                    "finalize_active_sessions: failed to end session for thread %s",
                    active_thread_id,
                    exc_info=True,
                )
