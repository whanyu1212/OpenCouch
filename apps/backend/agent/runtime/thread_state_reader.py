"""Read-side thread-state queries extracted from ``PersistentAgentRuntime``.

``ThreadStateReader`` owns the pure read queries over persisted thread state:
state snapshots, materialized history, active-session liveness, and thread
listing. It mutates nothing and acquires no per-thread lock, so it can be
exercised in isolation. ``PersistentAgentRuntime`` retains identical-signature
shims that delegate here, keeping the public API and voice facade unchanged.

Writers (``reset_thread`` and the session-lifecycle paths) stay on the runtime;
they depend on ``session_status_unlocked`` here for liveness checks.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import timedelta
from typing import Any

from agent.memory.modes import MemoryMode
from agent.models import Message, MessageRole
from agent.runtime.session import RuntimeSessionTracker
from agent.runtime.session.active_session import PersistedActiveSessionState
from agent.runtime.session.history import messages_from_transcript
from agent.runtime.session.state import session_has_expired
from agent.runtime.session.manager import ActiveSessionManager
from agent.runtime.session_store import TextSessionStore
from agent.runtime.state_store import RuntimeStateStore
from agent.runtime.types import SessionStatus, ThreadSummary
from agent.state import AgentState

logger = logging.getLogger(__name__)


def merge_history_response_styles(
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


class ThreadStateReader:
    """Pure read-side queries over persisted thread state.

    Holds no per-thread lock and mutates no state; every method is a read.
    """

    def __init__(
        self,
        *,
        state_store: RuntimeStateStore,
        text_session_store: TextSessionStore | None,
        active_session_manager: ActiveSessionManager,
        session_tracker: RuntimeSessionTracker,
        memory_mode: MemoryMode,
        session_timeout: timedelta,
    ) -> None:
        """Initialize the reader with the stores and trackers it reads.

        Args:
            state_store: Persisted runtime-state store.
            text_session_store: SDK-session transcript store, if configured.
            active_session_manager: Active-session liveness manager.
            session_tracker: In-process per-thread session tracker.
            memory_mode: Persistence tier for the runtime.
            session_timeout: Inactivity window before a session expires.
        """

        self._state_store = state_store
        self._text_session_store = text_session_store
        self._active_session_manager = active_session_manager
        self._session_tracker = session_tracker
        self._memory_mode = memory_mode
        self._session_timeout = session_timeout

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
            history = await self._text_session_store.get_history(thread_id, cache=False)
            if history:
                return merge_history_response_styles(history, state)
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

        return await self.session_status_unlocked(thread_id)

    async def session_status_unlocked(self, thread_id: str) -> SessionStatus:
        """Return session status without acquiring the per-thread lock.

        Args:
            thread_id: Thread identifier.

        Returns:
            The current session status.
        """

        if self._memory_mode == MemoryMode.INCOGNITO:
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

        if session_has_expired(
            session.last_active_at,
            session_timeout=self._session_timeout,
        ):
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


__all__ = [
    "ThreadStateReader",
    "merge_history_response_styles",
]
