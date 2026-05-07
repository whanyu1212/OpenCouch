"""Runtime-local active-session tracking helpers."""

from __future__ import annotations

from agent.runtime.active_session import PersistedActiveSessionState
from agent.memory.policy.candidates import SessionMemoryBuffer


class RuntimeSessionTracker:
    """Track in-process active-session state for runtime orchestration."""

    def __init__(self) -> None:
        """Initialize empty runtime session tracking dictionaries."""
        self.session_starts: dict[str, str] = {}
        self.max_crisis_levels: dict[str, int] = {}
        self.session_memory_buffers: dict[str, SessionMemoryBuffer] = {}
        self.session_transcript_starts: dict[str, int] = {}

    def clear(self, thread_id: str) -> None:
        """Drop runtime-local tracking for one thread.

        Args:
            thread_id (str): Thread identifier to clear.

        Returns:
            None: Mutates the tracker in place.
        """

        self.session_starts.pop(thread_id, None)
        self.max_crisis_levels.pop(thread_id, None)
        self.session_memory_buffers.pop(thread_id, None)
        self.session_transcript_starts.pop(thread_id, None)

    def thread_ids(self) -> list[str]:
        """Return thread ids with runtime-local session starts.

        Returns:
            list[str]: Thread ids currently tracked in process.
        """

        return list(self.session_starts)

    def has_tracking(self, thread_id: str) -> bool:
        """Return whether a thread has runtime-local session tracking.

        Args:
            thread_id (str): Thread identifier to check.

        Returns:
            bool: ``True`` when any runtime-local session tracker exists.
        """

        return (
            thread_id in self.session_starts
            or thread_id in self.session_transcript_starts
            or thread_id in self.session_memory_buffers
        )

    def hydrate(self, session: PersistedActiveSessionState) -> None:
        """Restore runtime-local tracking from persisted active-session state.

        Args:
            session (PersistedActiveSessionState): Persisted active-session record.

        Returns:
            None: Mutates the tracker in place.
        """

        self.session_starts[session.thread_id] = session.started_at
        self.max_crisis_levels[session.thread_id] = session.max_crisis_level
        self.session_transcript_starts[session.thread_id] = (
            session.transcript_start_index
        )
        self.session_memory_buffers[session.thread_id] = (
            session.session_buffer.model_copy(deep=True)
        )

    def start_session(
        self,
        thread_id: str,
        *,
        started_at: str,
        transcript_start_index: int,
    ) -> None:
        """Start runtime-local tracking for a new active session.

        Args:
            thread_id (str): Thread identifier.
            started_at (str): Session start timestamp.
            transcript_start_index (int): Transcript index where the session starts.

        Returns:
            None: Mutates the tracker in place.
        """

        self.session_starts[thread_id] = started_at
        self.max_crisis_levels[thread_id] = 0
        self.session_transcript_starts[thread_id] = transcript_start_index
        self.session_memory_buffers[thread_id] = SessionMemoryBuffer(
            session_id=thread_id
        )

    def session_memory_buffer_for_thread(
        self,
        thread_id: str,
    ) -> SessionMemoryBuffer:
        """Return the runtime-managed session buffer for a thread.

        Args:
            thread_id (str): Thread identifier.

        Returns:
            SessionMemoryBuffer: Per-thread session memory buffer.
        """

        if thread_id not in self.session_memory_buffers:
            self.session_memory_buffers[thread_id] = SessionMemoryBuffer(
                session_id=thread_id
            )
        return self.session_memory_buffers[thread_id]

    def record_crisis_level(self, thread_id: str, crisis_level: int) -> None:
        """Record the maximum crisis level observed for a session.

        Args:
            thread_id (str): Thread identifier.
            crisis_level (int): Crisis level observed in the latest turn.

        Returns:
            None: Mutates the tracker in place.
        """

        prior_max = self.max_crisis_levels.get(thread_id, 0)
        self.max_crisis_levels[thread_id] = max(prior_max, crisis_level)

    def transcript_start_index(self, thread_id: str) -> int:
        """Return the active-session transcript start index.

        Args:
            thread_id (str): Thread identifier.

        Returns:
            int: Transcript start index, defaulting to ``0``.
        """

        return self.session_transcript_starts.get(thread_id, 0)

    def started_at(self, thread_id: str, *, default: str) -> str:
        """Return the session start timestamp for a thread.

        Args:
            thread_id (str): Thread identifier.
            default (str): Fallback timestamp when the thread is untracked.

        Returns:
            str: Session start timestamp.
        """

        return self.session_starts.get(thread_id, default)

    def max_crisis_level(self, thread_id: str) -> int:
        """Return the maximum crisis level tracked for a thread.

        Args:
            thread_id (str): Thread identifier.

        Returns:
            int: Maximum crisis level, defaulting to ``0``.
        """

        return self.max_crisis_levels.get(thread_id, 0)

    def session_memory_buffer_or_none(
        self,
        thread_id: str,
    ) -> SessionMemoryBuffer | None:
        """Return an existing session buffer without creating one.

        Args:
            thread_id (str): Thread identifier.

        Returns:
            SessionMemoryBuffer | None: Existing buffer, if tracked.
        """

        return self.session_memory_buffers.get(thread_id)

    def to_persisted_session(
        self,
        thread_id: str,
        *,
        last_active_at: str,
    ) -> PersistedActiveSessionState | None:
        """Build persisted active-session state from runtime-local tracking.

        Args:
            thread_id (str): Thread identifier to persist.
            last_active_at (str): Last-active timestamp to write.

        Returns:
            PersistedActiveSessionState | None: Persistable session state when
                required tracking exists.
        """

        started_at = self.session_starts.get(thread_id)
        transcript_start_index = self.session_transcript_starts.get(thread_id)
        if started_at is None or transcript_start_index is None:
            return None

        return PersistedActiveSessionState(
            thread_id=thread_id,
            started_at=started_at,
            last_active_at=last_active_at,
            transcript_start_index=transcript_start_index,
            max_crisis_level=self.max_crisis_levels.get(thread_id, 0),
            session_buffer=self.session_memory_buffer_for_thread(thread_id).model_copy(
                deep=True
            ),
        )
