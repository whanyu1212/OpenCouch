"""Session-feedback backends.

Session feedback stores explicit end-of-session ratings recorded by
``PersistentAgentRuntime.record_session_feedback``. It is separate from
prompt memory and uses a session-keyed read path because feedback is
created at session close rather than during normal response generation.

Concrete backends share the same async protocol:

- ``InMemorySessionFeedbackBackend`` is ephemeral and used by tests and
  incognito runtimes.
- ``PostgresSessionFeedbackBackend`` is the supported durable implementation.
- ``NullSessionFeedbackBackend`` is reserved for explicit test fixtures.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from typing import TYPE_CHECKING, Protocol

from agent.feedback.models import SessionFeedbackRecord


class SessionFeedbackBackend(Protocol):
    """Protocol that any session-feedback backend must implement.

    The protocol has five methods: the runtime writes records
    (``aappend``), debugging / analytics code reads them back by
    session (``alist_by_session``), the CLI ``/memory status`` command
    reports the total count (``arecord_count``), retention purging
    drops old records (``apurge_before``), and the runtime lifecycle
    owns closing (``aclose``).

    ``alist_by_session`` is the session-keyed analogue of crisis_log's
    ``alist_by_date`` — feedback is written once per session, so
    per-session retrieval is the natural read pattern.
    """

    async def aappend(self, record: SessionFeedbackRecord) -> None:
        """Append one feedback record.

        Args:
            record (SessionFeedbackRecord): Feedback record to append.

        Returns:
            None: Persists the record in the backend.
        """
        ...

    async def alist_by_session(
        self, session_id_opaque: str
    ) -> list[SessionFeedbackRecord]:
        """List feedback records for one opaque session id.

        Args:
            session_id_opaque (str): Opaque session identifier to query.

        Returns:
            list[SessionFeedbackRecord]: Records for the session in insertion order.
        """
        ...

    async def arecord_count(self) -> int:
        """Count feedback records across the backend.

        Returns:
            int: Total feedback record count.
        """
        ...

    async def apurge_before(self, cutoff: date) -> int:
        """Purge feedback records older than a cutoff date.

        Args:
            cutoff (date): Exclusive cutoff date.

        Returns:
            int: Number of records deleted.
        """
        ...

    async def ensure_schema(self) -> None:
        """Prepare durable storage before the backend serves traffic.

        Durable backends connect and apply schema DDL here so request-time
        operations perform data work rather than migrations. Ephemeral
        backends implement this as a no-op.

        Returns:
            None: Prepares the backend.
        """
        ...

    async def aclose(self) -> None:
        """Release backend resources.

        Returns:
            None: Closes the backend.
        """
        ...


class InMemorySessionFeedbackBackend:
    """In-memory session-feedback backend for incognito mode and tests.

    Stores records in a per-instance dict keyed by ``session_id_opaque``.
    Nothing persists across runtime restarts. Adequate for incognito-
    mode runtimes where feedback shouldn't touch disk, and for unit
    tests that want to inspect what was written without opening a
    SQLite file.

    NOT thread-safe. Each runtime instance should own its own backend.
    """

    def __init__(self) -> None:
        """Initialize the in-memory feedback backend.

        Returns:
            None: Creates an empty in-memory feedback store.
        """

        self._records_by_session: dict[str, list[SessionFeedbackRecord]] = defaultdict(
            list
        )
        self._closed = False

    def _ensure_open(self) -> None:
        """Raise when the backend has already been closed.

        Raises:
            RuntimeError: If the backend is closed.

        Returns:
            None: The backend is open.
        """

        if self._closed:
            raise RuntimeError("InMemorySessionFeedbackBackend is closed.")

    async def aappend(self, record: SessionFeedbackRecord) -> None:
        """Append one in-memory feedback record.

        Args:
            record (SessionFeedbackRecord): Feedback record to append.

        Returns:
            None: Stores the record in memory.
        """

        self._ensure_open()
        self._records_by_session[record.session_id_opaque].append(record)

    async def alist_by_session(
        self, session_id_opaque: str
    ) -> list[SessionFeedbackRecord]:
        """List in-memory feedback records for one session.

        Args:
            session_id_opaque (str): Opaque session identifier to query.

        Returns:
            list[SessionFeedbackRecord]: Records for the session in insertion order.
        """

        self._ensure_open()
        return list(self._records_by_session.get(session_id_opaque, []))

    async def ensure_schema(self) -> None:
        """Prepare the in-memory feedback backend.

        Records live in per-instance dicts, so there is nothing to create and
        no connection to open.

        Returns:
            None: No preparation is required.
        """

        self._ensure_open()

    async def aclose(self) -> None:
        """Close the in-memory feedback backend.

        Returns:
            None: Marks the backend closed and clears in-memory data.
        """

        if self._closed:
            return
        self._closed = True
        self._records_by_session.clear()

    async def arecord_count(self) -> int:
        """Count in-memory feedback records.

        Returns:
            int: Total feedback record count.
        """

        if self._closed:
            return 0
        return sum(len(records) for records in self._records_by_session.values())

    async def apurge_before(self, cutoff: date) -> int:
        """Purge in-memory feedback records older than a cutoff date.

        Args:
            cutoff (date): Exclusive cutoff date.

        Returns:
            int: Number of records deleted.
        """

        if self._closed:
            return 0

        deleted = 0
        # Iterate over a list of keys because we may mutate the dict.
        for session_id in list(self._records_by_session.keys()):
            kept: list[SessionFeedbackRecord] = []
            for record in self._records_by_session[session_id]:
                recorded_date = datetime.fromisoformat(
                    record.recorded_at.replace("Z", "+00:00")
                ).date()
                if recorded_date < cutoff:
                    deleted += 1
                else:
                    kept.append(record)
            if kept:
                self._records_by_session[session_id] = kept
            else:
                del self._records_by_session[session_id]
        return deleted


class NullSessionFeedbackBackend:
    """No-op session-feedback backend.

    Reserved for test fixtures that want to assert "no feedback was
    written" or that want to explicitly disable feedback persistence
    in a test context.

    **NOT a valid production backend.** Production callers who want
    to disable feedback should use incognito mode, which selects the
    in-memory backend — feedback records are still created (and still
    reachable via the runtime accessor) but they don't persist across
    process restart. Using this null backend in production would
    silently drop feedback rather than respecting the user's rating.
    """

    async def aappend(self, record: SessionFeedbackRecord) -> None:  # noqa: ARG002
        """Discard a feedback record.

        Args:
            record (SessionFeedbackRecord): Feedback record to ignore.

        Returns:
            None: No-op for the null backend.
        """

        return None

    async def ensure_schema(self) -> None:
        """Prepare the null feedback backend.

        Returns:
            None: No-op for the null backend.
        """

        return None

    async def alist_by_session(
        self, session_id_opaque: str
    ) -> list[SessionFeedbackRecord]:  # noqa: ARG002
        """List feedback records for one session.

        Args:
            session_id_opaque (str): Opaque session identifier to query.

        Returns:
            list[SessionFeedbackRecord]: Always an empty list.
        """

        return []

    async def aclose(self) -> None:
        """Close the null feedback backend.

        Returns:
            None: No-op for the null backend.
        """

        return None

    async def arecord_count(self) -> int:
        """Count feedback records in the null backend.

        Returns:
            int: Always ``0``.
        """

        return 0

    async def apurge_before(self, cutoff: date) -> int:  # noqa: ARG002
        """Purge feedback records from the null backend.

        Args:
            cutoff (date): Exclusive cutoff date.

        Returns:
            int: Always ``0``.
        """

        return 0


if TYPE_CHECKING:
    _in_memory_backend: SessionFeedbackBackend = InMemorySessionFeedbackBackend()
    _null_backend: SessionFeedbackBackend = NullSessionFeedbackBackend()
