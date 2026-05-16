"""Always-on crisis safety log backends.

The crisis log is separate from prompt memory and writes regardless of
memory mode. The graph appends records only from ``crisis_log_node``;
retention purges are operator- or maintenance-driven and never happen
during normal turn processing.

Concrete backends share the same async protocol:

- ``InMemoryCrisisLogBackend`` is ephemeral and used by tests and
  incognito runtimes.
- ``SqliteCrisisLogBackend`` is the legacy SQLite fallback for local
  development and migration compatibility.
- ``NullCrisisLogBackend`` is reserved for explicit test fixtures.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import TYPE_CHECKING, Protocol

from agent.audit.models import CrisisLogRecord


class CrisisLogBackend(Protocol):
    """Protocol that any crisis-log backend must implement.

    ``crisis_log_node`` writes records, debugging and audit code reads
    them back by date, CLI status reports the total count, retention
    purging deletes expired records, and the runtime lifecycle owns
    closing.
    """

    async def aappend(self, record: CrisisLogRecord) -> None:
        """Append one crisis event record.

        Args:
            record (CrisisLogRecord): Crisis event record to append.

        Returns:
            None: Persists the record in the backend.
        """
        ...

    async def alist_by_date(self, day: date) -> list[CrisisLogRecord]:
        """List crisis records for one date.

        Args:
            day (date): Calendar day to query.

        Returns:
            list[CrisisLogRecord]: Records for the day in insertion order.
        """
        ...

    async def arecord_count(self) -> int:
        """Count crisis records across the backend.

        Returns:
            int: Total crisis-log record count.
        """
        ...

    async def apurge_before(self, cutoff: date) -> int:
        """Purge crisis records older than a cutoff date.

        Args:
            cutoff (date): Exclusive cutoff date.

        Returns:
            int: Number of records deleted.
        """
        ...

    async def aclose(self) -> None:
        """Release backend resources.

        Returns:
            None: Closes the backend.
        """
        ...


class InMemoryCrisisLogBackend:
    """In-memory crisis log backend for tests and incognito runtimes.

    Stores records in a per-instance dict keyed by ``datetime.date``.
    Nothing persists across runtime restarts.

    NOT thread-safe. Each runtime instance should own its own backend.
    """

    def __init__(self) -> None:
        """Initialize the in-memory crisis backend.

        Returns:
            None: Creates an empty in-memory record store.
        """

        self._records_by_date: dict[date, list[CrisisLogRecord]] = defaultdict(list)
        self._closed = False

    def _ensure_open(self) -> None:
        """Raise when the backend has already been closed.

        Raises:
            RuntimeError: If the backend is closed.

        Returns:
            None: The backend is open.
        """

        if self._closed:
            raise RuntimeError("InMemoryCrisisLogBackend is closed.")

    async def aappend(self, record: CrisisLogRecord) -> None:
        """Append one in-memory crisis record.

        Args:
            record (CrisisLogRecord): Crisis event record to append.

        Returns:
            None: Stores the record in memory.
        """

        self._ensure_open()
        day = date.fromisoformat(record.detected_at.split("T", 1)[0])
        self._records_by_date[day].append(record)

    async def alist_by_date(self, day: date) -> list[CrisisLogRecord]:
        """List in-memory crisis records for one date.

        Args:
            day (date): Calendar day to query.

        Returns:
            list[CrisisLogRecord]: Records for the day in insertion order.
        """

        self._ensure_open()
        return list(self._records_by_date.get(day, []))

    async def aclose(self) -> None:
        """Close the in-memory crisis backend.

        Returns:
            None: Marks the backend closed and clears in-memory data.
        """

        if self._closed:
            return
        self._closed = True
        self._records_by_date.clear()

    async def arecord_count(self) -> int:
        """Count in-memory crisis records.

        Returns:
            int: Total crisis-log record count.
        """

        if self._closed:
            return 0
        return sum(len(records) for records in self._records_by_date.values())

    async def apurge_before(self, cutoff: date) -> int:
        """Purge in-memory crisis records older than a cutoff date.

        Args:
            cutoff (date): Exclusive cutoff date.

        Returns:
            int: Number of records deleted.
        """

        if self._closed:
            return 0

        stale_dates = [day for day in self._records_by_date.keys() if day < cutoff]
        deleted = 0
        for day in stale_dates:
            deleted += len(self._records_by_date[day])
            del self._records_by_date[day]
        return deleted


class NullCrisisLogBackend:
    """No-op crisis log backend.

    Reserved for test fixtures that want to assert "no crisis events
    were logged" or want to explicitly disable logging. NOT a valid
    production backend — the always-on crisis log promise requires
    a real backend in all three memory modes.
    """

    async def aappend(self, record: CrisisLogRecord) -> None:
        """Discard a crisis record.

        Args:
            record (CrisisLogRecord): Crisis event record to ignore.

        Returns:
            None: No-op for the null backend.
        """

        return None

    async def alist_by_date(self, day: date) -> list[CrisisLogRecord]:
        """List crisis records for one date.

        Args:
            day (date): Calendar day to query.

        Returns:
            list[CrisisLogRecord]: Always an empty list.
        """

        return []

    async def aclose(self) -> None:
        """Close the null crisis backend.

        Returns:
            None: No-op for the null backend.
        """

        return None

    async def arecord_count(self) -> int:
        """Count crisis records in the null backend.

        Returns:
            int: Always ``0``.
        """

        return 0

    async def apurge_before(self, cutoff: date) -> int:  # noqa: ARG002
        """Purge crisis records from the null backend.

        Args:
            cutoff (date): Exclusive cutoff date.

        Returns:
            int: Always ``0``.
        """

        return 0


if TYPE_CHECKING:
    _in_memory_backend: CrisisLogBackend = InMemoryCrisisLogBackend()
    _null_backend: CrisisLogBackend = NullCrisisLogBackend()
