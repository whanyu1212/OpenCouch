"""Always-on crisis safety log backend.

The crisis log is the one memory channel that writes **regardless of
memory mode**. Even in incognito mode, every crisis event detected by
the gate is recorded — with ``user_id`` nulled out and the ``session_id``
replaced by its SHA-256 hash — so operators retain a safety audit trail
without storing user-identifying information.

The 90-day default retention policy and legal-review caveat are part of
the current memory design.

Phase 1 v0.8 scope:
- :class:`CrisisLogBackend` protocol that any backend must implement
  (append, list by date, arecord_count, apurge_before, aclose).
- :class:`InMemoryCrisisLogBackend` — original v0.1 implementation.
  Dict-backed, ephemeral, fast. Used for tests and incognito-mode
  CLI sessions where crisis events shouldn't persist to disk.
- :class:`agent.memory.sqlite_crisis_log.SqliteCrisisLogBackend`
  (v0.8) — aiosqlite-backed, durable across process restarts. Used
  by persistent-mode CLI sessions so the crisis log survives CLI
  exits.

v0.8.1 adds :meth:`CrisisLogBackend.apurge_before` to enforce the
90-day retention policy. The purge is exposed to the CLI via
``/memory purge-crisis [days]`` with a
typed ``purge`` confirmation to prevent accidental audit-trail
deletion — matching the v0.9 ``/memory clear`` UX pattern for
destructive operations. The append-only contract still holds
from the **agent's** perspective: graph nodes never delete crisis
records, only the operator-triggered purge command does.

Design decisions:
- The backend is **append-only from the agent's perspective**. Graph
  nodes never delete crisis records. The v0.8.1
  :meth:`apurge_before` method is a retention operation invoked by
  the operator (via the CLI) or by a future scheduled cleanup job,
  not by the agent runtime during normal turn processing.
- Records are stored using the :class:`CrisisLogRecord` pydantic model
  from :mod:`agent.memory.models`. The backend accepts model instances
  and returns them — no serialization gymnastics at the interface.
- The protocol lives here rather than in a separate ``protocols.py``
  file because there are only two backends and they're tightly
  coupled to the same record type. Split later if that changes.
- ``arecord_count`` is part of the protocol as of v0.8 (was a
  non-protocol debug helper before). Async for the same reason the
  memory store's ``arecord_count`` is async: the SQLite backend
  shares a single aiosqlite connection across its async methods,
  and a sync helper would either open a second connection (breaking
  ``:memory:`` databases) or require separate caching machinery.
- ``apurge_before`` is part of the protocol as of v0.8.1. All three
  concrete backends implement it. ``NullCrisisLogBackend`` returns
  0 without touching any state; the in-memory backend drops expired
  date buckets; the SQLite backend runs a single DELETE with a
  ``detected_date < ?`` predicate that uses the existing date index.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Protocol

from agent.memory.models import CrisisLogRecord


class CrisisLogBackend(Protocol):
    """Protocol that any crisis-log backend must implement.

    The protocol has four methods covering three use cases: the
    ``crisis_log_node`` writes records (``aappend``), debugging/audit
    code reads them back by date (``alist_by_date``), and the CLI
    ``/memory status`` command reports the total count
    (``arecord_count``). The runtime lifecycle owns closing
    (``aclose``). Retention purging is intentionally NOT in the
    protocol — it's a background-job concern handled separately.
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
    """In-memory crisis log backend for phase 1 v0.1.

    Stores records in a per-instance dict keyed by ISO date string.
    Nothing persists across runtime restarts. Adequate for v0.1 testing
    and for incognito-mode runtimes that explicitly don't persist
    anything.

    NOT thread-safe. Each runtime instance should own its own backend.
    """

    def __init__(self) -> None:
        self._records_by_date: dict[date, list[CrisisLogRecord]] = defaultdict(list)
        self._closed = False

    def _ensure_open(self) -> None:
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
        # detected_at is an ISO-8601 string per the pydantic model.
        # Extract the date portion without importing datetime parsing
        # machinery — the string is always in "YYYY-MM-DDTHH:MM:SSZ" form.
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

    async def apurge_before(self, cutoff: date) -> int:  # noqa: ARG002 — contract
        """Purge crisis records from the null backend.

        Args:
            cutoff (date): Exclusive cutoff date.

        Returns:
            int: Always ``0``.
        """

        return 0
