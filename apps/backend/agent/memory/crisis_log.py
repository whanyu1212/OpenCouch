"""Always-on crisis safety log backend.

The crisis log is the one memory channel that writes **regardless of
memory mode**. Even in incognito mode, every crisis event detected by
the gate is recorded — with ``user_id`` nulled out and the ``session_id``
replaced by its SHA-256 hash — so operators retain a safety audit trail
without storing user-identifying information.

See schema.yaml §2 namespaces.crisis_log and §9 q6 for the full rationale
and the legal-review caveat on the 90-day retention default.

Phase 1 v0.8 scope:
- :class:`CrisisLogBackend` protocol that any backend must implement
  (append, list by date, arecord_count, aclose).
- :class:`InMemoryCrisisLogBackend` — original v0.1 implementation.
  Dict-backed, ephemeral, fast. Used for tests and incognito-mode
  CLI sessions where crisis events shouldn't persist to disk.
- :class:`agent.memory.sqlite_crisis_log.SqliteCrisisLogBackend`
  (v0.8) — aiosqlite-backed, durable across process restarts. Used
  by persistent-mode CLI sessions so the crisis log survives CLI
  exits. The 90-day retention policy will be enforced by a
  background-job cleanup path in a later release (schema includes
  the columns needed; enforcement is deferred).

Design decisions:
- The backend is **append-only from the agent's perspective**. There is
  no ``delete`` or ``update`` method. Retention purging is a
  background-job concern, not a public method on the backend.
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
        """Append a crisis event record to the log.

        Implementations MUST be append-only — existing records are
        never modified. Duplicate ``record.id`` values on the same day
        should be treated as a bug by the caller; the backend should
        either overwrite (simplest, current InMemoryCrisisLogBackend
        behavior) or raise, but not silently merge.
        """
        ...

    async def alist_by_date(self, day: date) -> list[CrisisLogRecord]:
        """Return all crisis log records from a given day.

        Returns an empty list if no records exist for that day. Results
        are returned in insertion order.
        """
        ...

    async def arecord_count(self) -> int:
        """Return the total number of crisis records across all dates.

        Used by ``/memory status`` CLI command and by tests. Async as
        of v0.8 for the same reason the memory store's helper is
        async: the SQLite backend shares a single connection across
        its async methods, and a sync helper would either open a
        second connection (breaking ``:memory:`` databases) or require
        separate caching.
        """
        ...

    async def aclose(self) -> None:
        """Release any resources held by the backend.

        Safe to call on an already-closed backend (should be a no-op).
        Required by every implementation because the ``PersistentAgentRuntime``
        lifecycle calls it on ``__aexit__``.
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
        """Append a record under today's date bucket.

        The date bucket is derived from ``record.detected_at`` so
        backfilled records land in the correct day's bucket even if
        they're written later.
        """

        self._ensure_open()
        # detected_at is an ISO-8601 string per the pydantic model.
        # Extract the date portion without importing datetime parsing
        # machinery — the string is always in "YYYY-MM-DDTHH:MM:SSZ" form.
        day = date.fromisoformat(record.detected_at.split("T", 1)[0])
        self._records_by_date[day].append(record)

    async def alist_by_date(self, day: date) -> list[CrisisLogRecord]:
        """Return all records for a given day in insertion order."""

        self._ensure_open()
        return list(self._records_by_date.get(day, []))

    async def aclose(self) -> None:
        """Mark the backend as closed and clear its contents.

        Closed backends raise ``RuntimeError`` on any further access.
        Calling ``aclose`` on an already-closed backend is a no-op.
        """

        if self._closed:
            return
        self._closed = True
        self._records_by_date.clear()

    async def arecord_count(self) -> int:
        """Return the total number of records across all dates.

        Used by ``/memory status`` CLI command and by tests. Async
        as of v0.8 to match the protocol — see the module docstring
        for the rationale (matching the memory store's refactor).
        """

        if self._closed:
            return 0
        return sum(len(records) for records in self._records_by_date.values())


class NullCrisisLogBackend:
    """No-op crisis log backend.

    Reserved for test fixtures that want to assert "no crisis events
    were logged" or want to explicitly disable logging. NOT a valid
    production backend — the always-on crisis log promise requires
    a real backend in all three memory modes.
    """

    async def aappend(self, record: CrisisLogRecord) -> None:
        """Discard the record without storing it."""

        return None

    async def alist_by_date(self, day: date) -> list[CrisisLogRecord]:
        """Always return an empty list."""

        return []

    async def aclose(self) -> None:
        """No resources to release."""

        return None

    async def arecord_count(self) -> int:
        """Always zero."""

        return 0
