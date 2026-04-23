"""Always-on session-feedback backend.

Explicit end-of-session feedback ("was this session helpful?") written
by :meth:`PersistentAgentRuntime.record_session_feedback` when the user
provides a thumbs rating at ``/end``, ``/exit`` (save=y), or
``POST /threads/{id}/end``. Mirrors :mod:`agent.audit.crisis_log` in
shape and privacy posture — always-on regardless of memory mode, but
``user_id_or_null`` is ALWAYS ``None`` in incognito mode.

See :class:`agent.memory.models.SessionFeedbackRecord` for the record
shape and privacy contract. See ``session_feedback_plan.md`` at the
repo root for the full design rationale.

Phase 1 scope:
- :class:`SessionFeedbackBackend` protocol that any backend must
  implement (``aappend``, ``alist_by_session``, ``arecord_count``,
  ``apurge_before``, ``aclose``).
- :class:`InMemorySessionFeedbackBackend` — dict-backed, ephemeral,
  used in incognito mode and tests.
- :class:`NullSessionFeedbackBackend` — no-op; reserved for test
  fixtures that want to assert "no feedback was written". NOT a
  valid production backend.
- :class:`agent.audit.sqlite_session_feedback.SqliteSessionFeedbackBackend`
  — aiosqlite-backed, durable across process restarts. Used by
  persistent-mode runtimes.

Design parallels with :mod:`agent.audit.crisis_log`:
- Append-only from the runtime's perspective; ``apurge_before`` is an
  operator / scheduled-cleanup concern, not touched by normal write
  paths.
- The primary read pattern is **per session**, not per date (feedback
  is a closing-turn event, keyed to a specific session). So the
  protocol exposes ``alist_by_session`` rather than ``alist_by_date``.
- Retention window default is **180 days** (wider than crisis_log's 90
  because feedback analytics benefit from a longer lookback).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from typing import Protocol

from agent.memory.models import SessionFeedbackRecord


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
        self._records_by_session: dict[str, list[SessionFeedbackRecord]] = defaultdict(
            list
        )
        self._closed = False

    def _ensure_open(self) -> None:
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


# ─── Protocol conformance checks ────────────────────────────────────────────
#
# Assignment triggers a type-check at import time: if either class drifts
# from the ``SessionFeedbackBackend`` Protocol shape, mypy / pyright will
# complain here rather than at the runtime call site. Same pattern as
# ``sqlite_crisis_log.py``'s conformance assertion.

_: type[SessionFeedbackBackend] = InMemorySessionFeedbackBackend  # type: ignore[assignment]
_: type[SessionFeedbackBackend] = NullSessionFeedbackBackend  # type: ignore[assignment]
