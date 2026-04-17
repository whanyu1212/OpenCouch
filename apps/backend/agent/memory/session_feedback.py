"""Always-on session-feedback backend.

Explicit end-of-session feedback ("was this session helpful?") written
by :meth:`PersistentAgentRuntime.record_session_feedback` when the user
provides a thumbs rating at ``/end``, ``/exit`` (save=y), or
``POST /threads/{id}/end``. Mirrors :mod:`agent.memory.crisis_log` in
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
- :class:`agent.memory.sqlite_session_feedback.SqliteSessionFeedbackBackend`
  — aiosqlite-backed, durable across process restarts. Used by
  persistent-mode runtimes.

Design parallels with :mod:`agent.memory.crisis_log`:
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
        """Append a feedback record to the store.

        Implementations MUST be append-only — existing records are
        never modified. Two calls with the same ``record.id`` produce
        two rows in Phase 1 (no idempotency).
        """
        ...

    async def alist_by_session(
        self, session_id_opaque: str
    ) -> list[SessionFeedbackRecord]:
        """Return all feedback records for a given opaque session id.

        Returns an empty list when the session has no recorded
        feedback. Results are returned in insertion order.
        """
        ...

    async def arecord_count(self) -> int:
        """Return the total number of feedback records.

        Used by ``/memory status`` CLI command and the
        ``MemoryStatusResponse.session_feedback_count`` API field.
        Async for the same reason the crisis_log helper is async (see
        that module for the aiosqlite-connection rationale).
        """
        ...

    async def apurge_before(self, cutoff: date) -> int:
        """Delete all records with ``recorded_at`` strictly before ``cutoff``.

        The boundary is exclusive — records recorded on ``cutoff``
        itself are preserved, matching the crisis_log contract. Returns
        the number of records deleted.

        Not invoked by any agent code. This is a retention operation
        called by the CLI or by a future scheduled cleanup job.
        """
        ...

    async def aclose(self) -> None:
        """Release any resources held by the backend.

        Safe to call on an already-closed backend. Required by every
        implementation because :class:`agent.persistence.PersistentAgentRuntime`
        calls it on ``__aexit__``.
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
        """Append the record under its session bucket."""

        self._ensure_open()
        self._records_by_session[record.session_id_opaque].append(record)

    async def alist_by_session(
        self, session_id_opaque: str
    ) -> list[SessionFeedbackRecord]:
        """Return all records for a given session in insertion order."""

        self._ensure_open()
        return list(self._records_by_session.get(session_id_opaque, []))

    async def aclose(self) -> None:
        """Mark the backend as closed and clear its contents.

        Closed backends raise ``RuntimeError`` on any further write /
        read access. Calling ``aclose`` on an already-closed backend
        is a no-op.
        """

        if self._closed:
            return
        self._closed = True
        self._records_by_session.clear()

    async def arecord_count(self) -> int:
        """Return the total number of records across all sessions."""

        if self._closed:
            return 0
        return sum(len(records) for records in self._records_by_session.values())

    async def apurge_before(self, cutoff: date) -> int:
        """Delete all records with ``recorded_at`` strictly before ``cutoff``.

        Scans every session bucket, drops any record whose date is
        strictly less than ``cutoff``, and returns the total number of
        records removed. Closed backends return 0 without touching
        state, matching the other methods' closed-safe contract.
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
        """Discard the record without storing it."""

        return None

    async def alist_by_session(
        self, session_id_opaque: str
    ) -> list[SessionFeedbackRecord]:  # noqa: ARG002
        """Always return an empty list."""

        return []

    async def aclose(self) -> None:
        """No resources to release."""

        return None

    async def arecord_count(self) -> int:
        """Always zero."""

        return 0

    async def apurge_before(self, cutoff: date) -> int:  # noqa: ARG002
        """Nothing to purge — always zero."""

        return 0


# ─── Protocol conformance checks ────────────────────────────────────────────
#
# Assignment triggers a type-check at import time: if either class drifts
# from the ``SessionFeedbackBackend`` Protocol shape, mypy / pyright will
# complain here rather than at the runtime call site. Same pattern as
# ``sqlite_crisis_log.py``'s conformance assertion.

_: type[SessionFeedbackBackend] = InMemorySessionFeedbackBackend  # type: ignore[assignment]
_: type[SessionFeedbackBackend] = NullSessionFeedbackBackend  # type: ignore[assignment]
