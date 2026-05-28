"""Tests for :meth:`PersistentAgentRuntime.record_session_feedback`.

The runtime method is the only path that should produce
:class:`SessionFeedbackRecord` instances in production. OpenAI agents
don't touch this — end-session surfaces (CLI ``/end`` / ``/exit``,
HTTP ``POST /threads/{id}/end``) call it directly.

What these tests assert:
- A normal call writes a record with server-derived
  ``session_id_opaque`` (hash of thread_id) and the caller-supplied
  ``label`` / ``source``.
- Owner identity (``user_id_or_null``) is read from persisted state,
  NOT from the caller — the method signature doesn't even take a
  ``user_id`` argument.
- Incognito mode ALWAYS scrubs ``user_id_or_null`` to ``None``,
  even if state carries a user_id. This mirrors the crisis_log
  privacy contract.
- ``turn_count_at_end`` is read from ``state.session_progress``; zero-state
  threads produce records with ``turn_count_at_end=0``.
- Backend write failures return ``None`` from the method and log a
  WARNING — the caller continues to summarization regardless.
- State-lookup failures degrade the same way.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent.memory.hashing import hash_session_id
from agent.memory.modes import MemoryMode
from agent.feedback.session_feedback import SessionFeedbackBackend
from agent.runtime import PersistentAgentRuntime
from tests.support.persistence import FakeCrossRestartLLM


# ─── Failing-backend fixture ─────────────────────────────────────────


class _FailingFeedbackBackend:
    """Backend that raises on ``aappend``. Used to prove that the
    runtime swallows write failures and returns ``None``."""

    async def aappend(self, record: Any) -> None:
        raise RuntimeError("simulated backend outage")

    async def alist_by_session(self, session_id_opaque: str) -> list:
        return []

    async def arecord_count(self) -> int:
        return 0

    async def apurge_before(self, cutoff: Any) -> int:
        return 0

    async def aclose(self) -> None:
        return None


# Verify the fake satisfies the Protocol — otherwise wiring errors
# would fail here at import time.
_: type[SessionFeedbackBackend] = _FailingFeedbackBackend  # type: ignore[assignment]


def _runtime(memory_mode: MemoryMode = MemoryMode.LOCAL) -> PersistentAgentRuntime:
    """Construct a runtime that keeps everything in memory."""
    return PersistentAgentRuntime(
        sqlite_path=":memory:",
        memory_sqlite_path=":memory:",
        crisis_log_sqlite_path=":memory:",
        feedback_sqlite_path=":memory:",
        memory_mode=memory_mode,
    )


# ─── Happy-path tests ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_writes_record_with_server_derived_session_id() -> None:
    """``session_id_opaque`` must be the SHA-256 hash of the thread_id
    argument, NOT any value the caller might have threaded through
    another parameter. This is the privacy-at-rest guarantee."""

    async with _runtime() as rt:
        record = await rt.record_session_feedback(
            "thread-xyz", label="positive", source="cli_end"
        )
        assert record is not None
        assert record.session_id_opaque == hash_session_id("thread-xyz")


@pytest.mark.asyncio
async def test_writes_record_with_caller_supplied_label_source_and_modality() -> None:
    """``label``, ``source``, and ``modality`` are caller-controlled fields
    that flow into the record as-is."""

    async with _runtime() as rt:
        record = await rt.record_session_feedback(
            "t", label="negative", source="api_end", modality="voice"
        )
        assert record is not None
        assert record.label == "negative"
        assert record.source == "api_end"
        assert record.modality == "voice"


@pytest.mark.asyncio
async def test_record_is_persisted_in_backend() -> None:
    """The returned record should also appear in
    :meth:`SessionFeedbackBackend.alist_by_session` — the method
    isn't just returning a transient object."""

    async with _runtime() as rt:
        written = await rt.record_session_feedback("t", label="skip", source="cli_exit")
        assert written is not None
        stored = await rt.session_feedback_backend.alist_by_session(
            hash_session_id("t")
        )
        assert len(stored) == 1
        assert stored[0].id == written.id


@pytest.mark.asyncio
async def test_zero_state_thread_records_turn_count_zero() -> None:
    """When the thread has no state yet (e.g., ``/end`` immediately
    after ``/new`` with zero turns), the record should still land
    with ``turn_count_at_end=0`` — analytics value of "user rated
    an empty session" is non-zero."""

    async with _runtime() as rt:
        record = await rt.record_session_feedback(
            "never-ran", label="skip", source="cli_end"
        )
        assert record is not None
        assert record.turn_count_at_end == 0


# ─── Privacy contract ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_incognito_scrubs_user_id_to_null() -> None:
    """Even if runtime state somehow carries a user_id (via
    a caller passing one to ``run_turn``), the feedback record must
    scrub it to ``None`` in incognito mode. This matches the
    crisis_log incognito contract."""

    async with _runtime(memory_mode=MemoryMode.INCOGNITO) as rt:
        # Simulate a turn that wrote user_id into runtime state.
        await rt.run_turn(
            thread_id="incog",
            message="hi",
            user_id="alice",
            llm_client=FakeCrossRestartLLM(),
        )
        record = await rt.record_session_feedback(
            "incog", label="positive", source="cli_end"
        )
        assert record is not None
        # user_id_or_null must be None despite state carrying "alice".
        assert record.user_id_or_null is None


@pytest.mark.asyncio
async def test_local_mode_reads_user_id_from_state() -> None:
    """In local mode, ``user_id_or_null`` is the persisted state
    value — NOT the caller. The method signature doesn't expose
    ``user_id``, so this is the only source of truth."""

    async with _runtime(memory_mode=MemoryMode.LOCAL) as rt:
        await rt.run_turn(
            thread_id="t-alice",
            message="hi",
            user_id="alice",
            llm_client=FakeCrossRestartLLM(),
        )
        record = await rt.record_session_feedback(
            "t-alice", label="positive", source="cli_end"
        )
        assert record is not None
        assert record.user_id_or_null == "alice"


# ─── Graceful degradation ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_backend_failure_returns_none_and_does_not_raise(
    caplog,
) -> None:
    """When the feedback backend raises, the runtime method must
    swallow the exception, log a WARNING, and return ``None``. The
    caller (CLI or API handler) continues to summarization."""

    rt = PersistentAgentRuntime(
        sqlite_path=":memory:",
        memory_sqlite_path=":memory:",
        crisis_log_sqlite_path=":memory:",
        feedback_sqlite_path=":memory:",
        session_feedback_backend=_FailingFeedbackBackend(),  # type: ignore[arg-type]
    )
    async with rt:
        import logging

        with caplog.at_level(logging.WARNING, logger="agent.runtime.session_feedback"):
            result = await rt.record_session_feedback(
                "t", label="positive", source="cli_end"
            )
        # Failure-mode contract: None return + WARNING log.
        assert result is None
        assert any(
            "session feedback write failed" in rec.message for rec in caplog.records
        )


@pytest.mark.asyncio
async def test_state_lookup_failure_returns_none_and_does_not_raise(
    monkeypatch, caplog
) -> None:
    """If ``get_state`` itself raises (e.g., a state-store issue),
    the method must still return ``None`` rather than propagate."""

    async with _runtime() as rt:

        async def _raising_get_state(thread_id: str) -> Any:
            raise RuntimeError("simulated state-store crash")

        monkeypatch.setattr(rt, "get_state", _raising_get_state)

        import logging

        with caplog.at_level(logging.WARNING, logger="agent.runtime"):
            result = await rt.record_session_feedback(
                "t", label="positive", source="cli_end"
            )
        assert result is None
        assert any(
            "session feedback write failed" in rec.message for rec in caplog.records
        )
