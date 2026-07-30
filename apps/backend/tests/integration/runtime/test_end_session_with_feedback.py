"""Combined end-session-with-feedback holds one lock across both steps.

Recording feedback and finalizing as two operations let a concurrent turn
change the session between them, so the stored ``turn_count_at_end`` could
describe a window that was never the one summarized. These tests pin the
combined operation's atomicity and its best-effort feedback contract.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agent.memory.modes import MemoryMode
from agent.runtime import PersistentAgentRuntime
from tests.support.persistence import (
    in_memory_audit_feedback_dependencies,
    in_memory_runtime_storage_paths,
    runtime_persistence_config,
)

pytestmark = pytest.mark.asyncio


def _runtime(memory_mode: MemoryMode = MemoryMode.LOCAL) -> PersistentAgentRuntime:
    """Construct a runtime that keeps everything in memory."""

    return PersistentAgentRuntime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(memory_mode),
        dependencies=in_memory_audit_feedback_dependencies(),
    )


async def test_combined_end_records_feedback_and_finalizes() -> None:
    """The combined operation writes feedback and returns the finalize result."""

    async with _runtime() as rt:
        record, arc = await rt.end_session_with_feedback(
            "thread-combined",
            label="positive",
            source="api_end",
        )

        assert record is not None
        assert record.label == "positive"
        assert record.source == "api_end"
        # No LLM client and no turns, so summarization is skipped.
        assert arc is None

        stored = await rt.session_feedback_backend.alist_by_session(
            record.session_id_opaque
        )
        assert [entry.id for entry in stored] == [record.id]


async def test_concurrent_turn_cannot_run_between_feedback_and_finalize() -> None:
    """A waiter for the thread lock cannot observe the mid-operation gap.

    This is the defect itself: recording feedback read state without the
    lock, so a turn could land between that read and finalization and leave
    the feedback describing a window that was never summarized. The competing
    task holds the *same* thread lock the operation must own throughout, so
    it can only run once the whole operation completes.
    """

    async with _runtime() as rt:
        thread_id = "thread-locked"
        observed: list[str] = []
        feedback_read_started = asyncio.Event()

        original_get_state = rt.get_state

        async def _slow_get_state(target_thread_id: str) -> Any:
            # Widen the window between the feedback state read and finalize.
            if target_thread_id == thread_id and not feedback_read_started.is_set():
                feedback_read_started.set()
                observed.append("feedback-state-read")
                await asyncio.sleep(0.05)
            return await original_get_state(target_thread_id)

        async def _competing_turn() -> None:
            await feedback_read_started.wait()
            async with rt._thread_lock(thread_id):  # noqa: SLF001
                observed.append("concurrent-turn")

        rt.get_state = _slow_get_state  # type: ignore[method-assign]

        competitor = asyncio.create_task(_competing_turn())
        await rt.end_session_with_feedback(
            thread_id,
            label="negative",
            source="api_end",
        )
        observed.append("operation-complete")
        await competitor

        # The concurrent turn must land after the whole operation, never
        # between the feedback read and finalization.
        assert observed == [
            "feedback-state-read",
            "operation-complete",
            "concurrent-turn",
        ]


async def test_combined_end_finalizes_even_when_feedback_fails() -> None:
    """Feedback stays best-effort: a write failure must not block finalizing."""

    async with _runtime() as rt:
        finalize_calls: list[str] = []
        original_end_unlocked = rt._end_session_unlocked  # noqa: SLF001

        async def _tracking_end_unlocked(thread_id: str, **kwargs: Any) -> Any:
            finalize_calls.append(thread_id)
            return await original_end_unlocked(thread_id, **kwargs)

        async def _failing_aappend(record: Any) -> None:
            raise RuntimeError("simulated feedback backend outage")

        rt._end_session_unlocked = _tracking_end_unlocked  # type: ignore[method-assign]  # noqa: SLF001
        rt.session_feedback_backend.aappend = _failing_aappend  # type: ignore[method-assign]

        record, _ = await rt.end_session_with_feedback(
            "thread-feedback-fails",
            label="skip",
            source="api_end",
        )

        assert record is None
        assert finalize_calls == ["thread-feedback-fails"]


async def test_combined_end_finalizes_even_when_state_read_fails() -> None:
    """A failed state read degrades feedback only, never finalization."""

    async with _runtime() as rt:
        finalize_calls: list[str] = []
        original_end_unlocked = rt._end_session_unlocked  # noqa: SLF001

        async def _tracking_end_unlocked(thread_id: str, **kwargs: Any) -> Any:
            finalize_calls.append(thread_id)
            return await original_end_unlocked(thread_id, **kwargs)

        async def _failing_get_state(thread_id: str) -> Any:
            raise RuntimeError("simulated state read failure")

        rt._end_session_unlocked = _tracking_end_unlocked  # type: ignore[method-assign]  # noqa: SLF001
        rt.get_state = _failing_get_state  # type: ignore[method-assign]

        record, _ = await rt.end_session_with_feedback(
            "thread-state-fails",
            label="positive",
            source="api_end",
        )

        assert record is None
        assert finalize_calls == ["thread-state-fails"]


async def test_standalone_feedback_still_works_independently() -> None:
    """Post-session feedback remains supported without finalizing again."""

    async with _runtime() as rt:
        record = await rt.record_session_feedback(
            "thread-standalone",
            label="positive",
            source="api_end",
        )

        assert record is not None
        assert record.source == "api_end"


async def test_incognito_combined_end_scrubs_user_id() -> None:
    """Incognito keeps its privacy contract through the combined path."""

    async with _runtime(MemoryMode.INCOGNITO) as rt:
        record, _ = await rt.end_session_with_feedback(
            "thread-incognito",
            label="positive",
            source="api_end",
        )

        assert record is not None
        assert record.user_id_or_null is None
