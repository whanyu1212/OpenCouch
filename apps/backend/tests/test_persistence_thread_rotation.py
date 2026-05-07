"""Runtime liveness contracts used by Telegram thread rotation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.persistence import (
    ActiveSessionExists,
    PersistentAgentRuntime,
    SessionInterrupted,
    SessionLeaseExpired,
    SessionStatus,
)


class _FailingGraph:
    async def aget_state(self, config):  # noqa: ANN001
        return SimpleNamespace(values=None)

    async def ainvoke(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("graph failed")

    async def astream(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("graph failed")
        yield


def _runtime_paths(tmp_path: Path) -> dict[str, Path]:
    """Return isolated SQLite paths for one runtime test.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Runtime SQLite path mapping.
    """

    return {
        "sqlite_path": tmp_path / "threads.sqlite3",
        "memory_sqlite_path": tmp_path / "memory.sqlite3",
        "crisis_log_sqlite_path": tmp_path / "crisis.sqlite3",
    }


@pytest.mark.asyncio
async def test_soft_limit_marks_session_rotation_required(tmp_path: Path) -> None:
    async with PersistentAgentRuntime(
        **_runtime_paths(tmp_path),
        finalize_active_sessions_on_close=False,
    ) as runtime:
        await runtime.run_turn(
            thread_id="thread-rotation",
            message="hello",
            session_transcript_soft_limit=1,
        )

        assert await runtime.session_status("thread-rotation") == (
            SessionStatus.ROTATION_REQUIRED
        )
        with pytest.raises(SessionLeaseExpired):
            await runtime.run_turn(
                thread_id="thread-rotation",
                message="reuse should fail",
                expected_liveness="active",
            )

        await runtime.end_session("thread-rotation")
        assert await runtime.session_status("thread-rotation") == SessionStatus.ABSENT


@pytest.mark.asyncio
async def test_reset_thread_refuses_active_sessions(tmp_path: Path) -> None:
    async with PersistentAgentRuntime(
        **_runtime_paths(tmp_path),
        finalize_active_sessions_on_close=False,
    ) as runtime:
        await runtime.run_turn(thread_id="thread-reset", message="hello")

        with pytest.raises(ActiveSessionExists):
            await runtime.reset_thread("thread-reset")

        await runtime.end_session("thread-reset")
        await runtime.reset_thread("thread-reset")
        assert await runtime.session_status("thread-reset") == SessionStatus.ABSENT


@pytest.mark.asyncio
async def test_foreign_mutation_marker_reports_interrupted(tmp_path: Path) -> None:
    async with PersistentAgentRuntime(
        **_runtime_paths(tmp_path),
        finalize_active_sessions_on_close=False,
    ) as runtime:
        await runtime.run_turn(thread_id="thread-interrupted", message="hello")
        await runtime._active_session_manager.set_active_session_mutation(  # noqa: SLF001
            "thread-interrupted",
            mutation_token="foreign-runtime:other-instance:999999",
            mutation_kind="turn",
        )

        assert await runtime.session_status("thread-interrupted") == (
            SessionStatus.INTERRUPTED
        )
        with pytest.raises(SessionInterrupted):
            await runtime.run_turn(
                thread_id="thread-interrupted",
                message="reuse should fail",
                expected_liveness="active",
            )

        await runtime.end_session("thread-interrupted")
        assert await runtime.session_status("thread-interrupted") == (
            SessionStatus.ABSENT
        )


@pytest.mark.asyncio
async def test_failed_run_turn_leaves_interrupted_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "agent.persistence.build_agent_workflow", lambda checkpointer: _FailingGraph()
    )

    async with PersistentAgentRuntime(
        **_runtime_paths(tmp_path),
        finalize_active_sessions_on_close=False,
    ) as runtime:
        with pytest.raises(RuntimeError, match="graph failed"):
            await runtime.run_turn(thread_id="thread-failed-turn", message="hello")

        assert await runtime.session_status("thread-failed-turn") == (
            SessionStatus.INTERRUPTED
        )
        await runtime.end_session("thread-failed-turn")
        assert await runtime.session_status("thread-failed-turn") == (
            SessionStatus.ABSENT
        )
