"""Runtime liveness contracts for channel thread rotation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import pytest

from agent.runtime import (
    ActiveSessionExists,
    PersistentAgentRuntime,
    SessionInterrupted,
    SessionLeaseExpired,
    SessionStatus,
)
from tests.support.persistence import (
    FakeCrossRestartLLM,
    postgres_database_url,
    runtime_paths,
)


class _FailingTextRuntime:
    async def run_turn(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("runtime failed")

    async def run_turn_stream(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        raise RuntimeError("runtime failed")
        yield


@pytest.mark.asyncio
async def test_soft_limit_marks_session_rotation_required(tmp_path: Path) -> None:
    async with PersistentAgentRuntime(
        **runtime_paths(tmp_path),
        finalize_active_sessions_on_close=False,
    ) as runtime:
        await runtime.run_turn(
            thread_id="thread-rotation",
            message="hello",
            llm_client=FakeCrossRestartLLM(),
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
async def test_soft_limit_marks_session_rotation_required_in_postgres(
    tmp_path: Path,
) -> None:
    """Postgres-backed thread state should preserve rotation-required liveness."""
    memory_database_url = postgres_database_url()
    if not memory_database_url:
        pytest.skip(
            "Postgres integration tests are disabled; set "
            "OPENCOUCH_ENABLE_POSTGRES_INTEGRATION_TESTS=1 and "
            "OPENCOUCH_TEST_POSTGRES_URL"
        )

    paths = runtime_paths(tmp_path)

    async with PersistentAgentRuntime(
        **paths,
        memory_backend="postgres",
        memory_database_url=memory_database_url,
        thread_persistence_backend="postgres",
        thread_database_url=memory_database_url,
        finalize_active_sessions_on_close=False,
    ) as runtime:
        await runtime.run_turn(
            thread_id="thread-rotation-postgres",
            message="hello",
            llm_client=FakeCrossRestartLLM(),
            session_transcript_soft_limit=1,
        )

        assert await runtime.session_status("thread-rotation-postgres") == (
            SessionStatus.ROTATION_REQUIRED
        )

    async with PersistentAgentRuntime(
        **paths,
        memory_backend="postgres",
        memory_database_url=memory_database_url,
        thread_persistence_backend="postgres",
        thread_database_url=memory_database_url,
        finalize_active_sessions_on_close=False,
    ) as runtime:
        assert await runtime.session_status("thread-rotation-postgres") == (
            SessionStatus.ROTATION_REQUIRED
        )
        with pytest.raises(SessionLeaseExpired):
            await runtime.run_turn(
                thread_id="thread-rotation-postgres",
                message="reuse should fail",
                expected_liveness="active",
            )

        await runtime.end_session("thread-rotation-postgres")
        assert await runtime.session_status("thread-rotation-postgres") == (
            SessionStatus.ABSENT
        )


@pytest.mark.asyncio
async def test_reset_thread_refuses_active_sessions(tmp_path: Path) -> None:
    async with PersistentAgentRuntime(
        **runtime_paths(tmp_path),
        finalize_active_sessions_on_close=False,
    ) as runtime:
        await runtime.run_turn(
            thread_id="thread-reset",
            message="hello",
            llm_client=FakeCrossRestartLLM(),
        )

        with pytest.raises(ActiveSessionExists):
            await runtime.reset_thread("thread-reset")

        await runtime.end_session("thread-reset")
        await runtime.reset_thread("thread-reset")
        assert await runtime.session_status("thread-reset") == SessionStatus.ABSENT


@pytest.mark.asyncio
async def test_foreign_mutation_marker_reports_interrupted(tmp_path: Path) -> None:
    async with PersistentAgentRuntime(
        **runtime_paths(tmp_path),
        finalize_active_sessions_on_close=False,
    ) as runtime:
        await runtime.run_turn(
            thread_id="thread-interrupted",
            message="hello",
            llm_client=FakeCrossRestartLLM(),
        )
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
) -> None:
    async with PersistentAgentRuntime(
        **runtime_paths(tmp_path),
        finalize_active_sessions_on_close=False,
    ) as runtime:
        runtime._sdk_bridge._openai_text_runtime = cast(  # noqa: SLF001
            Any, _FailingTextRuntime()
        )

        with pytest.raises(RuntimeError, match="runtime failed"):
            await runtime.run_turn(thread_id="thread-failed-turn", message="hello")

        assert await runtime.session_status("thread-failed-turn") == (
            SessionStatus.INTERRUPTED
        )
        await runtime.end_session("thread-failed-turn")
        assert await runtime.session_status("thread-failed-turn") == (
            SessionStatus.ABSENT
        )
