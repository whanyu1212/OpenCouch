"""Tests for exception-safe persistent runtime shutdown."""

from __future__ import annotations

import asyncio
from builtins import BaseExceptionGroup
from types import SimpleNamespace

import pytest

from agent.runtime import runtime as runtime_module
from agent.runtime.runtime import PersistentAgentRuntime


def _shutdown_runtime(
    events: list[str],
    *,
    failures: dict[str, BaseException] | None = None,
    finalize_active_sessions: bool = True,
    cancellation_stage: str | None = None,
    cancellation_started: asyncio.Event | None = None,
    cancellation_release: asyncio.Event | None = None,
) -> PersistentAgentRuntime:
    failures = failures or {}

    async def run_stage(name: str) -> None:
        events.append(f"{name}:start")
        try:
            if name == cancellation_stage:
                assert cancellation_started is not None
                assert cancellation_release is not None
                cancellation_started.set()
                await cancellation_release.wait()
            failure = failures.get(name)
            if failure is not None:
                raise failure
        finally:
            events.append(f"{name}:done")

    runtime = object.__new__(PersistentAgentRuntime)
    runtime._session_lifecycle = SimpleNamespace(  # noqa: SLF001
        stop_background_tasks=lambda: run_stage("background_tasks")
    )
    runtime._finalize_active_sessions_on_close = finalize_active_sessions  # noqa: SLF001
    runtime._default_llm_client = None  # noqa: SLF001
    runtime.finalize_active_sessions = (  # type: ignore[method-assign]
        lambda *, llm_client: run_stage("active_sessions")
    )
    runtime.voice = SimpleNamespace(aclose=lambda: run_stage("voice"))
    runtime._resources = SimpleNamespace(  # noqa: SLF001
        aclose=lambda: run_stage("resources")
    )
    return runtime


@pytest.mark.asyncio
async def test_shutdown_attempts_later_stages_after_failure() -> None:
    events: list[str] = []
    runtime = _shutdown_runtime(
        events,
        failures={"background_tasks": RuntimeError("background stop failed")},
    )

    with pytest.raises(RuntimeError, match="background stop failed"):
        await runtime.__aexit__(None, None, None)

    assert events == [
        "background_tasks:start",
        "background_tasks:done",
        "active_sessions:start",
        "active_sessions:done",
        "voice:start",
        "voice:done",
        "resources:start",
        "resources:done",
    ]


@pytest.mark.asyncio
async def test_shutdown_groups_multiple_stage_failures() -> None:
    events: list[str] = []
    runtime = _shutdown_runtime(
        events,
        failures={
            "background_tasks": RuntimeError("background stop failed"),
            "resources": ValueError("resource close failed"),
        },
    )

    with pytest.raises(BaseExceptionGroup) as exc_info:
        await runtime.__aexit__(None, None, None)

    assert [type(error) for error in exc_info.value.exceptions] == [
        RuntimeError,
        ValueError,
    ]
    assert events[-2:] == ["resources:start", "resources:done"]


@pytest.mark.asyncio
async def test_shutdown_skips_optional_finalization_when_disabled() -> None:
    events: list[str] = []
    runtime = _shutdown_runtime(events, finalize_active_sessions=False)

    await runtime.__aexit__(None, None, None)

    assert events == [
        "background_tasks:start",
        "background_tasks:done",
        "voice:start",
        "voice:done",
        "resources:start",
        "resources:done",
    ]


@pytest.mark.asyncio
async def test_shutdown_times_out_a_cancelled_stuck_stage_and_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stuck stage must not prevent later runtime resources from closing."""

    events: list[str] = []
    started = asyncio.Event()
    release = asyncio.Event()
    runtime = _shutdown_runtime(
        events,
        cancellation_stage="background_tasks",
        cancellation_started=started,
        cancellation_release=release,
    )
    monkeypatch.setattr(runtime_module, "_SHUTDOWN_STAGE_TIMEOUT_SECONDS", 0.01)

    shutdown_task = asyncio.create_task(runtime.__aexit__(None, None, None))
    await started.wait()
    shutdown_task.cancel()

    with pytest.raises(BaseExceptionGroup) as exc_info:
        await shutdown_task

    assert started.is_set()
    assert [type(error) for error in exc_info.value.exceptions] == [
        asyncio.CancelledError,
        TimeoutError,
    ]
    assert events == [
        "background_tasks:start",
        "background_tasks:done",
        "active_sessions:start",
        "active_sessions:done",
        "voice:start",
        "voice:done",
        "resources:start",
        "resources:done",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cancellation_stage",
    ["background_tasks", "active_sessions", "voice", "resources"],
)
async def test_shutdown_finishes_all_stages_before_propagating_cancellation(
    cancellation_stage: str,
) -> None:
    events: list[str] = []
    started = asyncio.Event()
    release = asyncio.Event()
    runtime = _shutdown_runtime(
        events,
        cancellation_stage=cancellation_stage,
        cancellation_started=started,
        cancellation_release=release,
    )

    shutdown_task = asyncio.create_task(runtime.__aexit__(None, None, None))
    await started.wait()
    shutdown_task.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await shutdown_task

    assert events == [
        "background_tasks:start",
        "background_tasks:done",
        "active_sessions:start",
        "active_sessions:done",
        "voice:start",
        "voice:done",
        "resources:start",
        "resources:done",
    ]
