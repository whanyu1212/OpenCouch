"""Tests for the shared terminal-console runtime adapter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agent.models import DoneEvent, ResponseReadyEvent, StatusEvent


@pytest.mark.asyncio
async def test_console_runtime_streams_deterministic_guest_turn() -> None:
    """Deterministic guest mode should run without configured credentials."""

    from opencouch_tui.runtime import ConsoleConfig, ConsoleRuntime

    async with ConsoleRuntime(
        ConsoleConfig(
            requested_mode="deterministic",
            thread_id="tui-deterministic",
            user_id="alice",
            memory_mode="guest",
        )
    ) as runtime:
        events = [event async for event in runtime.run_turn_stream("hello")]
        session = runtime.session

    assert session is not None
    assert session.requested_mode == "deterministic"
    assert session.resolved_mode == "deterministic"
    assert session.memory_mode == "guest"
    assert session.user_id is None
    assert session.owner_id == "tui-deterministic"
    assert [event.stage for event in events if isinstance(event, StatusEvent)] == [
        "deterministic",
        "finalize",
    ]
    assert any(isinstance(event, ResponseReadyEvent) for event in events)
    done = next(event for event in events if isinstance(event, DoneEvent))
    assert "Deterministic smoke mode" in done.output.response_text
    assert len(session.history) == 2
    assert session.last_context is not None


@pytest.mark.asyncio
async def test_console_runtime_reports_recoverable_turn_errors(monkeypatch) -> None:
    """Runtime exceptions should become recoverable events for terminal UIs."""

    from opencouch_tui.runtime import (
        ConsoleConfig,
        ConsoleErrorEvent,
        ConsoleRuntime,
    )

    class _FailingPersistentRuntime:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get_history(self, thread_id):
            return []

        async def get_state(self, thread_id):
            return None

        async def run_turn_stream(self, **kwargs):
            yield StatusEvent(stage="triage")
            raise RuntimeError("crisis gate failed")

    monkeypatch.setattr(
        "opencouch_tui.runtime.get_settings",
        lambda: SimpleNamespace(
            persistence_backend="sqlite",
            memory_database_url=None,
            allow_legacy_sqlite=True,
            text_session_backend="disabled",
            text_session_database_url=None,
        ),
    )
    monkeypatch.setattr(
        "opencouch_tui.runtime.PersistentAgentRuntime",
        lambda *args, **kwargs: _FailingPersistentRuntime(),
    )

    async with ConsoleRuntime(
        ConsoleConfig(
            requested_mode="deterministic",
            thread_id="tui-error",
            memory_mode="guest",
        )
    ) as runtime:
        events = [event async for event in runtime.run_turn_stream("hello")]

    assert any(isinstance(event, StatusEvent) for event in events)
    error = next(event for event in events if isinstance(event, ConsoleErrorEvent))
    assert error.prefix == "Turn failed"
    assert error.exception_type == "RuntimeError"
    assert "crisis gate failed" in error.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("memory_mode", "expected_thread_path", "expected_user_id"),
    [
        ("guest", ":memory:", None),
        ("persistent", "/tmp/thread.sqlite3", "alice"),
    ],
)
async def test_console_runtime_uses_grouped_storage_paths(
    memory_mode: str,
    expected_thread_path: str | None,
    expected_user_id: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TUI wiring should use grouped storage paths, not legacy path kwargs."""

    from agent.runtime import RuntimeStoragePaths
    from opencouch_tui.runtime import ConsoleConfig, ConsoleRuntime

    captured: dict[str, Any] = {}

    class _RecordingPersistentRuntime:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["args"] = args
            captured["kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get_history(self, thread_id):
            return []

        async def get_state(self, thread_id):
            return None

    monkeypatch.setattr(
        "opencouch_tui.runtime.get_settings",
        lambda: SimpleNamespace(
            persistence_backend="sqlite",
            memory_database_url=None,
            allow_legacy_sqlite=True,
            text_session_backend="disabled",
            text_session_database_url=None,
        ),
    )
    monkeypatch.setattr(
        "opencouch_tui.runtime.PersistentAgentRuntime",
        _RecordingPersistentRuntime,
    )

    async with ConsoleRuntime(
        ConsoleConfig(
            requested_mode="deterministic",
            thread_id="tui-storage-paths",
            user_id="alice",
            memory_mode=memory_mode,
            sqlite_path="/tmp/thread.sqlite3",
            memory_sqlite_path="/tmp/memory.sqlite3",
            crisis_log_sqlite_path="/tmp/crisis.sqlite3",
        )
    ) as runtime:
        session = runtime.session

    assert captured["args"] == ()
    kwargs = captured["kwargs"]
    assert "storage_paths" in kwargs
    assert "sqlite_path" not in kwargs
    assert "memory_sqlite_path" not in kwargs
    assert "crisis_log_sqlite_path" not in kwargs
    storage_paths = kwargs["storage_paths"]
    assert isinstance(storage_paths, RuntimeStoragePaths)
    assert storage_paths.sqlite_path == expected_thread_path
    assert storage_paths.memory_sqlite_path == "/tmp/memory.sqlite3"
    assert storage_paths.crisis_log_sqlite_path == "/tmp/crisis.sqlite3"
    assert session is not None
    assert session.user_id == expected_user_id


@pytest.mark.asyncio
async def test_console_runtime_memory_snapshot_includes_notebook_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The TUI runtime snapshot should include raw records plus the notebook view."""

    from agent.memory.store import OpenCouchMemoryStore
    from opencouch_tui.runtime import ConsoleConfig, ConsoleRuntime

    class _NotebookPersistentRuntime:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.memory_store = OpenCouchMemoryStore()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get_history(self, thread_id):
            return []

        async def get_state(self, thread_id):
            return None

    monkeypatch.setattr(
        "opencouch_tui.runtime.get_settings",
        lambda: SimpleNamespace(
            persistence_backend="sqlite",
            memory_database_url=None,
            allow_legacy_sqlite=True,
            text_session_backend="disabled",
            text_session_database_url=None,
        ),
    )
    monkeypatch.setattr(
        "opencouch_tui.runtime.PersistentAgentRuntime",
        _NotebookPersistentRuntime,
    )

    async with ConsoleRuntime(
        ConsoleConfig(
            requested_mode="deterministic",
            thread_id="tui-notebook",
            user_id="alice",
            memory_mode="persistent",
        )
    ) as runtime:
        snapshot = await runtime.load_memory_snapshot()

    assert snapshot["owner_id"] == "alice"
    assert snapshot["semantic"] == []
    assert snapshot["episodic"] == []
    assert snapshot["procedural"] is None
    assert snapshot["notebook"]["owner_id"] == "alice"
    assert snapshot["notebook"]["counts"]["total_entries"] == 0
    assert snapshot["notebook"]["topics"] == []


def test_console_config_defaults_are_tui_safe() -> None:
    """The adapter defaults should be safe for credential-free TUI smoke runs."""

    from agent.runtime import DEFAULT_CRISIS_LOG_DB_PATH, DEFAULT_MEMORY_DB_PATH
    from opencouch_tui.runtime import ConsoleConfig

    config = ConsoleConfig(thread_id="tui-defaults")

    assert config.requested_mode == "auto"
    assert config.thread_id == "tui-defaults"
    assert config.user_id is None
    assert config.memory_mode == "guest"
    assert config.response_model_tier == "fast"
    assert config.memory_sqlite_path == str(DEFAULT_MEMORY_DB_PATH)
    assert config.crisis_log_sqlite_path == str(DEFAULT_CRISIS_LOG_DB_PATH)
