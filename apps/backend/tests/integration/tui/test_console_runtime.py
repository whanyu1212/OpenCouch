"""Tests for the shared terminal-console runtime adapter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agent.models import DoneEvent, ResponseReadyEvent, StatusEvent


@pytest.mark.asyncio
async def test_console_runtime_requires_postgres_dsn_for_persistent_mode() -> None:
    from config import Settings
    from opencouch_tui.runtime import ConsoleConfig, ConsoleRuntime

    runtime = ConsoleRuntime(
        ConsoleConfig(
            requested_mode="deterministic",
            thread_id="tui-legacy-sqlite",
            memory_mode="persistent",
        ),
        settings=Settings(
            allow_legacy_sqlite=True,
            text_session_backend="disabled",
        ),
    )

    with pytest.raises(ValueError, match="OPENCOUCH_MEMORY_DATABASE_URL"):
        await runtime.__aenter__()


@pytest.mark.asyncio
async def test_console_runtime_streams_deterministic_guest_turn() -> None:
    """Deterministic guest mode should run without configured credentials."""

    from agent.memory.store import OpenCouchMemoryStore
    from config import Settings
    from opencouch_tui.runtime import ConsoleConfig, ConsoleRuntime

    async with ConsoleRuntime(
        ConsoleConfig(
            requested_mode="deterministic",
            thread_id="tui-deterministic",
            user_id="alice",
            memory_mode="guest",
        ),
        settings=Settings(memory_database_url=None),
    ) as runtime:
        events = [event async for event in runtime.run_turn_stream("hello")]
        session = runtime.session
        memory_store = runtime._require_runtime().memory_store

    assert session is not None
    assert session.requested_mode == "deterministic"
    assert session.resolved_mode == "deterministic"
    assert session.memory_mode == "guest"
    assert isinstance(memory_store, OpenCouchMemoryStore)
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

    received: dict[str, Any] = {}

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
            received.update(kwargs)
            yield StatusEvent(stage="triage")
            raise RuntimeError("crisis gate failed")

    monkeypatch.setattr(
        "opencouch_tui.runtime.get_settings",
        lambda: SimpleNamespace(
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
    assert "CLI slash commands available in this TUI" in received["prompt_appendix"]
    assert "/summary [short|full]" in received["prompt_appendix"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("memory_mode", "expected_session_path", "expected_user_id"),
    [
        ("guest", ":memory:", None),
        ("persistent", "/tmp/text-sessions.sqlite3", "alice"),
    ],
)
async def test_console_runtime_uses_explicit_sdk_session_sqlite_path(
    memory_mode: str,
    expected_session_path: str | None,
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
            memory_database_url="postgresql://unused/opencouch",
            allow_legacy_sqlite=True,
            text_session_backend="sqlite",
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
            text_session_sqlite_path="/tmp/text-sessions.sqlite3",
        )
    ) as runtime:
        session = runtime.session

    assert captured["args"] == ()
    kwargs = captured["kwargs"]
    assert "storage_paths" in kwargs
    assert "sqlite_path" not in kwargs
    assert "memory_sqlite_path" not in kwargs
    storage_paths = kwargs["storage_paths"]
    assert isinstance(storage_paths, RuntimeStoragePaths)
    assert storage_paths.text_session_sqlite_path == expected_session_path
    persistence_config = kwargs["persistence_config"]
    assert persistence_config.memory_backend == "postgres"
    assert persistence_config.thread_persistence_backend == "postgres"
    assert persistence_config.text_session_backend == "sqlite"
    assert persistence_config.allow_legacy_sqlite is True
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
    assert snapshot["has_unprojected_legacy_memory"] is False


@pytest.mark.asyncio
async def test_console_runtime_memory_snapshot_preserves_raw_memory_when_notebook_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy procedural data should not block raw memory snapshot consumers."""

    from agent.memory.operations.procedural_profile import PROCEDURAL_KEY
    from agent.memory.store import OpenCouchMemoryStore
    from opencouch_tui.runtime import ConsoleConfig, ConsoleRuntime

    class _LegacyProceduralPersistentRuntime:
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
            memory_database_url=None,
            allow_legacy_sqlite=True,
            text_session_backend="disabled",
            text_session_database_url=None,
        ),
    )
    monkeypatch.setattr(
        "opencouch_tui.runtime.PersistentAgentRuntime",
        _LegacyProceduralPersistentRuntime,
    )

    legacy_profile = {
        "proactive_recall_enabled": True,
        "rules": ["Use short plans."],
    }
    async with ConsoleRuntime(
        ConsoleConfig(
            requested_mode="deterministic",
            thread_id="tui-legacy-notebook",
            user_id="alice",
            memory_mode="persistent",
        )
    ) as runtime:
        persistent_runtime = runtime._require_runtime()
        await persistent_runtime.memory_store.aput(
            ("alice", "procedural"),
            PROCEDURAL_KEY,
            legacy_profile,
        )

        snapshot = await runtime.load_memory_snapshot()

    assert snapshot["owner_id"] == "alice"
    assert snapshot["semantic"] == []
    assert snapshot["episodic"] == []
    assert snapshot["procedural"] == legacy_profile
    assert snapshot["notebook"] is None
    assert snapshot["has_unprojected_legacy_memory"] is False


@pytest.mark.asyncio
async def test_console_runtime_memory_snapshot_can_skip_notebook_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw memory command snapshots should not pay notebook projection cost."""

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

    async def _fail_if_called(*args: object, **kwargs: object) -> object:
        raise AssertionError("notebook projection should not be built")

    monkeypatch.setattr(
        "opencouch_tui.runtime.get_settings",
        lambda: SimpleNamespace(
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
    monkeypatch.setattr("opencouch_tui.runtime.build_memory_notebook", _fail_if_called)

    async with ConsoleRuntime(
        ConsoleConfig(
            requested_mode="deterministic",
            thread_id="tui-raw-memory",
            user_id="alice",
            memory_mode="persistent",
        )
    ) as runtime:
        snapshot = await runtime.load_memory_snapshot(include_notebook=False)

    assert snapshot["owner_id"] == "alice"
    assert snapshot["semantic"] == []
    assert snapshot["episodic"] == []
    assert snapshot["procedural"] is None
    assert snapshot["notebook"] is None
    assert snapshot["has_unprojected_legacy_memory"] is False


def test_console_runtime_detects_only_visible_unprojected_records() -> None:
    """Only visible invalid rows should request the legacy raw fallback."""

    from agent.memory.types import SemanticFact
    from opencouch_tui.runtime import _has_visible_unprojected_records

    visible_legacy = SimpleNamespace(
        value={
            "predicate": "USES",
            "object": {"identifier": "breathing exercises"},
        }
    )
    hidden_legacy = SimpleNamespace(
        value={
            "predicate": "USES",
            "object": {"identifier": "private strategy"},
            "user_visible": False,
        }
    )

    assert _has_visible_unprojected_records([visible_legacy], SemanticFact) is True
    assert _has_visible_unprojected_records([hidden_legacy], SemanticFact) is False


@pytest.mark.asyncio
async def test_console_runtime_notebook_snapshot_fetches_complete_raw_namespaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A legacy fallback should include records beyond the store default page."""

    from agent.memory.store import OpenCouchMemoryStore
    from opencouch_tui.runtime import ConsoleConfig, ConsoleRuntime

    class _LegacyRowsPersistentRuntime:
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
            memory_database_url=None,
            allow_legacy_sqlite=True,
            text_session_backend="disabled",
            text_session_database_url=None,
        ),
    )
    monkeypatch.setattr(
        "opencouch_tui.runtime.PersistentAgentRuntime",
        _LegacyRowsPersistentRuntime,
    )

    async with ConsoleRuntime(
        ConsoleConfig(
            requested_mode="deterministic",
            thread_id="tui-complete-fallback",
            user_id="alice",
            memory_mode="persistent",
        )
    ) as runtime:
        persistent_runtime = runtime._require_runtime()
        for index in range(12):
            await persistent_runtime.memory_store.aput(
                ("alice", "semantic"),
                f"legacy-{index}",
                {
                    "predicate": "WANTS",
                    "object": {"identifier": f"goal-{index}"},
                },
            )

        snapshot = await runtime.load_memory_snapshot()

    assert len(snapshot["semantic"]) == 12
    assert snapshot["semantic"][-1]["object"]["identifier"] == "goal-11"
    assert snapshot["has_unprojected_legacy_memory"] is True


def test_console_config_defaults_are_tui_safe() -> None:
    """The adapter defaults should be safe for credential-free TUI smoke runs."""

    from opencouch_tui.runtime import ConsoleConfig

    config = ConsoleConfig(thread_id="tui-defaults")

    assert config.requested_mode == "auto"
    assert config.thread_id == "tui-defaults"
    assert config.user_id is None
    assert config.memory_mode == "guest"
    assert config.response_model_tier == "fast"
    assert not hasattr(config, "memory_sqlite_path")
