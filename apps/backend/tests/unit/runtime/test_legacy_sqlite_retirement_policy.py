"""Policy contracts for the remaining OpenAI SDK SQLite surface."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.feedback.session_feedback import InMemorySessionFeedbackBackend
from agent.memory.modes import MemoryMode
from agent.runtime import (
    PersistentAgentRuntime,
    RuntimeDependencies,
    RuntimePersistenceConfig,
    RuntimeStoragePaths,
)
from agent.memory.store import OpenCouchMemoryStore


def test_removed_application_sqlite_is_rejected_even_with_sdk_opt_in() -> None:
    config = RuntimePersistenceConfig(
        thread_persistence_backend="memory",
        allow_legacy_sqlite=True,
    )
    config.thread_persistence_backend = "sqlite"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="runtime-state and active-session"):
        PersistentAgentRuntime(
            persistence_config=config,
            dependencies=RuntimeDependencies(memory_store=OpenCouchMemoryStore()),
        )


def test_disk_sdk_text_session_sqlite_remains_guarded(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SQLite fields: text_session_backend"):
        PersistentAgentRuntime(
            storage_paths=RuntimeStoragePaths(
                text_session_sqlite_path=tmp_path / "text-sessions.sqlite3"
            ),
            persistence_config=RuntimePersistenceConfig(
                thread_persistence_backend="memory",
                text_session_backend="sqlite",
            ),
            dependencies=RuntimeDependencies(memory_store=OpenCouchMemoryStore()),
        )


def test_in_memory_sdk_text_session_sqlite_is_allowed() -> None:
    runtime = PersistentAgentRuntime(
        storage_paths=RuntimeStoragePaths(text_session_sqlite_path=":memory:"),
        persistence_config=RuntimePersistenceConfig(
            thread_persistence_backend="memory",
            text_session_backend="sqlite",
        ),
        dependencies=RuntimeDependencies(
            memory_store=OpenCouchMemoryStore(),
            crisis_log_backend=InMemoryCrisisLogBackend(),
            session_feedback_backend=InMemorySessionFeedbackBackend(),
        ),
    )

    assert runtime._text_session_store is not None  # noqa: SLF001
    assert runtime._text_session_store._config.sqlite_path == ":memory:"  # noqa: SLF001


def test_incognito_forces_sdk_sqlite_to_memory(tmp_path: Path) -> None:
    runtime = PersistentAgentRuntime(
        storage_paths=RuntimeStoragePaths(
            text_session_sqlite_path=tmp_path / "must-not-exist.sqlite3"
        ),
        persistence_config=RuntimePersistenceConfig(
            memory_mode=MemoryMode.INCOGNITO,
            text_session_backend="sqlite",
        ),
    )

    assert runtime._text_session_store is not None  # noqa: SLF001
    assert runtime._text_session_store._config.sqlite_path == ":memory:"  # noqa: SLF001
    assert not (tmp_path / "must-not-exist.sqlite3").exists()
