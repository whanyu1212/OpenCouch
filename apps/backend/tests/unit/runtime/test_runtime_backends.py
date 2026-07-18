"""Tests for centralized runtime backend selection helpers."""

from __future__ import annotations

import pytest

from agent.memory.modes import MemoryMode
from agent.memory.store import OpenCouchMemoryStore
from agent.memory.store.postgres import PostgresMemoryStore
from agent.memory.store.sqlite import SqliteMemoryStore
from agent.runtime.backends import create_memory_store, select_runtime_backends


@pytest.mark.parametrize("memory_mode", [MemoryMode.LOCAL, MemoryMode.SYNCED])
def test_select_runtime_backends_preserves_configured_persistent_backends(
    memory_mode: MemoryMode,
) -> None:
    selection = select_runtime_backends(
        memory_mode=memory_mode,
        memory_backend="postgres",
        thread_persistence_backend="postgres",
        crisis_log_persistence_backend="postgres",
        session_feedback_persistence_backend="postgres",
    )

    assert selection.thread_persistence_backend == "postgres"
    assert selection.memory_store_backend == "postgres"
    assert selection.crisis_log_backend == "postgres"
    assert selection.session_feedback_backend == "postgres"


def test_select_runtime_backends_forces_incognito_to_ephemeral_backends() -> None:
    selection = select_runtime_backends(
        memory_mode=MemoryMode.INCOGNITO,
        memory_backend="postgres",
        thread_persistence_backend="postgres",
        crisis_log_persistence_backend="postgres",
        session_feedback_persistence_backend="postgres",
    )

    assert selection.thread_persistence_backend == "memory"
    assert selection.memory_store_backend == "memory"
    assert selection.crisis_log_backend == "memory"
    assert selection.session_feedback_backend == "memory"


def test_create_memory_store_handles_each_explicit_backend() -> None:
    memory = create_memory_store(
        memory_store=None,
        memory_backend="memory",
        memory_database_url=None,
        memory_sqlite_path=":memory:",
    )
    sqlite = create_memory_store(
        memory_store=None,
        memory_backend="sqlite",
        memory_database_url=None,
        memory_sqlite_path=":memory:",
    )
    postgres = create_memory_store(
        memory_store=None,
        memory_backend="postgres",
        memory_database_url="postgresql://unused/opencouch",
        memory_sqlite_path=":memory:",
    )

    assert isinstance(memory, OpenCouchMemoryStore)
    assert isinstance(sqlite, SqliteMemoryStore)
    assert isinstance(postgres, PostgresMemoryStore)


def test_create_memory_store_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported memory backend: unknown"):
        create_memory_store(
            memory_store=None,
            memory_backend="unknown",  # type: ignore[arg-type]
            memory_database_url=None,
            memory_sqlite_path=":memory:",
        )
