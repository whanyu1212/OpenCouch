"""Focused tests for grouped persistent runtime configuration."""

from __future__ import annotations

from dataclasses import fields
from datetime import timedelta
from pathlib import Path

import pytest

from agent.memory.modes import MemoryMode
from agent.runtime.configuration import (
    DEFAULT_TEXT_SESSION_DB_PATH,
    SESSION_TIMEOUT,
    RuntimeBehaviorConfig,
    RuntimeDependencies,
    RuntimePersistenceConfig,
    RuntimeStoragePaths,
    validate_runtime_configuration,
)


def test_public_configuration_exports_match_final_contract() -> None:
    import agent.runtime as runtime_package
    from agent.runtime import runtime as runtime_module

    assert runtime_package.RuntimeStoragePaths is RuntimeStoragePaths
    assert runtime_package.RuntimePersistenceConfig is RuntimePersistenceConfig
    assert runtime_package.RuntimeDependencies is RuntimeDependencies
    assert runtime_package.RuntimeBehaviorConfig is RuntimeBehaviorConfig
    assert runtime_package.DEFAULT_TEXT_SESSION_DB_PATH == DEFAULT_TEXT_SESSION_DB_PATH
    assert runtime_module.DEFAULT_TEXT_SESSION_DB_PATH == DEFAULT_TEXT_SESSION_DB_PATH
    for removed_name in (
        "DEFAULT_THREAD_DB_PATH",
        "DEFAULT_MEMORY_DB_PATH",
        "DEFAULT_CRISIS_LOG_DB_PATH",
        "DEFAULT_FEEDBACK_DB_PATH",
    ):
        assert not hasattr(runtime_package, removed_name)
        assert not hasattr(runtime_module, removed_name)


def test_grouped_configuration_has_concrete_defaults() -> None:
    storage = RuntimeStoragePaths()
    persistence = RuntimePersistenceConfig()
    dependencies = RuntimeDependencies()
    behavior = RuntimeBehaviorConfig()

    assert storage.text_session_sqlite_path is None
    assert [field.name for field in fields(storage)] == ["text_session_sqlite_path"]
    assert persistence.memory_mode is MemoryMode.LOCAL
    assert persistence.memory_backend == "postgres"
    assert persistence.thread_persistence_backend == "postgres"
    assert persistence.crisis_log_persistence_backend == "postgres"
    assert persistence.session_feedback_persistence_backend == "postgres"
    assert persistence.text_session_backend == "auto"
    assert dependencies.memory_store is None
    assert dependencies.default_llm_client is None
    assert behavior.session_timeout == SESSION_TIMEOUT
    assert behavior.session_sweep_interval_seconds == 30.0
    assert behavior.finalize_active_sessions_on_close is True
    assert behavior.speculative_memory_prefetch is True


def test_storage_paths_reject_removed_application_sqlite_fields() -> None:
    for removed_name in (
        "sqlite_path",
        "memory_sqlite_path",
        "crisis_log_sqlite_path",
        "feedback_sqlite_path",
    ):
        with pytest.raises(TypeError, match=removed_name):
            RuntimeStoragePaths(**{removed_name: ":memory:"})  # type: ignore[arg-type]


def test_shared_postgres_configuration_fans_out_database_url() -> None:
    database_url = "postgresql://unused/opencouch"
    config = RuntimePersistenceConfig.for_shared_backend(
        memory_mode=MemoryMode.LOCAL,
        persistence_backend="postgres",
        database_url=database_url,
    )

    assert config.memory_database_url == database_url
    assert config.thread_database_url == database_url
    assert config.crisis_log_database_url == database_url
    assert config.session_feedback_database_url == database_url
    assert config.text_session_database_url == database_url


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("memory_backend", "SQLite memory persistence has been removed"),
        (
            "thread_persistence_backend",
            "SQLite runtime-state and active-session persistence has been removed",
        ),
        ("crisis_log_persistence_backend", "SQLite crisis-audit persistence"),
        (
            "session_feedback_persistence_backend",
            "SQLite session-feedback persistence",
        ),
    ],
)
def test_removed_application_sqlite_selectors_are_rejected(
    field: str,
    message: str,
) -> None:
    config = RuntimePersistenceConfig()
    setattr(config, field, "sqlite")

    with pytest.raises(ValueError, match=message):
        validate_runtime_configuration(
            persistence=config,
            storage_paths=RuntimeStoragePaths(),
        )


def test_disk_sdk_sqlite_requires_explicit_legacy_opt_in(tmp_path: Path) -> None:
    config = RuntimePersistenceConfig(
        thread_persistence_backend="memory",
        text_session_backend="sqlite",
    )

    with pytest.raises(ValueError, match="allow_legacy_sqlite=True"):
        validate_runtime_configuration(
            persistence=config,
            storage_paths=RuntimeStoragePaths(
                text_session_sqlite_path=tmp_path / "text-sessions.sqlite3"
            ),
        )


@pytest.mark.parametrize("backend", ["auto", "sqlalchemy"])
@pytest.mark.parametrize(
    "database_url",
    [
        "sqlite+aiosqlite:///text-sessions.sqlite3",
        "sqlite+aiosqlite:///text-sessions.sqlite3?mode=memory",
        " sqlite+aiosqlite:///text-sessions.sqlite3 ",
        "sqlite+aiosqlite:///text-sessions.sqlite3?mode=memory&URI=true",
        "sqlite+aiosqlite:///text-sessions.sqlite3?Mode=memory&uri=true",
    ],
)
def test_sqlalchemy_disk_sqlite_requires_explicit_legacy_opt_in(
    backend: str,
    database_url: str,
) -> None:
    config = RuntimePersistenceConfig(
        thread_persistence_backend="memory",
        text_session_backend=backend,  # type: ignore[arg-type]
        text_session_database_url=database_url,
    )

    with pytest.raises(ValueError, match="allow_legacy_sqlite=True"):
        validate_runtime_configuration(
            persistence=config,
            storage_paths=RuntimeStoragePaths(),
        )


@pytest.mark.parametrize(
    "database_url",
    [
        "sqlite+aiosqlite://",
        "sqlite+aiosqlite:///:memory:",
        "sqlite+aiosqlite:///file:memdb1?mode=memory&uri=true",
        "sqlite+aiosqlite:///file:memdb1?mode=memory&uri=on",
    ],
)
def test_sqlalchemy_in_memory_sqlite_does_not_require_opt_in(
    database_url: str,
) -> None:
    validate_runtime_configuration(
        persistence=RuntimePersistenceConfig(
            thread_persistence_backend="memory",
            text_session_backend="sqlalchemy",
            text_session_database_url=database_url,
        ),
        storage_paths=RuntimeStoragePaths(),
    )


@pytest.mark.parametrize(
    "persistence",
    [
        RuntimePersistenceConfig(
            memory_mode=MemoryMode.INCOGNITO,
            text_session_backend="sqlite",
        ),
        RuntimePersistenceConfig(
            thread_persistence_backend="memory",
            text_session_backend="sqlite",
        ),
        RuntimePersistenceConfig(
            thread_persistence_backend="memory",
            text_session_backend="sqlite",
            allow_legacy_sqlite=True,
        ),
    ],
)
def test_incognito_in_memory_and_opted_in_sdk_sqlite_are_allowed(
    persistence: RuntimePersistenceConfig,
) -> None:
    path = (
        Path("text-sessions.sqlite3")
        if persistence.allow_legacy_sqlite
        else Path(":memory:")
    )
    validate_runtime_configuration(
        persistence=persistence,
        storage_paths=RuntimeStoragePaths(text_session_sqlite_path=path),
    )


def test_grouped_dependency_and_behavior_values_are_direct() -> None:
    timeout = timedelta(minutes=5)

    def excluded(thread_id: str) -> bool:
        return thread_id == "external"

    dependencies = RuntimeDependencies(auto_finalize_excluded=excluded)
    behavior = RuntimeBehaviorConfig(
        text_session_history_limit=20,
        session_timeout=timeout,
        speculative_memory_prefetch=False,
    )

    assert dependencies.auto_finalize_excluded is excluded
    assert behavior.text_session_history_limit == 20
    assert behavior.session_timeout == timeout
    assert behavior.speculative_memory_prefetch is False
