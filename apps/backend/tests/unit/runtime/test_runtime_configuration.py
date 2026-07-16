"""Focused tests for persistent runtime configuration resolution."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from agent.memory.modes import MemoryMode
from agent.runtime.configuration import (
    DEFAULT_CRISIS_LOG_DB_PATH,
    DEFAULT_FEEDBACK_DB_PATH,
    DEFAULT_MEMORY_DB_PATH,
    DEFAULT_TEXT_SESSION_DB_PATH,
    DEFAULT_THREAD_DB_PATH,
    RuntimeBehaviorConfig,
    RuntimeDependencies,
    RuntimePersistenceConfig,
    RuntimeStoragePaths,
    _UNSET,
    _resolve_runtime_behavior_config,
    _resolve_runtime_dependencies,
    _resolve_runtime_persistence_config,
    _resolve_runtime_storage_paths,
)


def _resolve_storage(**overrides: object):
    arguments = {
        "sqlite_path": _UNSET,
        "storage_paths": None,
        "memory_sqlite_path": _UNSET,
        "crisis_log_sqlite_path": _UNSET,
        "feedback_sqlite_path": _UNSET,
        "text_session_sqlite_path": _UNSET,
    }
    arguments.update(overrides)
    return _resolve_runtime_storage_paths(**arguments)


def _resolve_persistence(**overrides: object):
    arguments = {
        "persistence_config": RuntimePersistenceConfig(allow_legacy_sqlite=True),
        "memory_mode": MemoryMode.LOCAL,
        "memory_backend": "sqlite",
        "memory_database_url": None,
        "memory_sqlite_path": DEFAULT_MEMORY_DB_PATH,
        "memory_sqlite_path_configured": False,
        "memory_store": None,
        "thread_persistence_backend": "sqlite",
        "thread_database_url": None,
        "sqlite_path": DEFAULT_THREAD_DB_PATH,
        "sqlite_path_configured": False,
        "crisis_log_persistence_backend": "sqlite",
        "crisis_log_database_url": None,
        "crisis_log_sqlite_path": DEFAULT_CRISIS_LOG_DB_PATH,
        "crisis_log_sqlite_path_configured": False,
        "crisis_log_backend": None,
        "session_feedback_persistence_backend": "sqlite",
        "session_feedback_database_url": None,
        "feedback_sqlite_path": DEFAULT_FEEDBACK_DB_PATH,
        "feedback_sqlite_path_configured": False,
        "session_feedback_backend": None,
        "text_session_backend": "auto",
        "text_session_database_url": None,
        "text_session_sqlite_path": None,
        "text_session_sqlite_path_configured": False,
    }
    arguments.update(overrides)
    return _resolve_runtime_persistence_config(**arguments)


def test_public_configuration_exports_remain_compatible() -> None:
    import agent.runtime as runtime_package
    from agent.runtime import runtime as runtime_module

    assert runtime_package.RuntimeStoragePaths is RuntimeStoragePaths
    assert runtime_package.RuntimePersistenceConfig is RuntimePersistenceConfig
    assert runtime_package.RuntimeDependencies is RuntimeDependencies
    assert runtime_package.RuntimeBehaviorConfig is RuntimeBehaviorConfig
    assert runtime_package.DEFAULT_THREAD_DB_PATH == DEFAULT_THREAD_DB_PATH
    assert runtime_module.RuntimeStoragePaths is RuntimeStoragePaths
    assert runtime_module.RuntimePersistenceConfig is RuntimePersistenceConfig
    assert runtime_module.DEFAULT_THREAD_DB_PATH == DEFAULT_THREAD_DB_PATH
    assert runtime_module.DEFAULT_MEMORY_DB_PATH == DEFAULT_MEMORY_DB_PATH
    assert runtime_module.DEFAULT_TEXT_SESSION_DB_PATH == DEFAULT_TEXT_SESSION_DB_PATH
    assert runtime_module.DEFAULT_CRISIS_LOG_DB_PATH == DEFAULT_CRISIS_LOG_DB_PATH
    assert runtime_module.DEFAULT_FEEDBACK_DB_PATH == DEFAULT_FEEDBACK_DB_PATH


def test_storage_defaults_share_the_configuration_sentinel() -> None:
    paths = RuntimeStoragePaths()
    resolved = _resolve_storage()

    assert paths.sqlite_path is _UNSET
    assert paths.memory_sqlite_path is _UNSET
    assert resolved.sqlite_path == DEFAULT_THREAD_DB_PATH
    assert resolved.memory_sqlite_path == DEFAULT_MEMORY_DB_PATH
    assert resolved.crisis_log_sqlite_path == DEFAULT_CRISIS_LOG_DB_PATH
    assert resolved.feedback_sqlite_path == DEFAULT_FEEDBACK_DB_PATH
    assert resolved.text_session_sqlite_path is None
    assert resolved.sqlite_path_configured is False


def test_legacy_storage_warning_lists_supplied_args_in_stable_order(
    tmp_path: Path,
) -> None:
    with pytest.warns(DeprecationWarning) as warnings:
        resolved = _resolve_storage(
            sqlite_path=tmp_path / "threads.sqlite3",
            memory_sqlite_path=tmp_path / "memory.sqlite3",
            text_session_sqlite_path=None,
        )

    assert str(warnings[0].message).endswith(
        "Legacy args: sqlite_path, memory_sqlite_path, text_session_sqlite_path."
    )
    assert resolved.sqlite_path_configured is True
    assert resolved.memory_sqlite_path_configured is True
    assert resolved.text_session_sqlite_path is None
    assert resolved.text_session_sqlite_path_configured is False


def test_grouped_storage_overrides_legacy_and_preserves_unset_fields(
    tmp_path: Path,
) -> None:
    legacy_memory = tmp_path / "legacy-memory.sqlite3"
    grouped_thread = tmp_path / "grouped-thread.sqlite3"

    with pytest.warns(DeprecationWarning):
        resolved = _resolve_storage(
            sqlite_path=tmp_path / "legacy-thread.sqlite3",
            memory_sqlite_path=legacy_memory,
            storage_paths=RuntimeStoragePaths(sqlite_path=grouped_thread),
        )

    assert resolved.sqlite_path == grouped_thread
    assert resolved.memory_sqlite_path == legacy_memory
    assert resolved.sqlite_path_configured is True
    assert resolved.memory_sqlite_path_configured is True


def test_repeating_default_storage_path_is_not_a_custom_override() -> None:
    resolved = _resolve_storage(
        storage_paths=RuntimeStoragePaths(sqlite_path=DEFAULT_THREAD_DB_PATH)
    )

    assert resolved.sqlite_path == DEFAULT_THREAD_DB_PATH
    assert resolved.sqlite_path_configured is False


def test_partial_grouped_persistence_preserves_legacy_values() -> None:
    resolved = _resolve_persistence(
        thread_persistence_backend="postgres",
        thread_database_url="postgresql://legacy/thread",
        persistence_config=RuntimePersistenceConfig(
            memory_backend="postgres",
            memory_database_url="postgresql://grouped/memory",
            allow_legacy_sqlite=True,
        ),
    )

    assert resolved.memory_backend == "postgres"
    assert resolved.memory_database_url == "postgresql://grouped/memory"
    assert resolved.thread_persistence_backend == "postgres"
    assert resolved.thread_database_url == "postgresql://legacy/thread"


def test_sqlite_validation_message_preserves_field_order() -> None:
    with pytest.raises(ValueError) as exc_info:
        _resolve_persistence(persistence_config=None)

    assert str(exc_info.value).endswith(
        "SQLite fields: thread_persistence_backend, memory_backend, "
        "crisis_log_persistence_backend, session_feedback_persistence_backend, "
        "text_session_backend."
    )


def test_custom_durable_sqlite_paths_require_legacy_opt_in(tmp_path: Path) -> None:
    with pytest.raises(ValueError) as exc_info:
        _resolve_persistence(
            persistence_config=RuntimePersistenceConfig(allow_legacy_sqlite=False),
            sqlite_path=tmp_path / "threads.sqlite3",
            sqlite_path_configured=True,
            memory_sqlite_path=tmp_path / "memory.sqlite3",
            memory_sqlite_path_configured=True,
            crisis_log_sqlite_path=tmp_path / "crisis.sqlite3",
            crisis_log_sqlite_path_configured=True,
            feedback_sqlite_path=tmp_path / "feedback.sqlite3",
            feedback_sqlite_path_configured=True,
            text_session_sqlite_path=tmp_path / "text-sessions.sqlite3",
            text_session_sqlite_path_configured=True,
        )

    assert str(exc_info.value).endswith(
        "SQLite fields: thread_persistence_backend, memory_backend, "
        "crisis_log_persistence_backend, session_feedback_persistence_backend, "
        "text_session_backend."
    )


def test_incognito_and_in_memory_paths_bypass_durable_sqlite_guard() -> None:
    incognito = _resolve_persistence(
        persistence_config=None,
        memory_mode=MemoryMode.INCOGNITO,
    )
    in_memory = _resolve_persistence(
        persistence_config=None,
        sqlite_path=":memory:",
        memory_sqlite_path=":memory:",
        crisis_log_sqlite_path=":memory:",
        feedback_sqlite_path=":memory:",
        text_session_sqlite_path=":memory:",
    )

    assert incognito.memory_mode is MemoryMode.INCOGNITO
    assert in_memory.memory_backend == "sqlite"


def test_dependency_config_can_override_and_explicitly_clear_values() -> None:
    original_store = object()
    original_client = object()

    def excluded(thread_id: str) -> bool:
        return thread_id == "external"

    resolved = _resolve_runtime_dependencies(
        dependencies=RuntimeDependencies(
            memory_store=None,
            default_llm_client=None,
            auto_finalize_excluded=excluded,
        ),
        memory_store=original_store,  # type: ignore[arg-type]
        crisis_log_backend=None,
        session_feedback_backend=None,
        embedding_provider=None,
        default_llm_client=original_client,  # type: ignore[arg-type]
        auto_finalize_excluded=None,
    )

    assert resolved.memory_store is None
    assert resolved.default_llm_client is None
    assert resolved.auto_finalize_excluded is excluded


def test_unset_dependency_fields_preserve_legacy_values() -> None:
    store = object()
    resolved = _resolve_runtime_dependencies(
        dependencies=RuntimeDependencies(),
        memory_store=store,  # type: ignore[arg-type]
        crisis_log_backend=None,
        session_feedback_backend=None,
        embedding_provider=None,
        default_llm_client=None,
        auto_finalize_excluded=None,
    )

    assert resolved.memory_store is store


def test_behavior_config_overrides_only_explicit_fields() -> None:
    timeout = timedelta(minutes=5)
    resolved = _resolve_runtime_behavior_config(
        behavior_config=RuntimeBehaviorConfig(
            text_session_history_limit=None,
            session_timeout=timeout,
            speculative_memory_prefetch=False,
        ),
        text_session_create_tables=True,
        text_session_history_limit=20,
        session_timeout=timedelta(minutes=20),
        session_sweep_interval_seconds=30.0,
        finalize_active_sessions_on_close=True,
        speculative_memory_prefetch=True,
    )

    assert resolved.text_session_create_tables is True
    assert resolved.text_session_history_limit is None
    assert resolved.session_timeout == timeout
    assert resolved.session_sweep_interval_seconds == 30.0
    assert resolved.finalize_active_sessions_on_close is True
    assert resolved.speculative_memory_prefetch is False
