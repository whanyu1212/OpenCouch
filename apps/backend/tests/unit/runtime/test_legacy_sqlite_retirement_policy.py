"""Policy contracts for removed and remaining legacy SQLite surfaces."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.feedback.session_feedback import InMemorySessionFeedbackBackend
from agent.memory.modes import MemoryMode
from agent.runtime.configuration import (
    DEFAULT_THREAD_DB_PATH,
    RuntimePersistenceConfig,
    _resolve_runtime_persistence_config,
)

_POSTGRES_URL = "postgresql://unused:unused@test-host.invalid/opencouch_test"


def _resolve(**overrides: object):
    arguments = {
        "persistence_config": None,
        "memory_mode": MemoryMode.LOCAL,
        "memory_backend": "postgres",
        "memory_database_url": _POSTGRES_URL,
        "thread_persistence_backend": "postgres",
        "thread_database_url": _POSTGRES_URL,
        "sqlite_path": DEFAULT_THREAD_DB_PATH,
        "sqlite_path_configured": False,
        "crisis_log_persistence_backend": "postgres",
        "crisis_log_database_url": _POSTGRES_URL,
        "crisis_log_backend": None,
        "session_feedback_persistence_backend": "postgres",
        "session_feedback_database_url": _POSTGRES_URL,
        "session_feedback_backend": None,
        "text_session_backend": "disabled",
        "text_session_database_url": None,
        "text_session_sqlite_path": None,
        "text_session_sqlite_path_configured": False,
    }
    arguments.update(overrides)
    return _resolve_runtime_persistence_config(**arguments)


@pytest.mark.parametrize(
    ("field", "overrides"),
    [
        (
            "crisis_log_persistence_backend",
            {"crisis_log_persistence_backend": "sqlite"},
        ),
        (
            "session_feedback_persistence_backend",
            {"session_feedback_persistence_backend": "sqlite"},
        ),
    ],
)
def test_removed_sqlite_application_store_is_rejected_even_with_opt_in(
    field: str,
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError) as exc_info:
        _resolve(
            persistence_config=RuntimePersistenceConfig(allow_legacy_sqlite=True),
            **overrides,
        )

    message = str(exc_info.value)
    assert (
        "SQLite crisis-audit and session-feedback persistence has been removed"
        in message
    )
    assert f"Removed fields: {field}." in message


def test_removed_thread_sqlite_is_rejected_even_with_legacy_opt_in() -> None:
    with pytest.raises(ValueError, match="SQLite runtime-state and active-session"):
        _resolve(
            persistence_config=RuntimePersistenceConfig(allow_legacy_sqlite=True),
            thread_persistence_backend="sqlite",
            thread_database_url=None,
            sqlite_path=Path("threads.sqlite3"),
        )


def test_explicit_in_memory_audit_feedback_overrides_skip_removed_selectors() -> None:
    resolved = _resolve(
        crisis_log_persistence_backend="sqlite",
        crisis_log_database_url=None,
        crisis_log_backend=InMemoryCrisisLogBackend(),
        session_feedback_persistence_backend="sqlite",
        session_feedback_database_url=None,
        session_feedback_backend=InMemorySessionFeedbackBackend(),
    )

    assert resolved.crisis_log_persistence_backend == "sqlite"
    assert resolved.session_feedback_persistence_backend == "sqlite"


@pytest.mark.parametrize("memory_mode", [MemoryMode.LOCAL, MemoryMode.INCOGNITO])
@pytest.mark.parametrize("allow_legacy_sqlite", [False, True])
def test_flat_sqlite_memory_backend_is_always_rejected(
    memory_mode: MemoryMode,
    allow_legacy_sqlite: bool,
) -> None:
    with pytest.raises(ValueError, match="SQLite memory persistence has been removed"):
        _resolve(
            persistence_config=RuntimePersistenceConfig(
                memory_mode=memory_mode,
                allow_legacy_sqlite=allow_legacy_sqlite,
            ),
            memory_backend="sqlite",
            memory_database_url=None,
        )


def test_disk_sdk_text_session_sqlite_remains_a_separate_legacy_guard() -> None:
    with pytest.raises(ValueError) as exc_info:
        _resolve(
            text_session_backend="sqlite",
            text_session_sqlite_path=Path("text-sessions.sqlite3"),
        )

    assert str(exc_info.value).endswith("SQLite fields: text_session_backend.")


def test_in_memory_sdk_text_session_sqlite_is_allowed() -> None:
    resolved = _resolve(
        text_session_backend="sqlite",
        text_session_sqlite_path=":memory:",
    )
    assert resolved.text_session_backend == "sqlite"


def test_shared_postgres_configuration_is_the_durable_target() -> None:
    resolved = _resolve(
        persistence_config=RuntimePersistenceConfig.for_shared_backend(
            memory_mode=MemoryMode.LOCAL,
            persistence_backend="postgres",
            database_url=_POSTGRES_URL,
            text_session_backend="sqlalchemy",
        )
    )

    assert resolved.memory_backend == "postgres"
    assert resolved.thread_persistence_backend == "postgres"
    assert resolved.crisis_log_persistence_backend == "postgres"
    assert resolved.session_feedback_persistence_backend == "postgres"
    assert resolved.text_session_backend == "sqlalchemy"
