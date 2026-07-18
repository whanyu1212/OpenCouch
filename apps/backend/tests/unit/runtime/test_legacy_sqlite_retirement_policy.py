"""Characterize the policy boundary for retiring durable SQLite."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.feedback.session_feedback import InMemorySessionFeedbackBackend
from agent.memory.modes import MemoryMode
from agent.runtime.configuration import (
    DEFAULT_CRISIS_LOG_DB_PATH,
    DEFAULT_FEEDBACK_DB_PATH,
    DEFAULT_MEMORY_DB_PATH,
    DEFAULT_THREAD_DB_PATH,
    RuntimePersistenceConfig,
    _resolve_runtime_persistence_config,
)

# Resolution is structural and never opens a connection; use an unmistakably fake DSN.
_POSTGRES_URL = "postgresql://unused:unused@test-host.invalid/opencouch_test"


def _resolve(**overrides: object):
    arguments = {
        "persistence_config": None,
        "memory_mode": MemoryMode.LOCAL,
        "memory_backend": "postgres",
        "memory_database_url": _POSTGRES_URL,
        "memory_sqlite_path": DEFAULT_MEMORY_DB_PATH,
        "memory_sqlite_path_configured": False,
        "memory_store": None,
        "thread_persistence_backend": "postgres",
        "thread_database_url": _POSTGRES_URL,
        "sqlite_path": DEFAULT_THREAD_DB_PATH,
        "sqlite_path_configured": False,
        "crisis_log_persistence_backend": "postgres",
        "crisis_log_database_url": _POSTGRES_URL,
        "crisis_log_sqlite_path": DEFAULT_CRISIS_LOG_DB_PATH,
        "crisis_log_sqlite_path_configured": False,
        "crisis_log_backend": None,
        "session_feedback_persistence_backend": "postgres",
        "session_feedback_database_url": _POSTGRES_URL,
        "feedback_sqlite_path": DEFAULT_FEEDBACK_DB_PATH,
        "feedback_sqlite_path_configured": False,
        "session_feedback_backend": None,
        "text_session_backend": "disabled",
        "text_session_database_url": None,
        "text_session_sqlite_path": None,
        "text_session_sqlite_path_configured": False,
    }
    arguments.update(overrides)
    return _resolve_runtime_persistence_config(**arguments)


# Memory is intentionally deferred to #233 and remains covered by
# test_runtime_configuration.py::test_sqlite_validation_message_preserves_field_order
# and test_incognito_and_in_memory_paths_bypass_durable_sqlite_guard.
@pytest.mark.parametrize(
    ("field", "overrides"),
    [
        (
            "crisis_log_persistence_backend",
            {
                "crisis_log_persistence_backend": "sqlite",
                "crisis_log_database_url": None,
                "crisis_log_sqlite_path": Path("crisis.sqlite3"),
            },
        ),
        (
            "session_feedback_persistence_backend",
            {
                "session_feedback_persistence_backend": "sqlite",
                "session_feedback_database_url": None,
                "feedback_sqlite_path": Path("feedback.sqlite3"),
            },
        ),
    ],
)
def test_durable_sqlite_application_store_requires_legacy_opt_in(
    field: str,
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError) as exc_info:
        _resolve(**overrides)

    message = str(exc_info.value)
    assert "Durable SQLite persistence is legacy" in message
    assert f"SQLite fields: {field}." in message


def test_removed_thread_sqlite_is_rejected_even_with_legacy_opt_in() -> None:
    with pytest.raises(ValueError, match="SQLite runtime-state and active-session"):
        _resolve(
            persistence_config=RuntimePersistenceConfig(allow_legacy_sqlite=True),
            thread_persistence_backend="sqlite",
            thread_database_url=None,
            sqlite_path=Path("threads.sqlite3"),
        )


def test_thread_backend_is_the_single_selector_for_state_and_active_sessions() -> None:
    resolved = _resolve(
        thread_persistence_backend="postgres",
        thread_database_url=_POSTGRES_URL,
    )

    # Active-session store construction consumes this same resolved field in
    # build_runtime_resources; there is no independent active-session selector.
    assert resolved.thread_persistence_backend == "postgres"


def test_legacy_opt_in_temporarily_permits_durable_application_stores() -> None:
    resolved = _resolve(
        persistence_config=RuntimePersistenceConfig(allow_legacy_sqlite=True),
        thread_persistence_backend="memory",
        thread_database_url=None,
        crisis_log_persistence_backend="sqlite",
        crisis_log_database_url=None,
        crisis_log_sqlite_path=Path("crisis.sqlite3"),
        session_feedback_persistence_backend="sqlite",
        session_feedback_database_url=None,
        feedback_sqlite_path=Path("feedback.sqlite3"),
    )

    assert resolved.thread_persistence_backend == "memory"
    assert resolved.crisis_log_persistence_backend == "sqlite"
    assert resolved.session_feedback_persistence_backend == "sqlite"


def test_in_memory_sqlite_application_stores_are_not_durable() -> None:
    resolved = _resolve(
        thread_persistence_backend="memory",
        thread_database_url=None,
        crisis_log_persistence_backend="sqlite",
        crisis_log_database_url=None,
        crisis_log_sqlite_path=":memory:",
        session_feedback_persistence_backend="sqlite",
        session_feedback_database_url=None,
        feedback_sqlite_path=":memory:",
    )

    assert resolved.thread_persistence_backend == "memory"
    assert resolved.crisis_log_persistence_backend == "sqlite"
    assert resolved.session_feedback_persistence_backend == "sqlite"


def test_explicit_in_memory_audit_feedback_overrides_skip_unused_sqlite_guard() -> None:
    crisis_backend = InMemoryCrisisLogBackend()
    feedback_backend = InMemorySessionFeedbackBackend()

    # At this resolver boundary, explicit dependency overrides mean the
    # corresponding configured backend fields are unused and are not validated.
    resolved = _resolve(
        crisis_log_persistence_backend="sqlite",
        crisis_log_database_url=None,
        crisis_log_sqlite_path=Path("crisis.sqlite3"),
        crisis_log_backend=crisis_backend,
        session_feedback_persistence_backend="sqlite",
        session_feedback_database_url=None,
        feedback_sqlite_path=Path("feedback.sqlite3"),
        session_feedback_backend=feedback_backend,
    )

    assert resolved.crisis_log_persistence_backend == "sqlite"
    assert resolved.session_feedback_persistence_backend == "sqlite"


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
    assert resolved.text_session_database_url is None


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
    assert resolved.text_session_database_url == _POSTGRES_URL
