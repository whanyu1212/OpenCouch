"""Tests for PersistentAgentRuntime backend selection (v0.8 Stage D).

These tests verify that the runtime picks the right memory store and
crisis log backend based on ``memory_mode`` and whether the caller
passed explicit overrides. They cover the Stage D wiring logic
specifically — not the full runtime behavior (that's Stage E).

What these tests DON'T cover:
- End-to-end cross-restart persistence (Stage E smoke test)
- CLI integration (Stage E / F)
- Actual database writes (the Stage B / C test files cover the
  SQLite backends' own behavior)

Test strategy: construct the runtime without entering its async
context (no network provider call needed), check the concrete
type of ``runtime.memory_store`` and ``runtime.crisis_log_backend``.
The selection logic lives entirely in ``__init__``, so we don't
need to await anything to verify it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.runtime.session.active_session import (
    PostgresActiveSessionStore,
    SqliteActiveSessionStore,
)
from agent.audit.crisis_log import InMemoryCrisisLogBackend, NullCrisisLogBackend
from agent.memory.providers.embeddings import NullEmbeddingProvider
from agent.memory.modes import MemoryMode
from agent.feedback.session_feedback import (
    InMemorySessionFeedbackBackend,
    NullSessionFeedbackBackend,
)
from agent.audit.postgres_crisis_log import PostgresCrisisLogBackend
from agent.feedback.postgres_session_feedback import PostgresSessionFeedbackBackend
from agent.audit.sqlite_crisis_log import SqliteCrisisLogBackend
from agent.feedback.sqlite_session_feedback import SqliteSessionFeedbackBackend
from agent.memory.store.postgres import PostgresMemoryStore
from agent.memory.store.sqlite import SqliteMemoryStore
from agent.memory.store import OpenCouchMemoryStore
from agent.runtime import (
    DEFAULT_CRISIS_LOG_DB_PATH,
    DEFAULT_FEEDBACK_DB_PATH,
    DEFAULT_MEMORY_DB_PATH,
    DEFAULT_THREAD_DB_PATH,
    PersistentAgentRuntime,
    RuntimeDependencies,
    RuntimePersistenceConfig,
    RuntimeStoragePaths,
)
from tests.support.persistence import in_memory_runtime_storage_paths


def _legacy_sqlite_runtime(**kwargs) -> PersistentAgentRuntime:
    """Construct a runtime with temporary legacy SQLite opt-in for tests."""

    kwargs.setdefault(
        "persistence_config",
        RuntimePersistenceConfig(allow_legacy_sqlite=True),
    )
    return PersistentAgentRuntime(**kwargs)


# ─── Default-path constants ────────────────────────────────────────────


def test_default_memory_db_path_is_distinct_from_thread_db() -> None:
    """The memory SQLite file must not share a path with runtime state."""

    # All four OpenCouch-owned SQLite files must be distinct so schemas
    # cannot collide across stores.
    paths = {
        DEFAULT_THREAD_DB_PATH,
        DEFAULT_MEMORY_DB_PATH,
        DEFAULT_CRISIS_LOG_DB_PATH,
        DEFAULT_FEEDBACK_DB_PATH,
    }
    assert len(paths) == 4, f"expected 4 distinct paths, got {paths}"


def test_default_paths_live_next_to_thread_db() -> None:
    """All four OpenCouch-owned SQLite files should sit in the same
    directory (``apps/backend/``) so operators can find them together."""

    from agent.runtime import DEFAULT_THREAD_DB_PATH

    assert DEFAULT_MEMORY_DB_PATH.parent == DEFAULT_THREAD_DB_PATH.parent
    assert DEFAULT_CRISIS_LOG_DB_PATH.parent == DEFAULT_THREAD_DB_PATH.parent
    assert DEFAULT_FEEDBACK_DB_PATH.parent == DEFAULT_THREAD_DB_PATH.parent


def test_default_paths_are_in_store_dir() -> None:
    """The SQLite files should live under ``.store/`` so they
    don't clutter the backend root."""

    assert DEFAULT_MEMORY_DB_PATH.parent.name == ".store"
    assert DEFAULT_CRISIS_LOG_DB_PATH.parent.name == ".store"
    assert DEFAULT_FEEDBACK_DB_PATH.parent.name == ".store"
    assert DEFAULT_MEMORY_DB_PATH.suffix == ".sqlite3"
    assert DEFAULT_CRISIS_LOG_DB_PATH.suffix == ".sqlite3"
    assert DEFAULT_FEEDBACK_DB_PATH.suffix == ".sqlite3"


# ─── Incognito mode — in-memory backings ───────────────────────────────


def test_incognito_mode_uses_in_memory_store_by_default() -> None:
    """In incognito mode without an explicit override, the runtime
    should construct an in-memory store — nothing hits disk."""

    runtime = PersistentAgentRuntime(memory_mode=MemoryMode.INCOGNITO)
    assert isinstance(runtime.memory_store, OpenCouchMemoryStore)
    assert not isinstance(runtime.memory_store, SqliteMemoryStore)


def test_incognito_mode_uses_in_memory_crisis_log_by_default() -> None:
    """Same as the memory store — incognito crisis log must not
    touch disk."""

    runtime = PersistentAgentRuntime(memory_mode=MemoryMode.INCOGNITO)
    assert isinstance(runtime.crisis_log_backend, InMemoryCrisisLogBackend)
    assert not isinstance(runtime.crisis_log_backend, SqliteCrisisLogBackend)


def test_incognito_mode_sqlite_path_forced_to_memory() -> None:
    """The runtime state path should be ``:memory:`` in incognito mode."""

    runtime = PersistentAgentRuntime(
        storage_paths=RuntimeStoragePaths(sqlite_path="/tmp/should-be-ignored.sqlite3"),
        memory_mode=MemoryMode.INCOGNITO,
    )
    assert runtime.sqlite_path == Path(":memory:")


# ─── Local mode — legacy SQLite opt-in ─────────────────────────────────


def test_local_mode_rejects_durable_sqlite_without_legacy_opt_in() -> None:
    """Durable constructor SQLite must be explicitly marked legacy."""

    with pytest.raises(ValueError, match="Durable SQLite persistence is legacy"):
        PersistentAgentRuntime(memory_mode=MemoryMode.LOCAL)


def test_empty_grouped_storage_paths_do_not_opt_into_sqlite() -> None:
    """An empty grouped storage object should not bypass the SQLite guard."""

    with pytest.raises(ValueError, match="thread_persistence_backend"):
        PersistentAgentRuntime(
            memory_mode=MemoryMode.LOCAL,
            storage_paths=RuntimeStoragePaths(),
        )


def test_default_grouped_storage_path_does_not_opt_into_sqlite() -> None:
    """Restating the default path should not count as a concrete override."""

    with pytest.raises(ValueError, match="thread_persistence_backend"):
        PersistentAgentRuntime(
            memory_mode=MemoryMode.LOCAL,
            storage_paths=RuntimeStoragePaths(sqlite_path=DEFAULT_THREAD_DB_PATH),
        )


def test_partial_grouped_storage_paths_do_not_opt_into_default_sqlite(
    tmp_path: Path,
) -> None:
    """A single custom path should not allow unrelated default SQLite stores."""

    with pytest.raises(ValueError, match="memory_backend"):
        PersistentAgentRuntime(
            memory_mode=MemoryMode.LOCAL,
            storage_paths=RuntimeStoragePaths(sqlite_path=tmp_path / "threads.sqlite3"),
        )


def test_injected_backends_are_not_validated_as_sqlite_defaults() -> None:
    """Dependency overrides should bypass unused SQLite backend defaults."""

    runtime = PersistentAgentRuntime(
        memory_mode=MemoryMode.LOCAL,
        thread_persistence_backend="postgres",
        thread_database_url="postgresql://opencouch:opencouch@postgres:5432/opencouch",
        text_session_backend="disabled",
        dependencies=RuntimeDependencies(
            memory_store=OpenCouchMemoryStore(),
            crisis_log_backend=InMemoryCrisisLogBackend(),
            session_feedback_backend=InMemorySessionFeedbackBackend(),
        ),
    )

    assert isinstance(runtime._active_session_store, PostgresActiveSessionStore)  # noqa: SLF001
    assert isinstance(runtime.memory_store, OpenCouchMemoryStore)
    assert isinstance(runtime.crisis_log_backend, InMemoryCrisisLogBackend)
    assert isinstance(runtime.session_feedback_backend, InMemorySessionFeedbackBackend)


def test_local_mode_uses_sqlite_memory_store_with_legacy_opt_in() -> None:
    """With temporary legacy opt-in, local mode can still use SQLite memory."""

    runtime = _legacy_sqlite_runtime(memory_mode=MemoryMode.LOCAL)
    assert isinstance(runtime.memory_store, SqliteMemoryStore)
    assert isinstance(runtime._active_session_store, SqliteActiveSessionStore)  # noqa: SLF001
    # The SqliteMemoryStore's path should be the default memory path.
    assert runtime.memory_store.sqlite_path == Path(DEFAULT_MEMORY_DB_PATH)


def test_local_mode_uses_sqlite_crisis_log_with_legacy_opt_in() -> None:
    """With temporary legacy opt-in, local mode can still use SQLite audit."""

    runtime = _legacy_sqlite_runtime(memory_mode=MemoryMode.LOCAL)
    assert isinstance(runtime.crisis_log_backend, SqliteCrisisLogBackend)
    assert runtime.crisis_log_backend.sqlite_path == Path(DEFAULT_CRISIS_LOG_DB_PATH)


def test_local_mode_accepts_grouped_custom_sqlite_paths(tmp_path: Path) -> None:
    """Callers can override SQLite paths through the grouped config."""

    custom_memory = tmp_path / "custom_memory.sqlite3"
    custom_crisis = tmp_path / "custom_crisis.sqlite3"

    runtime = _legacy_sqlite_runtime(
        memory_mode=MemoryMode.LOCAL,
        storage_paths=RuntimeStoragePaths(
            memory_sqlite_path=custom_memory,
            crisis_log_sqlite_path=custom_crisis,
        ),
    )

    assert isinstance(runtime.memory_store, SqliteMemoryStore)
    assert runtime.memory_store.sqlite_path == custom_memory
    assert isinstance(runtime.crisis_log_backend, SqliteCrisisLogBackend)
    assert runtime.crisis_log_backend.sqlite_path == custom_crisis


def test_legacy_sqlite_path_kwargs_warn_and_still_work(tmp_path: Path) -> None:
    """Legacy direct path kwargs remain compatible during the migration window."""

    custom_thread = tmp_path / "legacy_threads.sqlite3"
    custom_memory = tmp_path / "legacy_memory.sqlite3"
    custom_crisis = tmp_path / "legacy_crisis.sqlite3"
    custom_feedback = tmp_path / "legacy_feedback.sqlite3"
    custom_text_session = tmp_path / "legacy_text_sessions.sqlite3"

    with pytest.warns(DeprecationWarning, match="RuntimeStoragePaths"):
        runtime = _legacy_sqlite_runtime(
            sqlite_path=custom_thread,
            memory_sqlite_path=custom_memory,
            crisis_log_sqlite_path=custom_crisis,
            feedback_sqlite_path=custom_feedback,
            text_session_backend="sqlite",
            text_session_sqlite_path=custom_text_session,
        )

    assert runtime.sqlite_path == custom_thread
    assert isinstance(runtime.memory_store, SqliteMemoryStore)
    assert runtime.memory_store.sqlite_path == custom_memory
    assert isinstance(runtime.crisis_log_backend, SqliteCrisisLogBackend)
    assert runtime.crisis_log_backend.sqlite_path == custom_crisis
    assert isinstance(runtime.session_feedback_backend, SqliteSessionFeedbackBackend)
    assert runtime.session_feedback_backend.sqlite_path == custom_feedback
    assert runtime._text_session_store is not None  # noqa: SLF001
    assert runtime._text_session_store._config.sqlite_path == custom_text_session  # noqa: SLF001


def test_grouped_storage_paths_override_legacy_sqlite_paths(tmp_path: Path) -> None:
    """Grouped storage paths should take precedence over legacy path args."""

    grouped_thread = tmp_path / "grouped_threads.sqlite3"
    grouped_memory = tmp_path / "grouped_memory.sqlite3"
    grouped_crisis = tmp_path / "grouped_crisis.sqlite3"
    grouped_feedback = tmp_path / "grouped_feedback.sqlite3"

    with pytest.warns(DeprecationWarning, match="RuntimeStoragePaths"):
        runtime = _legacy_sqlite_runtime(
            sqlite_path=tmp_path / "legacy_threads.sqlite3",
            memory_sqlite_path=tmp_path / "legacy_memory.sqlite3",
            crisis_log_sqlite_path=tmp_path / "legacy_crisis.sqlite3",
            feedback_sqlite_path=tmp_path / "legacy_feedback.sqlite3",
            storage_paths=RuntimeStoragePaths(
                sqlite_path=grouped_thread,
                memory_sqlite_path=grouped_memory,
                crisis_log_sqlite_path=grouped_crisis,
                feedback_sqlite_path=grouped_feedback,
            ),
        )

    assert runtime.sqlite_path == grouped_thread
    assert isinstance(runtime.memory_store, SqliteMemoryStore)
    assert runtime.memory_store.sqlite_path == grouped_memory
    assert isinstance(runtime.crisis_log_backend, SqliteCrisisLogBackend)
    assert runtime.crisis_log_backend.sqlite_path == grouped_crisis
    assert isinstance(runtime.session_feedback_backend, SqliteSessionFeedbackBackend)
    assert runtime.session_feedback_backend.sqlite_path == grouped_feedback


def test_synced_mode_can_use_legacy_sqlite_with_opt_in() -> None:
    """SYNCED mode can still use legacy SQLite during the migration window."""

    runtime = _legacy_sqlite_runtime(memory_mode=MemoryMode.SYNCED)
    assert isinstance(runtime.memory_store, SqliteMemoryStore)
    assert isinstance(runtime.crisis_log_backend, SqliteCrisisLogBackend)
    assert isinstance(runtime._active_session_store, SqliteActiveSessionStore)  # noqa: SLF001


def test_local_mode_can_select_postgres_memory_store() -> None:
    """When configured explicitly, local mode should construct a
    PostgresMemoryStore while leaving the other runtime-owned
    backends unchanged."""

    runtime = _legacy_sqlite_runtime(
        memory_mode=MemoryMode.LOCAL,
        memory_backend="postgres",
        memory_database_url="postgresql://opencouch:opencouch@postgres:5432/opencouch",
    )
    assert isinstance(runtime.memory_store, PostgresMemoryStore)
    assert runtime.memory_store.dsn == (
        "postgresql://opencouch:opencouch@postgres:5432/opencouch"
    )
    assert isinstance(runtime.crisis_log_backend, SqliteCrisisLogBackend)


def test_grouped_persistence_config_can_select_postgres_memory_store() -> None:
    """Grouped persistence config should drive backend selection."""

    runtime = PersistentAgentRuntime(
        persistence_config=RuntimePersistenceConfig(
            memory_mode=MemoryMode.LOCAL,
            memory_backend="postgres",
            memory_database_url="postgresql://opencouch:opencouch@postgres:5432/opencouch",
            allow_legacy_sqlite=True,
        )
    )

    assert isinstance(runtime.memory_store, PostgresMemoryStore)
    assert runtime.memory_store.dsn == (
        "postgresql://opencouch:opencouch@postgres:5432/opencouch"
    )
    assert isinstance(runtime.crisis_log_backend, SqliteCrisisLogBackend)


def test_shared_backend_persistence_config_fans_out_backend_and_database_url() -> None:
    """Shared backend config should populate every durable runtime store."""

    config = RuntimePersistenceConfig.for_shared_backend(
        memory_mode=MemoryMode.LOCAL,
        persistence_backend="postgres",
        database_url="postgresql://opencouch:opencouch@postgres:5432/opencouch",
    )

    assert config.memory_mode is MemoryMode.LOCAL
    assert config.memory_backend == "postgres"
    assert config.memory_database_url == (
        "postgresql://opencouch:opencouch@postgres:5432/opencouch"
    )
    assert config.thread_persistence_backend == "postgres"
    assert config.thread_database_url == (
        "postgresql://opencouch:opencouch@postgres:5432/opencouch"
    )
    assert config.crisis_log_persistence_backend == "postgres"
    assert config.crisis_log_database_url == (
        "postgresql://opencouch:opencouch@postgres:5432/opencouch"
    )
    assert config.session_feedback_persistence_backend == "postgres"
    assert config.session_feedback_database_url == (
        "postgresql://opencouch:opencouch@postgres:5432/opencouch"
    )
    assert config.text_session_backend == "auto"
    assert config.text_session_database_url == (
        "postgresql://opencouch:opencouch@postgres:5432/opencouch"
    )
    assert config.allow_legacy_sqlite is False


def test_shared_backend_persistence_config_rejects_sqlite_without_opt_in() -> None:
    """Grouped durable SQLite config must be explicitly marked legacy."""

    config = RuntimePersistenceConfig.for_shared_backend(
        memory_mode=MemoryMode.LOCAL,
        persistence_backend="sqlite",
        database_url=None,
    )

    with pytest.raises(ValueError, match="Durable SQLite persistence is legacy"):
        PersistentAgentRuntime(persistence_config=config)


def test_grouped_persistence_config_rejects_auto_text_sessions_without_dsn() -> None:
    """Auto SDK sessions resolve to SQLite without a database URL."""

    config = RuntimePersistenceConfig(
        memory_mode=MemoryMode.LOCAL,
        memory_backend="postgres",
        memory_database_url="postgresql://opencouch:opencouch@postgres:5432/opencouch",
        thread_persistence_backend="postgres",
        thread_database_url="postgresql://opencouch:opencouch@postgres:5432/opencouch",
        crisis_log_persistence_backend="postgres",
        crisis_log_database_url="postgresql://opencouch:opencouch@postgres:5432/opencouch",
        session_feedback_persistence_backend="postgres",
        session_feedback_database_url=(
            "postgresql://opencouch:opencouch@postgres:5432/opencouch"
        ),
        text_session_backend="auto",
        text_session_database_url=None,
    )

    with pytest.raises(ValueError, match="text_session_backend"):
        PersistentAgentRuntime(persistence_config=config)


def test_shared_backend_persistence_config_allows_sqlite_with_opt_in() -> None:
    """Temporary legacy SQLite config remains available with explicit opt-in."""

    runtime = PersistentAgentRuntime(
        persistence_config=RuntimePersistenceConfig.for_shared_backend(
            memory_mode=MemoryMode.LOCAL,
            persistence_backend="sqlite",
            database_url=None,
            allow_legacy_sqlite=True,
        )
    )

    assert isinstance(runtime.memory_store, SqliteMemoryStore)
    assert isinstance(runtime.crisis_log_backend, SqliteCrisisLogBackend)
    assert isinstance(runtime.session_feedback_backend, SqliteSessionFeedbackBackend)


def test_shared_backend_persistence_config_accepts_text_session_url_override() -> None:
    """SDK sessions can use a distinct SQLAlchemy URL when configured."""

    config = RuntimePersistenceConfig.for_shared_backend(
        memory_mode=MemoryMode.LOCAL,
        persistence_backend="postgres",
        database_url="postgresql://opencouch:opencouch@postgres:5432/opencouch",
        text_session_backend="sqlalchemy",
        text_session_database_url=(
            "postgresql+asyncpg://opencouch:opencouch@postgres:5432/opencouch"
        ),
    )

    assert config.text_session_backend == "sqlalchemy"
    assert config.text_session_database_url == (
        "postgresql+asyncpg://opencouch:opencouch@postgres:5432/opencouch"
    )
    assert config.memory_database_url == (
        "postgresql://opencouch:opencouch@postgres:5432/opencouch"
    )


def test_partial_grouped_persistence_config_preserves_legacy_thread_backend() -> None:
    """Unset grouped persistence fields should not clobber legacy kwargs."""

    runtime = PersistentAgentRuntime(
        thread_persistence_backend="postgres",
        thread_database_url="postgresql://opencouch:opencouch@postgres:5432/opencouch",
        persistence_config=RuntimePersistenceConfig(
            memory_backend="postgres",
            memory_database_url="postgresql://opencouch:opencouch@postgres:5432/opencouch",
            allow_legacy_sqlite=True,
        ),
    )

    assert isinstance(runtime.memory_store, PostgresMemoryStore)
    assert isinstance(runtime._active_session_store, PostgresActiveSessionStore)  # noqa: SLF001


def test_postgres_memory_backend_requires_database_url() -> None:
    """Selecting the Postgres memory backend without a DSN should fail
    fast at runtime construction rather than later on first query."""

    with pytest.raises(ValueError, match="OPENCOUCH_MEMORY_DATABASE_URL"):
        _legacy_sqlite_runtime(
            memory_mode=MemoryMode.LOCAL,
            memory_backend="postgres",
        )


def test_local_mode_can_select_postgres_thread_backend() -> None:
    """When configured explicitly, local mode should construct a
    Postgres-backed active-session store for runtime-owned thread state."""

    runtime = _legacy_sqlite_runtime(
        memory_mode=MemoryMode.LOCAL,
        thread_persistence_backend="postgres",
        thread_database_url="postgresql://opencouch:opencouch@postgres:5432/opencouch",
    )
    assert isinstance(runtime._active_session_store, PostgresActiveSessionStore)  # noqa: SLF001
    assert isinstance(runtime.memory_store, SqliteMemoryStore)
    assert isinstance(runtime.crisis_log_backend, SqliteCrisisLogBackend)


def test_postgres_thread_backend_requires_database_url() -> None:
    """Selecting the Postgres thread backend without a DSN should fail
    fast at runtime construction rather than later on first turn."""

    with pytest.raises(ValueError, match="OPENCOUCH_MEMORY_DATABASE_URL"):
        _legacy_sqlite_runtime(
            memory_mode=MemoryMode.LOCAL,
            thread_persistence_backend="postgres",
        )


def test_local_mode_can_select_postgres_crisis_log_backend() -> None:
    """When configured explicitly, local mode should construct a
    PostgresCrisisLogBackend while leaving the memory store unchanged."""

    runtime = _legacy_sqlite_runtime(
        memory_mode=MemoryMode.LOCAL,
        crisis_log_persistence_backend="postgres",
        crisis_log_database_url="postgresql://opencouch:opencouch@postgres:5432/opencouch",
    )
    assert isinstance(runtime.crisis_log_backend, PostgresCrisisLogBackend)
    assert runtime.crisis_log_backend.dsn == (
        "postgresql://opencouch:opencouch@postgres:5432/opencouch"
    )
    assert isinstance(runtime.memory_store, SqliteMemoryStore)


def test_postgres_crisis_log_backend_requires_database_url() -> None:
    """Selecting the Postgres crisis-log backend without a DSN should fail
    fast at runtime construction rather than later on first write."""

    with pytest.raises(ValueError, match="OPENCOUCH_MEMORY_DATABASE_URL"):
        _legacy_sqlite_runtime(
            memory_mode=MemoryMode.LOCAL,
            crisis_log_persistence_backend="postgres",
        )


# ─── Explicit overrides ────────────────────────────────────────────────


def test_explicit_memory_store_overrides_mode_based_selection() -> None:
    """Passing an explicit ``memory_store`` should bypass the
    mode-based defaults entirely. Tests rely on this — they
    construct an in-memory store directly and expect the runtime
    to use it regardless of the mode flag."""

    custom_store = OpenCouchMemoryStore()
    runtime = _legacy_sqlite_runtime(
        memory_mode=MemoryMode.LOCAL,  # would normally pick SqliteMemoryStore
        memory_store=custom_store,
    )
    # The runtime should hold the exact instance we passed in.
    assert runtime.memory_store is custom_store


def test_explicit_crisis_log_overrides_mode_based_selection() -> None:
    """Same pattern as the memory store — explicit crisis log
    backend bypasses the mode-based default."""

    custom_backend = NullCrisisLogBackend()
    runtime = _legacy_sqlite_runtime(
        memory_mode=MemoryMode.LOCAL,
        crisis_log_backend=custom_backend,
    )
    assert runtime.crisis_log_backend is custom_backend


def test_grouped_dependencies_can_override_default_llm_client() -> None:
    """Grouped dependencies should populate injected runtime services."""

    llm_client = object()

    runtime = _legacy_sqlite_runtime(
        dependencies=RuntimeDependencies(
            default_llm_client=llm_client,  # type: ignore[arg-type]
        )
    )

    assert runtime._default_llm_client is llm_client  # noqa: SLF001


def test_explicit_overrides_work_with_incognito_too() -> None:
    """Even in incognito mode, an explicit override should be
    respected. This lets test fixtures pass a specific backend
    (e.g., a mock or a failing backend) regardless of the mode."""

    custom_store = OpenCouchMemoryStore()
    custom_backend = NullCrisisLogBackend()
    runtime = PersistentAgentRuntime(
        memory_mode=MemoryMode.INCOGNITO,
        memory_store=custom_store,
        crisis_log_backend=custom_backend,
    )
    assert runtime.memory_store is custom_store
    assert runtime.crisis_log_backend is custom_backend


# ─── Mixed overrides ───────────────────────────────────────────────────


def test_can_override_only_memory_store() -> None:
    """The two backend parameters are independent. Overriding just
    the memory store should leave the crisis log to mode-based
    selection."""

    custom_store = OpenCouchMemoryStore()
    runtime = _legacy_sqlite_runtime(
        memory_mode=MemoryMode.LOCAL,
        memory_store=custom_store,
    )
    assert runtime.memory_store is custom_store
    # Crisis log still picks the SQLite default for LOCAL mode
    assert isinstance(runtime.crisis_log_backend, SqliteCrisisLogBackend)


def test_can_override_only_crisis_log() -> None:
    """Symmetric to the memory-store-only override."""

    custom_backend = NullCrisisLogBackend()
    runtime = _legacy_sqlite_runtime(
        memory_mode=MemoryMode.LOCAL,
        crisis_log_backend=custom_backend,
    )
    # Memory store still picks the SQLite default for LOCAL mode
    assert isinstance(runtime.memory_store, SqliteMemoryStore)
    assert runtime.crisis_log_backend is custom_backend


# ─── Connection is NOT opened in __init__ ──────────────────────────────


def test_init_does_not_open_sqlite_connections() -> None:
    """``__init__`` should be cheap and never touch disk. The SQLite
    backends use lazy connection opening (connection is created on
    first async method call, not during construction). This test
    verifies that by checking the internal ``_connection`` attribute
    on both SQLite backends — it should be None after init."""

    runtime = _legacy_sqlite_runtime(memory_mode=MemoryMode.LOCAL)
    # Both backends should have their _connection set to None.
    # These are internal attributes, not public API — this test
    # reaches into them specifically to verify the lazy-open
    # contract. If the internals change, the test needs to update.
    memory_store = runtime.memory_store
    crisis_log = runtime.crisis_log_backend
    assert isinstance(memory_store, SqliteMemoryStore)
    assert isinstance(crisis_log, SqliteCrisisLogBackend)
    assert memory_store._connection is None  # noqa: SLF001
    assert crisis_log._connection is None  # noqa: SLF001


# ─── Close safety ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_init_without_enter_still_closes_cleanly() -> None:
    """If a runtime is constructed but never entered (e.g., a test
    that only checks ``__init__`` behavior, or an error path that
    aborts before ``async with``), calling ``aclose`` on the
    backends should still be safe. The SQLite backends' lazy-open
    means "never opened" is just "no-op close"."""

    runtime = _legacy_sqlite_runtime(memory_mode=MemoryMode.LOCAL)
    # Neither backend has had a chance to open its connection.
    # Closing them should not raise.
    await runtime.memory_store.aclose()
    await runtime.crisis_log_backend.aclose()
    await runtime.session_feedback_backend.aclose()


# ─── v0.10 session-feedback backend selection ──────────────────────────


def test_incognito_mode_uses_in_memory_feedback_by_default() -> None:
    """Incognito mode should pick the in-memory feedback backend —
    nothing hits disk. The feedback collector is always-on regardless
    of mode, but incognito keeps it ephemeral."""

    runtime = PersistentAgentRuntime(memory_mode=MemoryMode.INCOGNITO)
    assert isinstance(runtime.session_feedback_backend, InMemorySessionFeedbackBackend)
    assert not isinstance(
        runtime.session_feedback_backend, SqliteSessionFeedbackBackend
    )


def test_local_mode_uses_sqlite_feedback_with_legacy_opt_in() -> None:
    """With temporary legacy opt-in, local mode can still use SQLite feedback."""

    runtime = _legacy_sqlite_runtime(memory_mode=MemoryMode.LOCAL)
    assert isinstance(runtime.session_feedback_backend, SqliteSessionFeedbackBackend)
    assert runtime.session_feedback_backend.sqlite_path == Path(
        DEFAULT_FEEDBACK_DB_PATH
    )


def test_synced_mode_uses_sqlite_feedback_with_legacy_opt_in() -> None:
    """SYNCED mode can still use legacy SQLite feedback with opt-in."""

    runtime = _legacy_sqlite_runtime(memory_mode=MemoryMode.SYNCED)
    assert isinstance(runtime.session_feedback_backend, SqliteSessionFeedbackBackend)


def test_local_mode_accepts_grouped_custom_feedback_sqlite_path(
    tmp_path: Path,
) -> None:
    """Operators and test fixtures can override feedback path via grouped config."""

    custom = tmp_path / "custom_feedback.sqlite3"
    runtime = _legacy_sqlite_runtime(
        memory_mode=MemoryMode.LOCAL,
        storage_paths=RuntimeStoragePaths(feedback_sqlite_path=custom),
    )
    assert isinstance(runtime.session_feedback_backend, SqliteSessionFeedbackBackend)
    assert runtime.session_feedback_backend.sqlite_path == custom


def test_local_mode_can_select_postgres_session_feedback_backend() -> None:
    """When configured explicitly, local mode should construct a
    PostgresSessionFeedbackBackend while leaving the other backends unchanged."""

    runtime = _legacy_sqlite_runtime(
        memory_mode=MemoryMode.LOCAL,
        session_feedback_persistence_backend="postgres",
        session_feedback_database_url="postgresql://opencouch:opencouch@postgres:5432/opencouch",
    )
    assert isinstance(runtime.session_feedback_backend, PostgresSessionFeedbackBackend)
    assert runtime.session_feedback_backend.dsn == (
        "postgresql://opencouch:opencouch@postgres:5432/opencouch"
    )
    assert isinstance(runtime.memory_store, SqliteMemoryStore)
    assert isinstance(runtime.crisis_log_backend, SqliteCrisisLogBackend)


def test_postgres_session_feedback_backend_requires_database_url() -> None:
    """Selecting the Postgres feedback backend without a DSN should fail
    fast at runtime construction rather than later on first write."""

    with pytest.raises(ValueError, match="OPENCOUCH_MEMORY_DATABASE_URL"):
        _legacy_sqlite_runtime(
            memory_mode=MemoryMode.LOCAL,
            session_feedback_persistence_backend="postgres",
        )


def test_explicit_feedback_backend_overrides_mode_based_selection() -> None:
    """Passing an explicit ``session_feedback_backend`` should bypass
    the mode-based default — matches the crisis_log and memory_store
    override contract so tests can inject Null/Mock backends
    regardless of mode."""

    custom_backend = NullSessionFeedbackBackend()
    runtime = _legacy_sqlite_runtime(
        memory_mode=MemoryMode.LOCAL,  # would normally pick SQLite
        session_feedback_backend=custom_backend,
    )
    assert runtime.session_feedback_backend is custom_backend


def test_feedback_backend_lazy_connection_in_init() -> None:
    """The SQLite feedback backend should NOT open its connection
    during ``__init__``. Matches the crisis_log and memory_store
    lazy-open contract."""

    runtime = _legacy_sqlite_runtime(memory_mode=MemoryMode.LOCAL)
    backend = runtime.session_feedback_backend
    assert isinstance(backend, SqliteSessionFeedbackBackend)
    assert backend._connection is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_aexit_closes_feedback_backend() -> None:
    """``PersistentAgentRuntime.__aexit__`` must call ``aclose`` on
    the feedback backend, symmetric to the memory store and
    crisis_log lifecycle. A leaked aiosqlite connection on exit
    would accumulate across CLI / API sessions."""

    class _CountingBackend(InMemorySessionFeedbackBackend):
        def __init__(self) -> None:
            super().__init__()
            self.close_calls = 0

        async def aclose(self) -> None:  # type: ignore[override]
            self.close_calls += 1
            await super().aclose()

    backend = _CountingBackend()
    runtime = PersistentAgentRuntime(
        storage_paths=in_memory_runtime_storage_paths(),
        session_feedback_backend=backend,
    )
    async with runtime:
        # Fine — nothing to assert here; we just want __aexit__ to fire.
        pass

    # Exactly one close call on the feedback backend.
    assert backend.close_calls == 1


@pytest.mark.asyncio
async def test_aenter_prewarms_embedding_provider_and_text_runtime() -> None:
    """``__aenter__`` should finish warmup before the runtime is usable."""

    class _WarmableEmbeddingProvider(NullEmbeddingProvider):
        def __init__(self) -> None:
            self.warmup_calls = 0

        async def awarmup(self) -> None:  # type: ignore[override]
            self.warmup_calls += 1

    embedding_provider = _WarmableEmbeddingProvider()

    runtime = PersistentAgentRuntime(
        storage_paths=in_memory_runtime_storage_paths(),
        embedding_provider=embedding_provider,
        finalize_active_sessions_on_close=False,
    )

    async with runtime:
        assert embedding_provider.warmup_calls == 1
        assert runtime._sdk_bridge._openai_text_runtime is not None  # noqa: SLF001
