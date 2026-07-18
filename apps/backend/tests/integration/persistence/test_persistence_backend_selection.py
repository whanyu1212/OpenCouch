"""Tests for current runtime persistence backend selection."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from agent.audit.crisis_log import InMemoryCrisisLogBackend, NullCrisisLogBackend
from agent.audit.postgres_crisis_log import PostgresCrisisLogBackend
from agent.feedback.postgres_session_feedback import PostgresSessionFeedbackBackend
from agent.feedback.session_feedback import (
    InMemorySessionFeedbackBackend,
    NullSessionFeedbackBackend,
)
from agent.memory.modes import MemoryMode
from agent.memory.providers.embeddings import NullEmbeddingProvider
from agent.memory.store import OpenCouchMemoryStore
from agent.memory.store.postgres import PostgresMemoryStore
from agent.runtime import (
    PersistentAgentRuntime,
    RuntimeBehaviorConfig,
    RuntimeDependencies,
    RuntimePersistenceConfig,
)
from agent.runtime.session.active_session import (
    InMemoryActiveSessionStore,
    NullActiveSessionStore,
    PostgresActiveSessionStore,
)
from tests.support.persistence import in_memory_runtime_storage_paths

_POSTGRES_URL = "postgresql://opencouch:opencouch@postgres:5432/opencouch"


def _local_ephemeral_runtime() -> PersistentAgentRuntime:
    """Construct a local runtime with non-durable application stores."""

    return PersistentAgentRuntime(
        persistence_config=RuntimePersistenceConfig(
            memory_mode=MemoryMode.LOCAL,
            memory_backend="postgres",
            thread_persistence_backend="memory",
            text_session_backend="disabled",
        ),
        dependencies=RuntimeDependencies(
            memory_store=OpenCouchMemoryStore(),
            crisis_log_backend=InMemoryCrisisLogBackend(),
            session_feedback_backend=InMemorySessionFeedbackBackend(),
        ),
    )


def test_runtime_defaults_application_stores_to_postgres() -> None:
    parameters = inspect.signature(PersistentAgentRuntime).parameters

    assert parameters["memory_backend"].default == "postgres"
    assert parameters["thread_persistence_backend"].default == "postgres"
    assert parameters["crisis_log_persistence_backend"].default == "postgres"
    assert parameters["session_feedback_persistence_backend"].default == "postgres"


def test_incognito_mode_uses_only_non_durable_application_stores() -> None:
    runtime = PersistentAgentRuntime(
        persistence_config=RuntimePersistenceConfig(memory_mode=MemoryMode.INCOGNITO)
    )

    assert runtime.sqlite_path == Path(":memory:")
    assert isinstance(runtime.memory_store, OpenCouchMemoryStore)
    assert isinstance(runtime.crisis_log_backend, InMemoryCrisisLogBackend)
    assert isinstance(runtime.session_feedback_backend, InMemorySessionFeedbackBackend)
    assert isinstance(runtime._active_session_store, NullActiveSessionStore)  # noqa: SLF001


@pytest.mark.parametrize(
    "config",
    [
        RuntimePersistenceConfig(
            memory_mode=MemoryMode.LOCAL,
            thread_persistence_backend="memory",
            crisis_log_persistence_backend="sqlite",
            session_feedback_persistence_backend="postgres",
            session_feedback_database_url=_POSTGRES_URL,
            allow_legacy_sqlite=True,
        ),
        RuntimePersistenceConfig(
            memory_mode=MemoryMode.LOCAL,
            thread_persistence_backend="memory",
            crisis_log_persistence_backend="postgres",
            crisis_log_database_url=_POSTGRES_URL,
            session_feedback_persistence_backend="sqlite",
            allow_legacy_sqlite=True,
        ),
    ],
)
def test_removed_sqlite_application_stores_fail_even_with_legacy_opt_in(
    config: RuntimePersistenceConfig,
) -> None:
    with pytest.raises(ValueError, match="SQLite crisis-audit and session-feedback"):
        PersistentAgentRuntime(
            persistence_config=config,
            dependencies=RuntimeDependencies(memory_store=OpenCouchMemoryStore()),
        )


def test_explicit_in_memory_backends_preserve_local_credential_free_runtime() -> None:
    runtime = _local_ephemeral_runtime()

    assert isinstance(runtime.memory_store, OpenCouchMemoryStore)
    assert isinstance(runtime._active_session_store, InMemoryActiveSessionStore)  # noqa: SLF001
    assert isinstance(runtime.crisis_log_backend, InMemoryCrisisLogBackend)
    assert isinstance(runtime.session_feedback_backend, InMemorySessionFeedbackBackend)


def test_shared_postgres_configuration_selects_every_durable_store() -> None:
    runtime = PersistentAgentRuntime(
        persistence_config=RuntimePersistenceConfig.for_shared_backend(
            memory_mode=MemoryMode.LOCAL,
            persistence_backend="postgres",
            database_url=_POSTGRES_URL,
        )
    )

    assert isinstance(runtime.memory_store, PostgresMemoryStore)
    assert isinstance(runtime._active_session_store, PostgresActiveSessionStore)  # noqa: SLF001
    assert isinstance(runtime.crisis_log_backend, PostgresCrisisLogBackend)
    assert isinstance(runtime.session_feedback_backend, PostgresSessionFeedbackBackend)


def test_shared_postgres_configuration_fans_out_database_url() -> None:
    config = RuntimePersistenceConfig.for_shared_backend(
        memory_mode=MemoryMode.LOCAL,
        persistence_backend="postgres",
        database_url=_POSTGRES_URL,
    )

    assert config.memory_database_url == _POSTGRES_URL
    assert config.thread_database_url == _POSTGRES_URL
    assert config.crisis_log_database_url == _POSTGRES_URL
    assert config.session_feedback_database_url == _POSTGRES_URL
    assert config.text_session_database_url == _POSTGRES_URL


def test_shared_backend_configuration_rejects_sqlite_with_opt_in() -> None:
    with pytest.raises(ValueError, match="Unsupported shared persistence backend"):
        RuntimePersistenceConfig.for_shared_backend(
            memory_mode=MemoryMode.LOCAL,
            persistence_backend="sqlite",  # type: ignore[arg-type]
            database_url=None,
            allow_legacy_sqlite=True,
        )


@pytest.mark.parametrize(
    ("config", "dependencies"),
    [
        (
            RuntimePersistenceConfig(
                memory_mode=MemoryMode.LOCAL,
                thread_persistence_backend="memory",
                crisis_log_persistence_backend="postgres",
                crisis_log_database_url=None,
                allow_legacy_sqlite=True,
            ),
            RuntimeDependencies(
                memory_store=OpenCouchMemoryStore(),
                session_feedback_backend=InMemorySessionFeedbackBackend(),
            ),
        ),
        (
            RuntimePersistenceConfig(
                memory_mode=MemoryMode.LOCAL,
                thread_persistence_backend="memory",
                session_feedback_persistence_backend="postgres",
                session_feedback_database_url=None,
                allow_legacy_sqlite=True,
            ),
            RuntimeDependencies(
                memory_store=OpenCouchMemoryStore(),
                crisis_log_backend=InMemoryCrisisLogBackend(),
            ),
        ),
    ],
)
def test_postgres_application_stores_require_database_url(
    config: RuntimePersistenceConfig,
    dependencies: RuntimeDependencies,
) -> None:
    with pytest.raises(ValueError, match="OPENCOUCH_MEMORY_DATABASE_URL"):
        PersistentAgentRuntime(
            persistence_config=config,
            dependencies=dependencies,
        )


def test_explicit_null_backends_override_configured_selection() -> None:
    crisis_backend = NullCrisisLogBackend()
    feedback_backend = NullSessionFeedbackBackend()
    runtime = PersistentAgentRuntime(
        persistence_config=RuntimePersistenceConfig(
            memory_mode=MemoryMode.LOCAL,
            thread_persistence_backend="memory",
            allow_legacy_sqlite=True,
        ),
        dependencies=RuntimeDependencies(
            memory_store=OpenCouchMemoryStore(),
            crisis_log_backend=crisis_backend,
            session_feedback_backend=feedback_backend,
        ),
    )

    assert runtime.crisis_log_backend is crisis_backend
    assert runtime.session_feedback_backend is feedback_backend


@pytest.mark.asyncio
async def test_aexit_closes_feedback_backend() -> None:
    class _CountingBackend(InMemorySessionFeedbackBackend):
        def __init__(self) -> None:
            super().__init__()
            self.close_calls = 0

        async def aclose(self) -> None:
            self.close_calls += 1
            await super().aclose()

    feedback_backend = _CountingBackend()
    runtime = PersistentAgentRuntime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=RuntimePersistenceConfig(
            memory_backend="postgres",
            thread_persistence_backend="memory",
        ),
        dependencies=RuntimeDependencies(
            memory_store=OpenCouchMemoryStore(),
            crisis_log_backend=InMemoryCrisisLogBackend(),
            session_feedback_backend=feedback_backend,
        ),
    )

    async with runtime:
        pass

    assert feedback_backend.close_calls == 1


@pytest.mark.asyncio
async def test_aenter_prewarms_embedding_provider_and_text_runtime() -> None:
    class _WarmableEmbeddingProvider(NullEmbeddingProvider):
        def __init__(self) -> None:
            self.warmup_calls = 0

        async def awarmup(self) -> None:
            self.warmup_calls += 1

    embedding_provider = _WarmableEmbeddingProvider()
    runtime = PersistentAgentRuntime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=RuntimePersistenceConfig(
            memory_backend="postgres",
            thread_persistence_backend="memory",
        ),
        dependencies=RuntimeDependencies(
            memory_store=OpenCouchMemoryStore(),
            crisis_log_backend=InMemoryCrisisLogBackend(),
            session_feedback_backend=InMemorySessionFeedbackBackend(),
            embedding_provider=embedding_provider,
        ),
        behavior_config=RuntimeBehaviorConfig(
            finalize_active_sessions_on_close=False,
        ),
    )

    async with runtime:
        assert embedding_provider.warmup_calls == 1
        assert runtime._sdk_bridge._openai_text_runtime is not None  # noqa: SLF001
