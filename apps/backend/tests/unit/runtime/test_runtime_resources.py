"""Tests for runtime resource bootstrap helpers."""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from agent.memory.modes import MemoryMode
from agent.memory.providers.embeddings import NullEmbeddingProvider
from agent.memory.store import OpenCouchMemoryStore
from agent.runtime.resources import RuntimeResources, build_runtime_resources
from agent.runtime.session.active_session import ActiveSessionManager


class _CountingStateStore:
    def __init__(self) -> None:
        self.ensure_schema_calls = 0
        self.close_calls = 0

    async def ensure_schema(self) -> None:
        self.ensure_schema_calls += 1

    async def aclose(self) -> None:
        self.close_calls += 1


class _CountingClosable:
    def __init__(self) -> None:
        self.close_calls = 0
        self.ensure_schema_calls = 0

    async def ensure_schema(self) -> None:
        self.ensure_schema_calls += 1

    async def aclose(self) -> None:
        self.close_calls += 1


class _FailingSchemaBackend(_CountingClosable):
    """Backend whose schema preparation always fails."""

    async def ensure_schema(self) -> None:
        self.ensure_schema_calls += 1
        raise RuntimeError("simulated schema preparation failure")


class _CountingActiveSessionManager:
    def __init__(self) -> None:
        self.ensure_schema_calls = 0

    async def ensure_schema(self) -> None:
        self.ensure_schema_calls += 1


def test_active_session_manager_accepts_deprecated_timeout_without_owning_policy() -> (
    None
):
    with pytest.warns(DeprecationWarning, match="deprecated and ignored"):
        manager = ActiveSessionManager(
            store=_CountingClosable(),  # type: ignore[arg-type]
            memory_mode=MemoryMode.LOCAL,
            session_timeout=timedelta(minutes=30),
        )

    assert not hasattr(manager, "_session_timeout")


class _WarmableEmbeddingProvider(NullEmbeddingProvider):
    def __init__(self) -> None:
        self.warmup_calls = 0

    async def awarmup(self) -> None:  # type: ignore[override]
        self.warmup_calls += 1


@pytest.mark.asyncio
async def test_runtime_resources_ensure_schema_prepares_every_durable_backend() -> None:
    """Startup prepares all owned backends, not just thread and session state."""

    state_store = _CountingStateStore()
    active_session_manager = _CountingActiveSessionManager()
    memory_store = _CountingClosable()
    crisis_log_backend = _CountingClosable()
    session_feedback_backend = _CountingClosable()
    resources = RuntimeResources(
        thread_persistence_backend="memory",
        thread_database_url=None,
        state_store=state_store,  # type: ignore[arg-type]
        text_session_store=None,
        memory_store=memory_store,  # type: ignore[arg-type]
        crisis_log_backend=crisis_log_backend,  # type: ignore[arg-type]
        session_feedback_backend=session_feedback_backend,  # type: ignore[arg-type]
        embedding_provider=NullEmbeddingProvider(),
        active_session_store=_CountingClosable(),  # type: ignore[arg-type]
        active_session_manager=active_session_manager,  # type: ignore[arg-type]
    )

    await resources.ensure_schema()

    assert state_store.ensure_schema_calls == 1
    assert active_session_manager.ensure_schema_calls == 1
    assert memory_store.ensure_schema_calls == 1
    assert crisis_log_backend.ensure_schema_calls == 1
    assert session_feedback_backend.ensure_schema_calls == 1
    # Preparation alone must not close anything.
    assert memory_store.close_calls == 0
    assert crisis_log_backend.close_calls == 0


@pytest.mark.asyncio
async def test_runtime_resources_ensure_schema_unwinds_opened_resources_on_failure() -> (
    None
):
    """A failed startup closes the backends it already opened."""

    state_store = _CountingStateStore()
    memory_store = _CountingClosable()
    crisis_log_backend = _FailingSchemaBackend()
    session_feedback_backend = _CountingClosable()
    active_session_store = _CountingClosable()
    resources = RuntimeResources(
        thread_persistence_backend="memory",
        thread_database_url=None,
        state_store=state_store,  # type: ignore[arg-type]
        text_session_store=None,
        memory_store=memory_store,  # type: ignore[arg-type]
        crisis_log_backend=crisis_log_backend,  # type: ignore[arg-type]
        session_feedback_backend=session_feedback_backend,  # type: ignore[arg-type]
        embedding_provider=NullEmbeddingProvider(),
        active_session_store=active_session_store,  # type: ignore[arg-type]
        active_session_manager=_CountingActiveSessionManager(),  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="simulated schema preparation failure"):
        await resources.ensure_schema()

    # Everything opened before the failure must be closed rather than leaked.
    assert state_store.close_calls == 1
    assert memory_store.close_calls == 1
    assert crisis_log_backend.close_calls == 1
    assert active_session_store.close_calls == 1
    # The backend after the failure was never prepared.
    assert session_feedback_backend.ensure_schema_calls == 0
    assert session_feedback_backend.close_calls == 1


@pytest.mark.asyncio
async def test_runtime_resources_prewarm_initializes_text_runtime_and_embedding_provider() -> (
    None
):
    embedding_provider = _WarmableEmbeddingProvider()
    resources = RuntimeResources(
        thread_persistence_backend="memory",
        thread_database_url=None,
        state_store=_CountingStateStore(),  # type: ignore[arg-type]
        text_session_store=None,
        memory_store=_CountingClosable(),  # type: ignore[arg-type]
        crisis_log_backend=_CountingClosable(),  # type: ignore[arg-type]
        session_feedback_backend=_CountingClosable(),  # type: ignore[arg-type]
        embedding_provider=embedding_provider,
        active_session_store=_CountingClosable(),  # type: ignore[arg-type]
        active_session_manager=_CountingActiveSessionManager(),  # type: ignore[arg-type]
    )
    calls = 0

    def _get_text_runtime() -> object:
        nonlocal calls
        calls += 1
        return object()

    await resources.prewarm(get_text_runtime=_get_text_runtime)

    assert calls == 1
    assert embedding_provider.warmup_calls == 1


@pytest.mark.asyncio
async def test_runtime_resources_aclose_closes_owned_resources_in_order() -> None:
    call_order: list[str] = []

    class _OrderedClosable:
        def __init__(self, name: str) -> None:
            self.name = name

        async def aclose(self) -> None:
            call_order.append(self.name)

    resources = RuntimeResources(
        thread_persistence_backend="memory",
        thread_database_url=None,
        state_store=_OrderedClosable("state_store"),  # type: ignore[arg-type]
        text_session_store=_OrderedClosable("text_session_store"),  # type: ignore[arg-type]
        memory_store=_OrderedClosable("memory_store"),  # type: ignore[arg-type]
        crisis_log_backend=_OrderedClosable("crisis_log_backend"),  # type: ignore[arg-type]
        session_feedback_backend=_OrderedClosable("session_feedback_backend"),  # type: ignore[arg-type]
        embedding_provider=NullEmbeddingProvider(),
        active_session_store=_OrderedClosable("active_session_store"),  # type: ignore[arg-type]
        active_session_manager=_CountingActiveSessionManager(),  # type: ignore[arg-type]
    )

    await resources.aclose()

    assert call_order == [
        "memory_store",
        "crisis_log_backend",
        "session_feedback_backend",
        "text_session_store",
        "state_store",
        "active_session_store",
    ]


@pytest.mark.asyncio
async def test_runtime_resources_aclose_releases_every_resource_despite_failure() -> (
    None
):
    """One backend raising on close must not strand the remaining ones."""

    class _RaisingClosable:
        def __init__(self) -> None:
            self.close_calls = 0

        async def ensure_schema(self) -> None:
            return None

        async def aclose(self) -> None:
            self.close_calls += 1
            raise RuntimeError("simulated close failure")

    memory_store = _RaisingClosable()
    crisis_log_backend = _CountingClosable()
    session_feedback_backend = _CountingClosable()
    state_store = _CountingStateStore()
    active_session_store = _CountingClosable()
    resources = RuntimeResources(
        thread_persistence_backend="memory",
        thread_database_url=None,
        state_store=state_store,  # type: ignore[arg-type]
        text_session_store=None,
        memory_store=memory_store,  # type: ignore[arg-type]
        crisis_log_backend=crisis_log_backend,  # type: ignore[arg-type]
        session_feedback_backend=session_feedback_backend,  # type: ignore[arg-type]
        embedding_provider=NullEmbeddingProvider(),
        active_session_store=active_session_store,  # type: ignore[arg-type]
        active_session_manager=_CountingActiveSessionManager(),  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="simulated close failure"):
        await resources.aclose()

    # Every resource after the failing one is still released.
    assert crisis_log_backend.close_calls == 1
    assert session_feedback_backend.close_calls == 1
    assert state_store.close_calls == 1
    assert active_session_store.close_calls == 1


@pytest.mark.asyncio
async def test_runtime_resources_aclose_releases_every_resource_despite_cancellation() -> (
    None
):
    """Cancellation from one closer must not strand the remaining resources."""

    class _CancellingClosable:
        async def aclose(self) -> None:
            raise asyncio.CancelledError()

    crisis_log_backend = _CountingClosable()
    session_feedback_backend = _CountingClosable()
    state_store = _CountingStateStore()
    active_session_store = _CountingClosable()
    resources = RuntimeResources(
        thread_persistence_backend="memory",
        thread_database_url=None,
        state_store=state_store,  # type: ignore[arg-type]
        text_session_store=None,
        memory_store=_CancellingClosable(),  # type: ignore[arg-type]
        crisis_log_backend=crisis_log_backend,  # type: ignore[arg-type]
        session_feedback_backend=session_feedback_backend,  # type: ignore[arg-type]
        embedding_provider=NullEmbeddingProvider(),
        active_session_store=active_session_store,  # type: ignore[arg-type]
        active_session_manager=_CountingActiveSessionManager(),  # type: ignore[arg-type]
    )

    with pytest.raises(asyncio.CancelledError):
        await resources.aclose()

    assert crisis_log_backend.close_calls == 1
    assert session_feedback_backend.close_calls == 1
    assert state_store.close_calls == 1
    assert active_session_store.close_calls == 1


@pytest.mark.asyncio
async def test_startup_unwind_releases_every_resource_despite_close_failure() -> None:
    """A failed startup releases all resources even if one close raises.

    Without this, a raising ``aclose`` mid-sequence would leave the remaining
    backends holding open connections while the startup error propagates,
    defeating the unwinding guarantee.
    """

    class _RaisingClosable:
        def __init__(self) -> None:
            self.close_calls = 0

        async def ensure_schema(self) -> None:
            return None

        async def aclose(self) -> None:
            self.close_calls += 1
            raise RuntimeError("simulated close failure")

    memory_store = _RaisingClosable()
    crisis_log_backend = _FailingSchemaBackend()
    session_feedback_backend = _CountingClosable()
    state_store = _CountingStateStore()
    active_session_store = _CountingClosable()
    resources = RuntimeResources(
        thread_persistence_backend="memory",
        thread_database_url=None,
        state_store=state_store,  # type: ignore[arg-type]
        text_session_store=None,
        memory_store=memory_store,  # type: ignore[arg-type]
        crisis_log_backend=crisis_log_backend,  # type: ignore[arg-type]
        session_feedback_backend=session_feedback_backend,  # type: ignore[arg-type]
        embedding_provider=NullEmbeddingProvider(),
        active_session_store=active_session_store,  # type: ignore[arg-type]
        active_session_manager=_CountingActiveSessionManager(),  # type: ignore[arg-type]
    )

    # The startup failure propagates, not the cleanup failure that followed it.
    with pytest.raises(RuntimeError, match="simulated schema preparation failure"):
        await resources.ensure_schema()

    assert memory_store.close_calls == 1
    assert crisis_log_backend.close_calls == 1
    assert session_feedback_backend.close_calls == 1
    assert state_store.close_calls == 1
    assert active_session_store.close_calls == 1


@pytest.mark.asyncio
async def test_aclose_quietly_releases_resources_after_post_preparation_failure() -> (
    None
):
    """Resources opened by preparation are released when a later step fails.

    ``__aenter__`` prepares schemas and then warms the runtime. Python does
    not call ``__aexit__`` when ``__aenter__`` raises, so a prewarm failure
    or cancellation must release the connections preparation just opened.
    """

    memory_store = _CountingClosable()
    crisis_log_backend = _CountingClosable()
    session_feedback_backend = _CountingClosable()
    state_store = _CountingStateStore()
    active_session_store = _CountingClosable()
    resources = RuntimeResources(
        thread_persistence_backend="memory",
        thread_database_url=None,
        state_store=state_store,  # type: ignore[arg-type]
        text_session_store=None,
        memory_store=memory_store,  # type: ignore[arg-type]
        crisis_log_backend=crisis_log_backend,  # type: ignore[arg-type]
        session_feedback_backend=session_feedback_backend,  # type: ignore[arg-type]
        embedding_provider=NullEmbeddingProvider(),
        active_session_store=active_session_store,  # type: ignore[arg-type]
        active_session_manager=_CountingActiveSessionManager(),  # type: ignore[arg-type]
    )

    await resources.ensure_schema()
    assert memory_store.ensure_schema_calls == 1
    assert memory_store.close_calls == 0

    # Simulate the unwind __aenter__ performs when a later step fails.
    await resources.aclose_quietly()

    assert memory_store.close_calls == 1
    assert crisis_log_backend.close_calls == 1
    assert session_feedback_backend.close_calls == 1
    assert state_store.close_calls == 1
    assert active_session_store.close_calls == 1


@pytest.mark.asyncio
async def test_ensure_schema_skips_injected_backends_without_the_hook() -> None:
    """Backends predating ``ensure_schema`` must not break runtime entry.

    ``RuntimeDependencies`` lets callers inject custom storage backends. An
    implementation that satisfied the pre-change protocols has no
    ``ensure_schema``, and unconditionally calling it would fail every
    runtime entry with ``AttributeError``.
    """

    class _LegacyBackend:
        """Backend implementing the protocol as it stood before this change."""

        def __init__(self) -> None:
            self.close_calls = 0

        async def aclose(self) -> None:
            self.close_calls += 1

    legacy_memory_store = _LegacyBackend()
    legacy_crisis_backend = _LegacyBackend()
    prepared_feedback_backend = _CountingClosable()
    resources = RuntimeResources(
        thread_persistence_backend="memory",
        thread_database_url=None,
        state_store=_CountingStateStore(),  # type: ignore[arg-type]
        text_session_store=None,
        memory_store=legacy_memory_store,  # type: ignore[arg-type]
        crisis_log_backend=legacy_crisis_backend,  # type: ignore[arg-type]
        session_feedback_backend=prepared_feedback_backend,  # type: ignore[arg-type]
        embedding_provider=NullEmbeddingProvider(),
        active_session_store=_CountingClosable(),  # type: ignore[arg-type]
        active_session_manager=_CountingActiveSessionManager(),  # type: ignore[arg-type]
    )

    await resources.ensure_schema()

    # Legacy backends are skipped, not fatal, and nothing is torn down.
    assert legacy_memory_store.close_calls == 0
    assert legacy_crisis_backend.close_calls == 0
    # Backends that do implement the hook are still prepared.
    assert prepared_feedback_backend.ensure_schema_calls == 1


def test_build_runtime_resources_requires_database_url_for_postgres_thread_backend() -> (
    None
):
    with pytest.raises(ValueError, match="OPENCOUCH_MEMORY_DATABASE_URL"):
        build_runtime_resources(
            memory_mode=MemoryMode.LOCAL,
            text_session_sqlite_path=None,
            thread_persistence_backend="postgres",
            thread_database_url=None,
            text_session_backend="disabled",
            text_session_database_url=None,
            text_session_create_tables=True,
            text_session_history_limit=None,
            memory_store=OpenCouchMemoryStore(),
            memory_backend="postgres",
            memory_database_url=None,
            crisis_log_backend=None,
            crisis_log_persistence_backend="postgres",
            crisis_log_database_url="postgresql://unused/crisis",
            session_feedback_backend=None,
            session_feedback_persistence_backend="postgres",
            session_feedback_database_url="postgresql://unused/feedback",
            embedding_provider=NullEmbeddingProvider(),
        )


def test_build_runtime_resources_uses_configured_sdk_sqlite_path(tmp_path) -> None:
    text_session_path = tmp_path / "text-sessions.sqlite3"
    resources = build_runtime_resources(
        memory_mode=MemoryMode.LOCAL,
        text_session_sqlite_path=text_session_path,
        thread_persistence_backend="memory",
        thread_database_url=None,
        text_session_backend="sqlite",
        text_session_database_url=None,
        text_session_create_tables=True,
        text_session_history_limit=None,
        memory_store=OpenCouchMemoryStore(),
        memory_backend="postgres",
        memory_database_url=None,
        crisis_log_backend=_CountingClosable(),  # type: ignore[arg-type]
        crisis_log_persistence_backend="postgres",
        crisis_log_database_url=None,
        session_feedback_backend=_CountingClosable(),  # type: ignore[arg-type]
        session_feedback_persistence_backend="postgres",
        session_feedback_database_url=None,
        embedding_provider=NullEmbeddingProvider(),
    )

    assert resources.text_session_store is not None
    assert resources.text_session_store._config.sqlite_path == text_session_path  # noqa: SLF001


def test_build_runtime_resources_forces_incognito_sdk_sessions_to_memory(
    tmp_path,
) -> None:
    resources = build_runtime_resources(
        memory_mode=MemoryMode.INCOGNITO,
        text_session_sqlite_path=tmp_path / "must-not-exist.sqlite3",
        thread_persistence_backend="postgres",
        thread_database_url=None,
        text_session_backend="sqlite",
        text_session_database_url=None,
        text_session_create_tables=True,
        text_session_history_limit=None,
        memory_store=None,
        memory_backend="postgres",
        memory_database_url=None,
        crisis_log_backend=None,
        crisis_log_persistence_backend="postgres",
        crisis_log_database_url=None,
        session_feedback_backend=None,
        session_feedback_persistence_backend="postgres",
        session_feedback_database_url=None,
        embedding_provider=None,
    )

    assert resources.text_session_store is not None
    assert resources.text_session_store._config.sqlite_path == ":memory:"  # noqa: SLF001
