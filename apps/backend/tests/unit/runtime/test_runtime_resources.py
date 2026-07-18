"""Tests for runtime resource bootstrap helpers."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

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

    async def aclose(self) -> None:
        self.close_calls += 1


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
async def test_runtime_resources_ensure_schema_calls_state_and_active_session() -> None:
    state_store = _CountingStateStore()
    active_session_manager = _CountingActiveSessionManager()
    resources = RuntimeResources(
        sqlite_path=Path(":memory:"),
        thread_persistence_backend="memory",
        thread_database_url=None,
        state_store=state_store,  # type: ignore[arg-type]
        text_session_store=None,
        memory_store=_CountingClosable(),  # type: ignore[arg-type]
        crisis_log_backend=_CountingClosable(),  # type: ignore[arg-type]
        session_feedback_backend=_CountingClosable(),  # type: ignore[arg-type]
        embedding_provider=NullEmbeddingProvider(),
        active_session_store=_CountingClosable(),  # type: ignore[arg-type]
        active_session_manager=active_session_manager,  # type: ignore[arg-type]
    )

    await resources.ensure_schema()

    assert state_store.ensure_schema_calls == 1
    assert active_session_manager.ensure_schema_calls == 1


@pytest.mark.asyncio
async def test_runtime_resources_prewarm_initializes_text_runtime_and_embedding_provider() -> (
    None
):
    embedding_provider = _WarmableEmbeddingProvider()
    resources = RuntimeResources(
        sqlite_path=Path(":memory:"),
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
        sqlite_path=Path(":memory:"),
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


def test_build_runtime_resources_requires_database_url_for_postgres_thread_backend() -> (
    None
):
    with pytest.raises(ValueError, match="OPENCOUCH_MEMORY_DATABASE_URL"):
        build_runtime_resources(
            memory_mode=MemoryMode.LOCAL,
            sqlite_path=":memory:",
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
