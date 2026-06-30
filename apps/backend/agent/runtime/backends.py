"""Backend factory helpers for the persistent agent runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent.audit.crisis_log import CrisisLogBackend, InMemoryCrisisLogBackend
from agent.audit.postgres_crisis_log import PostgresCrisisLogBackend
from agent.feedback.postgres_session_feedback import PostgresSessionFeedbackBackend
from agent.audit.sqlite_crisis_log import SqliteCrisisLogBackend
from agent.feedback.sqlite_session_feedback import SqliteSessionFeedbackBackend
from agent.feedback.session_feedback import (
    InMemorySessionFeedbackBackend,
    SessionFeedbackBackend,
)
from agent.memory.providers.embeddings import (
    EmbeddingProvider,
    NullEmbeddingProvider,
    create_configured_embedding_provider,
)
from agent.memory.store.sqlite import SqliteMemoryStore
from agent.memory.modes import MemoryMode
from agent.memory.store.postgres import PostgresMemoryStore
from agent.memory.store import MemoryStore, OpenCouchMemoryStore

PersistenceBackend = Literal["sqlite", "postgres"]
RuntimeStoreBackend = Literal["memory", "sqlite", "postgres"]


@dataclass(frozen=True, slots=True)
class RuntimeBackendSelection:
    """Effective runtime-owned backends after mode overrides are applied."""

    thread_persistence_backend: PersistenceBackend
    memory_store_backend: RuntimeStoreBackend
    crisis_log_backend: RuntimeStoreBackend
    session_feedback_backend: RuntimeStoreBackend


def select_runtime_backends(
    *,
    memory_mode: MemoryMode,
    memory_backend: PersistenceBackend,
    thread_persistence_backend: PersistenceBackend,
    crisis_log_persistence_backend: PersistenceBackend,
    session_feedback_persistence_backend: PersistenceBackend,
) -> RuntimeBackendSelection:
    """Return effective runtime-owned backends for the configured memory mode."""

    if memory_mode == MemoryMode.INCOGNITO:
        return RuntimeBackendSelection(
            thread_persistence_backend="sqlite",
            memory_store_backend="memory",
            crisis_log_backend="memory",
            session_feedback_backend="memory",
        )
    return RuntimeBackendSelection(
        thread_persistence_backend=thread_persistence_backend,
        memory_store_backend=memory_backend,
        crisis_log_backend=crisis_log_persistence_backend,
        session_feedback_backend=session_feedback_persistence_backend,
    )


def effective_thread_persistence_backend(
    *,
    memory_mode: MemoryMode,
    thread_persistence_backend: PersistenceBackend,
) -> PersistenceBackend:
    """Return the effective runtime-state backend for a memory mode.

    Args:
        memory_mode (MemoryMode): Runtime memory mode.
        thread_persistence_backend (PersistenceBackend): Configured persistent
            thread-state backend.

    Returns:
        PersistenceBackend: Backend to use for thread state snapshots.
    """

    return select_runtime_backends(
        memory_mode=memory_mode,
        memory_backend="sqlite",
        thread_persistence_backend=thread_persistence_backend,
        crisis_log_persistence_backend="sqlite",
        session_feedback_persistence_backend="sqlite",
    ).thread_persistence_backend


def create_memory_store(
    *,
    memory_store: MemoryStore | None,
    memory_backend: RuntimeStoreBackend,
    memory_database_url: str | None,
    memory_sqlite_path: str | Path,
) -> MemoryStore:
    """Create the runtime memory store.

    Args:
        memory_store (MemoryStore | None): Optional explicit store override.
        memory_backend (RuntimeStoreBackend): Selected memory backend.
        memory_database_url (str | None): PostgreSQL URL for persistent memory.
        memory_sqlite_path (str | Path): SQLite path for local memory.

    Returns:
        MemoryStore: Configured memory store.

    Raises:
        ValueError: If PostgreSQL memory is selected without a database URL.
    """

    if memory_store is not None:
        return memory_store
    if memory_backend == "memory":
        return OpenCouchMemoryStore()
    if memory_backend == "postgres":
        if not memory_database_url:
            raise ValueError(
                "memory_database_url is required when memory_backend='postgres'"
            )
        return PostgresMemoryStore(memory_database_url)
    return SqliteMemoryStore(memory_sqlite_path)


def create_crisis_log_backend(
    *,
    crisis_log_backend: CrisisLogBackend | None,
    crisis_log_persistence_backend: RuntimeStoreBackend,
    crisis_log_database_url: str | None,
    crisis_log_sqlite_path: str | Path,
) -> CrisisLogBackend:
    """Create the runtime crisis-log backend.

    Args:
        crisis_log_backend (CrisisLogBackend | None): Optional explicit backend
            override.
        crisis_log_persistence_backend (RuntimeStoreBackend): Selected crisis-log
            backend.
        crisis_log_database_url (str | None): PostgreSQL URL for crisis logs.
        crisis_log_sqlite_path (str | Path): SQLite path for local crisis logs.

    Returns:
        CrisisLogBackend: Configured crisis-log backend.

    Raises:
        ValueError: If PostgreSQL crisis logging is selected without a database
            URL.
    """

    if crisis_log_backend is not None:
        return crisis_log_backend
    if crisis_log_persistence_backend == "memory":
        return InMemoryCrisisLogBackend()
    if crisis_log_persistence_backend == "postgres":
        if not crisis_log_database_url:
            raise ValueError(
                "crisis_log_database_url is required when "
                "crisis_log_persistence_backend='postgres'"
            )
        return PostgresCrisisLogBackend(crisis_log_database_url)
    return SqliteCrisisLogBackend(crisis_log_sqlite_path)


def create_session_feedback_backend(
    *,
    session_feedback_backend: SessionFeedbackBackend | None,
    session_feedback_persistence_backend: RuntimeStoreBackend,
    session_feedback_database_url: str | None,
    feedback_sqlite_path: str | Path,
) -> SessionFeedbackBackend:
    """Create the runtime session-feedback backend.

    Args:
        session_feedback_backend (SessionFeedbackBackend | None): Optional
            explicit backend override.
        session_feedback_persistence_backend (RuntimeStoreBackend): Selected
            feedback backend.
        session_feedback_database_url (str | None): PostgreSQL URL for feedback.
        feedback_sqlite_path (str | Path): SQLite path for local feedback.

    Returns:
        SessionFeedbackBackend: Configured session-feedback backend.

    Raises:
        ValueError: If PostgreSQL session feedback is selected without a
            database URL.
    """

    if session_feedback_backend is not None:
        return session_feedback_backend
    if session_feedback_persistence_backend == "memory":
        return InMemorySessionFeedbackBackend()
    if session_feedback_persistence_backend == "postgres":
        if not session_feedback_database_url:
            raise ValueError(
                "session_feedback_database_url is required when "
                "session_feedback_persistence_backend='postgres'"
            )
        return PostgresSessionFeedbackBackend(session_feedback_database_url)
    return SqliteSessionFeedbackBackend(feedback_sqlite_path)


def create_embedding_provider(
    *,
    memory_mode: MemoryMode,
    embedding_provider: EmbeddingProvider | None,
) -> EmbeddingProvider:
    """Create the runtime embedding provider.

    Args:
        memory_mode (MemoryMode): Runtime memory mode.
        embedding_provider (EmbeddingProvider | None): Optional explicit
            provider override.

    Returns:
        EmbeddingProvider: Configured embedding provider.
    """

    if embedding_provider is not None:
        return embedding_provider
    if memory_mode == MemoryMode.INCOGNITO:
        return NullEmbeddingProvider()
    return create_configured_embedding_provider()
