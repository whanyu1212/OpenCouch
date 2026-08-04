"""Backend factory helpers for the persistent agent runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agent.audit.crisis_log import CrisisLogBackend, InMemoryCrisisLogBackend
from agent.audit.postgres_crisis_log import PostgresCrisisLogBackend
from agent.feedback.postgres_session_feedback import PostgresSessionFeedbackBackend
from agent.feedback.session_feedback import (
    InMemorySessionFeedbackBackend,
    SessionFeedbackBackend,
)
from agent.memory.providers.embeddings import (
    EmbeddingProvider,
    NullEmbeddingProvider,
    create_configured_embedding_provider,
)
from agent.memory.modes import (
    EPHEMERAL_CAPABILITY_ATTRIBUTE,
    MemoryMode,
    is_ephemeral_capable,
)
from agent.memory.store.postgres import PostgresMemoryStore
from agent.memory.store import MemoryStore, OpenCouchMemoryStore
from agent.runtime.postgres import require_postgres_database_url

PersistenceBackend = Literal["postgres"]
RuntimeStoreBackend = Literal["memory", "postgres"]
MemoryPersistenceBackend = Literal["postgres"]
MemoryStoreBackend = Literal["memory", "postgres"]


@dataclass(frozen=True, slots=True)
class RuntimeBackendSelection:
    """Effective runtime-owned backends after mode overrides are applied."""

    thread_persistence_backend: RuntimeStoreBackend
    memory_store_backend: MemoryStoreBackend
    crisis_log_backend: RuntimeStoreBackend
    session_feedback_backend: RuntimeStoreBackend


def require_ephemeral_dependency(
    dependency: object,
    *,
    memory_mode: MemoryMode,
    dependency_name: str,
) -> None:
    """Reject an injected dependency an incognito runtime must not use.

    Runtime-owned selection already picks ephemeral backends for incognito,
    but injected dependencies take precedence over that choice. Without this
    check, an incognito runtime silently accepts a durable store or a network
    embedding provider, violating the documented no-persistence contract.

    Validation fails closed: a dependency must declare
    ``supports_incognito = True`` to be accepted. Unrecognized
    implementations — including a caller's own durable wrapper — are rejected
    rather than assumed safe.

    Args:
        dependency (object): Injected dependency, or ``None`` when absent.
        memory_mode (MemoryMode): Runtime memory mode.
        dependency_name (str): Name used in the error message.

    Returns:
        None: Returns when the dependency is allowed.

    Raises:
        ValueError: If an incognito runtime is given a dependency that does
            not declare itself ephemeral.
    """

    if dependency is None or memory_mode != MemoryMode.INCOGNITO:
        return
    if is_ephemeral_capable(dependency):
        return
    raise ValueError(
        f"Incognito runtimes cannot use the injected {dependency_name} "
        f"{type(dependency).__name__!r}: incognito must not write to durable "
        "storage or call remote services. Inject an ephemeral implementation, "
        f"or set '{EPHEMERAL_CAPABILITY_ATTRIBUTE} = True' on the class to "
        "declare that it performs no durable writes and no network calls."
    )


def select_runtime_backends(
    *,
    memory_mode: MemoryMode,
    memory_backend: MemoryPersistenceBackend,
    thread_persistence_backend: RuntimeStoreBackend,
    crisis_log_persistence_backend: PersistenceBackend,
    session_feedback_persistence_backend: PersistenceBackend,
) -> RuntimeBackendSelection:
    """Return effective runtime-owned backends for the configured memory mode."""

    if memory_mode == MemoryMode.INCOGNITO:
        return RuntimeBackendSelection(
            thread_persistence_backend="memory",
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
    thread_persistence_backend: RuntimeStoreBackend,
) -> RuntimeStoreBackend:
    """Return the effective runtime-state backend for a memory mode.

    Args:
        memory_mode (MemoryMode): Runtime memory mode.
        thread_persistence_backend (PersistenceBackend): Configured persistent
            thread-state backend.

    Returns:
        RuntimeStoreBackend: Backend to use for thread state snapshots.
    """

    return select_runtime_backends(
        memory_mode=memory_mode,
        memory_backend="postgres",
        thread_persistence_backend=thread_persistence_backend,
        crisis_log_persistence_backend="postgres",
        session_feedback_persistence_backend="postgres",
    ).thread_persistence_backend


def create_memory_store(
    *,
    memory_store: MemoryStore | None,
    memory_backend: MemoryStoreBackend,
    memory_database_url: str | None,
    memory_mode: MemoryMode = MemoryMode.LOCAL,
) -> MemoryStore:
    """Create the runtime memory store.

    Args:
        memory_store (MemoryStore | None): Optional explicit store override.
        memory_backend (MemoryStoreBackend): Selected memory backend.
        memory_database_url (str | None): PostgreSQL URL for persistent memory.

    Returns:
        MemoryStore: Configured memory store.

    Raises:
        ValueError: If PostgreSQL memory is selected without a database URL.
    """

    require_ephemeral_dependency(
        memory_store,
        memory_mode=memory_mode,
        dependency_name="memory store",
    )
    if memory_store is not None:
        return memory_store
    if memory_backend == "memory":
        return OpenCouchMemoryStore()
    if memory_backend == "postgres":
        return PostgresMemoryStore(require_postgres_database_url(memory_database_url))
    raise ValueError(f"Unsupported memory backend: {memory_backend}")


def create_crisis_log_backend(
    *,
    crisis_log_backend: CrisisLogBackend | None,
    crisis_log_persistence_backend: RuntimeStoreBackend,
    crisis_log_database_url: str | None,
    memory_mode: MemoryMode = MemoryMode.LOCAL,
) -> CrisisLogBackend:
    """Create the runtime crisis-log backend.

    Args:
        crisis_log_backend (CrisisLogBackend | None): Optional explicit backend
            override.
        crisis_log_persistence_backend (RuntimeStoreBackend): Selected crisis-log
            backend.
        crisis_log_database_url (str | None): PostgreSQL URL for crisis logs.

    Returns:
        CrisisLogBackend: Configured crisis-log backend.

    Raises:
        ValueError: If PostgreSQL crisis logging is selected without a database
            URL.
    """

    require_ephemeral_dependency(
        crisis_log_backend,
        memory_mode=memory_mode,
        dependency_name="crisis-log backend",
    )
    if crisis_log_backend is not None:
        return crisis_log_backend
    if crisis_log_persistence_backend == "memory":
        return InMemoryCrisisLogBackend()
    if crisis_log_persistence_backend == "postgres":
        return PostgresCrisisLogBackend(
            require_postgres_database_url(crisis_log_database_url)
        )
    raise ValueError(
        f"Unsupported crisis-log backend: {crisis_log_persistence_backend}"
    )


def create_session_feedback_backend(
    *,
    session_feedback_backend: SessionFeedbackBackend | None,
    session_feedback_persistence_backend: RuntimeStoreBackend,
    session_feedback_database_url: str | None,
    memory_mode: MemoryMode = MemoryMode.LOCAL,
) -> SessionFeedbackBackend:
    """Create the runtime session-feedback backend.

    Args:
        session_feedback_backend (SessionFeedbackBackend | None): Optional
            explicit backend override.
        session_feedback_persistence_backend (RuntimeStoreBackend): Selected
            feedback backend.
        session_feedback_database_url (str | None): PostgreSQL URL for feedback.

    Returns:
        SessionFeedbackBackend: Configured session-feedback backend.

    Raises:
        ValueError: If PostgreSQL session feedback is selected without a
            database URL.
    """

    require_ephemeral_dependency(
        session_feedback_backend,
        memory_mode=memory_mode,
        dependency_name="session-feedback backend",
    )
    if session_feedback_backend is not None:
        return session_feedback_backend
    if session_feedback_persistence_backend == "memory":
        return InMemorySessionFeedbackBackend()
    if session_feedback_persistence_backend == "postgres":
        return PostgresSessionFeedbackBackend(
            require_postgres_database_url(session_feedback_database_url)
        )
    raise ValueError(
        f"Unsupported session-feedback backend: {session_feedback_persistence_backend}"
    )


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

    require_ephemeral_dependency(
        embedding_provider,
        memory_mode=memory_mode,
        dependency_name="embedding provider",
    )
    if embedding_provider is not None:
        return embedding_provider
    if memory_mode == MemoryMode.INCOGNITO:
        return NullEmbeddingProvider()
    return create_configured_embedding_provider()
