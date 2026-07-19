"""Runtime resource bootstrap and lifecycle helpers."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from agent.audit.crisis_log import CrisisLogBackend
from agent.feedback.session_feedback import SessionFeedbackBackend
from agent.memory.modes import MemoryMode
from agent.memory.providers.embeddings import EmbeddingProvider
from agent.memory.store import MemoryStore
from agent.runtime.backends import (
    MemoryPersistenceBackend,
    PersistenceBackend,
    RuntimeStoreBackend,
    create_crisis_log_backend,
    create_embedding_provider,
    create_memory_store,
    create_session_feedback_backend,
    select_runtime_backends,
)
from agent.runtime.session.active_session import (
    ActiveSessionManager,
    ActiveSessionStore,
    InMemoryActiveSessionStore,
    NullActiveSessionStore,
    PostgresActiveSessionStore,
)
from agent.runtime.postgres import require_postgres_database_url
from agent.runtime.configuration import DEFAULT_TEXT_SESSION_DB_PATH
from agent.runtime.session_store import (
    TextSessionBackend,
    TextSessionStore,
    create_text_session_store,
)
from agent.runtime.state_store import RuntimeStateStore, create_runtime_state_store

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RuntimeResources:
    """Runtime-owned storage backends and warmup/close helpers."""

    thread_persistence_backend: RuntimeStoreBackend
    thread_database_url: str | None
    state_store: RuntimeStateStore
    text_session_store: TextSessionStore | None
    memory_store: MemoryStore
    crisis_log_backend: CrisisLogBackend
    session_feedback_backend: SessionFeedbackBackend
    embedding_provider: EmbeddingProvider
    active_session_store: ActiveSessionStore
    active_session_manager: ActiveSessionManager

    async def ensure_schema(self) -> None:
        """Create runtime-owned tables."""
        await self.state_store.ensure_schema()
        await self.active_session_manager.ensure_schema()

    async def prewarm(self, *, get_text_runtime: Callable[[], object]) -> None:
        """Warm runtime resources before the first user turn."""
        get_text_runtime()

        try:
            await asyncio.wait_for(self.embedding_provider.awarmup(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(
                "PersistentAgentRuntime prewarm: embedding warmup timed out; "
                "continuing with cold provider."
            )
        except Exception:
            logger.warning(
                "PersistentAgentRuntime prewarm: embedding warmup failed; "
                "continuing with cold provider.",
                exc_info=True,
            )

    async def aclose(self) -> None:
        """Close runtime-owned resources in deterministic order."""
        await self.memory_store.aclose()
        await self.crisis_log_backend.aclose()
        await self.session_feedback_backend.aclose()
        if self.text_session_store is not None:
            await self.text_session_store.aclose()
        await self.state_store.aclose()
        await self.active_session_store.aclose()


def build_runtime_resources(
    *,
    memory_mode: MemoryMode,
    text_session_sqlite_path: str | Path | None,
    thread_persistence_backend: RuntimeStoreBackend,
    thread_database_url: str | None,
    text_session_backend: TextSessionBackend,
    text_session_database_url: str | None,
    text_session_create_tables: bool,
    text_session_history_limit: int | None,
    memory_store: MemoryStore | None,
    memory_backend: MemoryPersistenceBackend,
    memory_database_url: str | None,
    crisis_log_backend: CrisisLogBackend | None,
    crisis_log_persistence_backend: PersistenceBackend,
    crisis_log_database_url: str | None,
    session_feedback_backend: SessionFeedbackBackend | None,
    session_feedback_persistence_backend: PersistenceBackend,
    session_feedback_database_url: str | None,
    embedding_provider: EmbeddingProvider | None,
) -> RuntimeResources:
    """Build internal runtime-owned stores, backends, and providers."""
    is_incognito = memory_mode == MemoryMode.INCOGNITO
    if is_incognito:
        resolved_text_session_sqlite_path = ":memory:"
    elif text_session_sqlite_path is not None:
        resolved_text_session_sqlite_path = text_session_sqlite_path
    else:
        resolved_text_session_sqlite_path = DEFAULT_TEXT_SESSION_DB_PATH

    backend_selection = select_runtime_backends(
        memory_mode=memory_mode,
        memory_backend=memory_backend,
        thread_persistence_backend=thread_persistence_backend,
        crisis_log_persistence_backend=crisis_log_persistence_backend,
        session_feedback_persistence_backend=session_feedback_persistence_backend,
    )

    resolved_thread_database_url = (
        require_postgres_database_url(thread_database_url)
        if backend_selection.thread_persistence_backend == "postgres"
        else thread_database_url
    )

    state_store = create_runtime_state_store(
        backend=backend_selection.thread_persistence_backend,
        database_url=resolved_thread_database_url,
    )
    text_session_store = create_text_session_store(
        memory_mode=memory_mode,
        backend=text_session_backend,
        sqlite_path=resolved_text_session_sqlite_path,
        database_url=text_session_database_url,
        create_tables=text_session_create_tables,
        history_limit=text_session_history_limit,
    )
    resolved_memory_store = create_memory_store(
        memory_store=memory_store,
        memory_backend=backend_selection.memory_store_backend,
        memory_database_url=memory_database_url,
    )
    resolved_crisis_log_backend = create_crisis_log_backend(
        crisis_log_backend=crisis_log_backend,
        crisis_log_persistence_backend=backend_selection.crisis_log_backend,
        crisis_log_database_url=crisis_log_database_url,
    )
    resolved_session_feedback_backend = create_session_feedback_backend(
        session_feedback_backend=session_feedback_backend,
        session_feedback_persistence_backend=backend_selection.session_feedback_backend,
        session_feedback_database_url=session_feedback_database_url,
    )
    resolved_embedding_provider = create_embedding_provider(
        memory_mode=memory_mode,
        embedding_provider=embedding_provider,
    )

    if backend_selection.thread_persistence_backend == "postgres":
        active_session_store: ActiveSessionStore = PostgresActiveSessionStore(
            dsn=resolved_thread_database_url
        )
    elif backend_selection.thread_persistence_backend == "memory":
        active_session_store = (
            NullActiveSessionStore()
            if memory_mode == MemoryMode.INCOGNITO
            else InMemoryActiveSessionStore()
        )
    else:
        raise ValueError(
            "Unsupported active-session backend: "
            f"{backend_selection.thread_persistence_backend}"
        )

    active_session_manager = ActiveSessionManager(
        store=active_session_store,
        memory_mode=memory_mode,
    )

    return RuntimeResources(
        thread_persistence_backend=backend_selection.thread_persistence_backend,
        thread_database_url=resolved_thread_database_url,
        state_store=state_store,
        text_session_store=text_session_store,
        memory_store=resolved_memory_store,
        crisis_log_backend=resolved_crisis_log_backend,
        session_feedback_backend=resolved_session_feedback_backend,
        embedding_provider=resolved_embedding_provider,
        active_session_store=active_session_store,
        active_session_manager=active_session_manager,
    )
