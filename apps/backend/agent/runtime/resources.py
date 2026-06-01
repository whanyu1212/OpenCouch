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
    PersistenceBackend,
    create_crisis_log_backend,
    create_embedding_provider,
    create_memory_store,
    create_session_feedback_backend,
    effective_thread_persistence_backend,
)
from agent.runtime.session.active_session import (
    ActiveSessionManager,
    PostgresActiveSessionStore,
    SqliteActiveSessionStore,
)
from agent.runtime.session_store import (
    TextSessionBackend,
    TextSessionStore,
    create_text_session_store,
)
from agent.runtime.state_store import RuntimeStateStore, create_runtime_state_store

logger = logging.getLogger(__name__)

DEFAULT_TEXT_SESSION_DB_FILENAME = "text_sessions.sqlite3"


@dataclass(slots=True)
class RuntimeResources:
    """Runtime-owned storage backends and warmup/close helpers."""

    sqlite_path: Path
    thread_persistence_backend: PersistenceBackend
    thread_database_url: str | None
    state_store: RuntimeStateStore
    text_session_store: TextSessionStore | None
    memory_store: MemoryStore
    crisis_log_backend: CrisisLogBackend
    session_feedback_backend: SessionFeedbackBackend
    embedding_provider: EmbeddingProvider
    active_session_store: PostgresActiveSessionStore | SqliteActiveSessionStore
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
    sqlite_path: str | Path,
    text_session_sqlite_path: str | Path | None,
    thread_persistence_backend: PersistenceBackend,
    thread_database_url: str | None,
    text_session_backend: TextSessionBackend,
    text_session_database_url: str | None,
    text_session_create_tables: bool,
    text_session_history_limit: int | None,
    memory_store: MemoryStore | None,
    memory_backend: PersistenceBackend,
    memory_database_url: str | None,
    memory_sqlite_path: str | Path,
    crisis_log_backend: CrisisLogBackend | None,
    crisis_log_persistence_backend: PersistenceBackend,
    crisis_log_database_url: str | None,
    crisis_log_sqlite_path: str | Path,
    session_feedback_backend: SessionFeedbackBackend | None,
    session_feedback_persistence_backend: PersistenceBackend,
    session_feedback_database_url: str | None,
    feedback_sqlite_path: str | Path,
    embedding_provider: EmbeddingProvider | None,
    session_timeout,
) -> RuntimeResources:
    """Build the runtime-owned stores, backends, and providers."""
    is_incognito = memory_mode == MemoryMode.INCOGNITO
    resolved_sqlite = ":memory:" if is_incognito else sqlite_path
    resolved_runtime_sqlite_path = (
        Path(resolved_sqlite) if resolved_sqlite != ":memory:" else Path(":memory:")
    )

    if text_session_sqlite_path is not None:
        resolved_text_session_sqlite_path = text_session_sqlite_path
    elif resolved_sqlite == ":memory:":
        resolved_text_session_sqlite_path = ":memory:"
    else:
        resolved_text_session_sqlite_path = Path(resolved_sqlite).with_name(
            DEFAULT_TEXT_SESSION_DB_FILENAME
        )

    resolved_thread_persistence_backend = effective_thread_persistence_backend(
        memory_mode=memory_mode,
        thread_persistence_backend=thread_persistence_backend,
    )

    state_store = create_runtime_state_store(
        backend=resolved_thread_persistence_backend,
        sqlite_path=resolved_runtime_sqlite_path,
        database_url=thread_database_url,
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
        memory_mode=memory_mode,
        memory_store=memory_store,
        memory_backend=memory_backend,
        memory_database_url=memory_database_url,
        memory_sqlite_path=memory_sqlite_path,
    )
    resolved_crisis_log_backend = create_crisis_log_backend(
        memory_mode=memory_mode,
        crisis_log_backend=crisis_log_backend,
        crisis_log_persistence_backend=crisis_log_persistence_backend,
        crisis_log_database_url=crisis_log_database_url,
        crisis_log_sqlite_path=crisis_log_sqlite_path,
    )
    resolved_session_feedback_backend = create_session_feedback_backend(
        memory_mode=memory_mode,
        session_feedback_backend=session_feedback_backend,
        session_feedback_persistence_backend=session_feedback_persistence_backend,
        session_feedback_database_url=session_feedback_database_url,
        feedback_sqlite_path=feedback_sqlite_path,
    )
    resolved_embedding_provider = create_embedding_provider(
        memory_mode=memory_mode,
        embedding_provider=embedding_provider,
    )

    if resolved_thread_persistence_backend == "postgres":
        if not thread_database_url:
            raise ValueError(
                "thread_database_url is required when "
                "thread_persistence_backend='postgres'"
            )
        active_session_store = PostgresActiveSessionStore(dsn=thread_database_url)
    else:
        active_session_store = SqliteActiveSessionStore(
            sqlite_path=resolved_runtime_sqlite_path
        )

    active_session_manager = ActiveSessionManager(
        store=active_session_store,
        memory_mode=memory_mode,
        session_timeout=session_timeout,
    )

    return RuntimeResources(
        sqlite_path=resolved_runtime_sqlite_path,
        thread_persistence_backend=resolved_thread_persistence_backend,
        thread_database_url=thread_database_url,
        state_store=state_store,
        text_session_store=text_session_store,
        memory_store=resolved_memory_store,
        crisis_log_backend=resolved_crisis_log_backend,
        session_feedback_backend=resolved_session_feedback_backend,
        embedding_provider=resolved_embedding_provider,
        active_session_store=active_session_store,
        active_session_manager=active_session_manager,
    )
