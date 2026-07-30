"""Runtime resource bootstrap and lifecycle helpers."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

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


class _Closable(Protocol):
    """Minimal shape shared by every runtime-owned closable resource."""

    async def aclose(self) -> None: ...


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
        """Create runtime-owned tables before serving traffic.

        Every durable backend the runtime owns is prepared here so that
        request-time operations perform data work rather than connecting and
        migrating. This matters most for the crisis log, whose appends run
        inside a bounded safety-capture timeout, and for the memory store,
        whose preparation includes a vector-column backfill.

        Preparation is not atomic across backends: if one fails, the ones
        already prepared hold open connections. Close them before propagating
        so a failed startup does not leak resources.

        Returns:
            None: Prepares every runtime-owned durable backend.

        Raises:
            Exception: Re-raises the first preparation failure after unwinding.
        """

        prepared: list[str] = []
        try:
            await self.state_store.ensure_schema()
            prepared.append("state_store")
            await self.active_session_manager.ensure_schema()
            prepared.append("active_session_manager")
            await self.memory_store.ensure_schema()
            prepared.append("memory_store")
            await self.crisis_log_backend.ensure_schema()
            prepared.append("crisis_log_backend")
            await self.session_feedback_backend.ensure_schema()
        except BaseException:
            logger.warning(
                "RuntimeResources.ensure_schema: preparation failed after "
                "preparing %s; closing opened resources.",
                ", ".join(prepared) or "no backends",
                exc_info=True,
            )
            await self._aclose_quietly()
            raise

    async def _aclose_quietly(self) -> None:
        """Close runtime-owned resources, logging rather than raising.

        Used when unwinding a failed startup: the original failure must
        propagate, so cleanup errors are logged and swallowed instead of
        masking it. ``aclose`` already releases each resource independently,
        so one failure cannot strand the rest.

        Returns:
            None: Closes what can be closed.
        """

        try:
            await self.aclose()
        except Exception:
            logger.warning(
                "RuntimeResources: cleanup after failed startup raised; "
                "ignoring so the original failure propagates.",
                exc_info=True,
            )

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
        """Close runtime-owned resources in deterministic order.

        Each resource is closed independently: one backend raising must not
        strand the ones after it, since that would leak connections during
        both normal shutdown and startup unwinding. The first failure is
        re-raised once every resource has been given a chance to close.

        Returns:
            None: Closes every runtime-owned resource.

        Raises:
            Exception: Re-raises the first close failure, if any.
        """

        closables: list[tuple[str, _Closable]] = [
            ("memory_store", self.memory_store),
            ("crisis_log_backend", self.crisis_log_backend),
            ("session_feedback_backend", self.session_feedback_backend),
        ]
        if self.text_session_store is not None:
            closables.append(("text_session_store", self.text_session_store))
        closables.append(("state_store", self.state_store))
        closables.append(("active_session_store", self.active_session_store))

        first_failure: BaseException | None = None
        for name, closable in closables:
            try:
                await closable.aclose()
            except Exception as exc:
                logger.warning(
                    "RuntimeResources.aclose: closing %s raised; continuing "
                    "so remaining resources are still released.",
                    name,
                    exc_info=True,
                )
                if first_failure is None:
                    first_failure = exc

        if first_failure is not None:
            raise first_failure


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
