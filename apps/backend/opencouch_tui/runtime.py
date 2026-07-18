"""Shared runtime adapter for OpenCouch terminal surfaces."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError

from agent.memory.modes import MemoryMode
from agent.memory.notebook import build_memory_notebook
from agent.memory.store import MemoryStore, Namespace, StoreRecord
from agent.memory.types import SemanticFact, StoredSessionArc
from agent.models import Message, StreamEvent
from agent.runtime import (
    DEFAULT_CRISIS_LOG_DB_PATH,
    DEFAULT_MEMORY_DB_PATH,
    DEFAULT_THREAD_DB_PATH,
    PersistentAgentRuntime,
    RuntimeBehaviorConfig,
    RuntimeDependencies,
    RuntimePersistenceConfig,
    RuntimeStoragePaths,
    ThreadSummary,
)
from agent.state import AgentState
from config import (
    PersistenceBackend,
    ResponseModelTier,
    Settings,
    create_configured_control_llm_client,
    create_configured_response_llm_client,
    get_settings,
)
from llm.base import BaseLLMClient

MemoryModeName = Literal["guest", "persistent"]


async def _load_memory_snapshot_records(
    store: MemoryStore,
    namespace: Namespace,
    *,
    complete: bool,
) -> list[StoreRecord]:
    if not complete:
        return await store.asearch(namespace, query=None)
    record_count = await store.arecord_count(namespace)
    if record_count == 0:
        return []
    return await store.asearch(namespace, query=None, limit=record_count)


def _has_visible_unprojected_records(
    records: list[StoreRecord],
    model_type: type[SemanticFact] | type[StoredSessionArc],
) -> bool:
    for record in records:
        value = record.value
        if isinstance(value, dict) and (
            not value.get("user_visible", True)
            or value.get("dormant_at")
            or value.get("superseded_by")
        ):
            continue
        try:
            model_type.model_validate(value)
        except ValidationError:
            return True
    return False


@dataclass(slots=True)
class ConsoleConfig:
    """Configuration shared by local terminal console surfaces."""

    requested_mode: str = "auto"
    thread_id: str = "local-tui"
    user_id: str | None = None
    response_model_tier: ResponseModelTier = "fast"
    sqlite_path: str = str(DEFAULT_THREAD_DB_PATH)
    memory_mode: MemoryModeName = "guest"
    memory_sqlite_path: str = str(DEFAULT_MEMORY_DB_PATH)
    crisis_log_sqlite_path: str = str(DEFAULT_CRISIS_LOG_DB_PATH)


@dataclass
class ConsoleSession:
    """Mutable runtime/session state exposed to terminal UIs."""

    requested_mode: str
    resolved_mode: str
    thread_id: str
    memory_mode: MemoryModeName
    persistence_backend: PersistenceBackend
    user_id: str | None
    response_model_tier: ResponseModelTier
    llm_client: BaseLLMClient | None
    response_llm_client: BaseLLMClient | None
    history: list[Message]
    last_context: AgentState | None

    @property
    def owner_id(self) -> str:
        """Return the effective owner identifier for memory operations.

        Returns the explicit ``user_id`` when set, otherwise falls back to
        ``thread_id``. Mirrors ``RunnerSession.owner_id()`` in
        ``opencouch_tui.models`` so both session types use identical
        ownership semantics.
        """

        return self.user_id or self.thread_id


@dataclass(frozen=True, slots=True)
class ConsoleErrorEvent:
    """Recoverable runtime error surfaced as a stream event."""

    prefix: str
    message: str
    exception_type: str


ConsoleStreamEvent = StreamEvent | ConsoleErrorEvent


class ConsoleRuntime:
    """In-process runtime adapter for REPL/TUI-style terminal surfaces."""

    def __init__(
        self,
        config: ConsoleConfig,
        *,
        settings: Settings | None = None,
    ) -> None:
        self.config = config
        self._settings = settings
        self._runtime: PersistentAgentRuntime | None = None
        self.session: ConsoleSession | None = None

    async def __aenter__(self) -> ConsoleRuntime:
        """Create the persistent runtime and load initial session state."""

        settings = self._settings or get_settings()
        llm_client, resolved_mode = resolve_llm_client(
            self.config.requested_mode,
            settings=settings,
        )
        response_llm_client = (
            resolve_response_llm_client(
                self.config.requested_mode,
                self.config.response_model_tier,
                settings=settings,
            )
            if llm_client is not None
            else None
        )

        runtime_memory_mode = (
            MemoryMode.INCOGNITO
            if self.config.memory_mode == "guest"
            else MemoryMode.LOCAL
        )
        is_guest_mode = runtime_memory_mode == MemoryMode.INCOGNITO
        effective_user_id = None if is_guest_mode else self.config.user_id

        if settings.persistence_backend == "postgres":
            persistence_config = RuntimePersistenceConfig.for_shared_backend(
                memory_mode=runtime_memory_mode,
                persistence_backend="postgres",
                database_url=settings.memory_database_url,
                text_session_backend=settings.text_session_backend,
                text_session_database_url=settings.text_session_database_url,
                allow_legacy_sqlite=settings.allow_legacy_sqlite,
            )
        else:
            persistence_config = RuntimePersistenceConfig(
                memory_mode=runtime_memory_mode,
                memory_backend="sqlite",
                thread_persistence_backend="memory",
                crisis_log_persistence_backend="sqlite",
                session_feedback_persistence_backend="sqlite",
                text_session_backend=settings.text_session_backend,
                text_session_database_url=settings.text_session_database_url,
                allow_legacy_sqlite=settings.allow_legacy_sqlite,
            )

        runtime = PersistentAgentRuntime(
            storage_paths=RuntimeStoragePaths(
                sqlite_path=":memory:" if is_guest_mode else self.config.sqlite_path,
                memory_sqlite_path=self.config.memory_sqlite_path,
                crisis_log_sqlite_path=self.config.crisis_log_sqlite_path,
            ),
            persistence_config=persistence_config,
            dependencies=RuntimeDependencies(default_llm_client=llm_client),
            behavior_config=RuntimeBehaviorConfig(
                finalize_active_sessions_on_close=False,
            ),
        )
        self._runtime = await runtime.__aenter__()
        history = await self._runtime.get_history(self.config.thread_id)
        last_context = await self._runtime.get_state(self.config.thread_id)
        self.session = ConsoleSession(
            requested_mode=self.config.requested_mode,
            resolved_mode=resolved_mode,
            thread_id=self.config.thread_id,
            memory_mode=self.config.memory_mode,
            persistence_backend=settings.persistence_backend,
            user_id=effective_user_id,
            response_model_tier=self.config.response_model_tier,
            llm_client=llm_client,
            response_llm_client=response_llm_client,
            history=history,
            last_context=last_context,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Close the underlying runtime context."""

        if self._runtime is not None:
            await self._runtime.__aexit__(exc_type, exc, tb)
        self._runtime = None

    async def run_turn_stream(self, message: str) -> AsyncIterator[ConsoleStreamEvent]:
        """Run a user turn and surface recoverable runtime errors as events."""

        runtime = self._require_runtime()
        session = self._require_session()
        try:
            async for event in runtime.run_turn_stream(
                thread_id=session.thread_id,
                user_id=session.owner_id,
                message=message,
                llm_client=session.llm_client,
                response_llm_client=session.response_llm_client,
            ):
                yield event
        except Exception as exc:
            yield ConsoleErrorEvent(
                prefix="Turn failed",
                message=_recoverable_error_message("Turn failed", exc),
                exception_type=type(exc).__name__,
            )
        finally:
            await self.refresh()

    async def refresh(self) -> None:
        """Refresh session history and latest persisted state."""

        runtime = self._require_runtime()
        session = self._require_session()
        session.history = await runtime.get_history(session.thread_id)
        session.last_context = await runtime.get_state(session.thread_id)

    async def thread_exists(self, thread_id: str) -> bool:
        """Return whether a persisted thread already exists."""

        runtime = self._require_runtime()
        state = await runtime.get_state(thread_id)
        history = await runtime.get_history(thread_id)
        return state is not None or bool(history)

    async def switch_thread(
        self,
        thread_id: str,
        *,
        require_existing: bool = False,
    ) -> bool:
        """Switch the active session thread and refresh persisted state."""

        session = self._require_session()
        if require_existing and not await self.thread_exists(thread_id):
            return False
        session.thread_id = thread_id
        await self.refresh()
        return True

    async def list_threads(self, *, limit: int = 12) -> list[ThreadSummary]:
        """Return recent persisted thread summaries."""

        runtime = self._require_runtime()
        return await runtime.list_threads(limit=limit)

    async def load_memory_snapshot(
        self, *, include_notebook: bool = True
    ) -> dict[str, Any]:
        """Return semantic, episodic, and procedural memory for the active owner."""

        runtime = self._require_runtime()
        session = self._require_session()
        owner_id = session.owner_id
        semantic = await _load_memory_snapshot_records(
            runtime.memory_store,
            (owner_id, "semantic"),
            complete=include_notebook,
        )
        episodic = await _load_memory_snapshot_records(
            runtime.memory_store,
            (owner_id, "episodic"),
            complete=include_notebook,
        )
        procedural = await runtime.memory_store.aget(
            (owner_id, "procedural"),
            "user_response_style",
        )
        notebook = None
        has_unprojected_legacy_memory = False
        if include_notebook:
            has_unprojected_legacy_memory = _has_visible_unprojected_records(
                semantic, SemanticFact
            ) or _has_visible_unprojected_records(episodic, StoredSessionArc)
            try:
                notebook = (
                    await build_memory_notebook(runtime.memory_store, owner_id=owner_id)
                ).model_dump(mode="json")
            except Exception:
                notebook = None
        return {
            "owner_id": owner_id,
            "semantic": [record.value for record in semantic],
            "episodic": [record.value for record in episodic],
            "procedural": procedural.value if procedural is not None else None,
            "notebook": notebook,
            "has_unprojected_legacy_memory": has_unprojected_legacy_memory,
        }

    def _require_runtime(self) -> PersistentAgentRuntime:
        if self._runtime is None:
            raise RuntimeError("ConsoleRuntime has not been entered.")
        return self._runtime

    def _require_session(self) -> ConsoleSession:
        if self.session is None:
            raise RuntimeError("ConsoleRuntime has not been entered.")
        return self.session


def resolve_llm_client(
    mode: str,
    *,
    settings: Settings | None = None,
) -> tuple[BaseLLMClient | None, str]:
    """Resolve the control-plane LLM client for a terminal console mode."""

    if mode == "deterministic":
        return None, "deterministic"
    if mode == "hybrid":
        return create_configured_control_llm_client(settings=settings), "hybrid"
    try:
        return create_configured_control_llm_client(settings=settings), "hybrid"
    except Exception:
        return None, "deterministic"


def resolve_response_llm_client(
    mode: str,
    tier: ResponseModelTier,
    *,
    settings: Settings | None = None,
) -> BaseLLMClient | None:
    """Resolve the response-writing LLM client for a terminal console mode."""

    if mode == "deterministic":
        return None
    try:
        return create_configured_response_llm_client(tier, settings=settings)
    except Exception:
        return None


def recoverable_error_message(prefix: str, exc: Exception) -> str:
    """Return a compact user-facing error for recoverable terminal failures."""

    detail = str(exc).strip() or type(exc).__name__
    return (
        f"{prefix}: {detail}\n"
        "The session stayed open. Fix the runtime configuration or retry the turn."
    )


# Legacy private alias kept for any internal callers.
_recoverable_error_message = recoverable_error_message
