"""Persistent thread runtime for the fresh START -> load_memory -> END graph."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent.graph import build_agent_workflow, build_initial_state, state_to_output
from agent.memory.candidates import SessionMemoryBuffer
from agent.audit.crisis_log import CrisisLogBackend, InMemoryCrisisLogBackend
from agent.audit.session_feedback import (
    InMemorySessionFeedbackBackend,
    SessionFeedbackBackend,
)
from agent.audit.sqlite_crisis_log import SqliteCrisisLogBackend
from agent.audit.sqlite_session_feedback import SqliteSessionFeedbackBackend
from agent.memory.hashing import hash_session_id, iso_now
from agent.memory.hashing import iso_now as _iso_now
from agent.memory.embeddings import (
    EmbeddingProvider,
    NullEmbeddingProvider,
    create_configured_embedding_provider,
)
from agent.memory.models import (
    FeedbackLabel,
    FeedbackSource,
    SessionFeedbackRecord,
    StoredSessionArc,
)
from agent.memory.modes import MemoryMode
from agent.memory.sqlite_store import SqliteMemoryStore
from agent.memory.store import MemoryStore, OpenCouchMemoryStore
from agent.models import (
    AgentInput,
    AgentOutput,
    Channel,
    ChunkEvent,
    DoneEvent,
    Message,
    MessageRole,
    ResponseReadyEvent,
    StatusEvent,
    StreamEvent,
)
from agent.nodes.commit_session_memory import run_commit_session_memory
from agent.nodes.extract_facts import run_extract_semantic_facts_node
from agent.nodes.extract_procedural_rules import run_extract_procedural_rules_node
from agent.nodes.summarize_session import run_summarize_session
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph
from services.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[1]
_STORE_DIR = BACKEND_ROOT / ".store"
DEFAULT_THREAD_DB_PATH = _STORE_DIR / "threads.sqlite3"
# v0.8: SQLite file paths for the memory store and crisis log.
# Kept separate from the thread checkpointer file so LangGraph owns
# its schema and we own ours — no cross-table coupling or shared-
# transaction surprises when LangGraph bumps its schema. All three
# live under ``.store/`` to keep the backend root clean.
DEFAULT_MEMORY_DB_PATH = _STORE_DIR / "memory.sqlite3"
DEFAULT_CRISIS_LOG_DB_PATH = _STORE_DIR / "crisis.sqlite3"
# v0.10: session-feedback SQLite path. Fourth file under ``.store/``,
# separate from the other three for the same isolation reasons.
DEFAULT_FEEDBACK_DB_PATH = _STORE_DIR / "session_feedback.sqlite3"
ALLOWED_MSGPACK_MODULES = [
    ("agent.models", "Channel"),
    ("agent.models", "CrisisAssessment"),
    ("agent.models", "ModeType"),
    ("agent.models", "ResponseCategory"),
    ("agent.models", "ResponseKind"),
]
SESSION_TIMEOUT = timedelta(minutes=20)
ACTIVE_SESSION_STATE_DDL = """
CREATE TABLE IF NOT EXISTS opencouch_active_sessions (
    thread_id TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL
);
"""


@dataclass(slots=True)
class PersistentTurnResult:
    """Return value for one persisted conversation turn."""

    output: AgentOutput
    state: AgentState
    history: list[Message]


@dataclass(slots=True)
class ThreadSummary:
    """Compact persisted-thread summary for CLI thread management."""

    thread_id: str
    turn_count: int
    message_count: int
    has_context: bool


@dataclass(slots=True)
class _RuntimeShim:
    """Minimal runtime shim for direct node reuse outside LangGraph."""

    context: WorkflowContext


@dataclass(slots=True)
class PersistedActiveSessionState:
    """Durable runtime-owned session state for Phase E recovery."""

    thread_id: str
    started_at: str
    last_active_at: str
    transcript_start_index: int
    max_crisis_level: int
    session_buffer: SessionMemoryBuffer

    def to_json(self) -> str:
        return json.dumps(
            {
                "thread_id": self.thread_id,
                "started_at": self.started_at,
                "last_active_at": self.last_active_at,
                "transcript_start_index": self.transcript_start_index,
                "max_crisis_level": self.max_crisis_level,
                "session_buffer": self.session_buffer.model_dump(mode="json"),
            }
        )

    @classmethod
    def from_json(cls, payload_json: str) -> PersistedActiveSessionState:
        payload = json.loads(payload_json)
        return cls(
            thread_id=str(payload["thread_id"]),
            started_at=str(payload["started_at"]),
            last_active_at=str(payload["last_active_at"]),
            transcript_start_index=max(
                0, int(payload.get("transcript_start_index", 0) or 0)
            ),
            max_crisis_level=max(0, int(payload.get("max_crisis_level", 0) or 0)),
            session_buffer=SessionMemoryBuffer.model_validate(
                payload.get("session_buffer")
                or {"session_id": str(payload["thread_id"])}
            ),
        )


def _parse_iso_timestamp(value: str | None) -> datetime | None:
    """Parse a stored ISO timestamp, accepting ``Z`` UTC suffixes."""

    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class PersistentAgentRuntime:
    """Thread-backed runtime with incognito, local, or synced memory mode.

    v0.8 changed the default storage layer. Prior to v0.8, every mode
    used in-memory backings regardless of the ``memory_mode`` flag —
    the mode only affected the LangGraph conversation checkpointer's
    SQLite path. Now all three storage layers (thread checkpoint,
    memory store, crisis log) have mode-aware defaults:

    - ``INCOGNITO`` → every layer uses ``:memory:`` SQLite (or the
      in-memory sibling class for the memory store / crisis log).
      Nothing touches disk. Closes the incognito privacy contract.
    - ``LOCAL`` / ``SYNCED`` → thread checkpointer uses
      :data:`DEFAULT_THREAD_DB_PATH`, memory store uses
      :data:`DEFAULT_MEMORY_DB_PATH`, crisis log uses
      :data:`DEFAULT_CRISIS_LOG_DB_PATH`. All three are file-backed
      and survive CLI restarts.

    Explicit overrides still work for tests: if the caller passes
    ``memory_store=OpenCouchMemoryStore()`` or a specific
    ``crisis_log_backend``, the runtime uses that instance and
    doesn't open a SQLite connection for that layer. The existing
    test fixtures that construct in-memory backings directly
    continue to work unchanged.
    """

    def __init__(
        self,
        sqlite_path: str | Path = DEFAULT_THREAD_DB_PATH,
        *,
        memory_store: MemoryStore | None = None,
        crisis_log_backend: CrisisLogBackend | None = None,
        session_feedback_backend: SessionFeedbackBackend | None = None,
        memory_mode: MemoryMode = MemoryMode.LOCAL,
        memory_sqlite_path: str | Path = DEFAULT_MEMORY_DB_PATH,
        crisis_log_sqlite_path: str | Path = DEFAULT_CRISIS_LOG_DB_PATH,
        feedback_sqlite_path: str | Path = DEFAULT_FEEDBACK_DB_PATH,
        embedding_provider: "EmbeddingProvider | None" = None,
        default_llm_client: BaseLLMClient | None = None,
        session_timeout: timedelta = SESSION_TIMEOUT,
        session_sweep_interval_seconds: float = 30.0,
        finalize_active_sessions_on_close: bool = True,
    ) -> None:
        """Initialize the runtime.

        Args:
            sqlite_path: SQLite database path for LangGraph checkpoints.
                Forced to ``:memory:`` in incognito mode.
            memory_store: Optional unified memory store. If None, the
                runtime picks an implementation based on ``memory_mode``:
                :class:`OpenCouchMemoryStore` for INCOGNITO,
                :class:`SqliteMemoryStore` for LOCAL/SYNCED. Tests that
                want an explicit in-memory store can pass one directly
                to bypass the mode-based selection.
            crisis_log_backend: Optional crisis log backend. Same
                mode-based selection as ``memory_store``:
                :class:`InMemoryCrisisLogBackend` for INCOGNITO,
                :class:`SqliteCrisisLogBackend` for LOCAL/SYNCED. The
                crisis log is always-on regardless of memory_mode, but
                the *backend* still follows the mode: incognito means
                no crisis events hit disk; local means they do. Tests
                can override with NullCrisisLogBackend
                or a mock to assert specific behaviors.
            memory_mode: Persistence tier for the runtime. ``INCOGNITO``
                uses ephemeral in-memory stores only; ``LOCAL`` persists
                to the configured SQLite paths; ``SYNCED`` is reserved
                for a future remote backend and currently behaves the
                same as LOCAL.
            memory_sqlite_path: SQLite database path for the memory
                store. Only used when ``memory_mode`` is LOCAL/SYNCED
                and the caller didn't pass an explicit ``memory_store``.
                Defaults to :data:`DEFAULT_MEMORY_DB_PATH`.
            crisis_log_sqlite_path: SQLite database path for the crisis
                log. Only used when ``memory_mode`` is LOCAL/SYNCED and
                the caller didn't pass an explicit
                ``crisis_log_backend``. Defaults to
                :data:`DEFAULT_CRISIS_LOG_DB_PATH`.
            session_feedback_backend: Optional session-feedback backend.
                Same mode-based selection as ``crisis_log_backend``:
                :class:`InMemorySessionFeedbackBackend` for INCOGNITO,
                :class:`SqliteSessionFeedbackBackend` for LOCAL/SYNCED.
                Always-on regardless of memory mode; in incognito
                ``user_id_or_null`` is scrubbed to None by
                :meth:`record_session_feedback`. Tests can override
                with :class:`NullSessionFeedbackBackend` to assert
                "no feedback was written".
            feedback_sqlite_path: SQLite database path for the session
                feedback store. Only used when ``memory_mode`` is
                LOCAL/SYNCED and the caller didn't pass an explicit
                ``session_feedback_backend``. Defaults to
                :data:`DEFAULT_FEEDBACK_DB_PATH`.
        """

        self.memory_mode = memory_mode
        is_incognito = memory_mode == MemoryMode.INCOGNITO

        # Thread checkpointer path: incognito forces :memory: so the
        # LangGraph checkpointer doesn't leak conversation state to
        # disk. Non-incognito uses the caller-provided path.
        resolved_sqlite = ":memory:" if is_incognito else sqlite_path
        self.sqlite_path = (
            Path(resolved_sqlite) if resolved_sqlite != ":memory:" else Path(":memory:")
        )

        self._saver_cm = None
        self._checkpointer: AsyncSqliteSaver | None = None
        self._graph: CompiledStateGraph | None = None
        self._default_llm_client = default_llm_client
        self._session_timeout = session_timeout
        self._session_sweep_interval_seconds = max(
            1.0, float(session_sweep_interval_seconds)
        )
        self._finalize_active_sessions_on_close = finalize_active_sessions_on_close
        self._session_sweeper_task: asyncio.Task[None] | None = None
        self._thread_llm_clients: dict[str, BaseLLMClient | None] = {}

        # v0.8: pick memory store + crisis log backend based on mode
        # and whether the caller passed an explicit override. The
        # mode-based defaults give persistent-mode CLI sessions real
        # disk backing without requiring the caller to construct
        # SqliteMemoryStore themselves; the override path lets tests
        # pass in in-memory instances directly.
        if memory_store is not None:
            # Explicit override — trust the caller.
            self._memory_store = memory_store
        elif is_incognito:
            self._memory_store = OpenCouchMemoryStore()
        else:
            self._memory_store = SqliteMemoryStore(memory_sqlite_path)

        if crisis_log_backend is not None:
            self._crisis_log_backend = crisis_log_backend
        elif is_incognito:
            self._crisis_log_backend = InMemoryCrisisLogBackend()
        else:
            self._crisis_log_backend = SqliteCrisisLogBackend(crisis_log_sqlite_path)

        # v0.10: session-feedback backend. Same mode-based selection as
        # crisis_log — the two subsystems have parallel privacy and
        # persistence contracts. Explicit override wins for tests.
        if session_feedback_backend is not None:
            self._session_feedback_backend = session_feedback_backend
        elif is_incognito:
            self._session_feedback_backend = InMemorySessionFeedbackBackend()
        else:
            self._session_feedback_backend = SqliteSessionFeedbackBackend(
                feedback_sqlite_path,
            )

        # v0.8.1: embedding provider for hybrid retrieval. Resolution
        # order: (1) explicit override, (2) NullEmbeddingProvider in
        # incognito mode (no network calls, no embeddings to store),
        # (3) configured provider (Gemini) if an API key is set,
        # (4) NullEmbeddingProvider as the final fallback. The null
        # fallback means the store's hybrid retrieval gracefully
        # degrades to token-recall when no provider is available —
        # same contract as the extractor nodes which silently skip
        # when no LLM client is configured.
        if embedding_provider is not None:
            self._embedding_provider: EmbeddingProvider = embedding_provider
        elif is_incognito:
            self._embedding_provider = NullEmbeddingProvider()
        else:
            self._embedding_provider = create_configured_embedding_provider()

        # v0.4: per-process tracking of when each thread's current session
        # began. Used by the session summarizer to populate started_at on
        # the stored episodic arc. Reset on CLI restart — matches the Q4
        # scoping decision that "one continuous CLI process = one session,
        # continuation after /resume in a new CLI is a new session."
        #
        # Keys are thread_ids, values are ISO-8601 strings. Entries are
        # populated lazily on the first run_turn for each thread and
        # cleared by end_session after a successful summary write.
        self._session_starts: dict[str, str] = {}

        # v0.4: per-process tracking of the peak crisis-gate level
        # observed across the turns of each thread's current session.
        # The crisis gate is the canonical source of truth for crisis
        # severity (with regex fast paths, LLM classifier, and override
        # logic) — this dict is a simple max-of-seen rollup that the
        # summarizer reads at session end to populate the stored arc's
        # ``crisis_level_max`` field deterministically, rather than
        # asking the summarizer LLM to re-interpret the session.
        #
        # Keys are thread_ids, values are the max level seen so far
        # (0, 1, 2, or 3). Entries are populated on every run_turn
        # (updated via max) and cleared by end_session alongside
        # ``_session_starts``.
        self._max_crisis_levels: dict[str, int] = {}

        # Phase 2: per-thread buffer for memory candidates that should
        # not commit on the hot path. The turn nodes append held
        # semantic/procedural candidates here, and end_session runs the
        # session-level commit pass before clearing the buffer.
        self._session_memory_buffers: dict[str, SessionMemoryBuffer] = {}
        self._session_transcript_starts: dict[str, int] = {}

    async def __aenter__(self) -> PersistentAgentRuntime:
        """Open runtime resources.

        Only opens the LangGraph thread checkpointer connection
        eagerly — the memory store and crisis log backends (whether
        in-memory or SQLite) open their own resources lazily on first
        async method call. This asymmetry exists because the LangGraph
        checkpointer was designed with explicit ``__aenter__`` /
        ``__aexit__`` semantics, while the v0.8 SQLite stores use
        lazy connection initialization to keep ``__init__`` cheap
        and to support sync test fixtures that construct instances
        without an event loop.

        The practical effect is the same: by the time the runtime is
        usable (inside the ``async with`` block), all three storage
        layers are ready to accept reads and writes.
        """

        if self.sqlite_path != Path(":memory:"):
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)

        serde = JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSGPACK_MODULES)
        self._saver_cm = AsyncSqliteSaver.from_conn_string(str(self.sqlite_path))
        self._checkpointer = await self._saver_cm.__aenter__()
        self._checkpointer.serde = serde
        await self._ensure_runtime_schema()
        await self._prewarm()
        self._session_sweeper_task = asyncio.create_task(self._session_sweeper_loop())
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Close runtime resources."""

        if self._session_sweeper_task is not None:
            self._session_sweeper_task.cancel()
            try:
                await self._session_sweeper_task
            except asyncio.CancelledError:
                pass
            self._session_sweeper_task = None
        if self._finalize_active_sessions_on_close:
            await self.finalize_active_sessions(llm_client=self._default_llm_client)
        await self._memory_store.aclose()
        await self._crisis_log_backend.aclose()
        await self._session_feedback_backend.aclose()
        if self._saver_cm is not None:
            await self._saver_cm.__aexit__(exc_type, exc, tb)

    def _ensure_open(self) -> None:
        """Raise if runtime is used outside its async context."""

        if self._checkpointer is None:
            raise RuntimeError(
                "PersistentAgentRuntime must be used inside 'async with'."
            )

    async def _ensure_runtime_schema(self) -> None:
        """Create runtime-owned tables alongside LangGraph checkpoints."""

        self._ensure_open()
        await self._checkpointer.setup()

    async def _prewarm(self) -> None:
        """Warm one-time runtime resources before the first user turn."""

        # Graph compilation happens on the main event loop to avoid thread-safety
        # issues with the checkpointer's SQLite connection.
        self._get_graph()

        try:
            await asyncio.wait_for(self._embedding_provider.awarmup(), timeout=5.0)
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

        async with self._checkpointer.lock:
            await self._checkpointer.conn.execute(ACTIVE_SESSION_STATE_DDL)
            await self._checkpointer.conn.commit()

    async def _load_persisted_active_session(
        self,
        thread_id: str,
    ) -> PersistedActiveSessionState | None:
        """Return the persisted active-session record for ``thread_id``."""

        if self.memory_mode == MemoryMode.INCOGNITO:
            return None

        self._ensure_open()
        async with self._checkpointer.lock:
            async with self._checkpointer.conn.execute(
                """
                SELECT payload_json
                FROM opencouch_active_sessions
                WHERE thread_id = ?
                """,
                (thread_id,),
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return PersistedActiveSessionState.from_json(str(row[0]))

    async def _list_persisted_active_session_ids(self) -> list[str]:
        """Return every thread id with an unresolved active session."""

        if self.memory_mode == MemoryMode.INCOGNITO:
            return list(self._session_starts)

        self._ensure_open()
        async with self._checkpointer.lock:
            async with self._checkpointer.conn.execute(
                """
                SELECT thread_id
                FROM opencouch_active_sessions
                ORDER BY thread_id
                """
            ) as cursor:
                rows = await cursor.fetchall()
        return [str(row[0]) for row in rows]

    async def _save_persisted_active_session(
        self,
        session: PersistedActiveSessionState,
    ) -> None:
        """Upsert one runtime-owned active-session record."""

        if self.memory_mode == MemoryMode.INCOGNITO:
            return

        self._ensure_open()
        payload_json = session.to_json()
        async with self._checkpointer.lock:
            await self._checkpointer.conn.execute(
                """
                INSERT INTO opencouch_active_sessions(thread_id, payload_json)
                VALUES(?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    payload_json = excluded.payload_json
                """,
                (session.thread_id, payload_json),
            )
            await self._checkpointer.conn.commit()

    async def _delete_persisted_active_session(self, thread_id: str) -> None:
        """Delete the persisted active-session record for ``thread_id``."""

        if self.memory_mode == MemoryMode.INCOGNITO:
            return

        self._ensure_open()
        async with self._checkpointer.lock:
            await self._checkpointer.conn.execute(
                """
                DELETE FROM opencouch_active_sessions
                WHERE thread_id = ?
                """,
                (thread_id,),
            )
            await self._checkpointer.conn.commit()

    @staticmethod
    def _transcript_length(state: AgentState | None) -> int:
        """Return the current durable transcript length for a thread."""

        if state is None:
            return 0
        return len(state.get("transcript", []) or [])

    @staticmethod
    def _slice_state_to_active_session(
        state: AgentState,
        *,
        transcript_start_index: int,
    ) -> AgentState:
        """Return a shallow state copy limited to the active session window."""

        transcript = list(state.get("transcript", []) or [])
        start = min(max(transcript_start_index, 0), len(transcript))
        windowed: AgentState = dict(state)
        windowed["transcript"] = transcript[start:]

        if "history" in state:
            history = list(state.get("history", []) or [])
            history_start = min(start, len(history))
            windowed["history"] = history[history_start:]

        return windowed

    def _session_has_expired(self, session: PersistedActiveSessionState) -> bool:
        """Return whether the active session crossed the inactivity timeout."""

        last_active = _parse_iso_timestamp(session.last_active_at)
        if last_active is None:
            return True
        return (
            datetime.now(tz=last_active.tzinfo) - last_active >= self._session_timeout
        )

    def _clear_runtime_session_tracking(self, thread_id: str) -> None:
        """Drop all in-process session trackers for one thread."""

        self._session_starts.pop(thread_id, None)
        self._max_crisis_levels.pop(thread_id, None)
        self._session_memory_buffers.pop(thread_id, None)
        self._session_transcript_starts.pop(thread_id, None)
        self._thread_llm_clients.pop(thread_id, None)

    def _hydrate_runtime_session_tracking(
        self,
        session: PersistedActiveSessionState,
    ) -> None:
        """Restore the in-process trackers from a persisted session record."""

        self._session_starts[session.thread_id] = session.started_at
        self._max_crisis_levels[session.thread_id] = session.max_crisis_level
        self._session_transcript_starts[session.thread_id] = (
            session.transcript_start_index
        )
        self._session_memory_buffers[session.thread_id] = (
            session.session_buffer.model_copy(deep=True)
        )

    def _remember_llm_client(
        self,
        thread_id: str,
        llm_client: BaseLLMClient | None,
    ) -> None:
        """Remember the llm client for automatic lifecycle finalization."""

        if llm_client is not None:
            self._thread_llm_clients[thread_id] = llm_client

    def _effective_llm_client(
        self,
        thread_id: str,
        llm_client: BaseLLMClient | None = None,
    ) -> BaseLLMClient | None:
        """Resolve the llm client for timeout/shutdown finalization."""

        return (
            llm_client
            or self._thread_llm_clients.get(thread_id)
            or self._default_llm_client
        )

    async def _persist_runtime_session_tracking(
        self,
        thread_id: str,
        *,
        last_active_at: str | None = None,
    ) -> None:
        """Persist the current in-process session trackers for one thread."""

        started_at = self._session_starts.get(thread_id)
        transcript_start_index = self._session_transcript_starts.get(thread_id)
        if started_at is None or transcript_start_index is None:
            return

        session = PersistedActiveSessionState(
            thread_id=thread_id,
            started_at=started_at,
            last_active_at=last_active_at or _iso_now(),
            transcript_start_index=transcript_start_index,
            max_crisis_level=self._max_crisis_levels.get(thread_id, 0),
            session_buffer=self._session_memory_buffer_for_thread(thread_id).model_copy(
                deep=True
            ),
        )
        await self._save_persisted_active_session(session)

    async def _finalize_expired_sessions_once(self) -> None:
        """End any sessions that crossed the inactivity timeout."""

        try:
            active_thread_ids = await self._list_persisted_active_session_ids()
        except Exception:
            logger.warning(
                "finalize_expired_sessions_once: failed to list active sessions",
                exc_info=True,
            )
            return

        for active_thread_id in active_thread_ids:
            try:
                persisted = await self._load_persisted_active_session(active_thread_id)
                if persisted is None or not self._session_has_expired(persisted):
                    continue
                logger.info(
                    "session timeout reached for thread %s; auto-finalizing expired session",
                    active_thread_id,
                )
                await self.end_session(
                    active_thread_id,
                    llm_client=self._effective_llm_client(active_thread_id),
                )
            except Exception:
                logger.warning(
                    "finalize_expired_sessions_once: failed to end expired session for thread %s",
                    active_thread_id,
                    exc_info=True,
                )

    async def _session_sweeper_loop(self) -> None:
        """Background task that proactively closes expired sessions."""

        try:
            while True:
                await asyncio.sleep(self._session_sweep_interval_seconds)
                await self._finalize_expired_sessions_once()
        except asyncio.CancelledError:
            raise

    async def _prepare_session_for_turn(
        self,
        *,
        thread_id: str,
        prior_state: AgentState | None,
        llm_client: BaseLLMClient | None,
    ) -> None:
        """Restore or create the active session before a new turn runs."""

        persisted = await self._load_persisted_active_session(thread_id)
        if persisted is not None:
            self._hydrate_runtime_session_tracking(persisted)
            if self._session_has_expired(persisted):
                logger.info(
                    "session timeout reached for thread %s; ending prior session before new turn",
                    thread_id,
                )
                await self.end_session(thread_id, llm_client=llm_client)
                persisted = None

        if persisted is None:
            now = _iso_now()
            self._session_starts[thread_id] = now
            self._max_crisis_levels[thread_id] = 0
            self._session_transcript_starts[thread_id] = self._transcript_length(
                prior_state
            )
            self._session_memory_buffers[thread_id] = SessionMemoryBuffer(
                session_id=thread_id
            )
            await self._persist_runtime_session_tracking(
                thread_id,
                last_active_at=now,
            )

    async def _finalize_session_window(
        self,
        *,
        thread_id: str,
        state: AgentState,
        started_at: str,
        ended_at: str,
        crisis_level_max: int,
        session_buffer: SessionMemoryBuffer | None,
        llm_client: BaseLLMClient | None,
    ) -> StoredSessionArc | None:
        """Run the shared session-end summarization + commit seam."""

        # Compute dominant modality from the session buffer's per-turn
        # accumulator. Falls back to None when no buffer exists (e.g.,
        # end_transcript_session with a fresh buffer) or no modality was
        # dispatched during the session.
        approach_hint = session_buffer.dominant_approach() if session_buffer else None

        stored_arc = await run_summarize_session(
            state,
            llm_client=llm_client,
            memory_store=self._memory_store,
            memory_mode=self.memory_mode,
            session_id=thread_id,
            started_at=started_at,
            ended_at=ended_at,
            crisis_level_max=crisis_level_max,
            embedding_provider=self._embedding_provider,
            approach_hint=approach_hint,
        )

        commit_result = await run_commit_session_memory(
            state,
            memory_store=self._memory_store,
            session_buffer=session_buffer,
            stored_arc=stored_arc,
            embedding_provider=self._embedding_provider,
        )
        if commit_result is not None:
            logger.info(
                "end_session: committed %d semantic facts, %d procedural rules "
                "(%d semantic bumps, %d semantic skipped, %d procedural skipped)",
                commit_result.semantic_writes,
                commit_result.procedural_writes,
                commit_result.semantic_bumps,
                commit_result.semantic_skips,
                commit_result.procedural_skips,
            )
        return stored_arc

    async def _extract_memory_from_transcript(
        self,
        *,
        thread_id: str,
        user_id: str | None,
        transcript: list[dict[str, Any]],
        llm_client: BaseLLMClient | None,
        session_buffer: SessionMemoryBuffer,
    ) -> None:
        """Replay transcript user turns through the existing extractor nodes."""

        runtime = _RuntimeShim(
            context=WorkflowContext(
                llm_client=llm_client,
                memory_store=self._memory_store,
                crisis_log_backend=self._crisis_log_backend,
                memory_mode=self.memory_mode,
                embedding_provider=self._embedding_provider,
                session_memory_buffer=session_buffer,
            )
        )

        user_turn_count = 0
        for transcript_index, turn in enumerate(transcript):
            if turn.get("role") != "user":
                continue

            message = (turn.get("content") or "").strip()
            if not message:
                continue

            user_turn_count += 1
            state: AgentState = {
                "message": message,
                "user_id": user_id,
                "session_id": thread_id,
                "history": list(transcript[:transcript_index]),
                "transcript": list(transcript[: transcript_index + 1]),
                "progress": {"turn_count": user_turn_count},
                "routing": {"route": "therapeutic"},
            }
            await run_extract_semantic_facts_node(state, runtime)
            await run_extract_procedural_rules_node(state, runtime)

    @property
    def memory_store(self) -> MemoryStore:
        """Return the runtime's unified memory store.

        Exposed for CLI and debug-tooling use (e.g. ``/memory status``).
        The store is the same instance passed into node runtime contexts,
        so reads reflect the current live state. Typed as the
        :class:`MemoryStore` protocol so callers don't depend on
        whether the runtime is holding the in-memory or SQLite
        implementation.
        """

        return self._memory_store

    @property
    def crisis_log_backend(self) -> CrisisLogBackend:
        """Return the runtime's crisis log backend.

        Exposed for CLI and debug-tooling use. The backend is the same
        instance passed into node runtime contexts and is always-on
        regardless of memory mode.
        """

        return self._crisis_log_backend

    @property
    def session_feedback_backend(self) -> SessionFeedbackBackend:
        """Return the runtime's session-feedback backend.

        Exposed for CLI and debug-tooling use (``/memory status``,
        observability panels). Always-on regardless of memory mode;
        in incognito mode the ``record_session_feedback`` method
        scrubs ``user_id_or_null`` so feedback is still recorded but
        without owner-identifying information.
        """

        return self._session_feedback_backend

    def _config_for_thread(
        self,
        thread_id: str,
        *,
        channel: Channel | None = None,
        user_id: str | None = None,
        streaming: bool = False,
    ) -> dict[str, Any]:
        """Build LangGraph config payload for one thread."""

        metadata = {
            "thread_id": thread_id,
            "therapeutic_approach": "text",
            "streaming": streaming,
            "channel": channel.value if channel is not None else None,
            "user_scope": "persistent" if user_id else "guest",
            "memory_mode": self.memory_mode.value,
        }
        return {
            "configurable": {"thread_id": thread_id},
            "metadata": metadata,
        }

    def _session_memory_buffer_for_thread(self, thread_id: str) -> SessionMemoryBuffer:
        """Return the runtime-managed session buffer for one thread."""

        if thread_id not in self._session_memory_buffers:
            self._session_memory_buffers[thread_id] = SessionMemoryBuffer(
                session_id=thread_id
            )
        return self._session_memory_buffers[thread_id]

    def _context_for_turn(
        self,
        *,
        thread_id: str,
        llm_client: BaseLLMClient | None,
        response_llm_client: BaseLLMClient | None = None,
    ) -> WorkflowContext:
        """Build LangGraph runtime context for one turn."""

        return WorkflowContext(
            llm_client=llm_client,
            response_llm=response_llm_client or llm_client,
            memory_store=self._memory_store,
            crisis_log_backend=self._crisis_log_backend,
            memory_mode=self.memory_mode,
            # v0.8.1: make the embedding provider visible to graph
            # nodes via runtime.context. The extractor nodes read
            # this to compute embeddings at write time; the
            # load_memory node reads it to compute query embeddings
            # for the hybrid retrieval path. Always present (even as
            # NullEmbeddingProvider) so nodes don't need to guard
            # against the key being missing from the dict.
            embedding_provider=self._embedding_provider,
            session_memory_buffer=self._session_memory_buffer_for_thread(thread_id),
        )

    @staticmethod
    def _messages_from_transcript(
        transcript: list[dict[str, Any]],
    ) -> list[Message]:
        """Materialize validated messages from a serialized transcript.

        v0.8 observability pass: the transcript dicts now carry an
        optional ``mode`` field for assistant turns (written by
        ``run_finalize_turn_node``). We forward it into the Message
        pydantic model so the CLI's ``/history`` renderer can show
        it next to each assistant reply. User turns leave ``mode``
        unset, which we coerce to ``None``. Older checkpoints that
        predate this field just see ``None`` from ``.get()`` and the
        resulting Messages look identical to pre-v0.8 shape.

        The dict type annotation is ``dict[str, Any]`` rather than
        ``dict[str, str]`` because the ``mode`` field can be ``None``
        in user turns. The previous stricter annotation was technically
        wrong even before this change (LangGraph's JsonPlusSerializer
        can round-trip non-string values), but no caller noticed.
        """

        messages: list[Message] = []
        for turn in transcript:
            role = turn.get("role")
            content = (turn.get("content") or "").strip()
            if role not in {"system", "user", "assistant"} or not content:
                continue
            style = turn.get("response_style") if role == "assistant" else None
            messages.append(
                Message(role=MessageRole(role), content=content, response_style=style)
            )
        return messages

    def _get_graph(self) -> CompiledStateGraph:
        """Return the compiled LangGraph workflow for this runtime."""

        self._ensure_open()
        if self._graph is None:
            self._graph = build_agent_workflow(checkpointer=self._checkpointer)
        return self._graph

    async def get_state(self, thread_id: str) -> AgentState | None:
        """Load the latest persisted state snapshot for a thread."""

        graph = self._get_graph()
        snapshot = await graph.aget_state(self._config_for_thread(thread_id))
        values = snapshot.values or {}
        return values or None

    async def get_history(self, thread_id: str) -> list[Message]:
        """Load the full persisted transcript for a thread."""

        state = await self.get_state(thread_id)
        if state is None:
            return []
        return self._messages_from_transcript(state.get("transcript", []))

    async def has_active_session(self, thread_id: str) -> bool:
        """Return whether ``thread_id`` currently has an unresolved session."""

        persisted = await self._load_persisted_active_session(thread_id)
        if persisted is not None:
            return not self._session_has_expired(persisted)

        return (
            thread_id in self._session_starts
            or thread_id in self._session_transcript_starts
            or thread_id in self._session_memory_buffers
        )

    async def reset_thread(self, thread_id: str) -> None:
        """Delete all persisted checkpoints for a thread."""

        self._ensure_open()
        await self._checkpointer.adelete_thread(thread_id)
        await self._delete_persisted_active_session(thread_id)
        self._clear_runtime_session_tracking(thread_id)

    async def list_threads(self, *, limit: int = 20) -> list[ThreadSummary]:
        """List the most recent persisted threads in SQLite."""

        self._ensure_open()
        await self._checkpointer.setup()

        thread_ids: list[str] = []
        async with (
            self._checkpointer.lock,
            self._checkpointer.conn.execute(
                """
            SELECT thread_id
            FROM checkpoints
            GROUP BY thread_id
            ORDER BY MAX(rowid) DESC
            LIMIT ?
            """,
                (limit,),
            ) as cursor,
        ):
            async for row in cursor:
                thread_ids.append(row[0])

        summaries: list[ThreadSummary] = []
        for thread_id in thread_ids:
            state = await self.get_state(thread_id)
            history = self._messages_from_transcript(
                state.get("transcript", []) if state is not None else []
            )
            progress = state.get("progress", {}) if state is not None else {}
            turn_count = progress.get("turn_count", 0)
            summaries.append(
                ThreadSummary(
                    thread_id=thread_id,
                    turn_count=int(turn_count),
                    message_count=len(history),
                    has_context=state is not None,
                )
            )
        return summaries

    @staticmethod
    def _turn_count_from_state(state: AgentState | None) -> int:
        """Extract the persisted turn count from a checkpoint snapshot."""

        if state is None:
            return 0
        progress = state.get("progress", {}) or {}
        return int(progress.get("turn_count", 0) or 0)

    async def run_turn(
        self,
        *,
        thread_id: str,
        message: str,
        channel: Channel = Channel.TEST,
        user_id: str | None = None,
        installed_skills: list[str] | None = None,
        llm_client: BaseLLMClient | None = None,
        response_llm_client: BaseLLMClient | None = None,
    ) -> PersistentTurnResult:
        """Run one conversation turn through the minimal workflow."""

        graph = self._get_graph()
        self._remember_llm_client(thread_id, llm_client)

        # Load only the persisted turn counter. The checkpointer restores
        # the accumulated transcript/history via reducers, so we do NOT
        # need to deserialize the full transcript just to compute
        # ``progress.turn_count`` for the next turn.
        prior_state = await self.get_state(thread_id)
        await self._prepare_session_for_turn(
            thread_id=thread_id,
            prior_state=prior_state,
            llm_client=llm_client,
        )
        prior_turn_count = self._turn_count_from_state(prior_state)

        agent_input = AgentInput(
            message=message,
            channel=channel,
            user_id=user_id,
            session_id=thread_id,
            history=[],
            working_memory=[],
            installed_skills=list(installed_skills or []),
        )
        initial_state = build_initial_state(
            agent_input,
            prior_turn_count=prior_turn_count,
        )

        # v0.8 observability: time the whole turn for the CLI's
        # post-turn diagnostics panel. The per-node timings are
        # stamped into ``state["diagnostics"]`` inside each node,
        # and we add the outer total here. They're added
        # non-destructively by reading the final state's
        # diagnostics dict and merging.
        turn_start = time.monotonic()
        final_state = await graph.ainvoke(
            initial_state,
            config=self._config_for_thread(
                thread_id,
                channel=channel,
                user_id=user_id,
                streaming=False,
            ),
            context=self._context_for_turn(
                thread_id=thread_id,
                llm_client=llm_client,
                response_llm_client=response_llm_client,
            ),
        )
        turn_total_ms = round((time.monotonic() - turn_start) * 1000, 2)

        # Stamp the outer total into the final state's diagnostics
        # before we pass it to state_to_output. LangGraph returns a
        # plain dict for final_state, so we mutate it in place
        # rather than constructing a new one.
        if "diagnostics" not in final_state or final_state["diagnostics"] is None:
            final_state["diagnostics"] = {}
        final_state["diagnostics"]["turn_total_ms"] = turn_total_ms

        # v0.4: update the per-thread max crisis level so end_session
        # can pass it to the summarizer. The crisis gate writes a
        # CrisisAssessment into state["crisis"] on every turn; we
        # take the max across the session's lifetime. The lookup is
        # defensive because a turn that somehow skipped the crisis
        # gate entirely would leave the field None.
        turn_crisis = final_state.get("crisis")
        turn_level = 0
        if turn_crisis is not None:
            # CrisisAssessment is a pydantic model with a ``level`` attr,
            # but the checkpointed state may round-trip as a plain dict,
            # so read defensively.
            turn_level = (
                turn_crisis.level
                if hasattr(turn_crisis, "level")
                else int(turn_crisis.get("level", 0) or 0)
            )
        prior_max = self._max_crisis_levels.get(thread_id, 0)
        self._max_crisis_levels[thread_id] = max(prior_max, turn_level)

        # Record the dispatched modality for session-end dominant-modality
        # computation. Mirrors the crisis-level tracking pattern above.
        turn_routing = final_state.get("routing") or {}
        turn_modality = (
            turn_routing.get("therapeutic_approach")
            if isinstance(turn_routing, dict)
            else getattr(turn_routing, "therapeutic_approach", None)
        )
        self._session_memory_buffer_for_thread(thread_id).record_approach(turn_modality)

        await self._persist_runtime_session_tracking(thread_id)

        return PersistentTurnResult(
            output=state_to_output(final_state),
            state=final_state,
            history=self._messages_from_transcript(final_state.get("transcript", [])),
        )

    async def end_session(
        self,
        thread_id: str,
        *,
        llm_client: BaseLLMClient | None = None,
    ) -> StoredSessionArc | None:
        """Summarize the active session for ``thread_id`` and write it to memory.

        Loads the current state for the thread, invokes the session
        summarizer with the full transcript plus session metadata, and
        returns the written :class:`StoredSessionArc` on success.

        Skip conditions (return ``None``):
        - Incognito memory mode (symmetric with extract_facts — no
          episodic writes in incognito).
        - No LLM client provided (the summarizer needs one).
        - LLM judged the session too thin to summarize (returned
          ``arc=None``).
        - LLM call or store write failed (degrades silently; check
          the warning logs for diagnostics).
        - No state exists for ``thread_id`` (e.g., the thread has no
          turns yet).

        After a successful summarization, the tracked session_start for
        this thread is cleared, so a subsequent ``run_turn`` on the same
        thread_id starts a fresh session. This matches the v0.4 scoping
        decision that one session = one summary, and a continuation
        after end_session is a new session.

        The session summarizer is a standalone function rather than a
        graph node — see ``agent/nodes/summarize_session.py`` for the
        rationale. The runtime invokes it directly with explicit
        dependencies (llm_client, memory_store, memory_mode) rather
        than through the LangGraph runtime context.

        Args:
            thread_id: Which thread's session to summarize.
            llm_client: The LLM client to use for the summarization
                call. If None, the summarizer skips silently (same
                contract as extract_facts).

        Returns:
            The written :class:`StoredSessionArc` on success, or
            ``None`` on any legitimate skip / failure.
        """

        effective_llm_client = self._effective_llm_client(thread_id, llm_client)
        persisted = await self._load_persisted_active_session(thread_id)
        if persisted is not None:
            self._hydrate_runtime_session_tracking(persisted)
        has_active_session = (
            persisted is not None
            or thread_id in self._session_starts
            or thread_id in self._session_transcript_starts
            or thread_id in self._session_memory_buffers
        )

        if not has_active_session:
            return None

        state = await self.get_state(thread_id)

        if state is None:
            # No state yet — nothing to summarize. This happens if
            # end_session is called immediately after /new without any
            # turns having run.
            await self._delete_persisted_active_session(thread_id)
            self._clear_runtime_session_tracking(thread_id)
            return None

        transcript_start_index = self._session_transcript_starts.get(thread_id, 0)
        session_state = self._slice_state_to_active_session(
            state,
            transcript_start_index=transcript_start_index,
        )
        started_at = self._session_starts.get(thread_id, _iso_now())
        ended_at = _iso_now()
        crisis_level_max = self._max_crisis_levels.get(thread_id, 0)
        session_buffer = self._session_memory_buffers.get(thread_id)
        try:
            return await self._finalize_session_window(
                thread_id=thread_id,
                state=session_state,
                started_at=started_at,
                ended_at=ended_at,
                crisis_level_max=crisis_level_max,
                session_buffer=session_buffer,
                llm_client=effective_llm_client,
            )
        finally:
            await self._delete_persisted_active_session(thread_id)
            self._clear_runtime_session_tracking(thread_id)

    async def end_transcript_session(
        self,
        *,
        thread_id: str,
        user_id: str | None,
        transcript: list[dict[str, Any]],
        llm_client: BaseLLMClient | None = None,
        started_at: str | None = None,
        ended_at: str | None = None,
        crisis_level_max: int = 0,
    ) -> StoredSessionArc | None:
        """Finalize a non-graph session that only has a transcript."""

        if not transcript:
            return None

        self._remember_llm_client(thread_id, llm_client)
        session_buffer = SessionMemoryBuffer(session_id=thread_id)
        await self._extract_memory_from_transcript(
            thread_id=thread_id,
            user_id=user_id,
            transcript=transcript,
            llm_client=self._effective_llm_client(thread_id, llm_client),
            session_buffer=session_buffer,
        )
        session_state: AgentState = {
            "user_id": user_id,
            "session_id": thread_id,
            "transcript": list(transcript),
        }
        return await self._finalize_session_window(
            thread_id=thread_id,
            state=session_state,
            started_at=started_at or _iso_now(),
            ended_at=ended_at or _iso_now(),
            crisis_level_max=crisis_level_max,
            session_buffer=session_buffer,
            llm_client=self._effective_llm_client(thread_id, llm_client),
        )

    async def finalize_active_sessions(
        self,
        *,
        llm_client: BaseLLMClient | None = None,
    ) -> None:
        """Best-effort graceful shutdown hook for unresolved sessions."""

        try:
            active_thread_ids = await self._list_persisted_active_session_ids()
        except Exception:
            logger.warning(
                "finalize_active_sessions: failed to list active sessions",
                exc_info=True,
            )
            return

        for active_thread_id in active_thread_ids:
            try:
                await self.end_session(
                    active_thread_id,
                    llm_client=self._effective_llm_client(active_thread_id, llm_client),
                )
            except Exception:
                logger.warning(
                    "finalize_active_sessions: failed to end session for thread %s",
                    active_thread_id,
                    exc_info=True,
                )

    async def record_session_feedback(
        self,
        thread_id: str,
        *,
        label: FeedbackLabel,
        source: FeedbackSource,
    ) -> SessionFeedbackRecord | None:
        """Record an explicit end-of-session feedback label.

        Called by end-session surfaces (CLI ``/end``, CLI ``/exit`` with
        save=y, HTTP ``POST /threads/{id}/end`` with a feedback body)
        BEFORE invoking :meth:`end_session`. Never raises — any
        exception is logged and swallowed. Returns ``None`` on any
        failure. The end-session flow continues regardless, so a
        backend outage never blocks the summary or the farewell.

        Owner identity (``user_id_or_null``) is derived from persisted
        thread state, not from the caller — the authoritative source is
        the same ``state.user_id`` that powers memory and crisis_log
        writes. **Scrubbed to None in incognito mode** regardless of
        what state carries, mirroring the
        :class:`CrisisLogRecord` privacy contract.

        Turn count is read from ``state.progress.turn_count`` via the
        existing :meth:`_turn_count_from_state` helper. When the thread
        has no state (e.g., ``/end`` immediately after ``/new`` with
        zero turns) we still write the record with
        ``turn_count_at_end=0``.

        Idempotency: none in Phase 1. Two calls produce two rows. The
        CLI prompt flow is single-shot and the HTTP endpoint is
        single-request, so accidental duplicates require an explicit
        double-click / double-POST. If explicit idempotency becomes
        necessary later, the fix is an idempotency-key column on
        :class:`SessionFeedbackRecord`, not a UNIQUE constraint on
        the opaque ``id``.

        Args:
            thread_id: The thread whose session is ending. Hashed
                into ``session_id_opaque`` before the record is
                written — the raw thread id never touches the store.
            label: The explicit feedback label the user provided.
            source: Which end-session surface produced this feedback.

        Returns:
            The written :class:`SessionFeedbackRecord` on success, or
            ``None`` on any failure (backend write error, state
            lookup failure, etc.). Callers should ignore the return
            value for control flow — the contract is "best effort".
        """

        try:
            state = await self.get_state(thread_id)
            turn_count = self._turn_count_from_state(state)

            # Privacy: scrub user_id in incognito mode. Even though
            # state may carry a user_id in incognito (nothing
            # prevents the caller from passing one), we never
            # persist it here. Mirrors CrisisLogRecord's
            # incognito-scrub contract.
            if self.memory_mode == MemoryMode.INCOGNITO:
                user_id: str | None = None
            elif state is not None:
                user_id = state.get("user_id")
            else:
                user_id = None

            record = SessionFeedbackRecord(
                id=str(uuid4()),
                session_id_opaque=hash_session_id(thread_id),
                user_id_or_null=user_id,
                recorded_at=iso_now(),
                label=label,
                turn_count_at_end=turn_count,
                source=source,
                schema_version=1,
            )

            await self._session_feedback_backend.aappend(record)
            return record
        except Exception:
            logger.warning(
                "session feedback write failed for thread %s",
                thread_id,
                exc_info=True,
            )
            return None

    async def run_turn_stream(
        self,
        *,
        thread_id: str,
        message: str,
        channel: Channel = Channel.TEST,
        user_id: str | None = None,
        installed_skills: list[str] | None = None,
        llm_client: BaseLLMClient | None = None,
        response_llm_client: BaseLLMClient | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Run one turn and yield status, reply-ready, and done events.

        Streams the graph via ``astream(stream_mode=["updates", "values"])``
        so that per-node progress and the accumulated final state come out
        of a single pass. Each ``updates`` chunk is a ``{node_name: delta}``
        dict emitted when that node finishes — we map the node name to the
        CLI's stage label vocabulary (``load_memory``, ``crisis_gate``,
        ``therapeutic``, etc.) and yield a :class:`StatusEvent`. Each
        ``values`` chunk is the full merged state at that checkpoint —
        we keep the most recent one as the final state.

        Duplicates the session-tracking bookkeeping from :meth:`run_turn`
        (session start stamp, max crisis level, turn-total timing) so
        callers using ``run_turn_stream`` get the same side effects as
        callers using ``run_turn``. The stream emits a non-terminal
        :class:`ResponseReadyEvent` once ``finalize_turn_node`` seals the
        reply into the checkpointed transcript, then a terminal
        :class:`DoneEvent` after the post-response tail reaches ``END``.

        Why the per-node status labels don't exactly match node names:
        the CLI's ``_STAGE_LABELS`` vocabulary predates the graph
        refactor and uses short names like ``crisis_gate`` and
        ``therapeutic``. Rather than rename either side, we translate
        here. Unknown nodes fall through to the node name itself so
        future additions still render readably in the Live display.
        """

        graph = self._get_graph()

        # Same turn-count optimization as ``run_turn``: read only the
        # persisted counter from the checkpoint snapshot rather than
        # deserializing the full transcript.
        prior_state = await self.get_state(thread_id)
        await self._prepare_session_for_turn(
            thread_id=thread_id,
            prior_state=prior_state,
            llm_client=llm_client,
        )
        prior_turn_count = self._turn_count_from_state(prior_state)

        agent_input = AgentInput(
            message=message,
            channel=channel,
            user_id=user_id,
            session_id=thread_id,
            history=[],
            working_memory=[],
            installed_skills=list(installed_skills or []),
        )
        initial_state = build_initial_state(
            agent_input,
            prior_turn_count=prior_turn_count,
        )

        # Map internal graph node names → CLI stage labels. Keeps the
        # naming mismatch between graph internals and CLI vocabulary
        # contained to a single source of truth. Therapeutic subgraph
        # emits its own node updates from within the subgraph, but at
        # the parent level it reports as ``therapeutic_subgraph`` — we
        # surface that as ``therapeutic`` so the CLI label matches.
        _NODE_TO_STAGE = {
            "load_memory_node": "load_memory",
            "crisis_gate_node": "crisis_gate",
            "crisis_response_node": "crisis_response",
            "crisis_log_node": "crisis_log",
            "therapeutic_subgraph": "therapeutic",
            "extract_semantic_facts_node": "extract_facts",
            "extract_procedural_rules_node": "extract_procedural",
            "finalize_turn_node": "finalize",
        }

        turn_start = time.monotonic()
        final_state: AgentState | None = None
        chunks_emitted = False
        finalize_seen = False
        response_ready_emitted = False

        def _response_ready_output(state: AgentState | None) -> AgentOutput | None:
            """Return the partial output once finalize has made it durable."""

            if state is None or not finalize_seen or response_ready_emitted:
                return None
            response_text = str(state.get("response", {}).get("text", "") or "").strip()
            if not response_text:
                return None
            return state_to_output(state)

        # v0.9: subgraphs=True propagates custom stream events from
        # inside the compiled therapeutic_subgraph. version="v2" gives
        # a unified StreamPart dict format: chunk["type"], chunk["ns"],
        # chunk["data"]. Without subgraphs=True, get_stream_writer()
        # chunks from subgraph nodes are silently swallowed.
        async for chunk in graph.astream(
            initial_state,
            config=self._config_for_thread(
                thread_id,
                channel=channel,
                user_id=user_id,
                streaming=True,
            ),
            context=self._context_for_turn(
                thread_id=thread_id,
                llm_client=llm_client,
                response_llm_client=response_llm_client,
            ),
            stream_mode=["custom", "updates", "values"],
            subgraphs=True,
            version="v2",
        ):
            if chunk["type"] == "custom":
                # Token streaming: mode nodes emit per-token chunks via
                # get_stream_writer(). Forward from any namespace (root
                # or subgraph — all are response tokens).
                payload = chunk["data"]
                if isinstance(payload, dict) and payload.get("type") == "chunk":
                    yield ChunkEvent(text=payload["text"])
                    chunks_emitted = True
            elif chunk["type"] == "updates" and chunk["ns"] == ():
                # Root-level node completions only — skip subgraph
                # internals to avoid duplicate StatusEvents.
                for node_name in chunk["data"]:
                    stage = _NODE_TO_STAGE.get(node_name, node_name)
                    yield StatusEvent(stage=stage)
                    if node_name == "finalize_turn_node":
                        finalize_seen = True
                        ready_output = _response_ready_output(final_state)
                        if ready_output is not None:
                            # Fallback for deterministic mode (no LLM,
                            # no token chunks): emit the full response
                            # text once finalize seals it into the
                            # checkpointed state.
                            if not chunks_emitted:
                                yield ChunkEvent(text=ready_output.response_text)
                                chunks_emitted = True
                            yield ResponseReadyEvent(output=ready_output)
                            response_ready_emitted = True
            elif chunk["type"] == "values" and chunk["ns"] == ():
                # Root-level state snapshots only.
                final_state = chunk["data"]  # type: ignore[assignment]
                ready_output = _response_ready_output(final_state)
                if ready_output is not None:
                    if not chunks_emitted:
                        yield ChunkEvent(text=ready_output.response_text)
                        chunks_emitted = True
                    yield ResponseReadyEvent(output=ready_output)
                    response_ready_emitted = True

        turn_total_ms = round((time.monotonic() - turn_start) * 1000, 2)

        # Defensive guard: a graph that returns no values chunks (which
        # would be a LangGraph bug, not a legitimate runtime state) would
        # leave final_state as None. Fall back to reading the checkpoint
        # directly so the caller still gets a DoneEvent rather than a
        # crash deep in state_to_output.
        if final_state is None:
            fallback = await self.get_state(thread_id)
            if fallback is None:
                raise RuntimeError(
                    "run_turn_stream: graph stream yielded no values chunks "
                    "and no checkpoint was found for this thread."
                )
            final_state = fallback

        # Stamp the outer turn-total into diagnostics. Mirrors run_turn.
        if "diagnostics" not in final_state or final_state["diagnostics"] is None:
            final_state["diagnostics"] = {}
        final_state["diagnostics"]["turn_total_ms"] = turn_total_ms

        # v0.4: update max crisis level — same logic as run_turn.
        turn_crisis = final_state.get("crisis")
        turn_level = 0
        if turn_crisis is not None:
            turn_level = (
                turn_crisis.level
                if hasattr(turn_crisis, "level")
                else int(turn_crisis.get("level", 0) or 0)
            )
        prior_max = self._max_crisis_levels.get(thread_id, 0)
        self._max_crisis_levels[thread_id] = max(prior_max, turn_level)

        # Record dispatched modality — same logic as run_turn.
        turn_routing = final_state.get("routing") or {}
        turn_modality = (
            turn_routing.get("therapeutic_approach")
            if isinstance(turn_routing, dict)
            else getattr(turn_routing, "therapeutic_approach", None)
        )
        self._session_memory_buffer_for_thread(thread_id).record_approach(turn_modality)

        await self._persist_runtime_session_tracking(thread_id)

        yield DoneEvent(output=state_to_output(final_state))
