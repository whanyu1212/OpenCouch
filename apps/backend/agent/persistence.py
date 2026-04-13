"""Persistent thread runtime for the fresh START -> load_memory -> END graph."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent.graph import build_agent_workflow, build_initial_state, state_to_output
from agent.memory.crisis_log import CrisisLogBackend, InMemoryCrisisLogBackend
from agent.memory.embeddings import (
    EmbeddingProvider,
    NullEmbeddingProvider,
    create_configured_embedding_provider,
)
from agent.memory.models import StoredSessionArc
from agent.memory.modes import MemoryMode
from agent.memory.sqlite_crisis_log import SqliteCrisisLogBackend
from agent.memory.sqlite_store import SqliteMemoryStore
from agent.memory.store import MemoryStore, OpenCouchMemoryStore
from agent.models import (
    AgentInput,
    AgentOutput,
    Channel,
    DoneEvent,
    Message,
    MessageRole,
    StatusEvent,
    StreamEvent,
)
from agent.nodes.summarize_session import run_summarize_session
from agent.state import AgentState
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph
from services.llm.base import BaseLLMClient

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
ALLOWED_MSGPACK_MODULES = [
    ("agent.models", "Channel"),
    ("agent.models", "CrisisAssessment"),
    ("agent.models", "ModeType"),
    ("agent.models", "ResponseKind"),
]


def _iso_now() -> str:
    """Return the current UTC time in ISO-8601 format with 'Z' suffix."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


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
        memory_mode: MemoryMode = MemoryMode.LOCAL,
        memory_sqlite_path: str | Path = DEFAULT_MEMORY_DB_PATH,
        crisis_log_sqlite_path: str | Path = DEFAULT_CRISIS_LOG_DB_PATH,
        embedding_provider: "EmbeddingProvider | None" = None,
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
                crisis log is always-on regardless of memory_mode (see
                schema.yaml §2 namespaces.crisis_log for the privacy
                asymmetry), but the *backend* still follows the mode:
                incognito means no crisis events hit disk; local means
                they do. Tests can override with NullCrisisLogBackend
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
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Close runtime resources."""

        await self._memory_store.aclose()
        await self._crisis_log_backend.aclose()
        if self._saver_cm is not None:
            await self._saver_cm.__aexit__(exc_type, exc, tb)

    def _ensure_open(self) -> None:
        """Raise if runtime is used outside its async context."""

        if self._checkpointer is None:
            raise RuntimeError(
                "PersistentAgentRuntime must be used inside 'async with'."
            )

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

    def _config_for_thread(self, thread_id: str) -> dict[str, dict[str, str]]:
        """Build LangGraph config payload for one thread."""

        return {"configurable": {"thread_id": thread_id}}

    def _context_for_turn(
        self,
        *,
        llm_client: BaseLLMClient | None,
    ) -> dict[str, object]:
        """Build LangGraph runtime context for one turn."""

        return {
            "llm_client": llm_client,
            "memory_store": self._memory_store,
            "crisis_log_backend": self._crisis_log_backend,
            "memory_mode": self.memory_mode,
            # v0.8.1: make the embedding provider visible to graph
            # nodes via runtime.context. The extractor nodes read
            # this to compute embeddings at write time; the
            # load_memory node reads it to compute query embeddings
            # for the hybrid retrieval path. Always present (even as
            # NullEmbeddingProvider) so nodes don't need to guard
            # against the key being missing from the dict.
            "embedding_provider": self._embedding_provider,
        }

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
            mode = turn.get("mode") if role == "assistant" else None
            messages.append(Message(role=MessageRole(role), content=content, mode=mode))
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

    async def reset_thread(self, thread_id: str) -> None:
        """Delete all persisted checkpoints for a thread."""

        self._ensure_open()
        await self._checkpointer.adelete_thread(thread_id)

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

    async def run_turn(
        self,
        *,
        thread_id: str,
        message: str,
        channel: Channel = Channel.TEST,
        user_id: str | None = None,
        installed_skills: list[str] | None = None,
        llm_client: BaseLLMClient | None = None,
    ) -> PersistentTurnResult:
        """Run one conversation turn through the minimal workflow."""

        graph = self._get_graph()
        history = await self.get_history(thread_id)

        # v0.4: track the session start time for this thread. Populated
        # lazily on the first turn we see for each thread within the
        # lifetime of this runtime process. Used by end_session to
        # populate the session summary's started_at field. See the
        # docstring on self._session_starts for the "one CLI = one
        # session" definition.
        if thread_id not in self._session_starts:
            self._session_starts[thread_id] = _iso_now()

        agent_input = AgentInput(
            message=message,
            channel=channel,
            user_id=user_id,
            session_id=thread_id,
            history=history,
            working_memory=[],
            installed_skills=list(installed_skills or []),
        )
        initial_state = build_initial_state(agent_input)

        # Carry forward multi-turn exercise state from the previous
        # turn's checkpoint. build_initial_state creates a fresh
        # progress dict without exercise_type / exercise_step, so
        # without this merge the dispatcher's active-exercise fast-
        # path never fires and guided exercises restart every turn.
        prev_state = await self.get_state(thread_id)
        if prev_state is not None:
            prev_progress = prev_state.get("progress", {}) or {}
            if prev_progress.get("exercise_type") is not None:
                initial_state["progress"]["exercise_type"] = prev_progress[
                    "exercise_type"
                ]
                initial_state["progress"]["exercise_step"] = prev_progress.get(
                    "exercise_step", 0
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
            config=self._config_for_thread(thread_id),
            context=self._context_for_turn(llm_client=llm_client),
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

        state = await self.get_state(thread_id)
        if state is None:
            # No state yet — nothing to summarize. This happens if
            # end_session is called immediately after /new without any
            # turns having run.
            return None

        started_at = self._session_starts.get(thread_id, _iso_now())
        ended_at = _iso_now()
        # v0.4: pull the peak crisis level seen during this session
        # from the per-process tracker. Populated by run_turn after
        # every graph invocation. Defaults to 0 if this thread has
        # never run a turn in the current process — which is the
        # correct semantics for "no crisis events observed."
        crisis_level_max = self._max_crisis_levels.get(thread_id, 0)

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
        )

        # Clear the start time and crisis level tracker regardless of
        # outcome so subsequent turns on this thread begin a fresh
        # session. If we kept the old values around after a failed
        # summary, the next attempt would use stale metadata — better
        # to reset and treat the next turn as a new session start.
        self._session_starts.pop(thread_id, None)
        self._max_crisis_levels.pop(thread_id, None)

        return stored_arc

    async def run_turn_stream(
        self,
        *,
        thread_id: str,
        message: str,
        channel: Channel = Channel.TEST,
        user_id: str | None = None,
        installed_skills: list[str] | None = None,
        llm_client: BaseLLMClient | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Run one turn and yield per-node status events + a terminal done event.

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
        callers using ``run_turn``. The two paths converge on
        ``state_to_output`` for the terminal output.

        Why the per-node status labels don't exactly match node names:
        the CLI's ``_STAGE_LABELS`` vocabulary predates the graph
        refactor and uses short names like ``crisis_gate`` and
        ``therapeutic``. Rather than rename either side, we translate
        here. Unknown nodes fall through to the node name itself so
        future additions still render readably in the Live display.
        """

        graph = self._get_graph()
        history = await self.get_history(thread_id)

        # v0.4: track session start on the first turn we see for this
        # thread. Same logic as ``run_turn`` — see that method for the
        # "one CLI = one session" definition.
        if thread_id not in self._session_starts:
            self._session_starts[thread_id] = _iso_now()

        agent_input = AgentInput(
            message=message,
            channel=channel,
            user_id=user_id,
            session_id=thread_id,
            history=history,
            working_memory=[],
            installed_skills=list(installed_skills or []),
        )
        initial_state = build_initial_state(agent_input)

        # Carry forward multi-turn exercise state — same as run_turn.
        prev_state = await self.get_state(thread_id)
        if prev_state is not None:
            prev_progress = prev_state.get("progress", {}) or {}
            if prev_progress.get("exercise_type") is not None:
                initial_state["progress"]["exercise_type"] = prev_progress[
                    "exercise_type"
                ]
                initial_state["progress"]["exercise_step"] = prev_progress.get(
                    "exercise_step", 0
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

        async for mode, payload in graph.astream(
            initial_state,
            config=self._config_for_thread(thread_id),
            context=self._context_for_turn(llm_client=llm_client),
            stream_mode=["updates", "values"],
        ):
            if mode == "updates":
                # ``updates`` payloads are ``{node_name: delta}`` dicts,
                # usually with a single key. Emit one StatusEvent per
                # node completion. Unknown node names (e.g., therapeutic
                # subgraph internals) fall through to the raw name.
                for node_name in payload:
                    stage = _NODE_TO_STAGE.get(node_name, node_name)
                    yield StatusEvent(stage=stage)
            elif mode == "values":
                # ``values`` payloads carry the full accumulated state
                # after each step. We hold onto the latest one and use
                # it as the final state once the stream ends.
                final_state = payload  # type: ignore[assignment]

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

        yield DoneEvent(output=state_to_output(final_state))
