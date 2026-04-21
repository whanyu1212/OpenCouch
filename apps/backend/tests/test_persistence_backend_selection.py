"""Tests for PersistentAgentRuntime backend selection (v0.8 Stage D).

These tests verify that the runtime picks the right memory store and
crisis log backend based on ``memory_mode`` and whether the caller
passed explicit overrides. They cover the Stage D wiring logic
specifically — not the full runtime behavior (that's Stage E).

What these tests DON'T cover:
- End-to-end cross-restart persistence (Stage E smoke test)
- CLI integration (Stage E / F)
- Actual database writes (the Stage B / C test files cover the
  SQLite backends' own behavior)

Test strategy: construct the runtime without entering its async
context (no checkpointer connection needed), check the concrete
type of ``runtime.memory_store`` and ``runtime.crisis_log_backend``.
The selection logic lives entirely in ``__init__``, so we don't
need to await anything to verify it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.memory.crisis_log import InMemoryCrisisLogBackend, NullCrisisLogBackend
from agent.memory.embeddings import NullEmbeddingProvider
from agent.memory.modes import MemoryMode
from agent.memory.session_feedback import (
    InMemorySessionFeedbackBackend,
    NullSessionFeedbackBackend,
)
from agent.memory.sqlite_crisis_log import SqliteCrisisLogBackend
from agent.memory.sqlite_session_feedback import SqliteSessionFeedbackBackend
from agent.memory.sqlite_store import SqliteMemoryStore
from agent.memory.store import OpenCouchMemoryStore
from agent.persistence import (
    DEFAULT_CRISIS_LOG_DB_PATH,
    DEFAULT_FEEDBACK_DB_PATH,
    DEFAULT_MEMORY_DB_PATH,
    PersistentAgentRuntime,
)


# ─── Default-path constants ────────────────────────────────────────────


def test_default_memory_db_path_is_distinct_from_thread_db() -> None:
    """The v0.8 memory SQLite file must NOT share a path with the
    LangGraph thread checkpointer file. Mixing our tables with
    LangGraph's risks namespace collisions when LangGraph bumps
    its schema."""

    from agent.persistence import DEFAULT_THREAD_DB_PATH

    # All four OpenCouch-owned SQLite files must be distinct from
    # each other AND from the LangGraph thread DB. Crossed paths
    # would mix schemas in ways that break on future migrations.
    paths = {
        DEFAULT_THREAD_DB_PATH,
        DEFAULT_MEMORY_DB_PATH,
        DEFAULT_CRISIS_LOG_DB_PATH,
        DEFAULT_FEEDBACK_DB_PATH,
    }
    assert len(paths) == 4, f"expected 4 distinct paths, got {paths}"


def test_default_paths_live_next_to_thread_db() -> None:
    """All four OpenCouch-owned SQLite files should sit in the same
    directory (``apps/backend/``) so operators can find them together."""

    from agent.persistence import DEFAULT_THREAD_DB_PATH

    assert DEFAULT_MEMORY_DB_PATH.parent == DEFAULT_THREAD_DB_PATH.parent
    assert DEFAULT_CRISIS_LOG_DB_PATH.parent == DEFAULT_THREAD_DB_PATH.parent
    assert DEFAULT_FEEDBACK_DB_PATH.parent == DEFAULT_THREAD_DB_PATH.parent


def test_default_paths_are_in_store_dir() -> None:
    """The SQLite files should live under ``.store/`` so they
    don't clutter the backend root."""

    assert DEFAULT_MEMORY_DB_PATH.parent.name == ".store"
    assert DEFAULT_CRISIS_LOG_DB_PATH.parent.name == ".store"
    assert DEFAULT_FEEDBACK_DB_PATH.parent.name == ".store"
    assert DEFAULT_MEMORY_DB_PATH.suffix == ".sqlite3"
    assert DEFAULT_CRISIS_LOG_DB_PATH.suffix == ".sqlite3"
    assert DEFAULT_FEEDBACK_DB_PATH.suffix == ".sqlite3"


# ─── Incognito mode — in-memory backings ───────────────────────────────


def test_incognito_mode_uses_in_memory_store_by_default() -> None:
    """In incognito mode without an explicit override, the runtime
    should construct an in-memory store — nothing hits disk."""

    runtime = PersistentAgentRuntime(memory_response_style=MemoryMode.INCOGNITO)
    assert isinstance(runtime.memory_store, OpenCouchMemoryStore)
    assert not isinstance(runtime.memory_store, SqliteMemoryStore)


def test_incognito_mode_uses_in_memory_crisis_log_by_default() -> None:
    """Same as the memory store — incognito crisis log must not
    touch disk."""

    runtime = PersistentAgentRuntime(memory_response_style=MemoryMode.INCOGNITO)
    assert isinstance(runtime.crisis_log_backend, InMemoryCrisisLogBackend)
    assert not isinstance(runtime.crisis_log_backend, SqliteCrisisLogBackend)


def test_incognito_mode_sqlite_path_forced_to_memory() -> None:
    """The LangGraph thread checkpointer path should be ``:memory:``
    in incognito mode regardless of what the caller passed. This is
    the pre-v0.8 behavior; v0.8 just preserves it."""

    runtime = PersistentAgentRuntime(
        sqlite_path="/tmp/should-be-ignored.sqlite3",
        memory_response_style=MemoryMode.INCOGNITO,
    )
    assert runtime.sqlite_path == Path(":memory:")


# ─── Local mode — SQLite-backed defaults ───────────────────────────────


def test_local_mode_uses_sqlite_memory_store_by_default() -> None:
    """In local mode without an explicit override, the runtime should
    construct a :class:`SqliteMemoryStore` pointing at the default
    memory SQLite path. This is the core v0.8 Stage D wiring."""

    runtime = PersistentAgentRuntime(memory_response_style=MemoryMode.LOCAL)
    assert isinstance(runtime.memory_store, SqliteMemoryStore)
    # The SqliteMemoryStore's path should be the default memory path.
    assert runtime.memory_store.sqlite_path == Path(DEFAULT_MEMORY_DB_PATH)


def test_local_mode_uses_sqlite_crisis_log_by_default() -> None:
    """Same as the memory store — local mode crisis log should be
    SQLite-backed and pointed at the default crisis log path."""

    runtime = PersistentAgentRuntime(memory_response_style=MemoryMode.LOCAL)
    assert isinstance(runtime.crisis_log_backend, SqliteCrisisLogBackend)
    assert runtime.crisis_log_backend.sqlite_path == Path(DEFAULT_CRISIS_LOG_DB_PATH)


def test_local_mode_accepts_custom_sqlite_paths(tmp_path: Path) -> None:
    """Callers can override the default SQLite paths — useful for
    tests that want to isolate their files in a tmp directory, and
    for operators who want non-default locations."""

    custom_memory = tmp_path / "custom_memory.sqlite3"
    custom_crisis = tmp_path / "custom_crisis.sqlite3"

    runtime = PersistentAgentRuntime(
        memory_response_style=MemoryMode.LOCAL,
        memory_sqlite_path=custom_memory,
        crisis_log_sqlite_path=custom_crisis,
    )

    assert isinstance(runtime.memory_store, SqliteMemoryStore)
    assert runtime.memory_store.sqlite_path == custom_memory
    assert isinstance(runtime.crisis_log_backend, SqliteCrisisLogBackend)
    assert runtime.crisis_log_backend.sqlite_path == custom_crisis


def test_synced_mode_behaves_like_local_for_v0_8() -> None:
    """SYNCED mode is reserved for a future remote backend. For now
    (v0.8), it should behave identically to LOCAL — SQLite-backed
    memory store and crisis log. This test pins that behavior so
    we don't accidentally break SYNCED mode callers when the remote
    backend actually ships."""

    runtime = PersistentAgentRuntime(memory_response_style=MemoryMode.SYNCED)
    assert isinstance(runtime.memory_store, SqliteMemoryStore)
    assert isinstance(runtime.crisis_log_backend, SqliteCrisisLogBackend)


# ─── Explicit overrides ────────────────────────────────────────────────


def test_explicit_memory_store_overrides_mode_based_selection() -> None:
    """Passing an explicit ``memory_store`` should bypass the
    mode-based defaults entirely. Tests rely on this — they
    construct an in-memory store directly and expect the runtime
    to use it regardless of the mode flag."""

    custom_store = OpenCouchMemoryStore()
    runtime = PersistentAgentRuntime(
        memory_response_style=MemoryMode.LOCAL,  # would normally pick SqliteMemoryStore
        memory_store=custom_store,
    )
    # The runtime should hold the exact instance we passed in.
    assert runtime.memory_store is custom_store


def test_explicit_crisis_log_overrides_mode_based_selection() -> None:
    """Same pattern as the memory store — explicit crisis log
    backend bypasses the mode-based default."""

    custom_backend = NullCrisisLogBackend()
    runtime = PersistentAgentRuntime(
        memory_response_style=MemoryMode.LOCAL,
        crisis_log_backend=custom_backend,
    )
    assert runtime.crisis_log_backend is custom_backend


def test_explicit_overrides_work_with_incognito_too() -> None:
    """Even in incognito mode, an explicit override should be
    respected. This lets test fixtures pass a specific backend
    (e.g., a mock or a failing backend) regardless of the mode."""

    custom_store = OpenCouchMemoryStore()
    custom_backend = NullCrisisLogBackend()
    runtime = PersistentAgentRuntime(
        memory_response_style=MemoryMode.INCOGNITO,
        memory_store=custom_store,
        crisis_log_backend=custom_backend,
    )
    assert runtime.memory_store is custom_store
    assert runtime.crisis_log_backend is custom_backend


# ─── Mixed overrides ───────────────────────────────────────────────────


def test_can_override_only_memory_store() -> None:
    """The two backend parameters are independent. Overriding just
    the memory store should leave the crisis log to mode-based
    selection."""

    custom_store = OpenCouchMemoryStore()
    runtime = PersistentAgentRuntime(
        memory_response_style=MemoryMode.LOCAL,
        memory_store=custom_store,
    )
    assert runtime.memory_store is custom_store
    # Crisis log still picks the SQLite default for LOCAL mode
    assert isinstance(runtime.crisis_log_backend, SqliteCrisisLogBackend)


def test_can_override_only_crisis_log() -> None:
    """Symmetric to the memory-store-only override."""

    custom_backend = NullCrisisLogBackend()
    runtime = PersistentAgentRuntime(
        memory_response_style=MemoryMode.LOCAL,
        crisis_log_backend=custom_backend,
    )
    # Memory store still picks the SQLite default for LOCAL mode
    assert isinstance(runtime.memory_store, SqliteMemoryStore)
    assert runtime.crisis_log_backend is custom_backend


# ─── Connection is NOT opened in __init__ ──────────────────────────────


def test_init_does_not_open_sqlite_connections() -> None:
    """``__init__`` should be cheap and never touch disk. The SQLite
    backends use lazy connection opening (connection is created on
    first async method call, not during construction). This test
    verifies that by checking the internal ``_connection`` attribute
    on both SQLite backends — it should be None after init."""

    runtime = PersistentAgentRuntime(memory_response_style=MemoryMode.LOCAL)
    # Both backends should have their _connection set to None.
    # These are internal attributes, not public API — this test
    # reaches into them specifically to verify the lazy-open
    # contract. If the internals change, the test needs to update.
    memory_store = runtime.memory_store
    crisis_log = runtime.crisis_log_backend
    assert isinstance(memory_store, SqliteMemoryStore)
    assert isinstance(crisis_log, SqliteCrisisLogBackend)
    assert memory_store._connection is None  # noqa: SLF001
    assert crisis_log._connection is None  # noqa: SLF001


# ─── Close safety ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_init_without_enter_still_closes_cleanly() -> None:
    """If a runtime is constructed but never entered (e.g., a test
    that only checks ``__init__`` behavior, or an error path that
    aborts before ``async with``), calling ``aclose`` on the
    backends should still be safe. The SQLite backends' lazy-open
    means "never opened" is just "no-op close"."""

    runtime = PersistentAgentRuntime(memory_response_style=MemoryMode.LOCAL)
    # Neither backend has had a chance to open its connection.
    # Closing them should not raise.
    await runtime.memory_store.aclose()
    await runtime.crisis_log_backend.aclose()
    await runtime.session_feedback_backend.aclose()


# ─── v0.10 session-feedback backend selection ──────────────────────────


def test_incognito_mode_uses_in_memory_feedback_by_default() -> None:
    """Incognito mode should pick the in-memory feedback backend —
    nothing hits disk. The feedback collector is always-on regardless
    of mode, but incognito keeps it ephemeral."""

    runtime = PersistentAgentRuntime(memory_response_style=MemoryMode.INCOGNITO)
    assert isinstance(runtime.session_feedback_backend, InMemorySessionFeedbackBackend)
    assert not isinstance(
        runtime.session_feedback_backend, SqliteSessionFeedbackBackend
    )


def test_local_mode_uses_sqlite_feedback_by_default() -> None:
    """Local mode should pick the SQLite feedback backend at the
    default feedback path."""

    runtime = PersistentAgentRuntime(memory_response_style=MemoryMode.LOCAL)
    assert isinstance(runtime.session_feedback_backend, SqliteSessionFeedbackBackend)
    assert runtime.session_feedback_backend.sqlite_path == Path(
        DEFAULT_FEEDBACK_DB_PATH
    )


def test_synced_mode_uses_sqlite_feedback_by_default() -> None:
    """SYNCED mode behaves like LOCAL for v0.10 — SQLite-backed
    feedback at the default path. Pins the behavior so SYNCED mode
    callers don't silently lose feedback when the remote backend
    eventually ships."""

    runtime = PersistentAgentRuntime(memory_response_style=MemoryMode.SYNCED)
    assert isinstance(runtime.session_feedback_backend, SqliteSessionFeedbackBackend)


def test_local_mode_accepts_custom_feedback_sqlite_path(tmp_path: Path) -> None:
    """Operators and test fixtures can override the default path —
    useful for pointing feedback at an isolated tmp file."""

    custom = tmp_path / "custom_feedback.sqlite3"
    runtime = PersistentAgentRuntime(
        memory_response_style=MemoryMode.LOCAL,
        feedback_sqlite_path=custom,
    )
    assert isinstance(runtime.session_feedback_backend, SqliteSessionFeedbackBackend)
    assert runtime.session_feedback_backend.sqlite_path == custom


def test_explicit_feedback_backend_overrides_mode_based_selection() -> None:
    """Passing an explicit ``session_feedback_backend`` should bypass
    the mode-based default — matches the crisis_log and memory_store
    override contract so tests can inject Null/Mock backends
    regardless of mode."""

    custom_backend = NullSessionFeedbackBackend()
    runtime = PersistentAgentRuntime(
        memory_response_style=MemoryMode.LOCAL,  # would normally pick SQLite
        session_feedback_backend=custom_backend,
    )
    assert runtime.session_feedback_backend is custom_backend


def test_feedback_backend_lazy_connection_in_init() -> None:
    """The SQLite feedback backend should NOT open its connection
    during ``__init__``. Matches the crisis_log and memory_store
    lazy-open contract."""

    runtime = PersistentAgentRuntime(memory_response_style=MemoryMode.LOCAL)
    backend = runtime.session_feedback_backend
    assert isinstance(backend, SqliteSessionFeedbackBackend)
    assert backend._connection is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_aexit_closes_feedback_backend() -> None:
    """``PersistentAgentRuntime.__aexit__`` must call ``aclose`` on
    the feedback backend, symmetric to the memory store and
    crisis_log lifecycle. A leaked aiosqlite connection on exit
    would accumulate across CLI / API sessions."""

    class _CountingBackend(InMemorySessionFeedbackBackend):
        def __init__(self) -> None:
            super().__init__()
            self.close_calls = 0

        async def aclose(self) -> None:  # type: ignore[override]
            self.close_calls += 1
            await super().aclose()

    backend = _CountingBackend()
    runtime = PersistentAgentRuntime(
        sqlite_path=":memory:",
        memory_sqlite_path=":memory:",
        crisis_log_sqlite_path=":memory:",
        session_feedback_backend=backend,
    )
    async with runtime:
        # Fine — nothing to assert here; we just want __aexit__ to fire.
        pass

    # Exactly one close call on the feedback backend.
    assert backend.close_calls == 1


@pytest.mark.asyncio
async def test_aenter_prewarms_embedding_provider_and_graph(monkeypatch) -> None:
    """``__aenter__`` should finish warmup before the runtime is usable."""

    class _WarmableEmbeddingProvider(NullEmbeddingProvider):
        def __init__(self) -> None:
            self.warmup_calls = 0

        async def awarmup(self) -> None:  # type: ignore[override]
            self.warmup_calls += 1

    embedding_provider = _WarmableEmbeddingProvider()
    compiled_graph = object()

    monkeypatch.setattr(
        "agent.persistence.build_agent_workflow",
        lambda checkpointer: compiled_graph,
    )

    runtime = PersistentAgentRuntime(
        sqlite_path=":memory:",
        memory_sqlite_path=":memory:",
        crisis_log_sqlite_path=":memory:",
        embedding_provider=embedding_provider,
        finalize_active_sessions_on_close=False,
    )

    async with runtime:
        assert embedding_provider.warmup_calls == 1
        assert runtime._graph is compiled_graph  # noqa: SLF001
