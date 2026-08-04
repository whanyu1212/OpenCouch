"""Tests for OpenAI Agents SDK text session storage."""

from __future__ import annotations

import logging

import pytest

from agent.memory.modes import MemoryMode
from agent.models import Message, MessageRole
from agent.runtime.session_store import (
    TextSessionStore,
    TextSessionStoreConfig,
    create_text_session_store,
    normalize_sqlalchemy_async_url,
)
from agent.runtime.session.history import messages_from_sdk_session_items


@pytest.mark.asyncio
async def test_sqlite_text_session_store_persists_by_thread_id(tmp_path) -> None:
    """SQLite SDK sessions should survive store re-creation by thread id."""

    db_path = tmp_path / "text-sessions.sqlite3"
    store = TextSessionStore(
        TextSessionStoreConfig(backend="sqlite", sqlite_path=db_path)
    )
    session = store.session_for_thread("thread-1")
    await session.add_items(
        [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi"}],
            },
        ]
    )
    await store.aclose()

    reloaded = TextSessionStore(
        TextSessionStoreConfig(backend="sqlite", sqlite_path=db_path)
    )
    try:
        history = await reloaded.get_history("thread-1")
    finally:
        await reloaded.aclose()

    assert [(message.role.value, message.content) for message in history] == [
        ("user", "hello"),
        ("assistant", "hi"),
    ]


@pytest.mark.asyncio
async def test_sqlite_text_session_store_keeps_threads_separate(tmp_path) -> None:
    """Session storage should not bleed history across thread ids."""

    store = TextSessionStore(
        TextSessionStoreConfig(backend="sqlite", sqlite_path=tmp_path / "sessions.db")
    )
    try:
        await store.session_for_thread("thread-a").add_items(
            [{"role": "user", "content": "alpha"}]
        )
        await store.session_for_thread("thread-b").add_items(
            [{"role": "user", "content": "beta"}]
        )

        history = await store.get_history("thread-a")
    finally:
        await store.aclose()

    assert [message.content for message in history] == ["alpha"]


@pytest.mark.asyncio
async def test_clear_thread_removes_sdk_session_items(tmp_path) -> None:
    """Thread reset should be able to clear SDK session history."""

    store = TextSessionStore(
        TextSessionStoreConfig(backend="sqlite", sqlite_path=tmp_path / "sessions.db")
    )
    try:
        await store.session_for_thread("thread-1").add_items(
            [{"role": "user", "content": "hello"}]
        )

        await store.clear_thread("thread-1")

        assert "thread-1" not in store._sessions  # noqa: SLF001
        assert await store.get_history("thread-1") == []
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_evict_thread_releases_cache_without_deleting_history(tmp_path) -> None:
    """Eviction should close the object while retaining persisted history."""

    store = TextSessionStore(
        TextSessionStoreConfig(backend="sqlite", sqlite_path=tmp_path / "sessions.db")
    )
    try:
        session = store.session_for_thread("thread-1")
        await session.add_items([{"role": "user", "content": "hello"}])

        await store.evict_thread(" thread-1 ")

        replacement = store.session_for_thread("thread-1")
        history = await store.get_history("thread-1")
    finally:
        await store.aclose()

    assert replacement is not session
    assert [message.content for message in history] == ["hello"]


@pytest.mark.asyncio
async def test_uncached_history_read_does_not_retain_session(tmp_path) -> None:
    """Completed-thread history reads should close their transient SDK session."""

    store = TextSessionStore(
        TextSessionStoreConfig(backend="sqlite", sqlite_path=tmp_path / "sessions.db")
    )
    try:
        session = store.session_for_thread("thread-1")
        await session.add_items([{"role": "user", "content": "hello"}])
        await store.evict_thread("thread-1")

        history = await store.get_history("thread-1", cache=False)

        assert [message.content for message in history] == ["hello"]
        assert store._sessions == {}  # noqa: SLF001
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_uncached_history_read_does_not_use_cached_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unlocked history reads must not race cached-session eviction."""

    class _CachedSession:
        async def get_items(self, *, limit: int | None = None) -> list[dict[str, str]]:
            raise AssertionError("uncached history read used the cached session")

    class _TransientSession:
        def __init__(self) -> None:
            self.closed = False

        async def get_items(self, *, limit: int | None = None) -> list[dict[str, str]]:
            return [{"role": "user", "content": "hello"}]

        async def close(self) -> None:
            self.closed = True

    store = TextSessionStore(TextSessionStoreConfig())
    cached_session = _CachedSession()
    transient_session = _TransientSession()
    store._sessions["thread-1"] = cached_session  # noqa: SLF001
    monkeypatch.setattr(store, "_create_session", lambda _thread_id: transient_session)

    history = await store.get_history("thread-1", cache=False)

    assert [message.content for message in history] == ["hello"]
    assert store._sessions["thread-1"] is cached_session  # noqa: SLF001
    assert transient_session.closed is True


@pytest.mark.asyncio
async def test_evicting_many_threads_does_not_retain_session_objects(tmp_path) -> None:
    """Historical threads should not make the in-process cache unbounded."""

    store = TextSessionStore(
        TextSessionStoreConfig(backend="sqlite", sqlite_path=tmp_path / "sessions.db")
    )
    try:
        for index in range(100):
            thread_id = f"thread-{index}"
            store.session_for_thread(thread_id)
            await store.evict_thread(thread_id)

        assert store._sessions == {}  # noqa: SLF001
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_evict_thread_awaits_async_sdk_close() -> None:
    """Eviction should support SDK session implementations with async close."""

    class _AsyncCloseSession:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    store = TextSessionStore(TextSessionStoreConfig())
    session = _AsyncCloseSession()
    store._sessions["thread-1"] = session  # noqa: SLF001

    await store.evict_thread("thread-1")

    assert session.closed is True
    assert store._sessions == {}  # noqa: SLF001


@pytest.mark.asyncio
async def test_uncached_history_read_awaits_async_sdk_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transient history reads should await async SDK cleanup."""

    class _AsyncReadSession:
        def __init__(self) -> None:
            self.closed = False

        async def get_items(self, *, limit: int | None = None) -> list[dict[str, str]]:
            return [{"role": "user", "content": "hello"}]

        async def close(self) -> None:
            self.closed = True

    store = TextSessionStore(TextSessionStoreConfig())
    session = _AsyncReadSession()
    monkeypatch.setattr(store, "_create_session", lambda _thread_id: session)

    history = await store.get_history("thread-1", cache=False)

    assert [message.content for message in history] == ["hello"]
    assert session.closed is True
    assert store._sessions == {}  # noqa: SLF001


@pytest.mark.asyncio
async def test_uncached_history_read_preserves_result_when_close_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Transient cleanup should not invalidate a completed history read."""

    class _CloseFailingSession:
        async def get_items(self, *, limit: int | None = None) -> list[dict[str, str]]:
            return [{"role": "user", "content": "hello"}]

        async def close(self) -> None:
            raise RuntimeError("close failed")

    store = TextSessionStore(TextSessionStoreConfig())
    monkeypatch.setattr(
        store,
        "_create_session",
        lambda _thread_id: _CloseFailingSession(),
    )

    with caplog.at_level(logging.WARNING, logger="agent.runtime.session_store"):
        history = await store.get_history("thread-1", cache=False)

    assert [message.content for message in history] == ["hello"]
    assert "transient history session close failed" in caplog.text


@pytest.mark.asyncio
async def test_uncached_history_read_preserves_read_failure_when_close_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Transient cleanup should not mask the original history read failure."""

    class _ReadAndCloseFailingSession:
        async def get_items(self, *, limit: int | None = None) -> list[dict[str, str]]:
            raise ValueError("history read failed")

        async def close(self) -> None:
            raise RuntimeError("close failed")

    store = TextSessionStore(TextSessionStoreConfig())
    monkeypatch.setattr(
        store,
        "_create_session",
        lambda _thread_id: _ReadAndCloseFailingSession(),
    )

    with caplog.at_level(logging.WARNING, logger="agent.runtime.session_store"):
        with pytest.raises(ValueError, match="history read failed"):
            await store.get_history("thread-1", cache=False)

    assert "transient history session close failed" in caplog.text


@pytest.mark.asyncio
async def test_turn_session_stores_raw_user_message_not_runtime_prompt(
    tmp_path,
) -> None:
    """Runner prompt writes should become raw public conversation history."""

    store = TextSessionStore(
        TextSessionStoreConfig(backend="sqlite", sqlite_path=tmp_path / "sessions.db")
    )
    try:
        turn_session = store.turn_session_for_thread(
            "thread-1",
            current_user_message="I feel overwhelmed",
        )

        await turn_session.add_items(
            [
                {
                    "role": "user",
                    "content": "Write the next assistant message.\nCurrent user: ...",
                },
                {"type": "function_call", "name": "show_memory_status"},
                {"role": "assistant", "content": "That sounds heavy."},
            ]
        )
        await turn_session.add_items(
            [{"role": "user", "content": "duplicate runtime prompt"}]
        )

        history = await store.get_history("thread-1")
    finally:
        await store.aclose()

    assert [(message.role.value, message.content) for message in history] == [
        ("user", "I feel overwhelmed"),
        ("assistant", "That sounds heavy."),
    ]


@pytest.mark.asyncio
async def test_seed_thread_from_messages_only_seeds_empty_session(tmp_path) -> None:
    """Persisted transcripts can bootstrap an empty SDK session."""

    store = TextSessionStore(
        TextSessionStoreConfig(backend="sqlite", sqlite_path=tmp_path / "sessions.db")
    )
    try:
        seeded = await store.seed_thread_from_messages(
            "thread-1",
            [
                Message(role=MessageRole.USER, content="hello"),
                Message(role=MessageRole.ASSISTANT, content="hi"),
            ],
        )
        seeded_again = await store.seed_thread_from_messages(
            "thread-1",
            [Message(role=MessageRole.USER, content="should not append")],
        )

        history = await store.get_history("thread-1")
    finally:
        await store.aclose()

    assert seeded is True
    assert seeded_again is False
    assert [message.content for message in history] == ["hello", "hi"]


@pytest.mark.asyncio
async def test_ensure_turn_recorded_appends_missing_turn_once(tmp_path) -> None:
    """Runtime fallback should fill history when a runner fake does not persist."""

    store = TextSessionStore(
        TextSessionStoreConfig(backend="sqlite", sqlite_path=tmp_path / "sessions.db")
    )
    try:
        await store.ensure_turn_recorded(
            "thread-1",
            user_message="hello",
            assistant_message="hi",
        )
        await store.ensure_turn_recorded(
            "thread-1",
            user_message="hello",
            assistant_message="hi",
        )

        history = await store.get_history("thread-1")
    finally:
        await store.aclose()

    assert [(message.role.value, message.content) for message in history] == [
        ("user", "hello"),
        ("assistant", "hi"),
    ]


@pytest.mark.asyncio
async def test_incognito_text_session_store_uses_memory_sqlite() -> None:
    """Incognito mode must not write SDK sessions to configured disk storage."""

    store = create_text_session_store(
        memory_mode=MemoryMode.INCOGNITO,
        backend="sqlalchemy",
        database_url="postgresql://opencouch:opencouch@localhost/opencouch",
    )

    try:
        assert store is not None
        assert store.backend == "sqlite"
    finally:
        if store is not None:
            await store.aclose()


@pytest.mark.asyncio
async def test_auto_text_session_store_uses_sqlite_without_database_url() -> None:
    """Auto mode should work in local development without extra config."""

    store = create_text_session_store(
        memory_mode=MemoryMode.LOCAL,
        backend="auto",
    )

    try:
        assert store is not None
        assert store.backend == "sqlite"
    finally:
        if store is not None:
            await store.aclose()


def test_sqlalchemy_backend_requires_database_url() -> None:
    """SQLAlchemy SDK sessions should fail fast without a URL."""

    with pytest.raises(ValueError, match="text_session_database_url"):
        TextSessionStore(TextSessionStoreConfig(backend="sqlalchemy"))


def test_normalize_sqlalchemy_async_url_adapts_common_sync_urls() -> None:
    """Runtime config may reuse the app's existing Postgres-style URL."""

    assert normalize_sqlalchemy_async_url("postgresql://u:p@host/db") == (
        "postgresql+asyncpg://u:p@host/db"
    )
    assert normalize_sqlalchemy_async_url("postgres://u:p@host/db") == (
        "postgresql+asyncpg://u:p@host/db"
    )
    assert normalize_sqlalchemy_async_url("sqlite:///tmp/session.db") == (
        "sqlite+aiosqlite:///tmp/session.db"
    )


def test_messages_from_sdk_session_items_skips_non_message_items() -> None:
    """Public history conversion should ignore tool calls and empty messages."""

    messages = messages_from_sdk_session_items(
        [
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"type": "function_call", "name": "tool"},
            {"role": "assistant", "content": ""},
            {"role": "assistant", "content": "hello"},
        ]
    )

    assert [(message.role.value, message.content) for message in messages] == [
        ("user", "hi"),
        ("assistant", "hello"),
    ]
