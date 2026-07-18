"""Tests for the OpenAI text-agent runtime."""

from __future__ import annotations

import pytest

from agent.memory.modes import MemoryMode
from agent.runtime import (
    OpenAITextRuntime,
    PersistentAgentRuntime,
    RuntimePersistenceConfig,
    RuntimeStoragePaths,
)
from tests.support.persistence import in_memory_audit_feedback_dependencies


def _runtime(**kwargs) -> PersistentAgentRuntime:
    return PersistentAgentRuntime(
        dependencies=in_memory_audit_feedback_dependencies(),
        **kwargs,
    )


def test_persistent_runtime_defaults_to_openai_text_runtime() -> None:
    """PersistentAgentRuntime should use the OpenAI text runtime."""

    runtime = _runtime(
        storage_paths=RuntimeStoragePaths(
            sqlite_path=":memory:",
            text_session_sqlite_path=":memory:",
        ),
        persistence_config=RuntimePersistenceConfig(
            memory_mode=MemoryMode.INCOGNITO,
        ),
    )

    assert runtime._text_session_store is not None
    assert runtime._sdk_bridge._openai_text_runtime is None


@pytest.mark.asyncio
async def test_prewarm_initializes_openai_text_runtime(tmp_path) -> None:
    """Runtime prewarm should initialize the OpenAI runtime before use."""

    async with _runtime(
        storage_paths=RuntimeStoragePaths(
            sqlite_path=tmp_path / "threads.sqlite3",
            text_session_sqlite_path=tmp_path / "text-sessions.sqlite3",
        ),
        persistence_config=RuntimePersistenceConfig(
            thread_persistence_backend="memory",
            text_session_backend="sqlite",
            allow_legacy_sqlite=True,
        ),
    ) as runtime:
        assert isinstance(runtime._sdk_bridge._openai_text_runtime, OpenAITextRuntime)


@pytest.mark.asyncio
async def test_runtime_reset_clears_runtime_and_sdk_session_state(tmp_path) -> None:
    """Thread reset should remove runtime state and SDK session history."""

    async with _runtime(
        storage_paths=RuntimeStoragePaths(
            sqlite_path=tmp_path / "threads.sqlite3",
            text_session_sqlite_path=tmp_path / "text-sessions.sqlite3",
        ),
        persistence_config=RuntimePersistenceConfig(
            thread_persistence_backend="memory",
            text_session_backend="sqlite",
            allow_legacy_sqlite=True,
        ),
    ) as runtime:
        await runtime._state_store.save_state(
            "thread-1",
            {
                "transcript": [{"role": "user", "content": "hello"}],
                "session_progress": {"turn_count": 1},
            },
        )
        assert runtime._text_session_store is not None
        session = runtime._text_session_store.session_for_thread("thread-1")
        await session.add_items([{"role": "user", "content": "hello"}])

        await runtime.reset_thread("thread-1")

        assert await runtime.get_state("thread-1") is None
        assert await runtime._text_session_store.get_history("thread-1") == []


@pytest.mark.asyncio
async def test_runtime_history_falls_back_to_runtime_state_transcript(tmp_path) -> None:
    """History remains available from app-owned runtime state snapshots."""

    async with _runtime(
        storage_paths=RuntimeStoragePaths(
            sqlite_path=tmp_path / "threads.sqlite3",
            text_session_sqlite_path=tmp_path / "text-sessions.sqlite3",
        ),
        persistence_config=RuntimePersistenceConfig(
            thread_persistence_backend="memory",
            text_session_backend="sqlite",
            allow_legacy_sqlite=True,
        ),
    ) as runtime:
        await runtime._state_store.save_state(
            "thread-1",
            {
                "transcript": [
                    {"role": "user", "content": "saved user"},
                    {"role": "assistant", "content": "saved assistant"},
                ],
                "session_progress": {"turn_count": 1},
            },
        )

        history = await runtime.get_history("thread-1")

        assert [(message.role.value, message.content) for message in history] == [
            ("user", "saved user"),
            ("assistant", "saved assistant"),
        ]
        assert runtime._text_session_store is not None
        assert await runtime._text_session_store.get_history("thread-1") == []
