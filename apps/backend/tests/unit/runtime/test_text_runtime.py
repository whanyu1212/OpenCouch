"""Tests for the OpenAI text-agent runtime seam."""

from __future__ import annotations

import pytest

from agent.persistence import PersistentAgentRuntime
from agent.text_runtime import (
    DEFAULT_TEXT_AGENT_RUNTIME,
    OpenAITextAgentAdapter,
    create_text_agent_adapter,
    resolve_text_agent_runtime,
)


def test_resolve_text_agent_runtime_defaults_to_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default text runtime is OpenAI."""

    monkeypatch.delenv("OPENCOUCH_TEXT_AGENT_RUNTIME", raising=False)

    assert DEFAULT_TEXT_AGENT_RUNTIME == "openai"
    assert resolve_text_agent_runtime() == "openai"


def test_resolve_text_agent_runtime_accepts_openai_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime selection tolerates common env formatting noise."""

    monkeypatch.setenv("OPENCOUCH_TEXT_AGENT_RUNTIME", " OpenAI ")

    assert resolve_text_agent_runtime() == "openai"


def test_resolve_text_agent_runtime_rejects_langgraph_value() -> None:
    """Stale LangGraph runtime config should fail loudly."""

    with pytest.raises(ValueError, match="Supported value: openai"):
        resolve_text_agent_runtime("langgraph")


def test_resolve_text_agent_runtime_rejects_unknown_value() -> None:
    """Unsupported runtimes should fail before a turn starts."""

    with pytest.raises(ValueError, match="Supported value: openai"):
        resolve_text_agent_runtime("unknown")


def test_persistent_runtime_defaults_to_openai_text_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PersistentAgentRuntime should use OpenAI when no override is set."""

    monkeypatch.delenv("OPENCOUCH_TEXT_AGENT_RUNTIME", raising=False)

    runtime = PersistentAgentRuntime()

    assert runtime._text_agent_runtime == "openai"
    assert runtime._text_session_store is not None


def test_create_text_agent_adapter_builds_openai_adapter() -> None:
    """The factory should make OpenAI the only serving adapter."""

    adapter = create_text_agent_adapter()

    assert isinstance(adapter, OpenAITextAgentAdapter)


@pytest.mark.asyncio
async def test_runtime_reset_clears_runtime_and_sdk_session_state(tmp_path) -> None:
    """Thread reset should remove runtime state and SDK session history."""

    async with PersistentAgentRuntime(
        sqlite_path=tmp_path / "threads.sqlite3",
        text_session_backend="sqlite",
        text_session_sqlite_path=tmp_path / "text-sessions.sqlite3",
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

    async with PersistentAgentRuntime(
        sqlite_path=tmp_path / "threads.sqlite3",
        text_session_backend="sqlite",
        text_session_sqlite_path=tmp_path / "text-sessions.sqlite3",
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
