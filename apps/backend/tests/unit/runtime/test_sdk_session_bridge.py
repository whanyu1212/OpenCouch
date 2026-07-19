"""Unit tests for :class:`agent.runtime.sdk_session_bridge.SdkSessionBridge`."""

from __future__ import annotations

import pytest

from agent.runtime.openai_text_runtime import OpenAITextRuntime
from agent.runtime.sdk_session_bridge import SdkSessionBridge


def test_get_text_runtime_returns_same_instance() -> None:
    """The serving text runtime is lazily created once and reused."""

    bridge = SdkSessionBridge(text_session_store=None)

    first = bridge.get_text_runtime()
    second = bridge.get_text_runtime()

    assert isinstance(first, OpenAITextRuntime)
    assert first is second


@pytest.mark.asyncio
async def test_session_for_thread_returns_none_without_store() -> None:
    """No SDK session is produced when session persistence is disabled."""

    bridge = SdkSessionBridge(text_session_store=None)

    session = await bridge.session_for_thread(
        "thread-1",
        current_user_message="hello",
        prior_state=None,
    )

    assert session is None


@pytest.mark.asyncio
async def test_recover_empty_session_noops_without_store_or_state() -> None:
    """Recovery is a no-op when there is no store or no prior state."""

    bridge = SdkSessionBridge(text_session_store=None)

    assert await bridge.recover_empty_session_from_state("thread-1", None) is False


@pytest.mark.asyncio
async def test_ensure_turn_recorded_noops_without_store() -> None:
    """Turn recording is a no-op when session persistence is disabled."""

    bridge = SdkSessionBridge(text_session_store=None)

    # Should not raise.
    await bridge.ensure_turn_recorded(
        "thread-1",
        user_message="hi",
        final_state={"response_text": "hello"},  # type: ignore[arg-type]
    )
