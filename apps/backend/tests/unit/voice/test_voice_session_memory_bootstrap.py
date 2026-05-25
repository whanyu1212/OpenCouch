"""Regression tests for voice_session_memory_context bootstrap behavior.

Phase 2 A1: voice no longer performs semantic recall at session start with
a placeholder query. These tests guard that contract by mocking the heavy
recall helpers and asserting they are never called from the bootstrap path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from agent.memory.modes import MemoryMode


@dataclass
class _StubProceduralProfile:
    proactive_recall_enabled: bool = True
    rules: list[Any] = field(default_factory=list)


class _FakeRuntime:
    """Minimal stand-in exposing the surface voice_session_memory_context uses.

    Only ``_memory_store``, ``memory_mode``, and ``_build_turn_initial_state``
    are referenced by the function under test.
    """

    def __init__(self, *, memory_mode: MemoryMode = MemoryMode.LOCAL) -> None:
        self._memory_store: object = object()
        self.memory_mode = memory_mode

    def _build_turn_initial_state(self, **kwargs: object) -> dict[str, Any]:
        # Mirror the contract: returns an AgentState-shaped dict with at
        # minimum the fields resolve_owner_id needs.
        return {
            "user_id": kwargs.get("user_id"),
            "session_id": kwargs.get("thread_id"),
            "channel": kwargs.get("channel"),
            "message": kwargs.get("message"),
            "transcript": [],
        }


@pytest.fixture
def fake_runtime() -> _FakeRuntime:
    return _FakeRuntime()


@pytest.mark.asyncio
async def test_voice_bootstrap_does_not_call_load_memory_for_turn(
    fake_runtime: _FakeRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A1 regression: bootstrap must not run semantic recall."""

    from agent.runtime import runtime as runtime_module

    async def fail_loader(**_: object) -> Any:
        raise AssertionError(
            "voice_session_memory_context must not call load_memory_for_turn; "
            "session bootstrap should only fetch the procedural profile."
        )

    async def fail_delta(*_: object, **__: object) -> Any:
        raise AssertionError(
            "voice_session_memory_context must not call build_turn_memory_delta; "
            "session bootstrap should only fetch the procedural profile."
        )

    async def fake_profile(_store: object, *, user_id: str) -> _StubProceduralProfile:
        del user_id
        return _StubProceduralProfile(proactive_recall_enabled=True)

    monkeypatch.setattr(runtime_module, "load_memory_for_turn", fail_loader)
    monkeypatch.setattr(runtime_module, "build_turn_memory_delta", fail_delta)
    monkeypatch.setattr(runtime_module, "aget_procedural_profile", fake_profile)

    # Call the bound method directly with the fake runtime as self.
    result = await runtime_module.PersistentAgentRuntime.voice_session_memory_context(
        fake_runtime,
        thread_id="voice-thread",
        user_id="alice",
        memory_mode="persistent",
    )

    # Bootstrap returned without invoking either fail_* helper. Any return
    # value (empty string, procedural-rules block, etc.) is acceptable as
    # long as the heavy recall paths were skipped.
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_voice_bootstrap_returns_empty_for_incognito_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Incognito runtime must short-circuit before any store reads."""

    from agent.runtime import runtime as runtime_module

    async def fail_profile(_store: object, *, user_id: str) -> Any:
        raise AssertionError(
            "voice_session_memory_context must not fetch a profile in incognito."
        )

    monkeypatch.setattr(runtime_module, "aget_procedural_profile", fail_profile)

    fake_runtime = _FakeRuntime(memory_mode=MemoryMode.INCOGNITO)
    result = await runtime_module.PersistentAgentRuntime.voice_session_memory_context(
        fake_runtime,
        thread_id="voice-thread",
        user_id="alice",
        memory_mode="persistent",
    )

    assert result == ""
