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
from agent.runtime import PersistentAgentRuntime
from tests.support.persistence import (
    in_memory_runtime_storage_paths,
    runtime_persistence_config,
)
from agent.voice import runtime_facade as facade_module
from agent.voice.runtime_facade import VoiceRuntimeFacade


@dataclass
class _StubProceduralRule:
    rule: str


@dataclass
class _StubProceduralProfile:
    proactive_recall_enabled: bool = True
    rules: list[Any] = field(default_factory=list)


class _FakeBackingRuntime:
    """Minimal stand-in for the runtime back-pointer the facade uses."""

    def _build_turn_initial_state(self, **kwargs: object) -> dict[str, Any]:
        del kwargs
        raise AssertionError(
            "voice_session_memory_context must resolve the bootstrap owner "
            "without building an AgentInput turn."
        )


class _FakeFacade:
    """Minimal stand-in that can receive unbound-method calls to
    ``VoiceRuntimeFacade.voice_session_memory_context``."""

    def __init__(self, *, memory_mode: MemoryMode = MemoryMode.LOCAL) -> None:
        self._runtime = _FakeBackingRuntime()
        self._memory_store: object = object()
        self._memory_mode = memory_mode


@pytest.fixture
def fake_facade() -> _FakeFacade:
    return _FakeFacade()


@pytest.mark.asyncio
async def test_voice_bootstrap_does_not_call_load_memory_for_turn(
    fake_facade: _FakeFacade,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A1 regression: bootstrap must not run semantic recall.

    Enforced two ways:

    1. ``load_memory_for_turn`` must not be a module-level attribute on
       ``runtime_facade`` — if a future refactor re-imports it, this
       assertion fires before the test even runs the bootstrap.
    2. ``aget_procedural_profile`` is monkeypatched so the bootstrap can
       complete without a real memory store.
    """

    assert not hasattr(facade_module, "load_memory_for_turn"), (
        "voice_session_memory_context must not import load_memory_for_turn; "
        "session bootstrap should only fetch the procedural profile."
    )

    seen_user_id: str | None = None

    async def fake_profile(_store: object, *, user_id: str) -> _StubProceduralProfile:
        nonlocal seen_user_id
        seen_user_id = user_id
        return _StubProceduralProfile(
            proactive_recall_enabled=True,
            rules=[_StubProceduralRule(rule="Reply briefly.")],
        )

    monkeypatch.setattr(facade_module, "aget_procedural_profile", fake_profile)

    result = await VoiceRuntimeFacade.voice_session_memory_context(
        fake_facade,
        thread_id="voice-thread",
        user_id="alice",
        memory_mode="persistent",
    )

    assert isinstance(result, str)
    assert seen_user_id == "alice"
    assert "Saved response preferences:\n- Reply briefly." in result
    assert "Proactive memory recall is enabled." in result


@pytest.mark.asyncio
async def test_voice_bootstrap_real_runtime_uses_thread_id_without_user_utterance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent bootstrap must not require a non-empty user message."""

    seen_user_id: str | None = None

    async def fake_profile(_store: object, *, user_id: str) -> _StubProceduralProfile:
        nonlocal seen_user_id
        seen_user_id = user_id
        return _StubProceduralProfile(proactive_recall_enabled=False)

    monkeypatch.setattr(facade_module, "aget_procedural_profile", fake_profile)

    runtime = PersistentAgentRuntime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
    )
    async with runtime:
        result = await runtime.voice.voice_session_memory_context(
            thread_id="voice-thread",
            user_id=None,
            memory_mode="persistent",
        )

    assert isinstance(result, str)
    assert seen_user_id == "voice-thread"


@pytest.mark.asyncio
async def test_voice_bootstrap_returns_empty_for_incognito_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Incognito runtime must short-circuit before any store reads."""

    async def fail_profile(_store: object, *, user_id: str) -> Any:
        raise AssertionError(
            "voice_session_memory_context must not fetch a profile in incognito."
        )

    monkeypatch.setattr(facade_module, "aget_procedural_profile", fail_profile)

    fake = _FakeFacade(memory_mode=MemoryMode.INCOGNITO)
    result = await VoiceRuntimeFacade.voice_session_memory_context(
        fake,
        thread_id="voice-thread",
        user_id="alice",
        memory_mode="persistent",
    )

    assert result == ""
