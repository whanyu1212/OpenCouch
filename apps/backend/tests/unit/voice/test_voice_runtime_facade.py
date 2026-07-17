"""Guard tests for the VoiceRuntimeFacade wiring.

These tests protect two invariants that are easy to break during
future refactoring:

1. **Lock identity** — the facade and the runtime must share the same
   per-thread lock instances.  If they diverge, concurrent voice and
   text mutations on the same thread race.

2. **Facade existence** — ``PersistentAgentRuntime.voice`` must be a
   ``VoiceRuntimeFacade`` so callers can rely on attribute access.
"""

from __future__ import annotations

import pytest

from agent.memory.modes import MemoryMode
from agent.runtime import PersistentAgentRuntime
from tests.support.persistence import (
    in_memory_runtime_storage_paths,
    runtime_persistence_config,
)
from agent.voice.runtime_facade import VoiceRuntimeFacade


def test_runtime_exposes_voice_facade() -> None:
    runtime = PersistentAgentRuntime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.INCOGNITO),
    )
    assert isinstance(runtime.voice, VoiceRuntimeFacade)


@pytest.mark.asyncio
async def test_voice_facade_shares_thread_lock_with_runtime() -> None:
    runtime = PersistentAgentRuntime(
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.INCOGNITO),
    )
    async with runtime:
        assert runtime.voice._lock_for("t") is runtime._thread_lock("t")
