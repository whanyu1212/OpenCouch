"""Tests for centralized runtime backend selection helpers."""

from __future__ import annotations

import pytest

from agent.memory.modes import MemoryMode
from agent.runtime.backends import select_runtime_backends


@pytest.mark.parametrize("memory_mode", [MemoryMode.LOCAL, MemoryMode.SYNCED])
def test_select_runtime_backends_preserves_configured_persistent_backends(
    memory_mode: MemoryMode,
) -> None:
    selection = select_runtime_backends(
        memory_mode=memory_mode,
        memory_backend="postgres",
        thread_persistence_backend="postgres",
        crisis_log_persistence_backend="postgres",
        session_feedback_persistence_backend="postgres",
    )

    assert selection.thread_persistence_backend == "postgres"
    assert selection.memory_store_backend == "postgres"
    assert selection.crisis_log_backend == "postgres"
    assert selection.session_feedback_backend == "postgres"


def test_select_runtime_backends_forces_incognito_to_ephemeral_backends() -> None:
    selection = select_runtime_backends(
        memory_mode=MemoryMode.INCOGNITO,
        memory_backend="postgres",
        thread_persistence_backend="postgres",
        crisis_log_persistence_backend="postgres",
        session_feedback_persistence_backend="postgres",
    )

    assert selection.thread_persistence_backend == "memory"
    assert selection.memory_store_backend == "memory"
    assert selection.crisis_log_backend == "memory"
    assert selection.session_feedback_backend == "memory"
