"""Unit tests for resolve_effective_memory_mode (truth-table coverage)."""

from __future__ import annotations

import pytest

from agent.memory.modes import MemoryMode, resolve_effective_memory_mode


@pytest.mark.parametrize(
    ("runtime_mode", "requested_mode", "expected"),
    [
        # Runtime incognito always wins, regardless of request.
        (MemoryMode.INCOGNITO, "incognito", "incognito"),
        (MemoryMode.INCOGNITO, "persistent", "incognito"),
        (MemoryMode.INCOGNITO, None, "incognito"),
        # Persistent runtime defers to the request when present.
        (MemoryMode.LOCAL, "incognito", "incognito"),
        (MemoryMode.LOCAL, "persistent", "persistent"),
        (MemoryMode.LOCAL, None, "persistent"),
        # SYNCED collapses to persistent for the binary read/write surface.
        (MemoryMode.SYNCED, "incognito", "incognito"),
        (MemoryMode.SYNCED, "persistent", "persistent"),
        (MemoryMode.SYNCED, None, "persistent"),
        # String inputs are accepted (case- and whitespace-insensitive).
        ("incognito", "persistent", "incognito"),
        ("INCOGNITO", None, "incognito"),
        ("  local  ", "incognito", "incognito"),
        ("local", "persistent", "persistent"),
        # None runtime defaults to persistent (treat absent as not-strict).
        (None, "incognito", "incognito"),
        (None, "persistent", "persistent"),
        (None, None, "persistent"),
        # Unknown request strings collapse to persistent unless they say incognito.
        (MemoryMode.LOCAL, "guest", "persistent"),
        (MemoryMode.LOCAL, "", "persistent"),
    ],
)
def test_resolve_effective_memory_mode(
    runtime_mode: MemoryMode | str | None,
    requested_mode: str | None,
    expected: str,
) -> None:
    assert resolve_effective_memory_mode(runtime_mode, requested_mode) == expected
