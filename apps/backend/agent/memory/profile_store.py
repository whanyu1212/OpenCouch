"""Profile-memory stubs pending redesign.

The previous SQLite-backed implementation has been deleted as part of the
legacy cleanup. This module preserves the public surface that the rest of the
graph still imports, so callers keep working against no-op defaults until the
memory subsystem is rebuilt.
"""

from __future__ import annotations

from pathlib import Path


class SqliteProfileMemoryStore:
    """No-op profile-memory store scaffold used until the redesign.

    Accepts the same constructor argument as the original implementation
    (a sqlite path) so callers do not need to change their wiring.
    """

    def __init__(self, sqlite_path: Path | str) -> None:
        self._sqlite_path = Path(sqlite_path)

    async def initialize(self) -> None:
        """No-op initializer retained for interface parity."""

        return None

    async def list_memories(self, owner_id: str) -> list[str]:
        """Return an empty profile-memory list."""

        return []


def compile_working_memory(
    profile_memories: list[str],
    graph_memories: list[str],
) -> list[str]:
    """Merge profile and graph memory snippets into a single working list.

    The stub simply concatenates the two inputs; order and deduplication are
    responsibilities of the future redesigned implementation.
    """

    return [*profile_memories, *graph_memories]
