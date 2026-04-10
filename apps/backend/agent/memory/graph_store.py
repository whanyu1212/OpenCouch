"""Graph-memory stubs pending redesign.

The previous Graphiti-backed implementation has been deleted as part of the
legacy cleanup. This module preserves the public surface (class names, function
signatures) that the rest of the graph still imports, so callers keep working
against no-op defaults until the memory subsystem is rebuilt.

Every function here is intentionally minimal and side-effect free.
"""

from __future__ import annotations

from typing import Protocol

from agent.state import AgentState


class GraphMemoryStore(Protocol):
    """Protocol for episodic/graph memory stores."""

    async def retrieve(
        self,
        *,
        owner_id: str,
        query: str,
        limit: int = 3,
    ) -> list[str]: ...

    async def persist(self, *, owner_id: str, state: AgentState) -> None: ...


class NullGraphMemoryStore:
    """No-op graph-memory store used when graph memory is disabled or stubbed."""

    async def retrieve(
        self,
        *,
        owner_id: str,
        query: str,
        limit: int = 3,
    ) -> list[str]:
        """Return an empty retrieval result."""

        return []

    async def persist(self, *, owner_id: str, state: AgentState) -> None:
        """Ignore persistence requests in the stub implementation."""

        return None


def should_retrieve_graph_memory(
    *,
    message: str,
    prior_state: AgentState,
) -> bool:
    """Return whether graph memory should be queried for the current turn.

    The stub always returns ``False`` so `load_memory_node` short-circuits
    retrieval until the memory redesign lands.
    """

    return False


def build_graph_memory_query(
    *,
    message: str,
    prior_state: AgentState,
) -> str:
    """Return a retrieval query derived from the current turn.

    The stub passes the message through unchanged.
    """

    return message


def create_graph_memory_store_from_env() -> GraphMemoryStore:
    """Return a graph-memory store selected from environment configuration.

    The stub always returns a :class:`NullGraphMemoryStore`.
    """

    return NullGraphMemoryStore()
