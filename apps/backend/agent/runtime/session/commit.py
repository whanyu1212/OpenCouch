"""Session-end memory promotion adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent.memory.session_commit_service import (
    SessionMemoryCommitResult,
    commit_session_memory,
)
from agent.state import AgentState

if TYPE_CHECKING:
    from agent.memory.policy.candidates import SessionMemoryBuffer
    from agent.memory.embeddings import EmbeddingProvider
    from agent.memory.types import StoredSessionArc
    from agent.memory.store import MemoryStore
    from llm.base import BaseLLMClient


async def run_commit_session_memory(
    state: AgentState,
    *,
    memory_store: "MemoryStore",
    session_buffer: "SessionMemoryBuffer | None",
    stored_arc: "StoredSessionArc | None",
    embedding_provider: "EmbeddingProvider | None" = None,
    llm_client: "BaseLLMClient | None" = None,
) -> SessionMemoryCommitResult | None:
    """Commit buffered semantic/procedural candidates that survived review.

    Args:
        state: Current runtime state at session end.
        memory_store: Store used for semantic/procedural writes.
        session_buffer: Runtime buffer containing held memory candidates.
        stored_arc: Optional episodic arc generated for the completed session.
        embedding_provider: Optional provider for semantic fact embeddings.
        llm_client: Optional classifier client for reconciliation.

    Returns:
        Commit result when work was attempted, otherwise ``None``.
    """

    return await commit_session_memory(
        state,
        memory_store=memory_store,
        session_buffer=session_buffer,
        stored_arc=stored_arc,
        embedding_provider=embedding_provider,
        llm_client=llm_client,
    )
