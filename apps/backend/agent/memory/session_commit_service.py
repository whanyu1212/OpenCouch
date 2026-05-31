"""Compatibility facade for session-end memory commit service."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent.memory.commit import service as _service
from agent.memory.commit.scoring import _load_prior_session_support_texts
from agent.memory.procedural_profile import aupsert_procedural_rule

if TYPE_CHECKING:
    from agent.memory.embeddings import EmbeddingProvider
    from agent.memory.policy.candidates import SessionMemoryBuffer
    from agent.memory.store import MemoryStore
    from agent.memory.types import StoredSessionArc
    from agent.state import AgentState
    from llm.base import BaseLLMClient

SessionMemoryCommitResult = _service.SessionMemoryCommitResult


async def commit_session_memory(
    state: "AgentState",
    *,
    memory_store: "MemoryStore",
    session_buffer: "SessionMemoryBuffer | None",
    stored_arc: "StoredSessionArc | None",
    embedding_provider: "EmbeddingProvider | None" = None,
    llm_client: "BaseLLMClient | None" = None,
    user_turn_texts: list[str] | None = None,
) -> SessionMemoryCommitResult | None:
    """Compatibility wrapper that preserves legacy monkeypatch hooks for tests."""
    _service._load_prior_session_support_texts = _load_prior_session_support_texts
    _service.aupsert_procedural_rule = aupsert_procedural_rule
    return await _service.commit_session_memory(
        state,
        memory_store=memory_store,
        session_buffer=session_buffer,
        stored_arc=stored_arc,
        embedding_provider=embedding_provider,
        llm_client=llm_client,
        user_turn_texts=user_turn_texts,
    )


__all__ = [
    "SessionMemoryCommitResult",
    "commit_session_memory",
    "_load_prior_session_support_texts",
    "aupsert_procedural_rule",
]
