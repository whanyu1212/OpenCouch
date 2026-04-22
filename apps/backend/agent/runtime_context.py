"""Runtime-only dependencies injected into the LangGraph workflow.

This module owns the :class:`WorkflowContext` dataclass that the graph nodes
read via ``runtime.context``. It lives in its own module so both
:mod:`agent.graph` (which registers the context schema with the StateGraph)
and the node modules (which read from ``runtime.context``) can import it
without creating an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.memory.candidates import SessionMemoryBuffer
from agent.memory.crisis_log import CrisisLogBackend
from agent.memory.embeddings import EmbeddingProvider
from agent.memory.modes import MemoryMode
from agent.memory.store import MemoryStore
from services.llm.base import BaseLLMClient


@dataclass(slots=True, frozen=True)
class WorkflowContext:
    """Runtime-only dependencies injected into the LangGraph workflow."""

    llm_client: BaseLLMClient | None
    # The unified memory store is the single entry point for long-term
    # memory reads and writes. It fans out internally to the right
    # namespace (semantic / episodic / procedural). Typed as the
    # protocol so either implementation can be injected —
    # :class:`OpenCouchMemoryStore` for in-memory (tests, incognito)
    # or :class:`SqliteMemoryStore` for persistent (v0.8+, local mode).
    # See ``agent/memory/store.py`` for the protocol definition.
    memory_store: MemoryStore
    # The always-on crisis safety log. Writes regardless of memory_mode;
    # incognito mode still records events, but without user identity.
    crisis_log_backend: CrisisLogBackend
    memory_mode: MemoryMode
    response_llm: BaseLLMClient | None = None
    # v0.8.1: embedding provider for hybrid retrieval. When set,
    # the extractor nodes compute embeddings at write time and
    # the load_memory node computes query embeddings for the
    # RRF fusion path in the store's ``asearch_similar`` method.
    # When None or :class:`NullEmbeddingProvider`, retrieval
    # degrades to token-recall only — the v0.3.1 contract that
    # shipped before v0.8.1. Defaults to None so test helpers can
    # construct minimal contexts without specifying an embedding
    # provider explicitly.
    embedding_provider: EmbeddingProvider | None = None
    # Phase 2: held semantic/procedural candidates live here until
    # end_session decides whether they are durable enough to commit.
    session_memory_buffer: SessionMemoryBuffer | None = None

    @property
    def control_llm(self) -> BaseLLMClient | None:
        """Pinned infrastructure client used by safety, routing, and memory."""

        return self.llm_client
