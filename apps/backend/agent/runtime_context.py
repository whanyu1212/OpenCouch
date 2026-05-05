"""Runtime dependency contract for the OpenCouch LangGraph workflow.

Graph state in ``agent.state`` stores checkpointable conversation channels:
messages, routing outputs, prompt-visible memory, and diagnostics. This module
defines the separate runtime context for process-owned services that should not
be serialized into checkpoints.

``agent.graph`` and ``agent.therapeutic.graph`` register ``WorkflowContext`` as
their LangGraph ``context_schema``. Nodes then read dependencies from
``runtime.context``. One-shot calls build the context in ``agent.graph``;
thread-persistent sessions build it in ``agent.persistence`` so each thread can
share runtime stores while keeping graph state scoped to the checkpoint.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from agent.memory.policy.candidates import SessionMemoryBuffer
from agent.audit.crisis_log import CrisisLogBackend
from agent.memory.embeddings import EmbeddingProvider
from agent.memory.modes import MemoryMode
from agent.memory.recall import LoadMemoryResult
from agent.memory.store import MemoryStore
from services.llm.base import BaseLLMClient


@dataclass(slots=True, frozen=True)
class WorkflowContext:
    """Immutable dependency bundle injected through ``runtime.context``.

    Attributes:
        llm_client: Control-plane LLM used by safety classification, routing,
            memory extraction, and other structured/background tasks.
        memory_store: Shared semantic, episodic, and procedural memory store.
            ``load_memory_node`` reads from it; extractor and exercise nodes may
            write to it when memory mode allows.
        crisis_log_backend: Always-on audit backend used by
            ``crisis_log_node``. This remains available even when user memory
            is incognito.
        memory_mode: Current persistence tier. Memory-aware nodes use this to
            decide whether durable memory reads/writes are allowed.
        response_llm: Optional response-writing LLM. Therapeutic response nodes
            prefer this and fall back to ``llm_client`` when it is not set.
        embedding_provider: Optional embedding provider for hybrid retrieval
            and write-time indexing.
        session_memory_buffer: Optional per-thread candidate buffer. Hot-path
            extractors add held semantic/procedural candidates here so
            session-end commit can promote or drop them.
        pre_fetched_memory: Optional in-flight memory fetch the runtime
            scheduled at turn start so it overlaps with crisis/control/grounded
            gates. ``load_memory_node`` awaits it on the therapeutic path.
            ``None`` when speculation is disabled, the turn is incognito, or
            the runtime did not pre-schedule (e.g., one-shot calls via
            ``run_agent``).
    """

    llm_client: BaseLLMClient | None
    memory_store: MemoryStore
    crisis_log_backend: CrisisLogBackend
    memory_mode: MemoryMode
    response_llm: BaseLLMClient | None = None
    embedding_provider: EmbeddingProvider | None = None
    session_memory_buffer: SessionMemoryBuffer | None = None
    pre_fetched_memory: asyncio.Task[LoadMemoryResult] | None = None

    @property
    def control_llm(self) -> BaseLLMClient | None:
        """Return the infrastructure client used by safety, routing, and memory.

        Returns:
            Shared control LLM client, if configured.
        """

        return self.llm_client
