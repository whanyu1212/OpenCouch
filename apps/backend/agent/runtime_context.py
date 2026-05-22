"""Runtime dependency contract for OpenCouch text-agent services.

Agent state in ``agent.state`` stores conversation channels: messages, routing
outputs, prompt-visible memory, and diagnostics. This module defines the
separate runtime context for process-owned services that should not be
serialized into runtime state snapshots.

One-shot calls build the context in ``agent.runtime``; thread-persistent sessions
build it in ``agent.runtime`` so each thread can share runtime stores while
keeping state snapshots scoped to the thread.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from agent.memory.policy.candidates import SessionMemoryBuffer
from agent.audit.crisis_log import CrisisLogBackend
from agent.memory.embeddings import EmbeddingProvider
from agent.memory.modes import MemoryMode
from agent.memory.recall import LoadMemoryResult
from agent.memory.store import MemoryStore
from llm.base import BaseLLMClient


@dataclass(slots=True, frozen=True)
class PrefetchedTurnMemory:
    """In-flight turn-memory fetch plus the turn tuple it belongs to."""

    task: asyncio.Task[LoadMemoryResult]
    owner_id: str
    query: str
    is_first_turn: bool
    scheduled_at_monotonic: float = field(default_factory=time.monotonic)

    def matches(self, *, owner_id: str, query: str, is_first_turn: bool) -> bool:
        """Return whether this prefetch belongs to the current turn."""

        return (
            self.owner_id == owner_id
            and self.query == query
            and self.is_first_turn is is_first_turn
        )

    def cancel_if_pending(self) -> None:
        """Cancel unused speculative work so stale tasks do not linger."""

        if not self.task.done():
            self.task.cancel()
            return

        try:
            self.task.exception()
        except asyncio.CancelledError:
            pass


@dataclass(slots=True, frozen=True)
class WorkflowContext:
    """Immutable dependency bundle injected through ``runtime.context``.

    Attributes:
        llm_client: Control-plane LLM used by safety classification, routing,
            session finalization, and other structured/background tasks.
        memory_store: Shared semantic, episodic, and procedural memory store.
            The runtime turn memory context reads from it; explicit memory
            tools and exercise services may write to it when memory mode allows.
        crisis_log_backend: Always-on audit backend used by crisis-response
            side effects. This remains available even when user memory is
            incognito.
        memory_mode: Current persistence tier. Memory-aware nodes use this to
            decide whether durable memory reads/writes are allowed.
        response_llm: Optional response-writing LLM. Therapeutic response nodes
            prefer this and fall back to ``llm_client`` when it is not set.
        embedding_provider: Optional embedding provider for hybrid retrieval
            and write-time indexing.
        session_memory_buffer: Optional per-thread candidate buffer. Hot-path
            extractors add held semantic/procedural candidates here so
            session-end commit can promote or drop them.
        pre_fetched_memory: Optional in-flight memory fetch plus the
            owner/query/first-turn tuple it was scheduled for. The runtime turn
            memory context validates that tuple before awaiting it.
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
    pre_fetched_memory: PrefetchedTurnMemory | None = None

    @property
    def control_llm(self) -> BaseLLMClient | None:
        """Return the infrastructure client used by safety, routing, and memory.

        Returns:
            Shared control LLM client, if configured.
        """

        return self.llm_client
