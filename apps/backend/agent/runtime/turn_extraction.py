"""Background-extraction lifecycle coordinator for the persistent runtime.

After a turn's response finalizes, the runtime needs to extract semantic
facts and procedural rules from the conversation — but extraction's LLM
call is too slow (~3.8s median p50 measured on real OpenAI traffic) to
sit on the user-visible turn path. This module owns the lifecycle that
makes extraction a background concern:

- :meth:`TurnExtractionCoordinator.run_pair` runs both extractors in
  parallel (used by the foreground path that test fixtures opt into via
  ``extract_in_foreground=True``).
- :meth:`TurnExtractionCoordinator.schedule` dispatches extraction as an
  ``asyncio.Task`` and registers it so the next turn can drain it.
- :meth:`TurnExtractionCoordinator.drain` awaits one thread's pending
  task with a bounded timeout; called from the next turn's prepare-step
  and from ``end_session`` so that consumers see a coherent memory
  state.
- :meth:`TurnExtractionCoordinator.drain_all` is the shutdown path; the
  runtime calls it from ``__aexit__`` before closing the memory store.

The coordinator depends on the runtime by callback, not by reference:
the constructor takes a :class:`MemoryStore`, an embedding provider, the
:class:`MemoryMode`, and two thread-keyed callables — one to fetch the
session memory buffer, one to persist runtime session tracking after
extraction populates the buffer. That last callback is load-bearing:
without it, ``end_session`` would hydrate from a pre-extraction snapshot
and clobber the in-memory buffer that just got candidates added.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from agent.memory.embeddings import EmbeddingProvider
from agent.memory.extraction_service import (
    extract_procedural_rules,
    extract_semantic_facts,
)
from agent.memory.modes import MemoryMode
from agent.memory.policy.candidates import SessionMemoryBuffer
from agent.memory.store import MemoryStore
from agent.state import AgentState
from llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


# Maximum seconds to wait for a prior turn's background extraction to drain
# before the next turn proceeds with possibly-stale memory state. Set higher
# than the observed p95 (~17s in measurement campaigns) but bounded so an
# LLM provider stall cannot indefinitely block subsequent turns.
EXTRACTION_DRAIN_TIMEOUT_SECONDS = 30.0


class TurnExtractionCoordinator:
    """Owns the lifecycle of background memory-extraction tasks.

    Each in-flight extraction is tracked by ``thread_id`` so the next
    turn (or shutdown) can drain it with a bounded timeout. Extraction
    failure is logged but never raised — a side-effect path must not
    fail the parent turn.
    """

    def __init__(
        self,
        *,
        memory_store: MemoryStore,
        embedding_provider: EmbeddingProvider | None,
        memory_mode: MemoryMode,
        session_buffer_for: Callable[[str], SessionMemoryBuffer],
        persist_after_extraction: Callable[[str], Awaitable[None]],
    ) -> None:
        """Build the coordinator with explicit runtime dependencies.

        Args:
            memory_store: Memory backend that extraction writes to.
            embedding_provider: Embedding provider used by the semantic
                write path; ``None`` is acceptable when embeddings are
                disabled.
            memory_mode: Runtime memory mode. Extraction skips silently
                in :attr:`MemoryMode.INCOGNITO`.
            session_buffer_for: Thread-keyed lookup that returns the
                runtime-managed session buffer for a thread; the
                coordinator passes that buffer to extraction so held
                candidates accumulate in the same place
                ``end_session`` reads from.
            persist_after_extraction: Coroutine factory called after
                background extraction completes to re-persist the
                runtime's active-session record. Without this, an
                ``end_session`` between turns would hydrate from the
                pre-extraction snapshot and clobber the in-memory
                buffer just populated by extraction.
        """

        self._memory_store = memory_store
        self._embedding_provider = embedding_provider
        self._memory_mode = memory_mode
        self._session_buffer_for = session_buffer_for
        self._persist_after_extraction = persist_after_extraction
        # Per-thread in-flight extraction tasks. Drained by ``drain``
        # (next turn) and ``drain_all`` (shutdown).
        self._pending: dict[str, asyncio.Task[None]] = {}

    async def run_pair(
        self,
        *,
        state: AgentState,
        llm_client: BaseLLMClient | None,
        session_buffer: SessionMemoryBuffer,
    ) -> dict[str, Any]:
        """Run both extractors in parallel and return their merged diagnostics.

        The two extractors are independent — one's failure must not
        affect the other — so we use ``return_exceptions=True`` and log
        any failure rather than letting it propagate.

        Args:
            state: Final graph state for the turn.
            llm_client: Control LLM for extraction. ``None`` causes both
                extractors to skip silently.
            session_buffer: Session-scoped candidate buffer for held
                writes.

        Returns:
            Merged diagnostics from both extractors. Empty when both
            skipped.
        """

        results = await asyncio.gather(
            extract_semantic_facts(
                state,
                llm_client=llm_client,
                memory_store=self._memory_store,
                memory_mode=self._memory_mode,
                embedding_provider=self._embedding_provider,
                session_buffer=session_buffer,
            ),
            extract_procedural_rules(
                state,
                llm_client=llm_client,
                memory_store=self._memory_store,
                memory_mode=self._memory_mode,
                session_buffer=session_buffer,
            ),
            return_exceptions=True,
        )
        diagnostics: dict[str, Any] = {}
        for result in results:
            if isinstance(result, BaseException):
                logger.warning(
                    "Background extraction task raised; diagnostics partial.",
                    exc_info=result,
                )
                continue
            diagnostics.update(result.as_diagnostics())
        return diagnostics

    def schedule(
        self,
        *,
        thread_id: str,
        state: AgentState,
        llm_client: BaseLLMClient | None,
    ) -> None:
        """Dispatch extraction as a background task.

        Replaces the prior turn's pending task in the registry; the
        prior task should already be drained by :meth:`drain` before
        the next turn reaches this point, so this assignment shouldn't
        usually overwrite a live task. Defensive: if it does overwrite,
        the prior task is allowed to run to completion in the
        background — its diagnostics are simply discarded.

        Args:
            thread_id: Thread the extraction belongs to.
            state: Final graph state captured before extraction starts.
            llm_client: Control LLM for extraction.
        """

        session_buffer = self._session_buffer_for(thread_id)

        async def _run() -> None:
            try:
                await self.run_pair(
                    state=state,
                    llm_client=llm_client,
                    session_buffer=session_buffer,
                )
                # Re-persist the active-session record so the buffer's
                # newly-added candidates survive past this point. Without
                # this, end_session called between turns would hydrate
                # from the pre-extraction snapshot and clobber the
                # in-memory buffer that just got populated.
                try:
                    await self._persist_after_extraction(thread_id)
                except Exception:
                    logger.warning(
                        "Background extraction post-persist failed for thread %s; "
                        "in-memory buffer remains correct but persistence is stale.",
                        thread_id,
                        exc_info=True,
                    )
            finally:
                # Self-clean from registry on completion. Guard against the
                # registry having moved on if a later turn re-scheduled.
                current = self._pending.get(thread_id)
                if current is not None and current.done():
                    self._pending.pop(thread_id, None)

        task = asyncio.create_task(_run(), name=f"extract:{thread_id}")
        self._pending[thread_id] = task

    async def drain(self, thread_id: str) -> None:
        """Block until the prior turn's extraction task for this thread completes.

        Called from the runtime's ``_prepare_session_for_turn`` so that
        turn N+1 sees turn N's memory writes, and from
        ``_end_session_unlocked`` so finalization sees a coherent
        buffer. Bounded by ``EXTRACTION_DRAIN_TIMEOUT_SECONDS`` so a
        stalled extraction (e.g., an LLM provider hang — we observed
        100s+ outliers in the latency profile) cannot indefinitely
        block subsequent turns.

        On timeout the stalled task is cancelled to avoid orphaning it;
        a long stall almost always means the LLM call is hung, and
        letting the task run on indefinitely creates "Task was destroyed
        but it is pending!" noise at event-loop shutdown.

        Args:
            thread_id: Thread whose pending extraction should be
                drained.
        """

        task = self._pending.get(thread_id)
        if task is None or task.done():
            self._pending.pop(thread_id, None)
            return
        try:
            await asyncio.wait_for(
                asyncio.shield(task),
                timeout=EXTRACTION_DRAIN_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Extraction drain exceeded %.1fs for thread %s; cancelling "
                "the stalled task and proceeding with possibly-stale memory "
                "state. (A long stall almost always means the LLM call is "
                "hung; letting the task run on indefinitely orphans it.)",
                EXTRACTION_DRAIN_TIMEOUT_SECONDS,
                thread_id,
            )
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        except Exception:
            logger.warning(
                "Extraction drain raised for thread %s; proceeding.",
                thread_id,
                exc_info=True,
            )
        finally:
            self._pending.pop(thread_id, None)

    async def drain_all(self) -> None:
        """Drain every in-flight extraction task. Used at shutdown.

        Awaits all pending tasks with the same per-task timeout to keep
        shutdown bounded. Errors are logged but never raised, matching
        the existing best-effort shutdown contract used by
        :attr:`PersistentAgentRuntime._finalize_active_sessions_on_close`.
        """

        if not self._pending:
            return
        tasks = list(self._pending.values())
        self._pending.clear()
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=EXTRACTION_DRAIN_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Extraction shutdown drain exceeded %.1fs; cancelling "
                "stuck tasks so the event loop can close cleanly.",
                EXTRACTION_DRAIN_TIMEOUT_SECONDS,
            )
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Best-effort await of the cancellations so event-loop
            # shutdown doesn't see "Task was destroyed but it is
            # pending!" warnings. Suppress exceptions because cancelled
            # tasks raise CancelledError here.
            await asyncio.gather(*tasks, return_exceptions=True)

    @property
    def pending(self) -> dict[str, asyncio.Task[None]]:
        """Read-only view of the in-flight extraction registry.

        Test fixtures inspect this to assert lifecycle invariants
        (e.g., "after run_turn returns, the task exists" or "after
        drain returns, the task is removed"). Production callers
        should not mutate the returned dict.
        """

        return self._pending
