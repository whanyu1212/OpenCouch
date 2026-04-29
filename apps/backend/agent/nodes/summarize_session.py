"""Session summarizer that runs once per session at session end.

Unlike the other files in ``agent/nodes/``, this module does NOT export
a LangGraph node function. It exports a standalone async function
:func:`run_summarize_session` that's invoked directly by
:class:`agent.persistence.PersistentAgentRuntime` when a session ends
(via the CLI's ``/end`` command or a ``/exit`` confirmation).

Why not a graph node:

    Summarization runs at **session boundaries**, not per-turn. LangGraph's
    value - multi-node orchestration, per-turn state reducers, conditional
    routing - does not apply to a single end-of-session LLM call. Compiling
    a throwaway one-node graph for this work would add ceremony without
    any benefit. A bare async function with the same signature pattern as
    the extraction node is cleaner. The runtime already owns the store
    and the LLM client, so it can invoke the summarizer directly.

    This file lives in ``agent/nodes/`` anyway (not ``agent/memory/``)
    because it participates in the node-layer memory workflow alongside
    the per-turn extractor.

Design rules (mirror the extract_facts conventions):

1. **Conservative summarization.** The system prompt tells the LLM to
   return ``arc=None`` for sessions that don't have enough content
   (pure small talk, <3 substantive turns, no emotional arc). A missing
   summary is better than a fabricated one.

2. **Silent skip on incognito or no LLM.** Same contract as
   ``extract_facts`` — if the memory mode is INCOGNITO or no LLM client
   is available, the summarizer returns ``None`` without touching the
   store. The runtime should not treat either case as an error.

3. **Failures degrade silently.** LLM errors, schema validation errors,
   and store write errors are all logged at WARNING level but never
   propagate. A summarization failure must not fail the session-end
   flow; the user is trying to exit the CLI, not diagnose an LLM call.

4. **Returns the written record** (or ``None``). The runtime uses this
   return value to render a farewell panel showing the user the summary
   that was just saved. ``None`` means "nothing was written, render a
   plain farewell instead."

5. **Observability at INFO level.** The LLM's reason string is logged at
   INFO so summarizer decisions are visible during local evaluation.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from agent.memory.hashing import iso_now as _iso_now
from agent.memory.models import (
    SessionArc,
    StoredSessionArc,
    SummarizationResult,
)
from agent.memory.modes import MemoryMode
from agent.memory.store import MemoryStore
from agent.memory.summarization_prompts import (
    build_summarization_system_prompt,
    build_summarization_user_prompt,
)
from agent.state import AgentState, resolve_owner_id
from services.llm.base import BaseLLMClient

if TYPE_CHECKING:
    from agent.memory.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


def _session_arc_to_stored(
    arc: SessionArc,
    *,
    owner_id: str,
    crisis_level_max: int = 0,
) -> StoredSessionArc:
    """Convert an LLM-produced :class:`SessionArc` to a stored shape.

    Parallels ``_memory_write_to_semantic_fact`` in ``extract_facts.py``.
    The SessionArc has the narrative fields the summarizer produces;
    StoredSessionArc adds the store metadata (id, owner_id, timestamps,
    visibility flag) AND the runtime-computed ``crisis_level_max``.
    This helper generates a fresh ID and timestamps for a new record
    and takes the crisis level as an explicit parameter.

    The ``crisis_level_max`` parameter is the peak crisis-gate level
    observed during the session, passed in from the runtime's
    per-thread tracker rather than derived from the LLM's output.
    See the class docstring on :class:`StoredSessionArc` for the
    rationale — in short, the crisis gate is the canonical source
    of truth for crisis severity and the summarizer should not
    re-interpret it.

    Args:
        arc: LLM-produced session arc.
        owner_id: Owner id for the stored arc.
        crisis_level_max: Peak crisis level observed during the session.

    Returns:
        Stored session arc ready for persistence.
    """

    now = _iso_now()
    return StoredSessionArc(
        **arc.model_dump(),
        id=str(uuid4()),
        owner_id=owner_id,
        created_at=now,
        last_referenced_at=now,
        user_visible=True,
        write_timing="session_end",
        write_reason="session-end episodic summary written from completed session transcript",
        policy_version="phase5_v1",
        crisis_level_max=crisis_level_max,  # type: ignore[arg-type]
    )


async def _write_session_arc(
    store: MemoryStore,
    *,
    owner_id: str,
    stored_arc: StoredSessionArc,
    embedding: list[float] | None = None,
    embedding_model: str | None = None,
) -> None:
    """Persist a freshly-summarized StoredSessionArc to the episodic namespace.

    Separated from the main body so the error-handling scope is tight
    around the single store call. A failure here is logged but never
    raised to the caller — a summarization failure must not break the
    session-end flow.

    Args:
        store: Memory store to write to.
        owner_id: Owner whose episodic namespace receives the arc.
        stored_arc: Stored session arc to persist.
        embedding: Optional document embedding for hybrid retrieval.
        embedding_model: Optional embedding model identifier.

    Returns:
        None.
    """

    namespace = (owner_id, "episodic")
    await store.aput(
        namespace,
        key=stored_arc.id,
        value=stored_arc.model_dump(mode="json"),
        embedding=embedding,
        embedding_model=embedding_model,
    )


async def run_summarize_session(
    state: AgentState,
    *,
    llm_client: BaseLLMClient | None,
    memory_store: MemoryStore,
    memory_mode: MemoryMode,
    session_id: str,
    started_at: str,
    ended_at: str | None = None,
    crisis_level_max: int = 0,
    embedding_provider: "EmbeddingProvider | None" = None,
    approach_hint: str | None = None,
) -> StoredSessionArc | None:
    """Summarize a completed session and write the arc to episodic memory.

    Runs once per session at session end, invoked by the runtime rather
    than by the LangGraph compiled graph. Returns the written
    :class:`StoredSessionArc` on success, or ``None`` on any of the
    legitimate skip conditions: incognito mode, no LLM client, LLM
    returned ``arc=None``, LLM call failed, store write failed.

    The ``ended_at`` parameter defaults to "now" if the caller doesn't
    specify one, so the runtime doesn't need to compute it separately
    in the common case.

    Args:
        state: Current graph state at session end. Reads ``transcript``
            and ``user_id`` / ``session_id``. The transcript is the full
            session history (checkpointer-restored), not a window.
        llm_client: The runtime's LLM client, passed explicitly rather
            than pulled from ``runtime.context`` (since this isn't a
            graph node). When ``None``, the summarizer skips silently.
        memory_store: The runtime's memory store. The written arc lands
            in ``(owner_id, "episodic")``.
        memory_mode: The runtime's active memory mode. When INCOGNITO,
            the summarizer skips silently.
        session_id: The session identifier. Copied verbatim into the
            SessionArc's ``session_id`` field.
        started_at: ISO-8601 timestamp when the session started. The
            runtime tracks this; the summarizer takes it as a parameter
            rather than inferring from the transcript.
        ended_at: ISO-8601 timestamp for session end. Defaults to now().
        crisis_level_max: Peak crisis-gate level observed during the
            session (0-3). The runtime tracks this in its
            ``_max_crisis_levels`` dict, updated after every
            ``run_turn`` invocation. The LLM does NOT produce this
            field — it's a deterministic max-of-per-turn-crisis-gate-
            verdicts, so the crisis gate stays the single source of
            truth for crisis severity.
        embedding_provider: Optional provider for storing a retrievable
            embedding alongside the session arc.
        approach_hint: The dominant therapeutic approach used during
            the session (e.g., "cbt", "act"). Passed to the
            summarization prompt so the LLM extracts approach-specific
            structured context. When None, the summarizer produces a
            general arc (backward-compatible behavior).

    Returns:
        The written :class:`StoredSessionArc` on success, or ``None`` on
        any legitimate skip / failure.
    """

    if ended_at is None:
        ended_at = _iso_now()

    if llm_client is None:
        logger.debug("run_summarize_session: no llm_client; skipping")
        return None
    if memory_mode == MemoryMode.INCOGNITO:
        logger.debug(
            "run_summarize_session: incognito mode; skipping (no episodic "
            "writes in incognito)"
        )
        return None

    owner_id = resolve_owner_id(state)

    # Compute duration from started_at / ended_at. If parsing fails (for
    # example, malformed timestamps), degrade to 0 rather than crashing; the
    # summary is still useful without the duration field.
    duration_seconds = 0
    try:
        start_dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
        duration_seconds = max(0, int((end_dt - start_dt).total_seconds()))
    except (ValueError, AttributeError):
        logger.warning(
            "run_summarize_session: could not parse started_at/ended_at; "
            "duration will be 0. started_at=%r ended_at=%r",
            started_at,
            ended_at,
        )

    # Count user turns from the transcript. The state's session_progress.turn_count
    # is technically 1-indexed and may include the current turn; using the
    # transcript is more reliable.
    transcript = state.get("transcript", [])
    user_turn_count = sum(1 for turn in transcript if turn.get("role") == "user")

    try:
        result: SummarizationResult = await llm_client.generate_structured(
            prompt=build_summarization_user_prompt(
                state,
                session_id=session_id,
                started_at=started_at,
                ended_at=ended_at,
                duration_seconds=duration_seconds,
                turn_count=user_turn_count,
                approach_hint=approach_hint,
            ),
            response_schema=SummarizationResult,
            system_instruction=build_summarization_system_prompt(),
        )
    except Exception:
        logger.warning(
            "run_summarize_session: LLM structured-output call failed; "
            "skipping summarization for this session.",
            exc_info=True,
        )
        return None

    # Log the reason regardless; it is a useful signal for prompt tuning.
    logger.info(
        "run_summarize_session: arc=%s reason=%r",
        "present" if result.arc is not None else "None",
        result.reason,
    )

    if result.arc is None:
        # Legitimate skip: the LLM judged the session too thin to summarize.
        # The CLI should render a plain farewell without a summary panel.
        return None

    try:
        stored_arc = _session_arc_to_stored(
            result.arc,
            owner_id=owner_id,
            crisis_level_max=crisis_level_max,
        )
    except Exception:
        logger.warning(
            "run_summarize_session: failed to promote SessionArc to "
            "StoredSessionArc; skipping write.",
            exc_info=True,
        )
        return None

    # Compute an embedding for the summary so the arc participates in hybrid
    # retrieval when the next session opens.
    # The summary prose is the canonical document-side representation
    # of an arc: it is what the load_memory catch-up path renders as
    # "Last session (themes): <summary>" and what the user's next-
    # session query would be trying to match semantically.
    #
    # Embedding failures degrade to None so a transient provider
    # outage does not lose the arc; the store write still happens
    # via the token-recall path. Same contract as the extract_facts
    # embedding logic.
    arc_embedding: list[float] | None = None
    arc_embedding_model: str | None = None
    if embedding_provider is not None:
        try:
            embeddings = await embedding_provider.aembed(
                [stored_arc.summary],
                task_type="RETRIEVAL_DOCUMENT",
            )
            arc_embedding = embeddings[0] if embeddings else None
            if arc_embedding is not None:
                arc_embedding_model = embedding_provider.model_name
        except Exception:
            logger.warning(
                "run_summarize_session: embedding call failed; writing arc "
                "without embedding for this session.",
                exc_info=True,
            )

    try:
        await _write_session_arc(
            memory_store,
            owner_id=owner_id,
            stored_arc=stored_arc,
            embedding=arc_embedding,
            embedding_model=arc_embedding_model,
        )
    except Exception:
        logger.warning(
            "run_summarize_session: failed to write session arc to "
            "episodic namespace; arc was computed but not persisted.",
            exc_info=True,
        )
        return None

    logger.info(
        "run_summarize_session: wrote arc id=%s owner=%r session=%r "
        "themes=%s crisis_level_max=%d approach=%s",
        stored_arc.id,
        owner_id,
        session_id,
        stored_arc.primary_themes,
        stored_arc.crisis_level_max,
        stored_arc.approach_used or "none",
    )
    return stored_arc
