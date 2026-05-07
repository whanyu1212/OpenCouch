"""Session finalization helpers for the persistent agent runtime."""

from __future__ import annotations

import logging
from typing import Any, cast

from agent.memory.policy.candidates import SessionMemoryBuffer
from agent.memory.embeddings import EmbeddingProvider
from agent.memory.extraction_service import (
    extract_procedural_rules,
    extract_semantic_facts,
)
from agent.memory.models import StoredSessionArc
from agent.memory.modes import MemoryMode
from agent.memory.store import MemoryStore
from agent.runtime.session.commit import run_commit_session_memory
from agent.runtime.session.summarize import run_summarize_session
from agent.state import AgentState
from llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


async def finalize_session_window(
    state: AgentState,
    *,
    thread_id: str,
    started_at: str,
    ended_at: str,
    crisis_level_max: int,
    session_buffer: SessionMemoryBuffer | None,
    llm_client: BaseLLMClient | None,
    memory_store: MemoryStore,
    memory_mode: MemoryMode,
    embedding_provider: EmbeddingProvider | None,
) -> StoredSessionArc | None:
    """Run the shared session-end summarization and memory commit path.

    Args:
        state (AgentState): The state window to summarize.
        thread_id (str): The thread identifier being finalized.
        started_at (str): The session start timestamp.
        ended_at (str): The session end timestamp.
        crisis_level_max (int): The max crisis level observed in the session.
        session_buffer (SessionMemoryBuffer | None): The buffered session memory
            candidates.
        llm_client (BaseLLMClient | None): The LLM client used by finalization.
        memory_store (MemoryStore): Store used for episodic and promoted memory.
        memory_mode (MemoryMode): Runtime memory mode.
        embedding_provider (EmbeddingProvider | None): Optional embedding
            provider for memory writes.

    Returns:
        StoredSessionArc | None: The stored session arc, or ``None`` when
        summarization is skipped.
    """

    approach_hint = session_buffer.dominant_approach() if session_buffer else None

    stored_arc = await run_summarize_session(
        state,
        llm_client=llm_client,
        memory_store=memory_store,
        memory_mode=memory_mode,
        session_id=thread_id,
        started_at=started_at,
        ended_at=ended_at,
        crisis_level_max=crisis_level_max,
        embedding_provider=embedding_provider,
        approach_hint=approach_hint,
    )

    commit_result = await run_commit_session_memory(
        state,
        memory_store=memory_store,
        session_buffer=session_buffer,
        stored_arc=stored_arc,
        embedding_provider=embedding_provider,
        llm_client=llm_client,
    )
    if commit_result is not None:
        logger.info(
            "end_session: committed %d semantic facts, %d procedural rules "
            "(%d semantic bumps, %d semantic skipped, %d procedural skipped)",
            commit_result.semantic_writes,
            commit_result.procedural_writes,
            commit_result.semantic_bumps,
            commit_result.semantic_skips,
            commit_result.procedural_skips,
        )
    return stored_arc


async def extract_memory_from_transcript(
    *,
    thread_id: str,
    user_id: str | None,
    transcript: list[dict[str, Any]],
    llm_client: BaseLLMClient | None,
    session_buffer: SessionMemoryBuffer,
    memory_store: MemoryStore,
    memory_mode: MemoryMode,
    embedding_provider: EmbeddingProvider | None,
) -> None:
    """Replay transcript user turns through the extraction services.

    Args:
        thread_id (str): The thread identifier for provenance.
        user_id (str | None): The resolved user identifier, if any.
        transcript (list[dict[str, Any]]): The serialized transcript to replay.
        llm_client (BaseLLMClient | None): The LLM client used by extraction.
        session_buffer (SessionMemoryBuffer): The session buffer to populate.
        memory_store (MemoryStore): Store available to extraction.
        memory_mode (MemoryMode): Runtime memory mode.
        embedding_provider (EmbeddingProvider | None): Optional embedding
            provider for extraction writes.

    Returns:
        None: This helper only mutates the provided session buffer.
    """

    user_turn_count = 0
    for transcript_index, turn in enumerate(transcript):
        if turn.get("role") != "user":
            continue

        message = (turn.get("content") or "").strip()
        if not message:
            continue

        user_turn_count += 1
        state = cast(
            AgentState,
            {
                "message": message,
                "user_id": user_id,
                "session_id": thread_id,
                "transcript": list(transcript[: transcript_index + 1]),
                "session_progress": {"turn_count": user_turn_count},
                "route": "therapeutic",
            },
        )
        await extract_semantic_facts(
            state,
            llm_client=llm_client,
            memory_store=memory_store,
            memory_mode=memory_mode,
            embedding_provider=embedding_provider,
            session_buffer=session_buffer,
        )
        await extract_procedural_rules(
            state,
            llm_client=llm_client,
            memory_store=memory_store,
            memory_mode=memory_mode,
            session_buffer=session_buffer,
        )
