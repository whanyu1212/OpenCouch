"""Service for promoting buffered memory candidates at session end.

This module scores and commits semantic and procedural candidates that were held
during the session for repetition evidence or end-of-session review.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent.memory.commit.clustering import (
    procedural_cluster_text,
    semantic_cluster_text,
)
from agent.memory.commit.scoring import _load_prior_session_support_texts
from agent.memory.commit.selection import (
    _select_procedural_candidates_to_commit,
    _select_semantic_candidates_to_commit,
    _semantic_procedural_overlap_resolution,
)
from agent.memory.policy.candidates import (
    BufferedProceduralCandidate,
    BufferedSemanticCandidate,
    SessionMemoryBuffer,
)
from agent.memory.policy.write import (
    semantic_candidate_needs_repetition_guard,
    text_contains_memory_control_request,
)
from agent.memory.operations.procedural_profile import (
    aupsert_procedural_rule,
    build_procedural_rule,
)
from agent.memory.operations.semantic_writes import (
    BatchSemanticWriteOutcome,
    BatchWriteItem,
    apply_semantic_writes_batch,
)
from agent.memory.store import MemoryStore


if TYPE_CHECKING:
    from agent.memory.providers.embeddings import EmbeddingProvider
    from agent.memory.types import StoredSessionArc
    from llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SessionMemoryCommitResult:
    """Outcome of the session-end promotion pass."""

    semantic_writes: int = 0
    semantic_bumps: int = 0
    semantic_skips: int = 0
    semantic_failures: int = 0
    procedural_writes: int = 0
    procedural_skips: int = 0
    procedural_failures: int = 0
    support_load_failed: bool = False


def _current_session_ids_for_commit(
    session_id: str | None,
    *,
    stored_arc: "StoredSessionArc | None",
    session_buffer: SessionMemoryBuffer,
) -> set[str]:
    """Return the session IDs that should be excluded from prior-support lookup."""
    return {
        session_id
        for session_id in (
            session_id,
            stored_arc.session_id if stored_arc is not None else None,
            session_buffer.session_id,
        )
        if session_id
    }


async def _load_prior_support_texts_for_commit(
    memory_store: MemoryStore,
    *,
    owner_id: str,
    current_session_ids: set[str],
    result: SessionMemoryCommitResult,
) -> list[str]:
    """Load prior episodic support texts while preserving failure accounting."""
    try:
        return await _load_prior_session_support_texts(
            memory_store,
            owner_id=owner_id,
            current_session_ids=current_session_ids,
        )
    except Exception:
        result.support_load_failed = True
        logger.warning(
            "commit_session_memory: failed to load prior episodic support; "
            "continuing without cross-session repetition evidence.",
            exc_info=True,
        )
        return []


def _prepare_semantic_batch_items(
    semantic_candidates_to_commit: list[BufferedSemanticCandidate],
    *,
    procedural_candidates_to_commit: list[
        tuple[BufferedProceduralCandidate, list[str], int]
    ],
) -> tuple[list[BatchWriteItem], int]:
    """Build semantic batch items and count overlap skips."""
    batch_items: list[BatchWriteItem] = []
    overlap_skips = 0
    for record in semantic_candidates_to_commit:
        candidate = record.candidate
        if (
            _semantic_procedural_overlap_resolution(
                candidate,
                procedural_candidates_to_commit,
            )
            == "procedural"
        ):
            overlap_skips += 1
            continue
        write_timing = (
            "promotion"
            if record.hold_action == "require_repetition"
            or semantic_candidate_needs_repetition_guard(candidate)
            else "session_end"
        )
        write_reason = (
            "repetition-qualified semantic candidate promoted at session end"
            if write_timing == "promotion"
            else "session-end semantic candidate supported by transcript and episodic summary"
        )
        batch_items.append(
            BatchWriteItem(
                candidate=candidate,
                write_timing=write_timing,
                write_reason=write_reason,
                policy_version="phase3_v1",
            )
        )

    return batch_items, overlap_skips


def _apply_semantic_batch_outcome(
    result: SessionMemoryCommitResult,
    batch_outcome: BatchSemanticWriteOutcome,
) -> None:
    """Apply semantic batch counters to the session commit result."""
    result.semantic_writes += batch_outcome.written
    result.semantic_bumps += batch_outcome.bumped
    result.semantic_skips += batch_outcome.skipped
    result.semantic_failures += batch_outcome.failures


async def _commit_semantic_candidates(
    memory_store: MemoryStore,
    *,
    owner_id: str,
    semantic_candidates_to_commit: list[BufferedSemanticCandidate],
    procedural_candidates_to_commit: list[
        tuple[BufferedProceduralCandidate, list[str], int]
    ],
    embedding_provider: "EmbeddingProvider | None",
    llm_client: "BaseLLMClient | None",
    result: SessionMemoryCommitResult,
) -> None:
    """Commit selected semantic candidates and update result counters."""
    if not semantic_candidates_to_commit:
        return

    batch_items, overlap_skips = _prepare_semantic_batch_items(
        semantic_candidates_to_commit,
        procedural_candidates_to_commit=procedural_candidates_to_commit,
    )
    result.semantic_skips += overlap_skips
    if not batch_items:
        return

    batch_outcome = await apply_semantic_writes_batch(
        memory_store,
        owner_id=owner_id,
        items=batch_items,
        llm_client=llm_client,
        embedding_provider=embedding_provider,
        log_context="commit_session_memory",
    )
    _apply_semantic_batch_outcome(result, batch_outcome)


async def _commit_procedural_candidates(
    memory_store: MemoryStore,
    *,
    owner_id: str,
    procedural_candidates_to_commit: list[
        tuple[BufferedProceduralCandidate, list[str], int]
    ],
    llm_client: "BaseLLMClient | None",
    result: SessionMemoryCommitResult,
) -> None:
    """Commit selected procedural candidates and update result counters."""
    if not procedural_candidates_to_commit:
        return

    for (
        procedural_record,
        evidence,
        effective_support,
    ) in procedural_candidates_to_commit:
        procedural_candidate = procedural_record.candidate
        try:
            rule = build_procedural_rule(
                rule_text=procedural_candidate.payload.rule,
                evidence=evidence,
                confidence="high" if effective_support >= 3 else "medium",
                source="consolidation",
                write_timing="promotion",
                write_reason="repeated implicit procedural preference promoted at session end",
                policy_version="phase3_v1",
            )
            upsert = await aupsert_procedural_rule(
                memory_store,
                user_id=owner_id,
                rule=rule,
                llm_client=llm_client,
            )
            if upsert.action == "skipped":
                result.procedural_skips += 1
                continue
            result.procedural_writes += 1
        except Exception:
            logger.warning(
                "commit_session_memory: failed to promote buffered procedural rule %r.",
                procedural_candidate.payload.rule[:60],
                exc_info=True,
            )
            result.procedural_failures += 1


async def _embed_for_clustering(
    texts: list[str],
    embedding_provider: "EmbeddingProvider | None",
) -> list[list[float]] | None:
    """Batch-embed candidate texts for cosine clustering.

    Returns one vector per text, or ``None`` (clustering falls back to lexical)
    when there is no provider, the call fails, or any text fails to embed.
    """

    if embedding_provider is None or not texts:
        return None
    try:
        vectors = await embedding_provider.aembed(texts)
    except Exception:
        logger.warning(
            "clustering embedding failed; using lexical clustering", exc_info=True
        )
        return None
    if any(vector is None for vector in vectors) or len(vectors) != len(texts):
        return None
    return [vector for vector in vectors if vector is not None]


async def commit_session_memory(
    *,
    owner_id: str,
    session_id: str | None,
    user_turn_texts: list[str],
    memory_store: MemoryStore,
    session_buffer: SessionMemoryBuffer | None,
    stored_arc: "StoredSessionArc | None",
    embedding_provider: "EmbeddingProvider | None" = None,
    llm_client: "BaseLLMClient | None" = None,
) -> SessionMemoryCommitResult | None:
    """Commit buffered semantic/procedural candidates that survived review."""
    if (
        session_buffer is None
        or not session_buffer.held_semantic_candidates
        and not session_buffer.held_procedural_candidates
    ):
        return None

    user_turn_texts = list(user_turn_texts)
    result = SessionMemoryCommitResult()
    if any(text_contains_memory_control_request(text) for text in user_turn_texts):
        result.semantic_skips = len(session_buffer.held_semantic_candidates)
        result.procedural_skips = len(session_buffer.held_procedural_candidates)
        logger.info(
            "commit_session_memory: explicit memory-control request in transcript; "
            "dropping %d semantic and %d procedural held candidate(s).",
            result.semantic_skips,
            result.procedural_skips,
        )
        return result

    current_session_ids = _current_session_ids_for_commit(
        session_id,
        stored_arc=stored_arc,
        session_buffer=session_buffer,
    )
    prior_session_support_texts = await _load_prior_support_texts_for_commit(
        memory_store,
        owner_id=owner_id,
        current_session_ids=current_session_ids,
        result=result,
    )

    procedural_embeddings = await _embed_for_clustering(
        [
            procedural_cluster_text(record.candidate)
            for record in session_buffer.held_procedural_candidates
        ],
        embedding_provider,
    )
    procedural_candidates_to_commit, result.procedural_skips = (
        _select_procedural_candidates_to_commit(
            session_buffer.held_procedural_candidates,
            user_turn_texts=user_turn_texts,
            embeddings=procedural_embeddings,
        )
    )

    semantic_embeddings = await _embed_for_clustering(
        [
            semantic_cluster_text(record.candidate)
            for record in session_buffer.held_semantic_candidates
        ],
        embedding_provider,
    )
    semantic_candidates_to_commit, result.semantic_skips = (
        _select_semantic_candidates_to_commit(
            session_buffer.held_semantic_candidates,
            stored_arc=stored_arc,
            user_turn_texts=user_turn_texts,
            prior_session_support_texts=prior_session_support_texts,
            embeddings=semantic_embeddings,
        )
    )
    await _commit_semantic_candidates(
        memory_store,
        owner_id=owner_id,
        semantic_candidates_to_commit=semantic_candidates_to_commit,
        procedural_candidates_to_commit=procedural_candidates_to_commit,
        embedding_provider=embedding_provider,
        llm_client=llm_client,
        result=result,
    )
    await _commit_procedural_candidates(
        memory_store,
        owner_id=owner_id,
        procedural_candidates_to_commit=procedural_candidates_to_commit,
        llm_client=llm_client,
        result=result,
    )

    logger.info(
        "commit_session_memory: session-end promotion complete — %d semantic written, "
        "%d semantic bumped, %d semantic skipped, %d semantic failures, "
        "%d procedural written, %d procedural skipped, %d procedural failures",
        result.semantic_writes,
        result.semantic_bumps,
        result.semantic_skips,
        result.semantic_failures,
        result.procedural_writes,
        result.procedural_skips,
        result.procedural_failures,
    )
    return result
