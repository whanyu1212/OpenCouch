"""Service for promoting buffered memory candidates at session end.

This module scores and commits semantic and procedural candidates that were held
during the session for repetition evidence or end-of-session review.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent.memory.policy.candidates import (
    BufferedProceduralCandidate,
    BufferedSemanticCandidate,
    ProceduralCandidate,
    SemanticCandidate,
    SessionMemoryBuffer,
)
from agent.memory.procedural_profile import (
    aupsert_procedural_rule,
    build_procedural_rule,
)
from agent.memory.semantic_writes import (
    BatchWriteItem,
    apply_semantic_writes_batch,
)
from agent.memory.store import MemoryStore
from agent.memory.text_tokens import tokenize_meaningful
from agent.memory.policy.write import (
    semantic_candidate_is_turn_scoped,
    semantic_candidate_needs_repetition_guard,
    should_commit_implicit_procedural_preference,
    should_commit_pattern,
)
from agent.state import AgentState, resolve_owner_id

if TYPE_CHECKING:
    from agent.memory.embeddings import EmbeddingProvider
    from agent.memory.types import StoredSessionArc
    from llm.base import BaseLLMClient

logger = logging.getLogger(__name__)

_SEMANTIC_PROCEDURAL_OVERLAP_CUES = (
    "prefer",
    "help",
    "helps",
    "keep",
    "brief",
    "short",
    "direct",
    "respond",
    "response",
    "plan",
    "plans",
)
_SEMANTIC_BEHAVIOR_GUIDANCE_CATEGORIES = {
    "coping_strategy",
    "support_preference",
    "communication_preference",
}
_SEMANTIC_BEHAVIOR_GUIDANCE_OBJECT_TYPES = {
    "copingstrategy",
    "supportpreference",
    "communicationstyle",
}


@dataclass(slots=True)
class SessionMemoryCommitResult:
    """Outcome of the session-end promotion pass."""

    semantic_writes: int = 0
    semantic_bumps: int = 0
    semantic_skips: int = 0
    procedural_writes: int = 0
    procedural_skips: int = 0


def _semantic_group_key(candidate: SemanticCandidate) -> tuple[str, ...]:
    """Return the grouping key for repeated semantic candidates.

    Args:
        candidate: Semantic candidate to group.

    Returns:
        Tuple identity for semantic repetition checks.
    """

    payload = candidate.payload
    return (
        payload.category,
        payload.subject.type,
        payload.subject.identifier,
        payload.predicate,
        payload.object.type,
        payload.object.identifier,
    )


def _candidate_tokens(*parts: str) -> frozenset[str]:
    """Return one meaningful-token signature for candidate/support text.

    Args:
        parts: Text fragments to combine before tokenization.

    Returns:
        Meaningful-token signature.
    """

    return tokenize_meaningful(" ".join(part for part in parts if part))


def _user_turn_texts(state: AgentState) -> list[str]:
    """Return the user-turn transcript texts for session-end scoring.

    Args:
        state: Current runtime state containing the session transcript.

    Returns:
        Non-empty user turn texts.
    """

    transcript = state.get("transcript", [])
    return [
        (turn.get("content") or "").strip()
        for turn in transcript
        if turn.get("role") == "user" and (turn.get("content") or "").strip()
    ]


def _count_supported_user_turns(
    candidate_tokens: frozenset[str],
    user_turn_texts: list[str],
    *,
    exact_terms: tuple[str, ...] = (),
) -> int:
    """Count how many user turns materially support this candidate.

    Args:
        candidate_tokens: Candidate token signature.
        user_turn_texts: User transcript turns to scan.
        exact_terms: Terms that count as direct support when present.

    Returns:
        Number of user turns with material support.
    """

    if not candidate_tokens and not exact_terms:
        return 0

    supported = 0
    for text in user_turn_texts:
        lowered = text.lower()
        if any(term and term in lowered for term in exact_terms):
            supported += 1
            continue

        overlap = candidate_tokens & tokenize_meaningful(text)
        if len(overlap) >= 2:
            supported += 1
    return supported


def _count_supporting_session_texts(
    candidate_tokens: frozenset[str],
    support_texts: list[str],
    *,
    exact_terms: tuple[str, ...] = (),
) -> int:
    """Count how many session-level texts materially support this candidate.

    Args:
        candidate_tokens: Candidate token signature.
        support_texts: Session-level support texts to scan.
        exact_terms: Terms that count as direct support when present.

    Returns:
        Number of session-level texts with material support.
    """

    if not candidate_tokens and not exact_terms:
        return 0

    supported = 0
    for text in support_texts:
        lowered = text.lower()
        if any(term and term in lowered for term in exact_terms):
            supported += 1
            continue

        overlap = candidate_tokens & tokenize_meaningful(text)
        if len(overlap) >= 2:
            supported += 1
    return supported


def _session_support_text(stored_arc: "StoredSessionArc | None") -> str:
    """Flatten the stored session arc into one support text blob.

    Args:
        stored_arc: Optional stored session arc from the completed session.

    Returns:
        Combined summary/theme/open-loop text.
    """

    if stored_arc is None:
        return ""

    parts = [stored_arc.summary]
    parts.extend(stored_arc.primary_themes)
    parts.extend(stored_arc.open_loops)
    parts.extend(stored_arc.resolved_threads)
    return " ".join(part for part in parts if part).strip()


def _arc_support_score(
    candidate_tokens: frozenset[str],
    *,
    stored_arc: "StoredSessionArc | None",
    exact_terms: tuple[str, ...] = (),
) -> int:
    """Return a small support score from the episodic summary fields.

    Args:
        candidate_tokens: Candidate token signature.
        stored_arc: Optional stored session arc from the completed session.
        exact_terms: Terms that count as direct support when present.

    Returns:
        Small integer support score.
    """

    support_text = _session_support_text(stored_arc)
    if not support_text:
        return 0

    score = 0
    lowered_support = support_text.lower()
    if any(term and term in lowered_support for term in exact_terms):
        score += 2

    overlap = candidate_tokens & tokenize_meaningful(support_text)
    if len(overlap) >= 2:
        score += 1
    if len(overlap) >= 3:
        score += 1
    return score


async def _load_prior_session_support_texts(
    memory_store: MemoryStore,
    *,
    owner_id: str,
    current_session_ids: set[str],
) -> list[str]:
    """Return support texts from prior episodic arcs for this owner.

    Args:
        memory_store: Store containing episodic memory records.
        owner_id: Owner whose prior sessions should be loaded.
        current_session_ids: Session ids to exclude from prior support.

    Returns:
        Prior session support text blobs.
    """

    records = await memory_store.asearch((owner_id, "episodic"), query=None, limit=100)
    prior_texts: list[str] = []
    for record in records:
        value = record.value
        if value.get("session_id") in current_session_ids:
            continue

        parts = [value.get("summary", "")]
        parts.extend(value.get("primary_themes", []))
        parts.extend(value.get("open_loops", []))
        parts.extend(value.get("resolved_threads", []))
        support_text = " ".join(part for part in parts if part).strip()
        if support_text:
            prior_texts.append(support_text)
    return prior_texts


def _procedural_signature_tokens(candidate: ProceduralCandidate) -> frozenset[str]:
    """Return the similarity signature for a procedural candidate.

    Args:
        candidate: Procedural candidate to fingerprint.

    Returns:
        Meaningful-token signature for grouping/dedup.
    """

    return _candidate_tokens(candidate.payload.rule, *candidate.evidence_quotes)


def _semantic_signature_tokens(candidate: SemanticCandidate) -> frozenset[str]:
    """Return the similarity signature for a semantic candidate."""

    return _candidate_tokens(
        candidate.payload.evidence_quote,
        candidate.payload.object.identifier,
    )


def _token_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    """Return token-set similarity for lightweight grouping/dedup.

    Args:
        left: First token set.
        right: Second token set.

    Returns:
        Jaccard-style token similarity.
    """

    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _cluster_semantic_candidates(
    buffered_candidates: list[BufferedSemanticCandidate],
) -> list[list[BufferedSemanticCandidate]]:
    """Cluster semantic candidates that express the same support pattern."""

    groups: list[list[BufferedSemanticCandidate]] = []
    group_tokens: list[frozenset[str]] = []
    group_keys: list[set[tuple[str, ...]]] = []

    for record in buffered_candidates:
        key = _semantic_group_key(record.candidate)
        tokens = _semantic_signature_tokens(record.candidate)
        placed = False
        for index, existing_tokens in enumerate(group_tokens):
            overlap = len(tokens & existing_tokens)
            similarity = _token_similarity(tokens, existing_tokens)
            if key in group_keys[index] or similarity >= 0.5 or overlap >= 3:
                groups[index].append(record)
                group_tokens[index] = frozenset(existing_tokens | tokens)
                group_keys[index].add(key)
                placed = True
                break
        if not placed:
            groups.append([record])
            group_tokens.append(tokens)
            group_keys.append({key})

    return groups


def _cluster_procedural_candidates(
    buffered_candidates: list[BufferedProceduralCandidate],
) -> list[list[BufferedProceduralCandidate]]:
    """Cluster procedural candidates with similar repeated preferences.

    Args:
        buffered_candidates: Procedural records buffered during the session.

    Returns:
        Groups of similar procedural records.
    """

    groups: list[list[BufferedProceduralCandidate]] = []
    group_tokens: list[frozenset[str]] = []

    for record in buffered_candidates:
        tokens = _procedural_signature_tokens(record.candidate)
        placed = False
        for index, existing_tokens in enumerate(group_tokens):
            overlap = len(tokens & existing_tokens)
            similarity = _token_similarity(tokens, existing_tokens)
            if similarity >= 0.5 or overlap >= 3:
                groups[index].append(record)
                group_tokens[index] = frozenset(existing_tokens | tokens)
                placed = True
                break
        if not placed:
            groups.append([record])
            group_tokens.append(tokens)

    return groups


def _select_semantic_candidates_to_commit(
    buffered_candidates: list[BufferedSemanticCandidate],
    *,
    stored_arc: "StoredSessionArc | None",
    user_turn_texts: list[str],
    prior_session_support_texts: list[str],
) -> tuple[list[BufferedSemanticCandidate], int]:
    """Choose which buffered semantic candidates are durable enough to commit.

    Args:
        buffered_candidates: Semantic records buffered during the session.
        stored_arc: Optional session arc produced at session end.
        user_turn_texts: User transcript turns for repetition checks.
        prior_session_support_texts: Prior episodic support texts.

    Returns:
        Selected candidates plus skipped group count.
    """

    selected: list[BufferedSemanticCandidate] = []
    skipped = 0
    for group in _cluster_semantic_candidates(buffered_candidates):
        support_turn_count = len(
            {record.candidate.source_turn_index for record in group}
        )
        repetition_record = next(
            (
                record
                for record in reversed(group)
                if record.hold_action == "require_repetition"
            ),
            None,
        )
        representative = repetition_record or group[-1]
        candidate = representative.candidate
        if semantic_candidate_is_turn_scoped(candidate):
            skipped += 1
            continue
        object_identifier = candidate.payload.object.identifier.lower().strip()
        candidate_tokens = _candidate_tokens(
            candidate.payload.evidence_quote,
            candidate.payload.object.identifier,
        )
        transcript_support_turns = _count_supported_user_turns(
            candidate_tokens,
            user_turn_texts,
            exact_terms=(object_identifier,),
        )
        prior_session_supports = _count_supporting_session_texts(
            candidate_tokens,
            prior_session_support_texts,
            exact_terms=(object_identifier,),
        )
        effective_support = max(support_turn_count, transcript_support_turns)
        arc_support = _arc_support_score(
            candidate_tokens,
            stored_arc=stored_arc,
            exact_terms=(object_identifier,),
        )

        requires_repetition = (
            representative.hold_action == "require_repetition"
            or semantic_candidate_needs_repetition_guard(candidate)
        )
        should_commit = False
        if requires_repetition:
            should_commit = should_commit_pattern(
                hold_action="require_repetition",
                evidence_count=effective_support,
            ) or (effective_support >= 1 and prior_session_supports >= 1)
        else:
            should_commit = (
                effective_support >= 2
                or (transcript_support_turns >= 1 and arc_support >= 2)
                or (transcript_support_turns >= 1 and prior_session_supports >= 1)
            )

        if should_commit:
            selected.append(representative)
        else:
            skipped += 1

    return selected, skipped


def _select_procedural_candidates_to_commit(
    buffered_candidates: list[BufferedProceduralCandidate],
    *,
    user_turn_texts: list[str],
) -> tuple[list[tuple[BufferedProceduralCandidate, list[str], int]], int]:
    """Choose which buffered implicit procedural candidates can promote.

    Args:
        buffered_candidates: Procedural records buffered during the session.
        user_turn_texts: User transcript turns for repetition checks.

    Returns:
        Selected records with evidence/support counts plus skipped group count.
    """

    selected: list[tuple[BufferedProceduralCandidate, list[str], int]] = []
    skipped = 0

    for group in _cluster_procedural_candidates(buffered_candidates):
        representative = group[-1]
        candidate = representative.candidate
        candidate_tokens = _procedural_signature_tokens(candidate)
        transcript_support_turns = _count_supported_user_turns(
            candidate_tokens,
            user_turn_texts,
        )
        support_turn_count = len(
            {record.candidate.source_turn_index for record in group}
        )
        effective_support = max(support_turn_count, transcript_support_turns)
        if should_commit_implicit_procedural_preference(
            hold_action=representative.hold_action,
            evidence_count=effective_support,
        ):
            evidence = list(
                dict.fromkeys(
                    quote
                    for record in group
                    for quote in record.candidate.evidence_quotes
                    if quote
                )
            )
            selected.append((representative, evidence[:3], effective_support))
        else:
            skipped += 1

    return selected, skipped


def _semantic_candidate_prefers_assistant_behavior(
    candidate: SemanticCandidate,
) -> bool:
    """Return whether a semantic candidate mainly encodes response guidance."""

    payload = candidate.payload
    if payload.category.lower() in _SEMANTIC_BEHAVIOR_GUIDANCE_CATEGORIES:
        return True
    if payload.object.type.lower() in _SEMANTIC_BEHAVIOR_GUIDANCE_OBJECT_TYPES:
        return True

    text = " ".join(
        (
            payload.evidence_quote,
            payload.object.identifier,
        )
    ).lower()
    return any(cue in text for cue in _SEMANTIC_PROCEDURAL_OVERLAP_CUES)


def _semantic_procedural_overlap_resolution(
    candidate: SemanticCandidate,
    procedural_candidates: list[tuple[BufferedProceduralCandidate, list[str], int]],
) -> str:
    """Resolve whether a semantic candidate should yield to promoted procedural memory."""

    if not _semantic_candidate_prefers_assistant_behavior(candidate):
        return "semantic"

    semantic_tokens = _semantic_signature_tokens(candidate)
    semantic_text = " ".join(
        (
            candidate.payload.evidence_quote,
            candidate.payload.object.identifier,
        )
    ).lower()

    for procedural_record, evidence, _effective_support in procedural_candidates:
        procedural_tokens = _candidate_tokens(
            procedural_record.candidate.payload.rule,
            *evidence,
        )
        overlap = len(semantic_tokens & procedural_tokens)
        similarity = _token_similarity(semantic_tokens, procedural_tokens)

        if similarity >= 0.5 or overlap >= 3:
            return "procedural"

        procedural_text = " ".join(
            [procedural_record.candidate.payload.rule, *evidence]
        ).lower()
        shared_cues = {
            cue
            for cue in _SEMANTIC_PROCEDURAL_OVERLAP_CUES
            if cue in semantic_text and cue in procedural_text
        }
        if shared_cues and overlap >= 2:
            return "procedural"

    return "semantic"


async def commit_session_memory(
    state: AgentState,
    *,
    memory_store: MemoryStore,
    session_buffer: SessionMemoryBuffer | None,
    stored_arc: "StoredSessionArc | None",
    embedding_provider: "EmbeddingProvider | None" = None,
    llm_client: "BaseLLMClient | None" = None,
    user_turn_texts: list[str] | None = None,
) -> SessionMemoryCommitResult | None:
    """Commit buffered semantic/procedural candidates that survived review.

    Args:
        state: Current runtime state at session end.
        memory_store: Store used for semantic/procedural writes.
        session_buffer: Runtime buffer containing held memory candidates.
        stored_arc: Optional episodic arc generated for the completed session.
        embedding_provider: Optional provider for semantic fact embeddings.
        llm_client: Optional classifier client for reconciliation.
        user_turn_texts: Explicit canonical user-turn texts from the completed
            session. When omitted, derives them from ``state["transcript"]`` for
            direct callers.

    Returns:
        Commit result when work was attempted, otherwise ``None``.
    """

    if (
        session_buffer is None
        or not session_buffer.held_semantic_candidates
        and not session_buffer.held_procedural_candidates
    ):
        return None

    owner_id = resolve_owner_id(state)
    user_turn_texts = (
        list(user_turn_texts)
        if user_turn_texts is not None
        else _user_turn_texts(state)
    )
    result = SessionMemoryCommitResult()
    current_session_ids = {
        session_id
        for session_id in (
            state.get("session_id"),
            stored_arc.session_id if stored_arc is not None else None,
            session_buffer.session_id if session_buffer is not None else None,
        )
        if session_id
    }
    try:
        prior_session_support_texts = await _load_prior_session_support_texts(
            memory_store,
            owner_id=owner_id,
            current_session_ids=current_session_ids,
        )
    except Exception:
        logger.warning(
            "commit_session_memory: failed to load prior episodic support; "
            "continuing without cross-session repetition evidence.",
            exc_info=True,
        )
        prior_session_support_texts = []

    procedural_candidates_to_commit, result.procedural_skips = (
        _select_procedural_candidates_to_commit(
            session_buffer.held_procedural_candidates,
            user_turn_texts=user_turn_texts,
        )
    )

    semantic_candidates_to_commit, result.semantic_skips = (
        _select_semantic_candidates_to_commit(
            session_buffer.held_semantic_candidates,
            stored_arc=stored_arc,
            user_turn_texts=user_turn_texts,
            prior_session_support_texts=prior_session_support_texts,
        )
    )
    if semantic_candidates_to_commit:
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

        result.semantic_skips += overlap_skips
        if batch_items:
            batch_outcome = await apply_semantic_writes_batch(
                memory_store,
                owner_id=owner_id,
                items=batch_items,
                llm_client=llm_client,
                embedding_provider=embedding_provider,
                log_context="commit_session_memory",
            )
            result.semantic_writes += batch_outcome.written
            result.semantic_bumps += batch_outcome.bumped
            result.semantic_skips += batch_outcome.skipped
    if procedural_candidates_to_commit:
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
                result.procedural_skips += 1

    logger.info(
        "commit_session_memory: session-end promotion complete — %d semantic written, "
        "%d semantic bumped, %d semantic skipped, %d procedural written, "
        "%d procedural skipped",
        result.semantic_writes,
        result.semantic_bumps,
        result.semantic_skips,
        result.procedural_writes,
        result.procedural_skips,
    )
    return result
