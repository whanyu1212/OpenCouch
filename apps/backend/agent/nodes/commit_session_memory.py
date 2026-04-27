"""Session-end promotion pass for buffered memory candidates.

Turn nodes buffer semantic/procedural candidates that should not commit
immediately, and this module decides which buffered items have enough
support to become durable memory at session end.

The promotion policy is intentionally conservative:

- ``require_repetition`` semantic candidates only promote after support
  from at least two distinct user turns, or one turn plus support from
  a prior session summary.
- Other held semantic candidates can promote when they are repeated, or
  when both the transcript and the session summary clearly support them.
- Implicit procedural preferences only promote after repeated evidence.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent.memory.candidates import (
    ProceduralCandidate,
    SemanticCandidate,
    SessionMemoryBuffer,
)
from agent.memory.dedup import find_near_duplicate
from agent.memory.procedural import (
    aupsert_procedural_rule,
    build_procedural_rule,
)
from agent.memory.reconciliation import (
    filter_semantic_collision_candidates,
    plan_semantic_write_llm_primary,
)
from agent.memory.store import MemoryStore, StoreRecord
from agent.memory.text_tokens import tokenize_meaningful
from agent.memory.write_policy import (
    should_commit_implicit_procedural_preference,
    should_commit_pattern,
)
from agent.nodes.extract_facts import (
    _bump_last_referenced_at,
    _fetch_existing_user_records,
    _mark_fact_superseded,
    _memory_write_to_semantic_fact,
    _write_new_fact,
)
from agent.state import AgentState, resolve_owner_id

if TYPE_CHECKING:
    from agent.memory.embeddings import EmbeddingProvider
    from agent.memory.models import StoredSessionArc
    from services.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


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
        state: Current graph state containing the session transcript.

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
    current_session_id: str | None,
) -> list[str]:
    """Return support texts from prior episodic arcs for this owner.

    Args:
        memory_store: Store containing episodic memory records.
        owner_id: Owner whose prior sessions should be loaded.
        current_session_id: Session id to exclude from prior support.

    Returns:
        Prior session support text blobs.
    """

    records = await memory_store.asearch((owner_id, "episodic"), query=None, limit=100)
    prior_texts: list[str] = []
    for record in records:
        value = record.value
        if value.get("session_id") == current_session_id:
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


def _cluster_procedural_candidates(
    buffered_candidates: list[ProceduralCandidate],
) -> list[list[ProceduralCandidate]]:
    """Cluster procedural candidates with similar repeated preferences.

    Args:
        buffered_candidates: Procedural candidates buffered during the session.

    Returns:
        Groups of similar procedural candidates.
    """

    groups: list[list[ProceduralCandidate]] = []
    group_tokens: list[frozenset[str]] = []

    for candidate in buffered_candidates:
        tokens = _procedural_signature_tokens(candidate)
        placed = False
        for index, existing_tokens in enumerate(group_tokens):
            overlap = len(tokens & existing_tokens)
            similarity = _token_similarity(tokens, existing_tokens)
            if similarity >= 0.5 or overlap >= 3:
                groups[index].append(candidate)
                group_tokens[index] = frozenset(existing_tokens | tokens)
                placed = True
                break
        if not placed:
            groups.append([candidate])
            group_tokens.append(tokens)

    return groups


def _select_semantic_candidates_to_commit(
    buffered_candidates: list[SemanticCandidate],
    *,
    stored_arc: "StoredSessionArc | None",
    user_turn_texts: list[str],
    prior_session_support_texts: list[str],
) -> tuple[list[SemanticCandidate], int]:
    """Choose which buffered semantic candidates are durable enough to commit.

    Args:
        buffered_candidates: Semantic candidates buffered during the session.
        stored_arc: Optional session arc produced at session end.
        user_turn_texts: User transcript turns for repetition checks.
        prior_session_support_texts: Prior episodic support texts.

    Returns:
        Selected candidates plus skipped group count.
    """

    grouped: dict[tuple[str, ...], list[SemanticCandidate]] = defaultdict(list)
    for candidate in buffered_candidates:
        grouped[_semantic_group_key(candidate)].append(candidate)

    selected: list[SemanticCandidate] = []
    skipped = 0
    for group in grouped.values():
        support_turn_count = len({c.source_turn_index for c in group})
        repetition_candidate = next(
            (
                candidate
                for candidate in reversed(group)
                if candidate.policy_recommendation == "require_repetition"
            ),
            None,
        )
        representative = repetition_candidate or group[-1]
        object_identifier = representative.payload.object.identifier.lower().strip()
        candidate_tokens = _candidate_tokens(
            representative.payload.evidence_quote,
            representative.payload.object.identifier,
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

        should_commit = False
        if representative.policy_recommendation == "require_repetition":
            should_commit = should_commit_pattern(
                representative,
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
    buffered_candidates: list[ProceduralCandidate],
    *,
    user_turn_texts: list[str],
) -> tuple[list[tuple[ProceduralCandidate, list[str], int]], int]:
    """Choose which buffered implicit procedural candidates can promote.

    Args:
        buffered_candidates: Procedural candidates buffered during the session.
        user_turn_texts: User transcript turns for repetition checks.

    Returns:
        Selected candidates with evidence/support counts plus skipped group count.
    """

    selected: list[tuple[ProceduralCandidate, list[str], int]] = []
    skipped = 0

    for group in _cluster_procedural_candidates(buffered_candidates):
        representative = group[-1]
        candidate_tokens = _procedural_signature_tokens(representative)
        transcript_support_turns = _count_supported_user_turns(
            candidate_tokens,
            user_turn_texts,
        )
        support_turn_count = len({candidate.source_turn_index for candidate in group})
        effective_support = max(support_turn_count, transcript_support_turns)
        if should_commit_implicit_procedural_preference(
            representative,
            evidence_count=effective_support,
        ):
            evidence = list(
                dict.fromkeys(
                    quote
                    for candidate in group
                    for quote in candidate.evidence_quotes
                    if quote
                )
            )
            selected.append((representative, evidence[:3], effective_support))
        else:
            skipped += 1

    return selected, skipped


async def run_commit_session_memory(
    state: AgentState,
    *,
    memory_store: MemoryStore,
    session_buffer: SessionMemoryBuffer | None,
    stored_arc: "StoredSessionArc | None",
    embedding_provider: "EmbeddingProvider | None" = None,
    llm_client: "BaseLLMClient | None" = None,
) -> SessionMemoryCommitResult | None:
    """Commit buffered semantic/procedural candidates that survived review.

    Args:
        state: Current graph state at session end.
        memory_store: Store used for semantic/procedural writes.
        session_buffer: Runtime buffer containing held memory candidates.
        stored_arc: Optional episodic arc generated for the completed session.
        embedding_provider: Optional provider for semantic fact embeddings.
        llm_client: Optional classifier client for reconciliation.

    Returns:
        Commit result when work was attempted, otherwise ``None``.
    """

    if (
        session_buffer is None
        or not session_buffer.semantic_candidates
        and not session_buffer.procedural_candidates
    ):
        return None

    owner_id = resolve_owner_id(state)
    user_turn_texts = _user_turn_texts(state)
    result = SessionMemoryCommitResult()
    current_session_id = (
        state.get("session_id")
        or (stored_arc.session_id if stored_arc is not None else None)
        or (session_buffer.session_id if session_buffer is not None else None)
    )
    try:
        prior_session_support_texts = await _load_prior_session_support_texts(
            memory_store,
            owner_id=owner_id,
            current_session_id=current_session_id,
        )
    except Exception:
        logger.warning(
            "commit_session_memory: failed to load prior episodic support; "
            "continuing without cross-session repetition evidence.",
            exc_info=True,
        )
        prior_session_support_texts = []

    semantic_candidates_to_commit, result.semantic_skips = (
        _select_semantic_candidates_to_commit(
            session_buffer.semantic_candidates,
            stored_arc=stored_arc,
            user_turn_texts=user_turn_texts,
            prior_session_support_texts=prior_session_support_texts,
        )
    )
    if semantic_candidates_to_commit:
        try:
            existing_records = await _fetch_existing_user_records(
                memory_store,
                owner_id=owner_id,
            )
        except Exception:
            logger.warning(
                "commit_session_memory: failed to fetch existing semantic records; "
                "skipping session-end semantic commit.",
                exc_info=True,
            )
            result.semantic_skips += len(semantic_candidates_to_commit)
            semantic_candidates_to_commit = []
            existing_records = []

        candidate_embeddings: list[list[float] | None] = [None] * len(
            semantic_candidates_to_commit
        )
        embedding_model_name: str | None = None
        if semantic_candidates_to_commit and embedding_provider is not None:
            try:
                quotes = [
                    candidate.payload.evidence_quote
                    for candidate in semantic_candidates_to_commit
                ]
                candidate_embeddings = await embedding_provider.aembed(
                    quotes,
                    task_type="RETRIEVAL_DOCUMENT",
                )
                embedding_model_name = embedding_provider.model_name
                if all(embedding is None for embedding in candidate_embeddings):
                    embedding_model_name = None
            except Exception:
                logger.warning(
                    "commit_session_memory: embedding batch failed; writing session-end "
                    "semantic facts without embeddings.",
                    exc_info=True,
                )
                candidate_embeddings = [None] * len(semantic_candidates_to_commit)
                embedding_model_name = None

        for candidate_index, candidate in enumerate(semantic_candidates_to_commit):
            write = candidate.payload
            collision_records = filter_semantic_collision_candidates(
                write,
                existing_records,
            )
            try:
                matched = find_near_duplicate(write, collision_records)
            except Exception:
                logger.warning(
                    "commit_session_memory: dedup check failed for buffered semantic "
                    "candidate %r; skipping it.",
                    write.evidence_quote[:60],
                    exc_info=True,
                )
                result.semantic_skips += 1
                continue

            if matched is not None:
                try:
                    await _bump_last_referenced_at(
                        memory_store,
                        matched_record=matched,
                    )
                    result.semantic_bumps += 1
                except Exception:
                    logger.warning(
                        "commit_session_memory: failed to bump last_referenced_at on %r.",
                        matched.key,
                        exc_info=True,
                    )
                    result.semantic_skips += 1
                continue

            try:
                write_timing = (
                    "promotion"
                    if candidate.policy_recommendation == "require_repetition"
                    else "session_end"
                )
                write_reason = (
                    "repetition-qualified semantic candidate promoted at session end"
                    if write_timing == "promotion"
                    else "session-end semantic candidate supported by transcript and episodic summary"
                )
                fact = _memory_write_to_semantic_fact(
                    write,
                    write_timing=write_timing,
                    write_reason=write_reason,
                    policy_version="phase3_v1",
                )
                reconciliation = await plan_semantic_write_llm_primary(
                    fact,
                    collision_records,
                    llm_client=llm_client,
                )
                if reconciliation.bump_record is not None:
                    await _bump_last_referenced_at(
                        memory_store,
                        matched_record=reconciliation.bump_record,
                    )
                    result.semantic_bumps += 1
                    continue
                this_embedding = candidate_embeddings[candidate_index]
                this_model = (
                    embedding_model_name if this_embedding is not None else None
                )
                await _write_new_fact(
                    memory_store,
                    owner_id=owner_id,
                    fact=fact,
                    embedding=this_embedding,
                    embedding_model=this_model,
                )
                result.semantic_writes += 1
                existing_records.append(
                    StoreRecord(
                        namespace=(owner_id, "semantic"),
                        key=fact.id,
                        value=fact.model_dump(mode="json"),
                        embedding=this_embedding,
                        embedding_model=this_model,
                    )
                )
                for superseded_record in reconciliation.supersede_records:
                    try:
                        await _mark_fact_superseded(
                            memory_store,
                            matched_record=superseded_record,
                            replacement_fact_id=fact.id,
                        )
                        superseded_record.value["last_referenced_at"] = fact.created_at
                        superseded_record.value["dormant_at"] = fact.created_at
                        superseded_record.value["superseded_by"] = fact.id
                    except Exception:
                        logger.warning(
                            "commit_session_memory: failed to mark stale fact %r "
                            "as superseded after writing replacement.",
                            superseded_record.key,
                            exc_info=True,
                        )
            except Exception:
                logger.warning(
                    "commit_session_memory: failed to write buffered semantic candidate %r.",
                    write.evidence_quote[:60],
                    exc_info=True,
                )
                result.semantic_skips += 1

    procedural_candidates_to_commit, result.procedural_skips = (
        _select_procedural_candidates_to_commit(
            session_buffer.procedural_candidates,
            user_turn_texts=user_turn_texts,
        )
    )
    if procedural_candidates_to_commit:
        for (
            procedural_candidate,
            evidence,
            effective_support,
        ) in procedural_candidates_to_commit:
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
