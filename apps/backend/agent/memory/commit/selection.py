"""Candidate-selection helpers for session-end memory commit."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent.memory.commit.clustering import (
    _SEMANTIC_BEHAVIOR_GUIDANCE_CATEGORIES,
    _SEMANTIC_BEHAVIOR_GUIDANCE_OBJECT_TYPES,
    _SEMANTIC_PROCEDURAL_OVERLAP_CUES,
    _candidate_tokens,
    _cluster_procedural_candidates,
    _cluster_semantic_candidates,
    _semantic_signature_tokens,
    _token_similarity,
)
from agent.memory.commit.scoring import (
    _arc_support_score,
    _count_supported_user_turns,
    _count_supporting_session_texts,
)
from agent.memory.policy.candidates import (
    BufferedProceduralCandidate,
    BufferedSemanticCandidate,
    SemanticCandidate,
)
from agent.memory.policy.write import (
    semantic_candidate_is_turn_scoped,
    semantic_candidate_needs_repetition_guard,
    should_commit_implicit_procedural_preference,
    should_commit_pattern,
)

if TYPE_CHECKING:
    from agent.memory.types import StoredSessionArc


def _select_semantic_candidates_to_commit(
    buffered_candidates: list[BufferedSemanticCandidate],
    *,
    stored_arc: "StoredSessionArc | None",
    user_turn_texts: list[str],
    prior_session_support_texts: list[str],
) -> tuple[list[BufferedSemanticCandidate], int]:
    """Choose which buffered semantic candidates are durable enough to commit."""
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
    """Choose which buffered implicit procedural candidates can promote."""
    selected: list[tuple[BufferedProceduralCandidate, list[str], int]] = []
    skipped = 0

    for group in _cluster_procedural_candidates(buffered_candidates):
        representative = group[-1]
        candidate = representative.candidate
        candidate_tokens = _candidate_tokens(
            candidate.payload.rule,
            *candidate.evidence_quotes,
        )
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
