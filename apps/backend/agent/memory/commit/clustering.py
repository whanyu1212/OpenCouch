"""Token normalization and clustering helpers for session-end memory commit."""

from __future__ import annotations

from agent.memory.policy.candidates import (
    BufferedProceduralCandidate,
    BufferedSemanticCandidate,
    ProceduralCandidate,
    SemanticCandidate,
)
from agent.memory.text_tokens import tokenize_meaningful

# Cue/stopword/category vocabularies that calibrate signature normalization,
# semantic-vs-procedural overlap resolution, and behavior-guidance detection.
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
_NORMALIZATION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "been",
    "being",
    "feels",
    "for",
    "i",
    "im",
    "is",
    "it",
    "its",
    "me",
    "my",
    "please",
    "remember",
    "said",
    "that",
    "the",
    "to",
    "user",
    "very",
    "when",
}
_SEMANTIC_GENERIC_OBJECT_TOKENS = {
    "anxiety",
    "concern",
    "panic",
    "stress",
    "trigger",
    "worry",
}


def _semantic_group_key(candidate: SemanticCandidate) -> tuple[str, ...]:
    """Return the grouping key for repeated semantic candidates."""
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
    """Return one meaningful-token signature for candidate/support text."""
    return tokenize_meaningful(" ".join(part for part in parts if part))


def _normalize_token(token: str) -> str:
    """Return a lightweight normalized token for paraphrase clustering."""
    normalized = token.lower().strip()
    if len(normalized) > 5 and normalized.endswith("able"):
        normalized = normalized[:-4]
    elif len(normalized) > 5 and normalized.endswith("ing"):
        normalized = normalized[:-3]
    elif len(normalized) > 4 and normalized.endswith("ed"):
        normalized = normalized[:-2]
    elif len(normalized) > 4 and normalized.endswith("ly"):
        normalized = normalized[:-2]
    elif len(normalized) > 4 and normalized.endswith("es"):
        normalized = normalized[:-2]
    elif len(normalized) > 3 and normalized.endswith("s"):
        normalized = normalized[:-1]

    synonym_map = {
        "bite": "small",
        "bitesize": "small",
        "chunks": "small",
        "chunk": "small",
        "concise": "direct",
        "manageable": "manage",
        "panicked": "panic",
        "panicky": "panic",
        "presentations": "presentation",
        "prep": "preparation",
        "supportive": "support",
        "talks": "presentation",
        "tiny": "small",
        "overwhelmed": "overwhelm",
    }
    return synonym_map.get(normalized, normalized)


def _normalized_signature_tokens(*parts: str) -> frozenset[str]:
    """Return normalized signature tokens for paraphrase-aware clustering."""
    normalized_tokens = {
        _normalize_token(token)
        for token in _candidate_tokens(*parts)
        if token not in _NORMALIZATION_STOPWORDS
    }
    return frozenset(token for token in normalized_tokens if token)


def _semantic_normalization_signature(
    candidate: SemanticCandidate,
) -> frozenset[str]:
    """Return normalized semantic signature tokens for clustering."""
    return _normalized_signature_tokens(
        candidate.payload.evidence_quote,
        candidate.payload.object.identifier,
    )


def _procedural_normalization_signature(
    candidate: ProceduralCandidate,
) -> frozenset[str]:
    """Return normalized procedural signature tokens for clustering."""
    return _normalized_signature_tokens(
        candidate.payload.rule,
        *candidate.evidence_quotes,
    )


def _semantic_object_anchor_tokens(candidate: SemanticCandidate) -> frozenset[str]:
    """Return normalized object-identifier tokens minus generic affect labels."""
    return frozenset(
        token
        for token in _normalized_signature_tokens(candidate.payload.object.identifier)
        if token not in _SEMANTIC_GENERIC_OBJECT_TOKENS
    )


def _procedural_signature_tokens(candidate: ProceduralCandidate) -> frozenset[str]:
    """Return the similarity signature for a procedural candidate."""
    return _candidate_tokens(candidate.payload.rule, *candidate.evidence_quotes)


def _semantic_signature_tokens(candidate: SemanticCandidate) -> frozenset[str]:
    """Return the similarity signature for a semantic candidate."""
    return _candidate_tokens(
        candidate.payload.evidence_quote,
        candidate.payload.object.identifier,
    )


def _token_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    """Return token-set similarity for lightweight grouping/dedup."""
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
    group_normalized_tokens: list[frozenset[str]] = []
    group_object_anchors: list[frozenset[str]] = []
    group_keys: list[set[tuple[str, ...]]] = []

    for record in buffered_candidates:
        key = _semantic_group_key(record.candidate)
        tokens = _semantic_signature_tokens(record.candidate)
        normalized_tokens = _semantic_normalization_signature(record.candidate)
        object_anchor_tokens = _semantic_object_anchor_tokens(record.candidate)
        placed = False
        for index, existing_tokens in enumerate(group_tokens):
            overlap = len(tokens & existing_tokens)
            similarity = _token_similarity(tokens, existing_tokens)
            normalized_overlap = len(normalized_tokens & group_normalized_tokens[index])
            normalized_similarity = _token_similarity(
                normalized_tokens,
                group_normalized_tokens[index],
            )
            anchor_overlap = len(object_anchor_tokens & group_object_anchors[index])
            anchor_similarity = _token_similarity(
                object_anchor_tokens,
                group_object_anchors[index],
            )
            same_group_key = key in group_keys[index]
            semantically_aligned_object = (
                anchor_overlap >= 1 or anchor_similarity >= 0.5
            )
            if same_group_key or (
                semantically_aligned_object
                and (
                    similarity >= 0.5
                    or overlap >= 3
                    or normalized_similarity >= 0.5
                    or normalized_overlap >= 3
                )
            ):
                groups[index].append(record)
                group_tokens[index] = frozenset(existing_tokens | tokens)
                group_normalized_tokens[index] = frozenset(
                    group_normalized_tokens[index] | normalized_tokens
                )
                group_object_anchors[index] = frozenset(
                    group_object_anchors[index] | object_anchor_tokens
                )
                group_keys[index].add(key)
                placed = True
                break
        if not placed:
            groups.append([record])
            group_tokens.append(tokens)
            group_normalized_tokens.append(normalized_tokens)
            group_object_anchors.append(object_anchor_tokens)
            group_keys.append({key})

    return groups


def _cluster_procedural_candidates(
    buffered_candidates: list[BufferedProceduralCandidate],
) -> list[list[BufferedProceduralCandidate]]:
    """Cluster procedural candidates with similar repeated preferences."""
    groups: list[list[BufferedProceduralCandidate]] = []
    group_tokens: list[frozenset[str]] = []
    group_normalized_tokens: list[frozenset[str]] = []

    for record in buffered_candidates:
        tokens = _procedural_signature_tokens(record.candidate)
        normalized_tokens = _procedural_normalization_signature(record.candidate)
        placed = False
        for index, existing_tokens in enumerate(group_tokens):
            overlap = len(tokens & existing_tokens)
            similarity = _token_similarity(tokens, existing_tokens)
            normalized_overlap = len(normalized_tokens & group_normalized_tokens[index])
            normalized_similarity = _token_similarity(
                normalized_tokens,
                group_normalized_tokens[index],
            )
            if (
                similarity >= 0.5
                or overlap >= 3
                or normalized_similarity >= 0.5
                or normalized_overlap >= 3
            ):
                groups[index].append(record)
                group_tokens[index] = frozenset(existing_tokens | tokens)
                group_normalized_tokens[index] = frozenset(
                    group_normalized_tokens[index] | normalized_tokens
                )
                placed = True
                break
        if not placed:
            groups.append([record])
            group_tokens.append(tokens)
            group_normalized_tokens.append(normalized_tokens)

    return groups
