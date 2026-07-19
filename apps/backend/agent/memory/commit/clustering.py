"""Token normalization and clustering helpers for session-end memory commit.

Clustering groups near-duplicate candidates before promotion. By default it uses
lexical token-set similarity (no dependencies, deterministic). When precomputed
embeddings are supplied (session-end has an embedding provider), it groups by
embedding cosine instead — more robust to paraphrase than the hand-tuned synonym
map, with the lexical path retained as the no-embeddings fallback.
"""

from __future__ import annotations

from agent.memory.policy.candidates import (
    BufferedProceduralCandidate,
    BufferedSemanticCandidate,
    ProceduralCandidate,
    SemanticCandidate,
)
from agent.memory.retrieval.ranking import cosine_similarity
from agent.memory.text_tokens import tokenize_meaningful

# Two candidates whose embeddings exceed this cosine are treated as paraphrases
# of the same memory and grouped. Calibrated against the lexical thresholds
# (~0.5 Jaccard) but on embedding space, where paraphrases sit higher.
_CLUSTER_COSINE_THRESHOLD = 0.83


def semantic_cluster_text(candidate: SemanticCandidate) -> str:
    """Canonical text embedded for semantic clustering."""
    return (
        f"{candidate.payload.evidence_quote} {candidate.payload.object.identifier}"
    ).strip()


def procedural_cluster_text(candidate: ProceduralCandidate) -> str:
    """Canonical text embedded for procedural clustering."""
    return " ".join([candidate.payload.rule, *candidate.evidence_quotes]).strip()


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
    *,
    embeddings: list[list[float]] | None = None,
) -> list[list[BufferedSemanticCandidate]]:
    """Cluster semantic candidates that express the same support pattern.

    When ``embeddings`` is supplied (one vector per candidate, same order),
    paraphrase grouping uses embedding cosine; otherwise it falls back to the
    lexical token signatures. The object-anchor guard applies in BOTH modes:
    facts about different objects are never merged even if otherwise similar.
    """
    groups: list[list[BufferedSemanticCandidate]] = []
    group_tokens: list[frozenset[str]] = []
    group_normalized_tokens: list[frozenset[str]] = []
    group_object_anchors: list[frozenset[str]] = []
    group_keys: list[set[tuple[str, ...]]] = []
    group_embeddings: list[list[float]] = []

    for position, record in enumerate(buffered_candidates):
        key = _semantic_group_key(record.candidate)
        tokens = _semantic_signature_tokens(record.candidate)
        normalized_tokens = _semantic_normalization_signature(record.candidate)
        object_anchor_tokens = _semantic_object_anchor_tokens(record.candidate)
        record_embedding = embeddings[position] if embeddings is not None else None
        placed = False
        for index, existing_tokens in enumerate(group_tokens):
            anchor_overlap = len(object_anchor_tokens & group_object_anchors[index])
            anchor_similarity = _token_similarity(
                object_anchor_tokens,
                group_object_anchors[index],
            )
            same_group_key = key in group_keys[index]
            semantically_aligned_object = (
                anchor_overlap >= 1 or anchor_similarity >= 0.5
            )
            if record_embedding is not None and group_embeddings[index]:
                similar = (
                    cosine_similarity(record_embedding, group_embeddings[index])
                    >= _CLUSTER_COSINE_THRESHOLD
                )
            else:
                overlap = len(tokens & existing_tokens)
                similarity = _token_similarity(tokens, existing_tokens)
                normalized_overlap = len(
                    normalized_tokens & group_normalized_tokens[index]
                )
                normalized_similarity = _token_similarity(
                    normalized_tokens,
                    group_normalized_tokens[index],
                )
                similar = (
                    similarity >= 0.5
                    or overlap >= 3
                    or normalized_similarity >= 0.5
                    or normalized_overlap >= 3
                )
            if same_group_key or (semantically_aligned_object and similar):
                groups[index].append(record)
                group_tokens[index] = frozenset(existing_tokens | tokens)
                group_normalized_tokens[index] = frozenset(
                    group_normalized_tokens[index] | normalized_tokens
                )
                group_object_anchors[index] = frozenset(
                    group_object_anchors[index] | object_anchor_tokens
                )
                group_keys[index].add(key)
                # Group embedding stays the seed (first member); merging vectors
                # is not meaningful, and the seed is a stable group representative.
                placed = True
                break
        if not placed:
            groups.append([record])
            group_tokens.append(tokens)
            group_normalized_tokens.append(normalized_tokens)
            group_object_anchors.append(object_anchor_tokens)
            group_keys.append({key})
            group_embeddings.append(record_embedding if record_embedding else [])

    return groups


def _cluster_procedural_candidates(
    buffered_candidates: list[BufferedProceduralCandidate],
    *,
    embeddings: list[list[float]] | None = None,
) -> list[list[BufferedProceduralCandidate]]:
    """Cluster procedural candidates with similar repeated preferences.

    Uses embedding cosine when ``embeddings`` is supplied (one vector per
    candidate, same order), else lexical token signatures.
    """
    groups: list[list[BufferedProceduralCandidate]] = []
    group_tokens: list[frozenset[str]] = []
    group_normalized_tokens: list[frozenset[str]] = []
    group_embeddings: list[list[float]] = []

    for position, record in enumerate(buffered_candidates):
        tokens = _procedural_signature_tokens(record.candidate)
        normalized_tokens = _procedural_normalization_signature(record.candidate)
        record_embedding = embeddings[position] if embeddings is not None else None
        placed = False
        for index, existing_tokens in enumerate(group_tokens):
            if record_embedding is not None and group_embeddings[index]:
                similar = (
                    cosine_similarity(record_embedding, group_embeddings[index])
                    >= _CLUSTER_COSINE_THRESHOLD
                )
            else:
                overlap = len(tokens & existing_tokens)
                similarity = _token_similarity(tokens, existing_tokens)
                normalized_overlap = len(
                    normalized_tokens & group_normalized_tokens[index]
                )
                normalized_similarity = _token_similarity(
                    normalized_tokens,
                    group_normalized_tokens[index],
                )
                similar = (
                    similarity >= 0.5
                    or overlap >= 3
                    or normalized_similarity >= 0.5
                    or normalized_overlap >= 3
                )
            if similar:
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
            group_embeddings.append(record_embedding if record_embedding else [])

    return groups
