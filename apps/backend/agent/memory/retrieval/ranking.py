"""Retrieval scoring helpers for hybrid memory search.

This module owns the lexical scorer, dense scorer, Reciprocal Rank
Fusion (RRF) combiner, and cosine-similarity math used by both memory
stores. Lexical recall handles exact token overlap and proper nouns;
dense retrieval handles paraphrases when embeddings are available.

Why hybrid (RRF) rather than pure embedding:

1. **Proper nouns.** Embeddings treat named entities as "just
   another token" and dilute the name signal across the sentence
   context. Token-recall does the opposite — a query like "tell
   me about Sarah" against a stored fact "my sister Sarah visited"
   is an exact overlap win for token-recall and an ambiguous mid-
   tier score for embeddings. The extractor writes proper-noun-heavy
   facts (``KNOWS Sarah``, ``USES fluoxetine``), so lexical recall
   must remain part of the ranking.

2. **Short queries.** Embedding quality drops for short queries
   because there's not enough context for the model to disambiguate.
   Token-recall handles single-token queries cleanly.

3. **Fallback path.** If the embedding provider is unavailable
   (guest mode, no API key, API outage), hybrid with only the
   token-recall input gracefully degenerates to lexical search with
   no special-case logic needed.

Why RRF specifically (vs. weighted score fusion or cascade rerank):

- **Zero tuning.** RRF's constant ``k=60`` is well-calibrated
  across retrieval datasets (Cormack et al. 2009); you don't
  re-tune it per corpus. Weighted fusion needs per-dataset
  weight tuning because embedding cosine and token-recall
  scores have different distributions.
- **Rank-based, not score-based.** RRF depends only on the rank
  position of each record in each scorer, not on the raw score
  values. That makes it robust to score scale differences (an
  embedding cosine of 0.7 vs. a token-recall of 0.7 are not
  comparable as floats, but their rank positions are).
- **Standard practice.** RRF is the default hybrid mode in
  Elasticsearch, Vespa, Milvus, and the BEIR benchmark suite's
  hybrid baselines, so contributors see a familiar pattern.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from agent.memory.text_tokens import tokenize, tokenize_meaningful

if TYPE_CHECKING:
    from agent.memory.store import StoreRecord


# Reciprocal Rank Fusion smoothing constant. The canonical value
# from Cormack, Clarke, and Buettcher (2009) is 60, and it's been
# re-validated across retrieval benchmarks since. Lower values
# weight the top of each ranking more heavily; higher values
# flatten the contribution of rank position and let lower-ranked
# matches contribute more to the fused score.
#
# Don't tune this without a good reason. The default is chosen so
# that rank 1 contributes 1/61 ≈ 0.0164 and rank 10 contributes
# 1/70 ≈ 0.0143 — close enough that multiple scorers each ranking
# a record highly matters more than any single scorer ranking it
# very highly. That asymmetry is what makes RRF "hybrid" rather
# than "whichever scorer shouted loudest."
RRF_K = 60

# Minimum cosine similarity for an embedding match to count as a
# hit. Unlike the 0.33 token-recall threshold (which is chosen to
# land "1 topical token out of 3 meaningful query tokens"), this
# threshold is chosen to keep spurious near-zero matches out of
# the embedding-ranked list. A cosine of 0.5 is roughly "somewhat
# related"; below that, the embedding's signal is too weak to
# trust and we'd rather not contribute it to the fusion.
#
# Tuning note: this threshold is deliberately loose compared to
# dense-retrieval benchmarks that use 0.7+ cutoffs, because
# OpenCouch's retrieval path feeds the results into the response
# prompt as "Previously noted: X" context — the cost of a false
# positive (irrelevant context in the prompt) is bounded, while
# the cost of a false negative (missed relevant memory) hits
# user experience directly. Tune retrieval loose, tune dedup tight
# — same asymmetry as the token-recall threshold discussion in
# store.py.
EMBEDDING_MATCH_THRESHOLD = 0.5
DENSE_CANDIDATE_MULTIPLIER = 5
MIN_DENSE_CANDIDATES = 50

# Secondary lexical path for wordy queries against compact records.
#
# Query recall alone penalizes natural questions such as "who did I say I
# reach out to when panic starts?" because the denominator is the whole
# question. If two or more query terms cover a meaningful share of a short
# memory record, the record is usually relevant even when query recall is just
# below SEARCH_MATCH_THRESHOLD.
SHORT_RECORD_MIN_SHARED_TOKENS = 2
SHORT_RECORD_MIN_RECORD_RECALL = 0.4
_PRIMARY_TEXT_FIELDS = ("evidence_quote", "summary", "text", "content")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute the cosine similarity between two equal-length vectors.

    Returns a float in ``[-1.0, 1.0]`` where 1.0 is perfectly
    aligned, 0.0 is orthogonal, and -1.0 is opposite. Returns
    ``0.0`` for either vector being empty or all-zero (no
    meaningful similarity to compute).

    Pure Python implementation with no numpy dependency. At the
    current expected scale of hundreds of records per user, this is
    fast enough for a namespace scan. This function is the boundary
    where a vectorized implementation can replace the loop if needed.

    Args:
        a: First vector as a list of floats.
        b: Second vector as a list of floats.

    Returns:
        Cosine similarity in ``[-1.0, 1.0]``, or ``0.0`` if
        either input is invalid or mismatched in length.
    """

    if not a or not b:
        return 0.0
    if len(a) != len(b):
        return 0.0

    dot = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a_sq += x * x
        norm_b_sq += y * y

    if norm_a_sq == 0.0 or norm_b_sq == 0.0:
        return 0.0

    return dot / (math.sqrt(norm_a_sq) * math.sqrt(norm_b_sq))


def dense_candidate_limit(limit: int) -> int:
    """Return the shared pre-fusion dense candidate bound."""

    return max(limit * DENSE_CANDIDATE_MULTIPLIER, MIN_DENSE_CANDIDATES)


@dataclass(slots=True)
class ScoredRecord:
    """A store record paired with a retrieval score.

    Used as the intermediate type in the retrieval path before
    the final fusion + limit slice. The ``insertion_index`` field
    is used as the tiebreaker when two records share the same
    score, matching the lexical-search contract so behavior is
    deterministic across runs.
    """

    record: "StoreRecord"
    score: float
    insertion_index: int


@dataclass(slots=True)
class IndexedRecord:
    """A store record paired with a caller-defined insertion index.

    The store implementations choose what "insertion order" means for
    a given scan and pass that through explicitly. This lets the shared
    retrieval helpers preserve each store's current tiebreaker semantics
    instead of inventing a new indexing policy.
    """

    record: "StoreRecord"
    insertion_index: int


def _record_haystack(record: "StoreRecord") -> str:
    """Build the lexical haystack string for a store record.

    Args:
        record (StoreRecord): Store record to serialize.

    Returns:
        str: Concatenated non-null field values used for lexical scoring.
    """

    return " ".join(str(value) for value in record.value.values() if value is not None)


def _record_primary_text(record: "StoreRecord") -> str:
    """Return the compact text field that best represents a memory record.

    Args:
        record (StoreRecord): Store record to inspect.

    Returns:
        str: Primary searchable text, or the full haystack when no known
        primary text field exists.
    """

    for field in _PRIMARY_TEXT_FIELDS:
        value = record.value.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return _record_haystack(record)


def lexical_rank(
    candidates: Sequence[IndexedRecord],
    *,
    query_text: str,
    match_threshold: float,
) -> list[ScoredRecord]:
    """Rank candidates by lexical token recall.

    Args:
        candidates: Candidate records with caller-defined insertion indices.
        query_text: Raw query text from the caller.
        match_threshold: Minimum recall score required for a hit.

    Returns:
        Ranked lexical hits sorted by score descending, then insertion
        index ascending. Returns an empty list when the query has no
        meaningful tokens or when no candidate clears the threshold.
    """

    query_tokens = tokenize_meaningful(query_text)
    if not query_tokens:
        return []

    query_token_count = len(query_tokens)
    ranked: list[ScoredRecord] = []
    for candidate in candidates:
        haystack = _record_haystack(candidate.record)
        haystack_tokens = tokenize(haystack)
        if not haystack_tokens:
            continue
        meaningful_primary_tokens = tokenize_meaningful(
            _record_primary_text(candidate.record)
        )
        overlap = query_tokens & haystack_tokens
        query_recall = len(overlap) / query_token_count
        record_overlap = query_tokens & meaningful_primary_tokens
        record_recall = (
            len(record_overlap) / len(meaningful_primary_tokens)
            if meaningful_primary_tokens
            else 0.0
        )
        if query_recall >= match_threshold or (
            len(record_overlap) >= SHORT_RECORD_MIN_SHARED_TOKENS
            and record_recall >= SHORT_RECORD_MIN_RECORD_RECALL
        ):
            score = query_recall if query_recall >= match_threshold else record_recall
            ranked.append(
                ScoredRecord(
                    record=candidate.record,
                    score=score,
                    insertion_index=candidate.insertion_index,
                )
            )

    ranked.sort(key=lambda scored: (-scored.score, scored.insertion_index))
    return ranked


def dense_rank(
    candidates: Sequence[IndexedRecord],
    *,
    query_embedding: list[float] | None,
    embedding_model: str | None,
) -> list[ScoredRecord]:
    """Rank candidates by embedding cosine similarity.

    Args:
        candidates: Candidate records with caller-defined insertion indices.
        query_embedding: Query embedding to compare against stored vectors.
        embedding_model: Optional model identifier used to skip cross-model
            similarity comparisons.

    Returns:
        Ranked dense hits sorted by score descending, then insertion
        index ascending. Returns an empty list when no query embedding
        is available or when no candidate clears the embedding threshold.
    """

    if query_embedding is None:
        return []

    ranked: list[ScoredRecord] = []
    for candidate in candidates:
        record = candidate.record
        if record.embedding is None:
            continue
        if embedding_model is not None and record.embedding_model != embedding_model:
            continue
        if len(record.embedding) != len(query_embedding):
            continue
        similarity = cosine_similarity(query_embedding, record.embedding)
        if similarity >= EMBEDDING_MATCH_THRESHOLD:
            ranked.append(
                ScoredRecord(
                    record=record,
                    score=similarity,
                    insertion_index=candidate.insertion_index,
                )
            )

    ranked.sort(key=lambda scored: (-scored.score, scored.insertion_index))
    return ranked


def rrf_fuse(
    *,
    lexical_ranked: list[ScoredRecord],
    dense_ranked: list[ScoredRecord],
    limit: int,
    k: int = RRF_K,
) -> list["StoreRecord"]:
    """Combine two ranked lists via Reciprocal Rank Fusion.

    Each record gets an RRF score equal to the sum of ``1 / (k + rank)``
    across the lists it appears in, where ``rank`` is its 1-indexed
    position in that list. Records appearing in both lists accumulate
    contributions from both; records appearing in only one list get
    only that list's contribution. Records in neither list are
    excluded from the output.

    The constant ``k`` dampens the contribution of top-of-list
    positions so multiple scorers each ranking a record highly
    matters more than any single scorer ranking it very highly.
    The canonical value of 60 is a well-known default from the
    original RRF paper (Cormack et al. 2009) and is the common
    choice across Elasticsearch, Vespa, and BEIR benchmark baselines.

    Ties in RRF score are broken by the record's ``insertion_index``
    (ascending) so behavior stays deterministic across runs. This
    matches the lexical-search contract where insertion order is the
    tiebreaker for equal-recall records.

    When one of the input lists is empty (e.g., no embedding
    provider configured, so ``dense_ranked`` is empty), RRF
    degenerates cleanly to the remaining ranking — you just get
    the token-recall or embedding results as-is, normalized by the
    rank-based scoring. That's the "graceful fallback" path: no
    special-case code is needed because RRF with one input list
    is still RRF.

    Args:
        lexical_ranked: The token-recall ranked results, sorted
            best-first. Usually the output of the lexical scan.
        dense_ranked: The embedding-similarity ranked results, sorted
            best-first. Empty when no embedding provider is available.
        limit: Maximum number of records to return after fusion.
        k: The RRF smoothing constant. Defaults to :data:`RRF_K` (60);
            overridable for tuning experiments but should not be
            changed casually.

    Returns:
        A list of :class:`StoreRecord` objects ordered by RRF score
        descending, with insertion-order tiebreaking, truncated to
        ``limit``. May be shorter than ``limit`` if the input lists
        had fewer unique records combined.
    """

    # Build a dict keyed by ``(namespace, key)`` tuples so we can
    # accumulate scores for records that appear in both lists. The
    # namespace+key is the unique identity — same record under
    # different namespaces (e.g., the same id across semantic and
    # episodic) must NOT collapse, so the key includes the namespace.
    fused: dict[tuple, tuple[float, int, "StoreRecord"]] = {}

    for rank, scored in enumerate(lexical_ranked, start=1):
        record = scored.record
        identity = (record.namespace, record.key)
        contribution = 1.0 / (k + rank)
        existing = fused.get(identity)
        if existing is None:
            fused[identity] = (contribution, scored.insertion_index, record)
        else:
            prev_score, prev_idx, prev_record = existing
            # Keep the earlier insertion_index so the tiebreaker
            # stays consistent across fusion inputs.
            fused[identity] = (
                prev_score + contribution,
                min(prev_idx, scored.insertion_index),
                prev_record,
            )

    for rank, scored in enumerate(dense_ranked, start=1):
        record = scored.record
        identity = (record.namespace, record.key)
        contribution = 1.0 / (k + rank)
        existing = fused.get(identity)
        if existing is None:
            fused[identity] = (contribution, scored.insertion_index, record)
        else:
            prev_score, prev_idx, prev_record = existing
            fused[identity] = (
                prev_score + contribution,
                min(prev_idx, scored.insertion_index),
                prev_record,
            )

    # Sort by fused score desc, then insertion_index asc for stable
    # ties. Slice to limit. Return plain StoreRecords — the caller
    # doesn't need to see the fused scores.
    ordered = sorted(
        fused.values(),
        key=lambda item: (-item[0], item[1]),
    )
    return [record for _, _, record in ordered[:limit]]
