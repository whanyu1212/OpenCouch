"""Hot-path deduplication for semantic fact writes.

When the extraction node produces a new :class:`MemoryWrite` candidate,
we check whether a near-duplicate fact already exists for this user
before writing. If a duplicate is found, the caller bumps the existing
record's ``last_referenced_at`` instead of inserting a new row. This
prevents the semantic namespace from filling up with restatements of
the same fact across multiple turns.

Design:
- **Comparison is field-aware.** Two facts are duplicates only when
  their structural fields match (same subject, predicate, object) AND
  their evidence quotes are highly similar. This matches the schema's
  intent that a semantic fact is a ``(user, predicate, target)`` triple
  backed by direct user quotes — both the triple and the evidence have
  to align for a collision to count.
- **Similarity is token-set Jaccard on the evidence quote.** Strict
  word-overlap matching. Two paraphrases with different word choices
  are not considered duplicates under Jaccard. This is conservative by
  design: false-negative duplicates produce a redundant-looking
  ``/memory list`` entry, while false-positive merges lose signal
  permanently.
- **Threshold is 0.85.** Higher than the 0.95 target we'd use for
  vector similarity, because Jaccard is stricter: 0.85
  Jaccard roughly corresponds to "almost all the same words, maybe
  one or two different." 0.95 would require near-identical quotes,
  which is rarely achievable across two distinct turns.

The public interface (``find_near_duplicate``) stays narrow so callers
do not need to know which duplicate-detection heuristic is active.
"""

from __future__ import annotations

from agent.memory.models import MemoryWrite
from agent.memory.store import StoreRecord
from agent.memory.text_tokens import tokenize as _tokenize

# Jaccard similarity threshold for considering two evidence quotes a
# duplicate. See module docstring for rationale on the value.
JACCARD_DUPLICATE_THRESHOLD = 0.85

# ``_tokenize`` is imported from ``agent.memory.text_tokens`` and kept
# under this private name for compatibility with tests that import it
# from this module. Dedup uses the **full** token set — no stopword
# filtering — because even pronouns and articles matter when deciding
# whether two evidence quotes describe the same fact ("I feel sad" vs
# "she feels sad" must remain distinct, so dropping pronouns would
# collapse them).


def _jaccard_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    """Compute Jaccard similarity for two token sets.

    Args:
        a (frozenset[str]): First token set.
        b (frozenset[str]): Second token set.

    Returns:
        float: Jaccard similarity in ``[0.0, 1.0]``.
    """

    if not a and not b:
        return 0.0
    intersection = a & b
    union = a | b
    return len(intersection) / len(union)


def _triples_match(
    candidate: MemoryWrite,
    existing: StoreRecord,
) -> bool:
    """Return whether two facts have matching semantic triples.

    Args:
        candidate (MemoryWrite): Candidate fact to compare.
        existing (StoreRecord): Existing stored fact record.

    Returns:
        bool: ``True`` when subject, predicate, and object all align.
    """

    value = existing.value
    candidate_subject = (candidate.subject.type, candidate.subject.identifier)
    candidate_object = (candidate.object.type, candidate.object.identifier)

    existing_subject = (
        value.get("subject", {}).get("type"),
        value.get("subject", {}).get("identifier"),
    )
    existing_object = (
        value.get("object", {}).get("type"),
        value.get("object", {}).get("identifier"),
    )
    existing_predicate = value.get("predicate")

    return (
        candidate_subject == existing_subject
        and candidate_object == existing_object
        and candidate.predicate == existing_predicate
    )


def find_near_duplicate(
    candidate: MemoryWrite,
    existing_records: list[StoreRecord],
    *,
    threshold: float = JACCARD_DUPLICATE_THRESHOLD,
) -> StoreRecord | None:
    """Return the first existing record that's a near-duplicate of the candidate.

    A record is a duplicate when BOTH conditions are true:

    1. Its (subject, predicate, object) triple matches the candidate's.
    2. Its ``evidence_quote`` has Jaccard token-set similarity ≥ threshold
       to the candidate's evidence quote.

    Returns ``None`` when no duplicate is found. The caller should write
    the candidate as a new record in that case. When a duplicate IS
    found, the caller should bump the matching record's
    ``last_referenced_at`` timestamp instead of writing the new one.

    Performance note: this function iterates the full ``existing_records``
    list. That is acceptable for the current expected scale of hundreds
    of records per user. A store-specific implementation can push the
    triple match into a WHERE clause if this path becomes hot.

    Args:
        candidate: The new fact the extractor wants to write.
        existing_records: All records currently in the user's semantic
            namespace. Typically obtained via
            ``store.asearch(namespace, query=None, limit=<large>)``.
        threshold: Minimum Jaccard similarity to count as a duplicate.
            Defaults to :data:`JACCARD_DUPLICATE_THRESHOLD` (0.85).
            Higher values are stricter (fewer merges); lower values are
            more aggressive (more merges).

    Returns:
        The first matching ``StoreRecord``, or ``None`` if no duplicate
        exists. "First" means the earliest in list order — existing
        records are typically stored in insertion order, so the returned
        match is the oldest duplicate.
    """

    candidate_tokens = _tokenize(candidate.evidence_quote)

    for record in existing_records:
        if not _triples_match(candidate, record):
            continue
        existing_quote = record.value.get("evidence_quote", "")
        existing_tokens = _tokenize(existing_quote)
        similarity = _jaccard_similarity(candidate_tokens, existing_tokens)
        if similarity >= threshold:
            return record

    return None
