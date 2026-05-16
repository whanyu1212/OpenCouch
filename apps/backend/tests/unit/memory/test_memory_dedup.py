"""Unit tests for the semantic extraction dedup helper.

Covers the pure helpers (``_tokenize``, ``_jaccard_similarity``,
``_triples_match``) and the public ``find_near_duplicate`` function
with a range of scenarios: no duplicates, strict matches, paraphrase
edge cases around the Jaccard threshold, structural (triple) mismatch,
and threshold tuning.

All tests are pure — no LLM client, no store instance, no async. The
dedup helper is deliberately a pure function so it can be unit-tested
in isolation before Stage C wires it into the extraction node.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent.memory.dedup import (
    JACCARD_DUPLICATE_THRESHOLD,
    _jaccard_similarity,
    _tokenize,
    _triples_match,
    find_near_duplicate,
)
from agent.memory.models import EntityRef, MemoryWrite
from agent.memory.store import StoreRecord


# ─── Test fixtures ──────────────────────────────────────────────────────


def _make_memory_write(
    *,
    evidence_quote: str = "my sister Sarah came over last night",
    subject_identifier: str = "user-1",
    object_identifier: str = "Sarah",
    predicate: str = "KNOWS",
    category: str = "relationship",
) -> MemoryWrite:
    """Build a MemoryWrite with sensible defaults for testing."""

    return MemoryWrite(
        category=category,  # type: ignore[arg-type]
        subject=EntityRef(type="User", identifier=subject_identifier),
        predicate=predicate,  # type: ignore[arg-type]
        object=EntityRef(type="Person", identifier=object_identifier),
        evidence_quote=evidence_quote,
        confidence="high",
        source_session_id="thread-test",
        source_turn_index=0,
    )


def _make_store_record(
    *,
    evidence_quote: str = "my sister Sarah came over last night",
    subject_identifier: str = "user-1",
    object_identifier: str = "Sarah",
    predicate: str = "KNOWS",
    key: str = "fact-1",
    subject_type: str = "User",
    object_type: str = "Person",
) -> StoreRecord:
    """Build a StoreRecord whose value dict mimics a serialized SemanticFact."""

    value: dict[str, Any] = {
        "id": key,
        "category": "relationship",
        "subject": {"type": subject_type, "identifier": subject_identifier},
        "predicate": predicate,
        "object": {"type": object_type, "identifier": object_identifier},
        "evidence_quote": evidence_quote,
        "confidence": "high",
        "source_session_id": "thread-test",
        "source_turn_index": 0,
        "created_at": "2026-04-10T12:00:00Z",
        "last_referenced_at": "2026-04-10T12:00:00Z",
        "dormant_at": None,
        "superseded_by": None,
        "user_visible": True,
    }
    return StoreRecord(
        namespace=("user-1", "semantic"),
        key=key,
        value=value,
    )


# ─── _tokenize tests ────────────────────────────────────────────────────


class TestTokenize:
    """Unit tests for the token-extraction helper."""

    def test_lowercases_and_splits_on_word_boundaries(self) -> None:
        assert _tokenize("Hello World") == frozenset({"hello", "world"})

    def test_ignores_punctuation(self) -> None:
        assert _tokenize("hello, world!") == frozenset({"hello", "world"})

    def test_apostrophes_split_contractions(self) -> None:
        # "i'm" produces ["i", "m"] — that's how `\b[a-z0-9]+\b` works.
        # Stable, if not semantically ideal.
        tokens = _tokenize("I'm anxious")
        assert "anxious" in tokens
        # Accept either tokenization — the important thing is that
        # "I'm anxious" and "im anxious" produce comparable token sets.
        assert len(tokens) >= 2

    def test_numbers_are_tokens(self) -> None:
        assert _tokenize("I had 3 cups of coffee") == frozenset(
            {"i", "had", "3", "cups", "of", "coffee"}
        )

    def test_empty_string_returns_empty_set(self) -> None:
        assert _tokenize("") == frozenset()

    def test_repeated_words_deduplicate_via_set(self) -> None:
        # "work work work" → set has one "work" element
        assert _tokenize("work work work") == frozenset({"work"})


# ─── _jaccard_similarity tests ──────────────────────────────────────────


class TestJaccardSimilarity:
    """Unit tests for the Jaccard similarity helper."""

    def test_identical_sets_return_one(self) -> None:
        s = frozenset({"a", "b", "c"})
        assert _jaccard_similarity(s, s) == 1.0

    def test_disjoint_sets_return_zero(self) -> None:
        a = frozenset({"a", "b"})
        b = frozenset({"c", "d"})
        assert _jaccard_similarity(a, b) == 0.0

    def test_partial_overlap(self) -> None:
        # Intersection: {a, b}. Union: {a, b, c, d}. 2/4 = 0.5
        a = frozenset({"a", "b", "c"})
        b = frozenset({"a", "b", "d"})
        assert _jaccard_similarity(a, b) == 0.5

    def test_empty_both_returns_zero_not_error(self) -> None:
        # Guards against division by zero.
        assert _jaccard_similarity(frozenset(), frozenset()) == 0.0

    def test_one_empty_one_populated_returns_zero(self) -> None:
        assert _jaccard_similarity(frozenset(), frozenset({"a"})) == 0.0

    def test_near_identity(self) -> None:
        # 7 overlapping, 1 extra: 7/8 = 0.875
        a = frozenset({"my", "sister", "sarah", "came", "over", "last", "night"})
        b = frozenset(
            {"my", "sister", "sarah", "came", "over", "last", "night", "again"}
        )
        assert _jaccard_similarity(a, b) == pytest.approx(7 / 8)


# ─── _triples_match tests ───────────────────────────────────────────────


class TestTriplesMatch:
    """Unit tests for the (subject, predicate, object) triple comparator."""

    def test_identical_triples_match(self) -> None:
        candidate = _make_memory_write()
        existing = _make_store_record()
        assert _triples_match(candidate, existing) is True

    def test_different_predicate_mismatches(self) -> None:
        candidate = _make_memory_write(predicate="WORRIES_ABOUT")
        existing = _make_store_record(predicate="KNOWS")
        assert _triples_match(candidate, existing) is False

    def test_different_subject_identifier_mismatches(self) -> None:
        candidate = _make_memory_write(subject_identifier="user-1")
        existing = _make_store_record(subject_identifier="user-2")
        assert _triples_match(candidate, existing) is False

    def test_different_object_identifier_mismatches(self) -> None:
        candidate = _make_memory_write(object_identifier="Sarah")
        existing = _make_store_record(object_identifier="Emma")
        assert _triples_match(candidate, existing) is False

    def test_different_object_type_mismatches(self) -> None:
        candidate = _make_memory_write(object_identifier="work stress")
        existing = _make_store_record(
            object_identifier="work stress",
            object_type="Concern",
        )
        # candidate defaults to object type Person; existing is Concern.
        # Even though identifiers match, type differences matter.
        assert _triples_match(candidate, existing) is False

    def test_missing_subject_in_existing_record_mismatches_gracefully(self) -> None:
        """Malformed records should mismatch, not crash."""
        record = _make_store_record()
        # Corrupt the value to simulate a malformed stored record.
        record.value.pop("subject")
        candidate = _make_memory_write()
        assert _triples_match(candidate, record) is False


# ─── find_near_duplicate integration tests ──────────────────────────────


class TestFindNearDuplicate:
    """Integration tests for the public dedup entry point."""

    def test_empty_records_returns_none(self) -> None:
        candidate = _make_memory_write()
        assert find_near_duplicate(candidate, []) is None

    def test_no_triple_match_returns_none(self) -> None:
        """Even identical quotes don't dedup if triples differ."""
        candidate = _make_memory_write(
            evidence_quote="my sister Sarah came over",
            predicate="KNOWS",
        )
        existing = [
            _make_store_record(
                evidence_quote="my sister Sarah came over",
                predicate="WORRIES_ABOUT",
            )
        ]
        assert find_near_duplicate(candidate, existing) is None

    def test_exact_match_is_duplicate(self) -> None:
        """Identical candidate + existing → duplicate detected."""
        candidate = _make_memory_write()
        existing = [_make_store_record()]
        result = find_near_duplicate(candidate, existing)
        assert result is not None
        assert result.key == "fact-1"

    def test_near_paraphrase_above_threshold_is_duplicate(self) -> None:
        """7/8 = 0.875 Jaccard > 0.85 threshold → duplicate."""
        candidate = _make_memory_write(
            evidence_quote="my sister Sarah came over last night"
        )
        existing = [
            _make_store_record(
                evidence_quote="my sister Sarah came over last night again"
            )
        ]
        result = find_near_duplicate(candidate, existing)
        assert result is not None

    def test_paraphrase_below_threshold_is_not_duplicate(self) -> None:
        """Different word choice → Jaccard below threshold → keep both."""
        candidate = _make_memory_write(
            evidence_quote="my sister Sarah came over last night"
        )
        existing = [
            _make_store_record(evidence_quote="my sister Sarah visited yesterday")
        ]
        # Tokens: {my, sister, sarah, came, over, last, night} vs
        #         {my, sister, sarah, visited, yesterday}
        # Intersection: {my, sister, sarah} (3). Union: 9. 3/9 = 0.33 < 0.85
        assert find_near_duplicate(candidate, existing) is None

    def test_returns_first_matching_record_in_list_order(self) -> None:
        """When multiple duplicates exist, the earliest one is returned."""
        candidate = _make_memory_write()
        existing = [
            _make_store_record(key="fact-a"),
            _make_store_record(key="fact-b"),
            _make_store_record(key="fact-c"),
        ]
        result = find_near_duplicate(candidate, existing)
        assert result is not None
        assert result.key == "fact-a"

    def test_triple_match_only_scanned_for_jaccard(self) -> None:
        """Non-matching triples are skipped before Jaccard is computed."""
        candidate = _make_memory_write(evidence_quote="my sister Sarah came over")
        existing = [
            # Wrong predicate — should be skipped even though quote is identical
            _make_store_record(
                evidence_quote="my sister Sarah came over",
                predicate="WORRIES_ABOUT",
                key="fact-wrong",
            ),
            # Right predicate, right quote
            _make_store_record(
                evidence_quote="my sister Sarah came over",
                predicate="KNOWS",
                key="fact-right",
            ),
        ]
        result = find_near_duplicate(candidate, existing)
        assert result is not None
        assert result.key == "fact-right"

    def test_custom_threshold_tuning(self) -> None:
        """Lowering the threshold catches more paraphrases."""
        candidate = _make_memory_write(evidence_quote="my sister Sarah visited me")
        existing = [_make_store_record(evidence_quote="my sister Sarah came over")]
        # Tokens: {my, sister, sarah, visited, me} vs
        #         {my, sister, sarah, came, over}
        # Intersection: {my, sister, sarah} (3). Union: 7. 3/7 ≈ 0.43
        # Default threshold 0.85 → no match
        assert find_near_duplicate(candidate, existing) is None
        # Lowered threshold 0.4 → match
        result = find_near_duplicate(candidate, existing, threshold=0.4)
        assert result is not None


# ─── Threshold constant sanity check ────────────────────────────────────


def test_jaccard_duplicate_threshold_is_sane() -> None:
    """The default threshold should be in a reasonable range for token Jaccard.

    This is a regression guard: if a future tuning change accidentally
    sets the threshold too high (≥ 0.99, so nothing dedups) or too low
    (< 0.5, so unrelated facts merge), the test catches it.
    """

    assert 0.5 <= JACCARD_DUPLICATE_THRESHOLD <= 0.95
