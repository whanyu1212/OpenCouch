"""Tests for embedding-cosine clustering in session-end consolidation.

The default (no embeddings) lexical path is covered by the existing
test_commit_session_memory suite, which stays green unchanged. These tests pin
the NEW cosine path: when per-candidate vectors are supplied, paraphrases group
by cosine and distinct facts stay separate, with deterministic hand-chosen
vectors (real-embedding quality is the live eval's job, not a unit test's).
"""

from __future__ import annotations

from agent.memory.commit.clustering import (
    _CLUSTER_COSINE_THRESHOLD,
    _cluster_procedural_candidates,
    _cluster_semantic_candidates,
)
from agent.memory.policy.candidates import (
    BufferedProceduralCandidate,
    BufferedSemanticCandidate,
    ProceduralCandidate,
    SemanticCandidate,
)
from agent.memory.types.procedural import ProceduralRuleDraft
from agent.memory.types.semantic import MemoryWrite


def _semantic(
    quote: str, *, obj: str = "presentations", turn: int = 0
) -> BufferedSemanticCandidate:
    write = MemoryWrite(
        category="trigger",
        subject={"type": "User", "identifier": "user-1"},
        predicate="WORRIES_ABOUT",
        object={"type": "Concern", "identifier": obj},
        evidence_quote=quote,
        confidence="high",
        source_session_id="s",
        source_turn_index=turn,
    )
    return BufferedSemanticCandidate(
        candidate=SemanticCandidate(
            evidence_quotes=[quote],
            source_session_id="s",
            source_turn_index=turn,
            payload=write,
        ),
        hold_action="commit_at_session_end",
        policy_reason="test",
        policy_version="test",
    )


def _procedural(rule: str) -> BufferedProceduralCandidate:
    return BufferedProceduralCandidate(
        candidate=ProceduralCandidate(
            evidence_quotes=[rule],
            source_session_id="s",
            source_turn_index=0,
            payload=ProceduralRuleDraft(rule=rule, evidence=[rule], confidence="high"),
        ),
        hold_action="commit_at_session_end",
        policy_reason="test",
        policy_version="test",
    )


# Hand-chosen vectors: v_a and v_a2 are nearly identical (cosine ~1.0, > threshold
# -> merge); v_b is orthogonal (cosine 0 -> separate).
_V_A = [1.0, 0.0, 0.0]
_V_A2 = [0.99, 0.14, 0.0]  # cosine with _V_A ~= 0.99
_V_B = [0.0, 1.0, 0.0]  # cosine with _V_A = 0.0


def test_cosine_merges_paraphrases_that_lexical_would_miss() -> None:
    # Two paraphrases with NO shared meaningful tokens — lexical clustering would
    # keep them apart, but their embeddings are near-identical -> one cluster.
    a = _semantic("Big talks make me panic", obj="presentations")
    a2 = _semantic("Speaking to a crowd terrifies me", obj="presentations")
    groups = _cluster_semantic_candidates([a, a2], embeddings=[_V_A, _V_A2])
    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_cosine_keeps_distinct_facts_separate() -> None:
    a = _semantic("Big talks make me panic", obj="presentations")
    b = _semantic("My grandmother passed away", obj="grief")
    groups = _cluster_semantic_candidates([a, b], embeddings=[_V_A, _V_B])
    assert len(groups) == 2


def test_object_anchor_guard_blocks_merge_even_when_cosine_high() -> None:
    # Same near-identical vectors, but DIFFERENT objects -> the anchor guard must
    # keep them apart (a fact about presentations is not a fact about grief, even
    # if phrased similarly).
    a = _semantic("It really gets to me", obj="presentations")
    b = _semantic("It really gets to me", obj="grief")
    groups = _cluster_semantic_candidates([a, b], embeddings=[_V_A, _V_A2])
    assert len(groups) == 2


def test_procedural_cosine_merges_paraphrases() -> None:
    a = _procedural("Keep replies short")
    a2 = _procedural("Be concise and brief")
    groups = _cluster_procedural_candidates([a, a2], embeddings=[_V_A, _V_A2])
    assert len(groups) == 1


def test_threshold_is_respected() -> None:
    # A vector just BELOW the cosine threshold must NOT merge. Distinct objects so
    # the same-object group-key path doesn't merge them independently of cosine —
    # this isolates the threshold check itself.
    import math

    angle = math.acos(_CLUSTER_COSINE_THRESHOLD) + 0.05  # past the threshold angle
    below = [math.cos(angle), math.sin(angle), 0.0]  # cosine with _V_A < threshold
    a = _semantic("x", obj="presentations")
    b = _semantic("y", obj="grief")
    groups = _cluster_semantic_candidates([a, b], embeddings=[_V_A, below])
    assert len(groups) == 2
