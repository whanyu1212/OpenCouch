"""Unit tests for the surviving hard-guard / memory-control write-policy helpers.

The per-candidate LLM write-policy classifiers (decide_*_candidate_llm_primary)
were removed when session-end consolidation moved to a whole-transcript extractor;
their tests went with them. What remains here covers the helpers that are still
live in production: the semantic hard-policy guard and memory-control detection.
"""

from __future__ import annotations

from agent.memory.policy.candidates import build_semantic_candidate
from agent.memory.policy.write import (
    semantic_hard_policy_guard,
    text_contains_memory_control_request,
)
from agent.memory.types import EntityRef, MemoryWrite


def _semantic_write(
    *,
    category: str,
    predicate: str = "WORRIES_ABOUT",
    object_type: str = "Concern",
    object_identifier: str = "work stress",
    evidence_quote: str,
) -> MemoryWrite:
    return MemoryWrite(
        category=category,  # type: ignore[arg-type]
        subject=EntityRef(type="User", identifier="user-1"),
        predicate=predicate,  # type: ignore[arg-type]
        object=EntityRef(type=object_type, identifier=object_identifier),  # type: ignore[arg-type]
        evidence_quote=evidence_quote,
        confidence="high",
        source_session_id="session-1",
        source_turn_index=2,
    )


def test_explicit_stable_semantic_fact_has_no_hard_guard() -> None:
    candidate = build_semantic_candidate(
        _semantic_write(
            category="relationship",
            predicate="KNOWS",
            object_type="Person",
            object_identifier="Sarah",
            evidence_quote="My sister Sarah lives nearby.",
        ),
        message="My sister Sarah lives nearby.",
    )

    assert semantic_hard_policy_guard(candidate) is None


def test_provenance_semantic_predicate_drops() -> None:
    candidate = build_semantic_candidate(
        _semantic_write(
            category="context",
            predicate="MENTIONED_IN",
            object_type="Person",
            object_identifier="Sarah",
            evidence_quote="My sister Sarah lives nearby.",
        ),
        message="My sister Sarah lives nearby.",
    )

    decision = semantic_hard_policy_guard(candidate)

    assert decision is not None
    assert decision.action == "drop"


def test_memory_control_text_requires_assistant_directed_intent() -> None:
    assert text_contains_memory_control_request("Please don't save this.")
    assert text_contains_memory_control_request(
        "Actually, do not save or remember any of that after this conversation."
    )
    assert text_contains_memory_control_request("Forget this after today.")
    assert text_contains_memory_control_request("Use incognito mode for this.")

    assert not text_contains_memory_control_request("I can't forget that argument.")
    assert not text_contains_memory_control_request("I don't remember details.")
    assert not text_contains_memory_control_request("I forgot this happened before.")
    assert not text_contains_memory_control_request("I want to remember this feeling.")


def test_memory_control_semantic_candidate_drops() -> None:
    candidate = build_semantic_candidate(
        _semantic_write(
            category="preference",
            predicate="WANTS",
            object_type="Concern",
            object_identifier="do-not-save-or-remember-that-after-this-conversation",
            evidence_quote="do not save or remember any of that after this conversation",
        ),
        message="Actually, do not save or remember any of that after this conversation.",
    )

    decision = semantic_hard_policy_guard(candidate)

    assert decision is not None
    assert decision.action == "drop"
    assert (
        decision.reason == "memory-control requests should not become semantic memory"
    )
