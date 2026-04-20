"""Unit tests for the upgraded session trajectory eval memory helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from agent.memory.models import (
    EntityRef,
    MoodArc,
    ProceduralRule,
    SemanticFact,
    StoredSessionArc,
)
from agent.memory.procedural import aadd_procedural_rule
from agent.memory.store import OpenCouchMemoryStore

RUNNER_PATH = (
    Path(__file__).resolve().parents[3]
    / "eval"
    / "runners"
    / "session_trajectory_eval.py"
)
_SPEC = importlib.util.spec_from_file_location("session_trajectory_eval", RUNNER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("session_trajectory_eval", _MODULE)
_SPEC.loader.exec_module(_MODULE)

_snapshot_memory_state = _MODULE._snapshot_memory_state
_diff_memory_state = _MODULE._diff_memory_state
_check_turn_expectation = _MODULE._check_turn_expectation
_check_final_expectation = _MODULE._check_final_expectation


@pytest.mark.asyncio
async def test_snapshot_memory_state_reads_actual_store_contents() -> None:
    store = OpenCouchMemoryStore()
    owner_id = "eval-user"

    fact = SemanticFact(
        id="fact-1",
        category="relationship",
        subject=EntityRef(type="User", identifier=owner_id),
        predicate="KNOWS",
        object=EntityRef(type="Person", identifier="Sarah"),
        evidence_quote="My sister Sarah lives nearby.",
        confidence="high",
        source_session_id="session-1",
        source_turn_index=0,
        created_at="2026-04-19T10:00:00Z",
        last_referenced_at="2026-04-19T10:00:00Z",
        dormant_at=None,
        superseded_by=None,
        user_visible=True,
    )
    await store.aput(
        (owner_id, "semantic"),
        fact.id,
        fact.model_dump(mode="json"),
    )

    await aadd_procedural_rule(
        store,
        user_id=owner_id,
        rule=ProceduralRule(
            rule="You prefer shorter responses.",
            evidence=["Please keep responses shorter."],
            confidence="high",
            added_at="2026-04-19T10:01:00Z",
            source="explicit_user",
        ),
    )

    arc = StoredSessionArc(
        id="arc-1",
        owner_id=owner_id,
        session_id="session-1",
        started_at="2026-04-19T10:00:00Z",
        ended_at="2026-04-19T10:10:00Z",
        duration_seconds=600,
        turn_count=4,
        primary_themes=["work"],
        summary="Talked about work stress and wanting one practical takeaway.",
        mood_arc=MoodArc(opened="overwhelmed", closed="steadier"),
        open_loops=["Notice the shame spiral earlier."],
        resolved_threads=[],
        created_at="2026-04-19T10:10:00Z",
        last_referenced_at="2026-04-19T10:10:00Z",
        user_visible=True,
        crisis_level_max=0,
    )
    await store.aput(
        (owner_id, "episodic"),
        arc.id,
        arc.model_dump(mode="json"),
    )

    snapshot = await _snapshot_memory_state(store, owner_id=owner_id)

    assert len(snapshot["semantic_facts"]) == 1
    assert snapshot["semantic_facts"][0]["object_identifier"] == "Sarah"
    assert len(snapshot["procedural_rules"]) == 1
    assert snapshot["procedural_rules"][0]["rule"] == "You prefer shorter responses."
    assert len(snapshot["episodic_arcs"]) == 1
    assert "work stress" in snapshot["episodic_arcs"][0]["summary"]


def test_diff_memory_state_reports_actual_namespace_deltas() -> None:
    before = {
        "semantic_facts": [],
        "procedural_rules": [],
        "episodic_arcs": [],
    }
    after = {
        "semantic_facts": [
            {
                "id": "fact-1",
                "category": "semantic",
                "semantic_category": "relationship",
                "object_identifier": "Sarah",
                "evidence_quote": "My sister Sarah lives nearby.",
            }
        ],
        "procedural_rules": [
            {
                "category": "procedural",
                "rule": "You prefer shorter responses.",
                "added_at": "2026-04-19T10:01:00Z",
                "source": "explicit_user",
            }
        ],
        "episodic_arcs": [
            {
                "id": "arc-1",
                "category": "episodic",
                "summary": "Talked about work stress.",
            }
        ],
    }

    delta = _diff_memory_state(before, after)

    assert delta["semantic_fact_count_delta"] == 1
    assert delta["procedural_rule_count_delta"] == 1
    assert delta["episodic_arc_count_delta"] == 1
    assert [write["category"] for write in delta["memory_writes"]] == [
        "semantic",
        "procedural",
        "episodic",
    ]


def test_turn_expectation_grades_actual_memory_fields() -> None:
    record = {
        "mode": "supportive",
        "crisis_level": 0,
        "needs_crisis_response": False,
        "needs_clarification": False,
        "session_intent": None,
        "session_intent_source": None,
        "session_stage": None,
        "modality": None,
        "response_type": "therapeutic",
        "response_guidance": None,
        "assistant_text": "Okay.",
        "exercise_type": None,
        "exercise_step": None,
        "exercise_active": False,
        "memory_writes": [
            {
                "category": "semantic",
                "object_identifier": "Sarah",
                "evidence_quote": "My sister Sarah lives nearby.",
            },
            {
                "category": "procedural",
                "rule": "You prefer shorter responses.",
            },
        ],
        "semantic_fact_count_delta": 1,
        "procedural_rule_count_delta": 1,
        "episodic_arc_count_delta": 0,
        "semantic_facts_total": 1,
        "procedural_rules_total": 1,
        "episodic_arcs_total": 0,
    }

    failures = _check_turn_expectation(
        "case-1",
        1,
        record,
        {
            "memory_write_expected": True,
            "memory_write_category_in": ["semantic", "procedural"],
            "semantic_fact_count_delta": 1,
            "procedural_rule_count_delta": 1,
            "semantic_fact_object_contains_any": ["Sarah"],
            "procedural_rule_contains_any": ["shorter responses"],
        },
    )

    assert failures == []


def test_final_expectation_grades_session_end_and_totals() -> None:
    record = {
        "summary_text": "Talked about work stress and shame.",
        "summary_expected": True,
        "end_session_run": True,
        "open_loops": ["Notice the spiral earlier."],
        "resolved_threads": [],
        "exercise_active": False,
        "exercise_type": None,
        "memory_writes_total": 2,
        "session_end_memory_writes": [
            {"category": "episodic", "summary": "Talked about work stress and shame."}
        ],
        "session_end_semantic_fact_count_delta": 0,
        "session_end_procedural_rule_count_delta": 0,
        "session_end_episodic_arc_count_delta": 1,
        "semantic_facts_total": 1,
        "procedural_rules_total": 1,
        "episodic_arcs_total": 1,
        "semantic_facts": [{"object_identifier": "Sarah"}],
        "procedural_rules": [{"rule": "You prefer shorter responses."}],
        "session_intent": None,
        "session_stage": None,
        "mode": "supportive",
        "modality": None,
        "response_guidance": None,
        "assistant_text": "Here is the main takeaway.",
        "needs_clarification": False,
        "needs_crisis_response": False,
    }

    failures = _check_final_expectation(
        "case-final",
        record,
        {
            "summary_expected": True,
            "session_end_memory_write_expected": True,
            "session_end_memory_write_category_in": ["episodic"],
            "session_end_episodic_arc_count_delta": 1,
            "semantic_facts_total": 1,
            "procedural_rules_total": 1,
            "episodic_arcs_total": 1,
            "semantic_fact_object_contains_any": ["Sarah"],
            "procedural_rule_contains_any": ["shorter responses"],
        },
    )

    assert failures == []
