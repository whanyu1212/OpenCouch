"""Unit tests for the upgraded session trajectory eval memory helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.memory.models import (
    EntityRef,
    MoodArc,
    ProceduralRule,
    SemanticFact,
    StoredSessionArc,
)
from agent.memory.procedural import aadd_procedural_rule, aget_procedural_profile
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
_normalize_turn_record = _MODULE._normalize_turn_record
_check_turn_expectation = _MODULE._check_turn_expectation
_check_final_expectation = _MODULE._check_final_expectation
_seed_case_memory = _MODULE._seed_case_memory


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


def test_normalize_turn_record_sets_mode_and_modality_aliases() -> None:
    result = SimpleNamespace(
        output=SimpleNamespace(
            response_text="Let's stay with one thing at a time.",
            response_style="supportive",
            response_type=SimpleNamespace(value="therapeutic"),
            crisis=SimpleNamespace(
                level=0,
                needs_crisis_response=False,
                needs_clarification=False,
            ),
            therapeutic_approach="act",
            response_style_type=None,
            response_style_source="therapeutic",
            diagnostics={},
        ),
        state={
            "session_progress": {
                "intent": "support",
                "intent_source": "dispatcher",
                "stage": "exploration",
            },
            "procedural_profile": {
                "proactive_recall_enabled": True,
            },
            "working_memory": [
                {
                    "type": "semantic",
                    "evidence_quote": "Family conflict is a big trigger for panic.",
                    "object": "panic",
                },
                {
                    "type": "episodic",
                    "summary": "Last session focused on work anxiety.",
                    "primary_themes": ["work"],
                    "is_catch_up": True,
                },
            ],
            "therapeutic_approach": "act",
        },
    )

    record = _normalize_turn_record(
        0,
        "I'm overwhelmed.",
        result,
        memory_snapshot={
            "semantic_facts": [],
            "procedural_rules": [],
            "episodic_arcs": [],
        },
        memory_delta={
            "memory_writes": [],
            "semantic_fact_count_delta": 0,
            "procedural_rule_count_delta": 0,
            "episodic_arc_count_delta": 0,
        },
    )

    assert record["mode"] == "supportive"
    assert record["response_style"] == "supportive"
    assert record["modality"] == "act"
    assert record["therapeutic_approach"] == "act"
    assert record["working_memory_types"] == ["semantic", "episodic"]
    assert record["working_memory_objects"] == ["panic"]
    assert record["working_memory_evidence_quotes"] == [
        "Family conflict is a big trigger for panic."
    ]
    assert record["working_memory_summaries"] == [
        "Last session focused on work anxiety."
    ]
    assert record["proactive_recall_enabled"] is True


def test_turn_expectation_grades_actual_memory_fields() -> None:
    record = {
        "response_style": "supportive",
        "crisis_level": 0,
        "needs_crisis_response": False,
        "needs_clarification": False,
        "session_intent": None,
        "session_intent_source": None,
        "session_stage": None,
        "therapeutic_approach": None,
        "response_type": "therapeutic",
        "assistant_text": "Okay.",
        "exercise_type": None,
        "exercise_step": None,
        "exercise_active": False,
        "working_memory_types": ["episodic", "semantic"],
        "working_memory_objects": ["panic"],
        "working_memory_evidence_quotes": [
            "Family conflict is a big trigger for panic."
        ],
        "working_memory_summaries": ["Last session focused on work anxiety."],
        "proactive_recall_enabled": True,
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
            "working_memory_types_contains_any": ["episodic"],
            "working_memory_types_not_contains_any": ["procedural"],
            "working_memory_object_contains_any": ["panic"],
            "working_memory_evidence_contains_any": ["family conflict"],
            "working_memory_summary_contains_any": ["work anxiety"],
            "proactive_recall_enabled": True,
            "semantic_fact_object_contains_any": ["Sarah"],
            "procedural_rule_contains_any": ["shorter responses"],
        },
    )

    assert failures == []


@pytest.mark.asyncio
async def test_seed_case_memory_populates_store_and_recall_toggle() -> None:
    store = OpenCouchMemoryStore()
    owner_id = "eval-user"

    await _seed_case_memory(
        store,
        owner_id=owner_id,
        thread_id="thread-seed",
        seed_memory={
            "proactive_recall_enabled": True,
            "procedural_rules": [
                {
                    "rule": "Keep responses shorter.",
                    "evidence": ["Please keep it short."],
                }
            ],
            "semantic_facts": [
                {
                    "category": "relationship",
                    "predicate": "KNOWS",
                    "object_type": "Person",
                    "object_identifier": "Sarah",
                    "evidence_quote": "My sister Sarah lives nearby.",
                }
            ],
            "episodic_arcs": [
                {
                    "summary": "User talked about presentation anxiety and steadied with concrete facts.",
                    "primary_themes": ["work", "anxiety"],
                    "approach_used": "cbt",
                }
            ],
        },
    )

    procedural_profile = await aget_procedural_profile(store, user_id=owner_id)
    assert procedural_profile.proactive_recall_enabled is True
    assert len(procedural_profile.rules) == 1
    assert procedural_profile.rules[0].rule == "Keep responses shorter."

    semantic_records = await store.asearch((owner_id, "semantic"), query=None, limit=10)
    episodic_records = await store.asearch((owner_id, "episodic"), query=None, limit=10)

    assert len(semantic_records) == 1
    assert semantic_records[0].value["object"]["identifier"] == "Sarah"
    assert len(episodic_records) == 1
    assert "presentation anxiety" in episodic_records[0].value["summary"]


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
        "response_style": "supportive",
        "therapeutic_approach": None,
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
