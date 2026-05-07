"""Unit tests for the phase-D reconciliation helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from agent.memory.models import EntityRef, SemanticFact
from agent.memory.procedural_profile import build_procedural_rule
from agent.memory.reconciliation import (
    filter_active_semantic_records,
    is_active_semantic_record_value,
    plan_procedural_rule_write_llm_primary,
    plan_procedural_rule_write,
    plan_semantic_write_llm_primary,
    plan_semantic_write,
)
from agent.memory.store import StoreRecord
from llm.base import BaseLLMClient, StructuredResponseT


class _FakeReconciliationLLM(BaseLLMClient):
    """Fake structured client for reconciliation classifier tests."""

    def __init__(self, decision: dict[str, Any]) -> None:
        self.decision = decision
        self.structured_calls = 0

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        return "unused"

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        yield "unused"

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
    ) -> StructuredResponseT:
        self.structured_calls += 1
        return cast(StructuredResponseT, response_schema(**self.decision))


def _semantic_fact(
    *,
    fact_id: str,
    object_identifier: str,
    evidence_quote: str,
) -> SemanticFact:
    return SemanticFact(
        id=fact_id,
        category="relationship",
        subject=EntityRef(type="User", identifier="user-1"),
        predicate="KNOWS",
        object=EntityRef(type="Person", identifier=object_identifier),
        evidence_quote=evidence_quote,
        confidence="high",
        source_session_id="thread-test",
        source_turn_index=0,
        created_at="2026-04-19T12:00:00Z",
        last_referenced_at="2026-04-19T12:00:00Z",
        dormant_at=None,
        superseded_by=None,
        user_visible=True,
    )


def _store_record(fact: SemanticFact) -> StoreRecord:
    return StoreRecord(
        namespace=("user-1", "semantic"),
        key=fact.id,
        value=fact.model_dump(mode="json"),
    )


def test_same_identifier_reconciles_to_bump() -> None:
    existing = _store_record(
        _semantic_fact(
            fact_id="fact-old",
            object_identifier="Sarah",
            evidence_quote="I have a sister named Sarah.",
        )
    )
    new_fact = _semantic_fact(
        fact_id="fact-new",
        object_identifier="Sarah",
        evidence_quote="My sister Sarah came over this weekend.",
    )

    plan = plan_semantic_write(new_fact, [existing])

    assert plan.bump_record is existing
    assert plan.supersede_records == []


def test_more_specific_identifier_supersedes_generic_representation() -> None:
    existing = _store_record(
        _semantic_fact(
            fact_id="fact-old",
            object_identifier="Sarah",
            evidence_quote="I talked to Sarah yesterday.",
        )
    )
    new_fact = _semantic_fact(
        fact_id="fact-new",
        object_identifier="my sister Sarah",
        evidence_quote="My sister Sarah called last night.",
    )

    plan = plan_semantic_write(new_fact, [existing])

    assert plan.bump_record is None
    assert plan.supersede_records == [existing]


def test_correction_marker_supersedes_old_active_fact() -> None:
    existing = _store_record(
        _semantic_fact(
            fact_id="fact-old",
            object_identifier="sister moved out",
            evidence_quote="My sister moved out last month.",
        )
    )
    new_fact = _semantic_fact(
        fact_id="fact-new",
        object_identifier="sister moved back in",
        evidence_quote="Actually, my sister moved back in this week.",
    )
    new_fact.category = "context"  # type: ignore[assignment]
    new_fact.predicate = "EXPERIENCED"  # type: ignore[assignment]
    new_fact.object.type = "Event"
    existing.value["category"] = "context"
    existing.value["predicate"] = "EXPERIENCED"
    existing.value["object"]["type"] = "Event"

    plan = plan_semantic_write(new_fact, [existing])

    assert plan.bump_record is None
    assert plan.supersede_records == [existing]


def test_inactive_semantic_records_are_filtered() -> None:
    active = _store_record(
        _semantic_fact(
            fact_id="fact-active",
            object_identifier="Sarah",
            evidence_quote="I have a sister named Sarah.",
        )
    )
    superseded = _store_record(
        _semantic_fact(
            fact_id="fact-superseded",
            object_identifier="Sarah",
            evidence_quote="Old Sarah fact.",
        )
    )
    superseded.value["superseded_by"] = "fact-active"
    superseded.value["dormant_at"] = "2026-04-19T13:00:00Z"

    assert is_active_semantic_record_value(active.value) is True
    assert is_active_semantic_record_value(superseded.value) is False
    assert filter_active_semantic_records([active, superseded]) == [active]


def test_procedural_rule_replaces_weaker_overlapping_rule() -> None:
    existing = build_procedural_rule(
        rule_text="You prefer short replies.",
        evidence=["Please keep it short."],
    )
    new_rule = build_procedural_rule(
        rule_text="You prefer short, direct replies without extra validation.",
        evidence=["Please be short and direct."],
    )

    plan = plan_procedural_rule_write(new_rule, [existing])

    assert plan.action == "replace"
    assert plan.replace_indexes == [0]


def test_procedural_rule_replaces_conflicting_old_rule() -> None:
    existing = build_procedural_rule(
        rule_text="Suggest meditation when it seems useful.",
        evidence=["Meditation is okay."],
    )
    new_rule = build_procedural_rule(
        rule_text="Don't suggest meditation again.",
        evidence=["Please don't suggest meditation again."],
    )

    plan = plan_procedural_rule_write(new_rule, [existing])

    assert plan.action == "replace"
    assert plan.replace_indexes == [0]


@pytest.mark.asyncio
async def test_llm_semantic_reconciliation_can_supersede_without_marker() -> None:
    existing = _store_record(
        _semantic_fact(
            fact_id="fact-old",
            object_identifier="sister moved out",
            evidence_quote="My sister moved out last month.",
        )
    )
    new_fact = _semantic_fact(
        fact_id="fact-new",
        object_identifier="sister moved back in",
        evidence_quote="My sister moved back in this week.",
    )
    new_fact.category = "context"  # type: ignore[assignment]
    new_fact.predicate = "EXPERIENCED"  # type: ignore[assignment]
    new_fact.object.type = "Event"
    existing.value["category"] = "context"
    existing.value["predicate"] = "EXPERIENCED"
    existing.value["object"]["type"] = "Event"
    llm = _FakeReconciliationLLM(
        {
            "action": "supersede",
            "record_indexes": [0],
            "reason": "new living situation replaces the older one",
            "confidence": "high",
        }
    )

    plan = await plan_semantic_write_llm_primary(
        new_fact,
        [existing],
        llm_client=llm,
    )

    assert llm.structured_calls == 1
    assert plan.bump_record is None
    assert plan.supersede_records == [existing]


@pytest.mark.asyncio
async def test_llm_procedural_reconciliation_can_replace_weaker_rule() -> None:
    existing = build_procedural_rule(
        rule_text="Use a gentle tone.",
        evidence=["Please be gentle."],
    )
    new_rule = build_procedural_rule(
        rule_text="Use a direct tone instead of a gentle one.",
        evidence=["Be direct with me, not gentle."],
    )
    llm = _FakeReconciliationLLM(
        {
            "action": "replace",
            "replace_indexes": [0],
            "reason": "new rule explicitly replaces older tone preference",
            "confidence": "high",
        }
    )

    plan = await plan_procedural_rule_write_llm_primary(
        new_rule,
        [existing],
        llm_client=llm,
    )

    assert llm.structured_calls == 1
    assert plan.action == "replace"
    assert plan.replace_indexes == [0]
