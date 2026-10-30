"""Types select resources and guidance without changing risk decisions."""

from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta
from itertools import combinations
from typing import get_args

import pytest
from pydantic import ValidationError

from opencouch_agent.safety.crisis import turn_instructions
from opencouch_agent.safety.crisis_guidance import crisis_fallback
from opencouch_agent.safety.crisis_reply_check import allowed_numbers, reply_problem
from opencouch_agent.safety.crisis_resources import (
    CrisisType,
    load_shipped_catalog,
    select_resources,
)
from opencouch_agent.safety.policy import RiskLevel
from opencouch_agent.safety.questions import UserSignals
from opencouch_agent.safety.risk import RiskState, update_risk
from opencouch_agent.safety.risk_record import dump_risk_record, load_risk_record

CATALOG = load_shipped_catalog()
NOW = datetime(2026, 10, 30, tzinfo=UTC)
CALM = UserSignals(**{field.name: 0.0 for field in fields(UserSignals)})


def test_types_survive_serialization_calm_turns_and_step_down() -> None:
    state = update_risk(RiskState(), replace(CALM, medical_emergency=0.95), at=NOW)
    state = load_risk_record(dump_risk_record(state))
    later = NOW + timedelta(minutes=11)
    for _ in range(3):
        state = update_risk(state, CALM, at=later)
    state = update_risk(state, CALM, at=later, check_in="safe")

    assert state.level is RiskLevel.ELEVATED
    assert state.crisis_types == {"medical_emergency"}


def test_legacy_records_keep_their_level_and_use_general_resources() -> None:
    record = dump_risk_record(RiskState(level=RiskLevel.CRISIS, peak=RiskLevel.CRISIS))
    stored = record["state"]
    assert isinstance(stored, dict)
    del stored["crisis_types"]
    state = load_risk_record(record)

    assert state.level is RiskLevel.CRISIS
    assert not state.crisis_types
    services = select_resources(CATALOG, "SG", state.crisis_types, imminent=False)
    assert [s.key for s in services] == ["sg-sos", "sg-mindline", "sg-emergency"]
    assert "Do not assume" in turn_instructions(
        services, types=state.crisis_types, imminent=False
    )


def test_invalid_stored_types_are_rejected_at_the_storage_boundary() -> None:
    record = dump_risk_record(RiskState())
    stored = record["state"]
    assert isinstance(stored, dict)
    stored["crisis_types"] = ["made-up"]
    with pytest.raises(ValidationError):
        load_risk_record(record)


def test_mixed_types_are_ordered_without_duplicate_resources() -> None:
    types: frozenset[CrisisType] = frozenset({"suicide_self_harm", "abuse_violence"})
    services = select_resources(CATALOG, "SG", types, imminent=True)
    assert [s.key for s in services] == ["sg-emergency", "sg-navh", "sg-sos"]
    instructions = turn_instructions(services, types=types, imminent=True)
    assert instructions.index("## Abuse or violence") < instructions.index(
        "## Suicide or self-harm"
    )
    assert "999 for imminent violence" in instructions


def test_every_type_combination_has_vetted_fallback_numbers_and_unique_resources() -> (
    None
):
    all_types: tuple[CrisisType, ...] = get_args(CrisisType)
    numbers = allowed_numbers(CATALOG)
    for size in range(len(all_types) + 1):
        for combination in combinations(all_types, size):
            types = frozenset(combination)
            fallback = crisis_fallback(types)
            assert reply_problem(fallback, numbers) is None
            assert fallback.count(".") + fallback.count("?") <= 4
            assert fallback.count("995") <= 1
            assert fallback.count("999") <= 2  # 999 and SMS 70999
            services = select_resources(CATALOG, "SG", types, imminent=False)
            keys = [s.key for s in services]
            assert len(keys) == len(set(keys))
            assert keys[-1] == "sg-emergency"
            assert turn_instructions(services, types=types, imminent=False)


@pytest.mark.parametrize(
    ("types", "question"),
    [
        (
            {"medical_emergency", "abuse_violence"},
            "Have you been able to call for medical help?",
        ),
        (
            {"harm_to_others", "abuse_violence"},
            "Is anyone in immediate danger right now?",
        ),
        ({"abuse_violence", "suicide_self_harm"}, "Is it safe to use this device?"),
        ({"suicide_self_harm"}, "Are you safe right now?"),
    ],
)
def test_fallback_question_matches_the_priority(
    types: set[CrisisType], question: str
) -> None:
    assert crisis_fallback(frozenset(types)).endswith(question)
