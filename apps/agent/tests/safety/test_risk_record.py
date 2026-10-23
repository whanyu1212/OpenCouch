from datetime import UTC, datetime

import pytest
from pydantic import JsonValue, ValidationError

from opencouch_agent.safety.policy import RiskLevel
from opencouch_agent.safety.risk import RiskChange, RiskState
from opencouch_agent.safety.risk_record import (
    RISK_RECORD_VERSION,
    dump_risk_record,
    load_risk_record,
)

IN_CRISIS = RiskState(
    level=RiskLevel.CRISIS,
    peak=RiskLevel.CRISIS,
    score=0.7,
    turns=3,
    calm_turns=0,
    last_risk_at=datetime(2026, 10, 23, 9, 0, tzinfo=UTC),
    last_turn_at=datetime(2026, 10, 23, 9, 5, tzinfo=UTC),
    changes=(
        RiskChange(
            2, RiskLevel.NONE, RiskLevel.ELEVATED, "this message read as ELEVATED"
        ),
        RiskChange(3, RiskLevel.ELEVATED, RiskLevel.CRISIS, "CRISIS: intent"),
    ),
)


def test_a_record_loads_back_to_the_same_state() -> None:
    assert load_risk_record(dump_risk_record(IN_CRISIS)) == IN_CRISIS


def test_a_record_carries_its_version() -> None:
    assert dump_risk_record(RiskState())["version"] == RISK_RECORD_VERSION


def test_a_record_from_another_version_fails_to_load() -> None:
    record = dump_risk_record(IN_CRISIS) | {"version": RISK_RECORD_VERSION + 1}

    with pytest.raises(ValidationError):
        load_risk_record(record)


def test_a_record_without_a_version_fails_to_load() -> None:
    with pytest.raises(ValidationError):
        load_risk_record({"state": dump_risk_record(IN_CRISIS)["state"]})


def _with_state(**fields: JsonValue) -> dict[str, JsonValue]:
    """A stored record of IN_CRISIS with some state fields replaced."""
    record = dump_risk_record(IN_CRISIS)
    state = record["state"]
    assert isinstance(state, dict)
    return record | {"state": state | fields}


def test_a_malformed_record_fails_instead_of_reading_as_no_risk() -> None:
    with pytest.raises(ValidationError):
        load_risk_record(_with_state(level="unknown"))


@pytest.mark.parametrize("field", ["last_risk_at", "last_turn_at"])
def test_a_record_with_a_time_without_a_timezone_fails_to_load(field: str) -> None:
    with pytest.raises(ValidationError):
        load_risk_record(_with_state(**{field: "2026-10-23T09:00:00"}))
