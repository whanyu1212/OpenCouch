"""The stored form of a conversation's risk: a versioned `RiskState` record.

The record holds levels, scores, times and the generated reason for each
change, never conversation text. Bump `RISK_RECORD_VERSION` when `RiskState`
changes shape, so old records fail to load instead of loading wrong. A new
field whose default is right for every old record needs no bump.
"""

from typing import Final

from pydantic import BaseModel, ConfigDict, JsonValue, field_validator

from opencouch_agent.safety.risk import RiskState

RISK_RECORD_VERSION: Final = 1


class _RiskRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    state: RiskState

    @field_validator("version")
    @classmethod
    def _current_version(cls, version: int) -> int:
        if version != RISK_RECORD_VERSION:
            raise ValueError(f"unsupported risk record version {version}")
        return version

    @field_validator("state")
    @classmethod
    def _timezone_aware(cls, state: RiskState) -> RiskState:
        # `update_risk` compares these with an aware time; a naive one would
        # fail mid-turn instead of here.
        for moment in (
            state.last_risk_at,
            state.last_turn_at,
            state.resources_pinned_until,
        ):
            if moment is not None and moment.tzinfo is None:
                raise ValueError("risk record times must be timezone-aware")
        return state


def dump_risk_record(state: RiskState) -> dict[str, JsonValue]:
    """The JSON-compatible record to store for `state`."""
    record = _RiskRecord(version=RISK_RECORD_VERSION, state=state)
    return record.model_dump(mode="json")


def load_risk_record(record: object) -> RiskState:
    """The risk state in a stored record.

    Raises:
        pydantic.ValidationError: If the record is malformed, from another
            version, or has a time without a timezone. Safety: it never falls
            back to an empty state, which would read as no risk.
    """
    return _RiskRecord.model_validate(record).state
