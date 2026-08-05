"""Versioned serialization boundary for persisted runtime state snapshots."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from enum import Enum
from typing import Any, TypeAlias, cast, get_args

from pydantic import BaseModel, ValidationError

from agent.models import AgentRoute, Channel, CrisisAssessment
from agent.state import AgentState

CURRENT_AGENT_STATE_SCHEMA_VERSION = 1

AgentStateSnapshot: TypeAlias = dict[str, Any]

_ROUTES = set(get_args(AgentRoute))
_SESSION_ACTIONS = {"none", "suggest_end_session"}
_SESSION_INTENTS = {
    "vent",
    "understand",
    "reflect",
    "work",
    "regulate",
    "repair",
    "close",
}
_SESSION_STAGES = {"opening", "deepening", "stabilizing", "closing"}
_GUIDANCE_PERMISSIONS = {"unknown", "not_yet", "granted"}
_ACTIVE_FLOWS = {"none", "guided_exercise", "pending_memory_action"}
_ACTIVE_FLOW_ACTIONS = {
    "none",
    "start",
    "continue",
    "preserve",
    "resume",
    "clear",
}
_TURN_ROUTES = {"therapeutic", "memory_control", "grounded_lookup", "guided_exercise"}
_CONFIDENCE_LEVELS = {"low", "medium", "high"}
_CLARIFICATION_KINDS = {"none", "blocking", "soft"}
_MEMORY_REFERENCE_MODES = {"none", "explicit"}
_CRISIS_OVERRIDE_OUTCOMES = {"none"}
_CRISIS_CLASSIFIER_PATHS = {"llm_primary", "voice_concurrent", "voice_post_turn"}
_RESOURCE_LOOKUP_STATUSES = {
    "not_attempted",
    "pending",
    "found",
    "no_location",
    "location_refused",
    "no_verified_results",
    "lookup_error",
}
_MAPPING_FIELDS = {
    "session_memory",
    "procedural_profile",
    "session_progress",
    "exercise_state",
    "memory_control",
    "grounded_lookup",
    "turn_lifecycle",
    "memory_reference",
    "crisis_audit",
    "diagnostics",
}
_LIST_FIELDS = {"installed_skills", "transcript", "working_memory", "found_resources"}


class RuntimeStateCodecError(ValueError):
    """Base error for invalid persisted runtime state."""


class RuntimeStateEncodeError(RuntimeStateCodecError):
    """Raised when runtime state cannot be represented as JSON."""


class RuntimeStateDecodeError(RuntimeStateCodecError):
    """Raised when persisted runtime state violates the current contract."""


class UnsupportedRuntimeStateVersion(RuntimeStateDecodeError):
    """Raised when a snapshot was written by a newer unsupported codec."""


def encode_agent_state_snapshot(state: Mapping[str, Any]) -> AgentStateSnapshot:
    """Encode one runtime state mapping in the current versioned envelope."""

    validated = _validate_state(copy.deepcopy(dict(state)))
    return {
        "schema_version": CURRENT_AGENT_STATE_SCHEMA_VERSION,
        "state": cast(dict[str, Any], _json_value(validated, path="state")),
    }


def decode_agent_state_snapshot(payload: Any) -> AgentState:
    """Decode and validate a current or legacy runtime state snapshot."""

    if not isinstance(payload, Mapping):
        raise RuntimeStateDecodeError("snapshot must be a mapping")

    if "schema_version" not in payload:
        version = 0
        state = copy.deepcopy(dict(payload))
    else:
        version = payload.get("schema_version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise RuntimeStateDecodeError("schema_version must be an integer")
        raw_state = payload.get("state")
        if not isinstance(raw_state, Mapping):
            raise RuntimeStateDecodeError("state must be a mapping")
        state = copy.deepcopy(dict(raw_state))

    if version > CURRENT_AGENT_STATE_SCHEMA_VERSION:
        raise UnsupportedRuntimeStateVersion(
            f"unsupported runtime state schema version {version}; "
            f"current version is {CURRENT_AGENT_STATE_SCHEMA_VERSION}"
        )
    if version < 0:
        raise RuntimeStateDecodeError("schema_version must not be negative")

    while version < CURRENT_AGENT_STATE_SCHEMA_VERSION:
        migration = _MIGRATIONS.get(version)
        if migration is None:
            raise RuntimeStateDecodeError(
                f"no runtime state migration from schema version {version}"
            )
        state = migration(state)
        version += 1

    return _validate_state(state)


def _migrate_v0_to_v1(state: dict[str, Any]) -> dict[str, Any]:
    """Migrate the legacy unversioned state body without discarding fields."""

    return dict(state)


_MIGRATIONS = {0: _migrate_v0_to_v1}


def _validate_state(raw: dict[str, Any]) -> AgentState:
    state = dict(raw)

    for field in _MAPPING_FIELDS:
        if field in state and not isinstance(state[field], Mapping):
            raise RuntimeStateDecodeError(f"state.{field} must be a mapping")
    for field in _LIST_FIELDS:
        if field in state and not isinstance(state[field], list):
            raise RuntimeStateDecodeError(f"state.{field} must be a list")

    if "channel" in state:
        try:
            state["channel"] = Channel(state["channel"])
        except (TypeError, ValueError) as exc:
            raise RuntimeStateDecodeError("state.channel has an invalid value") from exc

    if "crisis" in state:
        try:
            state["crisis"] = CrisisAssessment.model_validate(
                state["crisis"], strict=True
            )
        except ValidationError as exc:
            raise RuntimeStateDecodeError("state.crisis is invalid") from exc

    _validate_optional_value(state, "route", _ROUTES)
    _validate_optional_value(state, "session_action", _SESSION_ACTIONS)

    session_progress = _mapping(state, "session_progress")
    if session_progress is not None:
        turn_count = session_progress.get("turn_count")
        if turn_count is not None and (
            isinstance(turn_count, bool)
            or not isinstance(turn_count, int)
            or turn_count < 0
        ):
            raise RuntimeStateDecodeError(
                "state.session_progress.turn_count must be a non-negative integer"
            )
        _validate_optional_value(
            session_progress,
            "session_intent",
            _SESSION_INTENTS,
            path="state.session_progress",
        )
        _validate_optional_value(
            session_progress,
            "session_stage",
            _SESSION_STAGES,
            path="state.session_progress",
        )
        _validate_optional_value(
            session_progress,
            "guidance_permission",
            _GUIDANCE_PERMISSIONS,
            path="state.session_progress",
        )

    lifecycle = _mapping(state, "turn_lifecycle")
    if lifecycle is not None:
        _validate_optional_value(
            lifecycle, "active_flow", _ACTIVE_FLOWS, path="state.turn_lifecycle"
        )
        _validate_optional_value(
            lifecycle, "action", _ACTIVE_FLOW_ACTIONS, path="state.turn_lifecycle"
        )
        _validate_optional_value(
            lifecycle,
            "tentative_route",
            _TURN_ROUTES,
            path="state.turn_lifecycle",
            allow_none=True,
        )
        _validate_optional_value(
            lifecycle,
            "secondary_route",
            _TURN_ROUTES,
            path="state.turn_lifecycle",
            allow_none=True,
        )
        _validate_optional_value(
            lifecycle,
            "triage_confidence",
            _CONFIDENCE_LEVELS,
            path="state.turn_lifecycle",
            allow_none=True,
        )
        _validate_optional_value(
            lifecycle,
            "clarification_kind",
            _CLARIFICATION_KINDS,
            path="state.turn_lifecycle",
        )

    memory_reference = _mapping(state, "memory_reference")
    if memory_reference is not None:
        _validate_optional_value(
            memory_reference,
            "mode",
            _MEMORY_REFERENCE_MODES,
            path="state.memory_reference",
        )

    crisis_audit = _mapping(state, "crisis_audit")
    if crisis_audit is not None:
        _validate_optional_value(
            crisis_audit,
            "crisis_override_kind",
            _CRISIS_OVERRIDE_OUTCOMES,
            path="state.crisis_audit",
        )
        _validate_optional_value(
            crisis_audit,
            "crisis_classifier_path",
            _CRISIS_CLASSIFIER_PATHS,
            path="state.crisis_audit",
        )

    _validate_optional_value(state, "resource_lookup_status", _RESOURCE_LOOKUP_STATUSES)
    return cast(AgentState, state)


def _mapping(state: Mapping[str, Any], field: str) -> dict[str, Any] | None:
    value = state.get(field)
    return dict(value) if isinstance(value, Mapping) else None


def _validate_optional_value(
    state: Mapping[str, Any],
    field: str,
    allowed: set[str],
    *,
    path: str = "state",
    allow_none: bool = False,
) -> None:
    if field not in state:
        return
    value = state[field]
    if allow_none and value is None:
        return
    if not isinstance(value, str) or value not in allowed:
        raise RuntimeStateDecodeError(f"{path}.{field} has an invalid value")


def _json_value(value: Any, *, path: str) -> Any:
    if isinstance(value, BaseModel):
        return _json_value(value.model_dump(mode="json"), path=path)
    if isinstance(value, Enum):
        return _json_value(value.value, path=path)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item, path=path) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item, path=path) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise RuntimeStateEncodeError(
        f"{path} contains non-JSON value of type {type(value).__name__}"
    )


__all__ = [
    "CURRENT_AGENT_STATE_SCHEMA_VERSION",
    "RuntimeStateCodecError",
    "RuntimeStateDecodeError",
    "RuntimeStateEncodeError",
    "UnsupportedRuntimeStateVersion",
    "decode_agent_state_snapshot",
    "encode_agent_state_snapshot",
]
