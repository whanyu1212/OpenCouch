"""Tests for the versioned runtime-state persistence boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.models import Channel, CrisisAssessment
from agent.runtime.state_codec import (
    CURRENT_AGENT_STATE_SCHEMA_VERSION,
    RuntimeStateDecodeError,
    RuntimeStateEncodeError,
    UnsupportedRuntimeStateVersion,
    decode_agent_state_snapshot,
    encode_agent_state_snapshot,
)

_FIXTURES = Path(__file__).parents[2] / "fixtures" / "runtime_state"


def _fixture(name: str) -> object:
    return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


def test_current_snapshot_restores_models_and_preserves_unknown_fields() -> None:
    loaded = decode_agent_state_snapshot(_fixture("current_v1.json"))

    assert loaded["channel"] is Channel.WEB
    assert isinstance(loaded["crisis"], CrisisAssessment)
    assert loaded["session_progress"]["future_progress_field"] == "preserved"
    assert loaded["future_state_field"] == {"enabled": True}

    encoded = encode_agent_state_snapshot(loaded)

    assert encoded["schema_version"] == CURRENT_AGENT_STATE_SCHEMA_VERSION
    assert encoded["state"]["channel"] == "web"
    assert encoded["state"]["crisis"]["level"] == 1
    assert encoded["state"]["future_state_field"] == {"enabled": True}


def test_unversioned_snapshot_migrates_as_v0_without_losing_fields() -> None:
    loaded = decode_agent_state_snapshot(_fixture("legacy_v0.json"))

    assert loaded["channel"] is Channel.VOICE
    assert isinstance(loaded["crisis"], CrisisAssessment)
    assert loaded["legacy_unknown_field"] == "preserved"
    assert encode_agent_state_snapshot(loaded)["schema_version"] == 1


def test_partial_state_snapshot_remains_supported() -> None:
    assert decode_agent_state_snapshot(
        {"schema_version": 1, "state": {"diagnostics": {"source": "test"}}}
    ) == {"diagnostics": {"source": "test"}}


@pytest.mark.parametrize(
    ("state", "path"),
    [
        ({"route": "unknown"}, "state.route"),
        ({"session_action": "finish"}, "state.session_action"),
        (
            {"session_progress": {"turn_count": -1}},
            "state.session_progress.turn_count",
        ),
        (
            {"session_progress": {"session_stage": "finished"}},
            "state.session_progress.session_stage",
        ),
        (
            {"turn_lifecycle": {"active_flow": "other"}},
            "state.turn_lifecycle.active_flow",
        ),
        (
            {"turn_lifecycle": {"action": "restart"}},
            "state.turn_lifecycle.action",
        ),
        (
            {"memory_reference": {"mode": "automatic"}},
            "state.memory_reference.mode",
        ),
        (
            {"crisis_audit": {"crisis_classifier_path": "fallback"}},
            "state.crisis_audit.crisis_classifier_path",
        ),
        (
            {"resource_lookup_status": "unverified"},
            "state.resource_lookup_status",
        ),
    ],
)
def test_invalid_known_values_are_rejected(state: dict[str, object], path: str) -> None:
    with pytest.raises(RuntimeStateDecodeError, match=path):
        decode_agent_state_snapshot({"schema_version": 1, "state": state})


def test_invalid_channel_is_not_silently_mapped_and_error_omits_payload() -> None:
    with pytest.raises(RuntimeStateDecodeError) as caught:
        decode_agent_state_snapshot(_fixture("malformed_v1.json"))

    assert "state.channel" in str(caught.value)
    assert "private fixture sentinel" not in str(caught.value)


def test_invalid_crisis_assessment_is_rejected() -> None:
    with pytest.raises(RuntimeStateDecodeError, match="state.crisis"):
        decode_agent_state_snapshot(
            {
                "schema_version": 1,
                "state": {"crisis": {"level": 9, "confidence": "certain"}},
            }
        )


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"schema_version": "1", "state": {}},
        {"schema_version": 1},
        {"schema_version": -1, "state": {}},
        {"schema_version": 1, "state": {"transcript": {}}},
    ],
)
def test_malformed_envelopes_and_known_containers_are_rejected(payload: object) -> None:
    with pytest.raises(RuntimeStateDecodeError):
        decode_agent_state_snapshot(payload)


def test_unsupported_future_snapshot_is_rejected() -> None:
    with pytest.raises(UnsupportedRuntimeStateVersion, match="version 2"):
        decode_agent_state_snapshot(_fixture("unsupported_future.json"))


def test_encoder_recursively_serializes_models_and_enums() -> None:
    encoded = encode_agent_state_snapshot(
        {
            "diagnostics": {
                "channel": Channel.VOICE,
                "assessment": CrisisAssessment(level=1),
            }
        }
    )

    assert encoded["state"]["diagnostics"] == {
        "channel": "voice",
        "assessment": {
            "level": 1,
            "confidence": "low",
            "reason": "",
            "needs_crisis_response": False,
            "needs_clarification": False,
        },
    }


def test_encoder_rejects_non_json_values_without_exposing_mapping_keys() -> None:
    sensitive_key = "private user supplied key"
    with pytest.raises(RuntimeStateEncodeError, match="state") as caught:
        encode_agent_state_snapshot({"diagnostics": {sensitive_key: object()}})

    assert sensitive_key not in str(caught.value)
