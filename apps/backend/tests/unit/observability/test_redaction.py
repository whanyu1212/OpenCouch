"""Tests for trace attribute redaction."""

from __future__ import annotations

from agent.observability.config import TraceConfig
from agent.observability.redaction import sanitize_attributes, sanitize_value


def test_trace_config_validates_sample_rate() -> None:
    assert TraceConfig(sample_rate=0.0).sample_rate == 0.0
    assert TraceConfig(sample_rate=1.0).sample_rate == 1.0

    try:
        TraceConfig(sample_rate=1.1)
    except ValueError as exc:
        assert "sample_rate" in str(exc)
    else:
        raise AssertionError("expected invalid sample_rate to raise")


def test_sanitize_attributes_redacts_sensitive_keys_and_truncates_values() -> None:
    sanitized = sanitize_attributes(
        {
            "user_message": "please keep this private",
            "api_key": "abc",
            "route": "ground",
            "long": "x" * 20,
        },
        max_string_length=8,
    )

    assert sanitized["user_message"] == "[redacted]"
    assert sanitized["api_key"] == "[redacted]"
    assert sanitized["route"] == "ground"
    assert sanitized["long"] == "xxxxx..."


def test_sanitize_value_handles_nested_and_unknown_objects() -> None:
    class CustomObject:
        def __str__(self) -> str:
            return "custom object value"

    sanitized = sanitize_value(
        {
            "items": [" one\n two ", CustomObject()],
            "memory_text": "secret memory",
        },
        max_string_length=12,
    )

    assert sanitized["items"] == ["one two", "custom ob..."]
    assert sanitized["memory_text"] == "[redacted]"
