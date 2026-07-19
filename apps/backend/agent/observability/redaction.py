"""Privacy-preserving attribute sanitization for traces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DEFAULT_MAX_STRING_LENGTH = 240
REDACTED = "[redacted]"

_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "assistant_response",
    "audio",
    "crisis_text",
    "crisis_user_content",
    "memory_content",
    "memory_text",
    "password",
    "raw_prompt",
    "raw_response",
    "secret",
    "token",
    "transcript",
    "user_message",
)


def sanitize_attributes(
    attributes: Mapping[str, Any] | None,
    *,
    max_string_length: int = DEFAULT_MAX_STRING_LENGTH,
) -> dict[str, Any]:
    """Return privacy-safe trace attributes.

    Args:
        attributes: Raw attributes supplied by instrumentation.
        max_string_length: Maximum length for string values.

    Returns:
        Sanitized attribute mapping safe for generic observability sinks.
    """

    if not attributes:
        return {}

    sanitized: dict[str, Any] = {}
    for key, value in attributes.items():
        clean_key = _clean_key(key)
        if not clean_key:
            continue
        if _is_sensitive_key(clean_key):
            sanitized[clean_key] = REDACTED
            continue
        sanitized[clean_key] = sanitize_value(
            value,
            max_string_length=max_string_length,
        )
    return sanitized


def sanitize_value(
    value: Any, *, max_string_length: int = DEFAULT_MAX_STRING_LENGTH
) -> Any:
    """Return a compact trace-safe representation for one value."""

    if value is None or isinstance(value, bool | int | float):
        return value

    if isinstance(value, str):
        return _truncate(" ".join(value.strip().split()), max_string_length)

    if isinstance(value, tuple | list):
        return [
            sanitize_value(item, max_string_length=max_string_length)
            for item in value[:20]
        ]

    if isinstance(value, Mapping):
        return sanitize_attributes(value, max_string_length=max_string_length)

    return _truncate(str(value), max_string_length)


def _clean_key(key: Any) -> str:
    cleaned = " ".join(str(key).strip().split())
    return cleaned.replace(" ", "_")


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower()
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _truncate(value: str, max_length: int) -> str:
    if max_length < 4:
        return value[:max_length]
    if len(value) <= max_length:
        return value
    return f"{value[: max_length - 3]}..."
