"""Shared fixtures for safety-event capture regression tests."""

from __future__ import annotations

from datetime import datetime, timezone

from agent.audit.crisis_log import CrisisLogBackend
from agent.audit.models import CrisisLogRecord

CRISIS_USER_TEXT = "I'm in Singapore and I will end my life tonight."
CRISIS_VOICE_USER_TEXT = "I might hurt myself tonight."
CRISIS_RESPONSE_TEXT = "Please contact local emergency services now."
CRISIS_VOICE_RESPONSE_TEXT = "I'm here with you. Your safety matters most right now."
CRISIS_TOOL_NAME = "lookup_crisis_resources"


def openai_crisis_tool_calls() -> list[tuple[str, dict[str, object]]]:
    """Return SDK-runner tool calls for a text crisis response turn."""

    return [(CRISIS_TOOL_NAME, {})]


def voice_crisis_lookup_tool_call(
    *,
    resource_lookup_status: str = "found",
    found_resources: list[dict[str, object]] | None = None,
    inferred_location: str = "Singapore",
) -> dict[str, object]:
    """Return one Realtime voice crisis-resource lookup tool call payload."""

    return {
        "tool_name": CRISIS_TOOL_NAME,
        "status": "completed",
        "output": {
            "inferred_location": inferred_location,
            "found_resources": list(found_resources or []),
            "resource_lookup_status": resource_lookup_status,
        },
    }


async def utc_crisis_records(
    backend: CrisisLogBackend,
) -> list[CrisisLogRecord]:
    """Read crisis records for today's UTC ledger bucket."""

    return await backend.alist_by_date(datetime.now(timezone.utc).date())
