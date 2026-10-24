"""Builders for AG-UI request bodies, and a reader for replies, used across tests."""

import json
from typing import Any

import httpx2


def run_input(
    *messages: dict[str, Any],
    thread_id: str = "thread-1",
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an AG-UI `RunAgentInput` request body."""
    return {
        "threadId": thread_id,
        "runId": "run-1",
        "state": state or {},
        "messages": list(messages),
        "tools": [],
        "context": [],
        "forwardedProps": {},
    }


def user_message(content: str, message_id: str = "msg-1") -> dict[str, Any]:
    return {"id": message_id, "role": "user", "content": content}


def assistant_message(content: str, message_id: str = "msg-a") -> dict[str, Any]:
    return {"id": message_id, "role": "assistant", "content": content}


def reply_text(response: httpx2.Response) -> str:
    """The text the agent streamed in an AG-UI response."""
    events = _events(response.text)
    return "".join(e["delta"] for e in events if e["type"] == "TEXT_MESSAGE_CONTENT")


def event_types(body: str) -> list[str]:
    """The type of each event in a streamed AG-UI response body, in order."""
    return [event["type"] for event in _events(body)]


def state_snapshots(body: str) -> list[dict[str, Any]]:
    """The state in each snapshot event of a streamed AG-UI response, in order."""
    return [e["snapshot"] for e in _events(body) if e["type"] == "STATE_SNAPSHOT"]


def _events(body: str) -> list[dict[str, Any]]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ")
    ]
