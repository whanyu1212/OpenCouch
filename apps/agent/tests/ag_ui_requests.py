"""Builders for AG-UI request bodies used across tests."""

from typing import Any


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
