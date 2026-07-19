"""Shared session builders for integration suites."""

from __future__ import annotations

from agent.models import Message, MessageRole
from opencouch_tui.models import RunnerSession


def make_session() -> RunnerSession:
    """Return a baseline CLI session for command tests."""

    return RunnerSession(
        requested_mode="deterministic",
        resolved_mode="deterministic",
        llm_client=None,
        thread_id="thread-a",
        memory_mode="persistent",
        history=[
            Message(role=MessageRole.USER, content="first"),
            Message(role=MessageRole.ASSISTANT, content="reply"),
        ],
        last_context={"session_progress": {"turn_count": 2}, "transcript": []},
    )
