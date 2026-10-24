"""A conversation's saved history, shaped for the chat to show again (HY-350).

Only what was said goes out: the user's messages and the coach's text. Tool
calls, instructions, shared state and risk stay on the server.
"""

import asyncio
from collections.abc import Sequence
from datetime import timedelta
from typing import Final, Literal

from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse

from opencouch_agent.safety import conversation_turns
from opencouch_agent.sessions import SessionStore

# Where `AGUIAdapter` keeps the id the client gave a user message.
_UI_METADATA_KEY: Final = "__pydantic_ai__"
_POLL_SECONDS: Final = 0.1


class ShownMessage(BaseModel):
    """One message as the chat shows it."""

    id: str
    role: Literal["user", "assistant"]
    content: str
    interrupted: bool = False
    """The reply was cut off by an error, a timeout or a disconnect (HY-204)."""


class ConversationHistory(BaseModel):
    """The messages of one conversation, oldest first, and its resource card."""

    messages: list[ShownMessage]
    crisis_resources: list[str]
    """Catalog keys for the crisis resource card shown after the latest reply."""


async def wait_for_turn(store: SessionStore, thread_id: str, limit: timedelta) -> None:
    """Wait until no turn holds the thread, or `limit` has passed.

    A reload drops the connection of a reply still streaming, and that turn
    saves the cut-off reply while it stops, after the client has gone. Loading
    before then would leave the reply out of the reloaded conversation. The
    limit stops a reply streaming in another tab from holding the load up.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + limit.total_seconds()
    # Polled: the claim lives in the store, which another worker may hold, so
    # there's no in-process event to wait on.
    while await store.turn_in_progress(thread_id) and loop.time() < deadline:  # noqa: ASYNC110
        await asyncio.sleep(_POLL_SECONDS)


def shown_history(
    history: Sequence[ModelMessage], *, crisis_resources: Sequence[str] = ()
) -> ConversationHistory:
    """The user and coach messages in a stored history, oldest first."""
    shown: list[ShownMessage] = []
    for index, message in enumerate(history):
        message_id = _message_id(message, index)
        interrupted = (
            isinstance(message, ModelResponse) and message.state == "interrupted"
        )
        # `conversation_turns` already pulls out what was said, the same way
        # the safety gate reads it. A request rarely holds more than one
        # prompt, but each still needs its own id.
        for part_index, turn in enumerate(conversation_turns([message])):
            shown.append(
                ShownMessage(
                    id=message_id if part_index == 0 else f"{message_id}-{part_index}",
                    role=turn.role,
                    content=turn.text,
                    interrupted=interrupted,
                )
            )
    return ConversationHistory(messages=shown, crisis_resources=list(crisis_resources))


def _message_id(message: ModelMessage, index: int) -> str:
    # Keep the client's id for user messages, so a reload matches what the
    # client already had; replies get one stable per position in the history.
    if isinstance(message, ModelRequest) and message.metadata:
        ui = message.metadata.get(_UI_METADATA_KEY)
        if isinstance(ui, dict) and isinstance(ui.get("ui_message_id"), str):
            return str(ui["ui_message_id"])
    return f"history-{index}"
