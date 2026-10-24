"""A conversation's saved history, shaped for the chat to show again (HY-350).

Only what was said goes out: the user's messages and the coach's text. Tool
calls, instructions, shared state and risk stay on the server.
"""

from collections.abc import Sequence
from typing import Final, Literal

from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse

from opencouch_agent.safety import conversation_turns

# Where `AGUIAdapter` keeps the id the client gave a user message.
_UI_METADATA_KEY: Final = "__pydantic_ai__"


class ShownMessage(BaseModel):
    """One message as the chat shows it."""

    id: str
    role: Literal["user", "assistant"]
    content: str
    interrupted: bool = False
    """The reply was cut off by an error, a timeout or a disconnect (HY-204)."""


class ConversationHistory(BaseModel):
    """The messages of one conversation, oldest first."""

    messages: list[ShownMessage]


def shown_history(history: Sequence[ModelMessage]) -> ConversationHistory:
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
    return ConversationHistory(messages=shown)


def _message_id(message: ModelMessage, index: int) -> str:
    # Keep the client's id for user messages, so a reload matches what the
    # client already had; replies get one stable per position in the history.
    if isinstance(message, ModelRequest) and message.metadata:
        ui = message.metadata.get(_UI_METADATA_KEY)
        if isinstance(ui, dict) and isinstance(ui.get("ui_message_id"), str):
            return str(ui["ui_message_id"])
    return f"history-{index}"
