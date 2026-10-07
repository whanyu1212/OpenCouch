"""The user check: screen the newest user message with Jev.

The transcript framing and window are ported from the FirstRead repo's guard
(whanyu1212/firstread, `packages/firstread/src/guard.ts` at `25b8aef`). The
questions were tuned on text in exactly this shape, so keep it in step with
them.
"""

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    TextContent,
    TextPart,
    UserPromptPart,
)

from opencouch_agent.safety.jev import FailureReason, JevClient, failure_reason
from opencouch_agent.safety.policy import RiskLevel, assess_level
from opencouch_agent.safety.questions import USER_TURN_QUESTIONS, UserSignals

logger = logging.getLogger(__name__)

# How many recent turns Jev sees, so risk that builds across turns isn't judged
# one message at a time. Eight turns is four exchanges.
WINDOW_TURNS: Final = 8
# The whole check, retries included. Past this the turn goes on without it.
SCREENING_DEADLINE_SECONDS: Final = 3.0


@dataclass(frozen=True)
class Turn:
    """One message in the transcript Jev reads."""

    role: Literal["user", "assistant"]
    text: str


@dataclass(frozen=True)
class Screened:
    """Jev answered: the turn's signals and the level they assess to."""

    signals: UserSignals
    level: RiskLevel


@dataclass(frozen=True)
class ScreeningFailed:
    """Jev didn't answer in time, or answered with an error."""

    reason: FailureReason


ScreeningResult = Screened | ScreeningFailed


async def screen_user_message(
    client: JevClient,
    history: Sequence[Turn],
    latest_message: str,
    *,
    deadline_seconds: float = SCREENING_DEADLINE_SECONDS,
) -> ScreeningResult:
    """Run the user check on the newest message. Never raises on a failed check.

    Args:
        client: The Jev client.
        history: The conversation before `latest_message`, oldest first.
        latest_message: The message to screen.
        deadline_seconds: Limit on the whole check, retries included.

    Returns:
        The signals and assessed level, or why the check failed. The caller
        must still pick a crisis-aware path on failure.
    """
    state = user_check_state(history, latest_message)
    try:
        async with asyncio.timeout(deadline_seconds):
            probabilities = await client.ask(state, USER_TURN_QUESTIONS)
        signals = UserSignals.from_probabilities(probabilities)
    except Exception as error:
        # Fail closed: every failure becomes a result the caller has to handle,
        # so a broken check can't crash the turn or pass as "no risk".
        reason = failure_reason(error)
        # Only the type and reason: SDK error messages can echo response bodies.
        logger.warning("Jev user check failed (%s): %s", reason, type(error).__name__)
        return ScreeningFailed(reason)
    return Screened(signals=signals, level=assess_level(signals))


def user_check_state(history: Sequence[Turn], latest_message: str) -> str:
    """The text Jev reads for the user check."""
    return f"{_transcript(history)}\n\nLATEST USER MESSAGE:\n{latest_message}"


def conversation_turns(messages: Sequence[ModelMessage]) -> list[Turn]:
    """The user and assistant text in a stored history, oldest first.

    Tool calls, tool results and instructions are left out: Jev reads only
    what was said.
    """
    turns: list[Turn] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            turns.extend(
                Turn("user", _prompt_text(part))
                for part in message.parts
                if isinstance(part, UserPromptPart) and _prompt_text(part)
            )
        else:
            text = "".join(
                part.content for part in message.parts if isinstance(part, TextPart)
            )
            if text:
                turns.append(Turn("assistant", text))
    return turns


def _prompt_text(part: UserPromptPart) -> str:
    # Multimodal prompts keep only their text, so a message with an image
    # attached is still screened.
    if isinstance(part.content, str):
        return part.content
    texts = (
        item.content if isinstance(item, TextContent) else item
        for item in part.content
        if isinstance(item, str | TextContent)
    )
    return "\n".join(texts)


def _transcript(history: Sequence[Turn]) -> str:
    recent = history[-WINDOW_TURNS:]
    if not recent:
        return "CONVERSATION SO FAR: (none)"
    lines = "\n".join(f"{turn.role.upper()}: {turn.text}" for turn in recent)
    return f"CONVERSATION SO FAR:\n{lines}"
