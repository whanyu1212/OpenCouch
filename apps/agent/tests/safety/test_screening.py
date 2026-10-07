import asyncio
import logging
from collections.abc import Mapping

import httpx2
import pytest
from pydantic_ai.messages import (
    CachePoint,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextContent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from typesafe_sdk import Noul, TypeSafeAPIError

from opencouch_agent.safety.jev import FailureReason
from opencouch_agent.safety.policy import RiskLevel
from opencouch_agent.safety.questions import USER_TURN_QUESTIONS
from opencouch_agent.safety.screening import (
    WINDOW_TURNS,
    Screened,
    ScreeningFailed,
    Turn,
    conversation_turns,
    screen_user_message,
    user_check_state,
)

QUIET_ANSWERS = {name: 0.0 for name in USER_TURN_QUESTIONS}


class FakeJev:
    """Answers with fixed probabilities, or fails, and records what it was asked."""

    def __init__(
        self,
        answers: Mapping[str, float] = QUIET_ANSWERS,
        *,
        error: Exception | None = None,
        delay_seconds: float = 0.0,
    ) -> None:
        self.answers = dict(answers)
        self.error = error
        self.delay_seconds = delay_seconds
        self.states: list[str] = []
        self.questions: list[Mapping[str, Noul]] = []

    async def ask(self, state: str, questions: Mapping[str, Noul]) -> dict[str, float]:
        self.states.append(state)
        self.questions.append(questions)
        await asyncio.sleep(self.delay_seconds)
        if self.error is not None:
            raise self.error
        return self.answers


async def test_quiet_message_screens_as_none() -> None:
    jev = FakeJev()
    result = await screen_user_message(jev, [], "I had a long day at work.")
    assert isinstance(result, Screened)
    assert result.level is RiskLevel.NONE
    [questions] = jev.questions
    assert {name: q.instructions for name, q in questions.items()} == dict(
        USER_TURN_QUESTIONS
    )


async def test_risky_message_screens_at_the_assessed_level() -> None:
    jev = FakeJev({**QUIET_ANSWERS, "intent": 0.7})
    result = await screen_user_message(jev, [], "message")
    assert isinstance(result, Screened)
    assert result.level is RiskLevel.CRISIS
    assert result.signals.intent == 0.7


async def test_jev_error_is_returned_as_a_failure_not_raised() -> None:
    jev = FakeJev(error=TypeSafeAPIError(503, None, httpx2.Headers()))
    result = await screen_user_message(jev, [], "message")
    assert result == ScreeningFailed(FailureReason.UNAVAILABLE)


async def test_jev_slower_than_the_deadline_fails_as_timeout() -> None:
    jev = FakeJev(delay_seconds=1.0)
    result = await screen_user_message(jev, [], "message", deadline_seconds=0.01)
    assert result == ScreeningFailed(FailureReason.TIMEOUT)


async def test_nan_answer_fails_instead_of_reading_as_no_risk() -> None:
    answers = {**QUIET_ANSWERS, "intent": float("nan")}
    result = await screen_user_message(FakeJev(answers), [], "message")
    assert result == ScreeningFailed(FailureReason.ERROR)


async def test_missing_answer_fails_as_error() -> None:
    answers = {**QUIET_ANSWERS}
    del answers["intent"]
    result = await screen_user_message(FakeJev(answers), [], "message")
    assert result == ScreeningFailed(FailureReason.ERROR)


async def test_failure_log_leaves_out_the_error_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    error = TypeSafeAPIError(
        422, {"message": "echoed: I want to die"}, httpx2.Headers()
    )
    with caplog.at_level(logging.WARNING):
        await screen_user_message(FakeJev(error=error), [], "I want to die")
    assert "TypeSafeAPIError" in caplog.text
    assert "die" not in caplog.text


async def test_cancellation_is_not_swallowed() -> None:
    task = asyncio.create_task(
        screen_user_message(FakeJev(delay_seconds=1.0), [], "message")
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_jev_reads_the_recent_conversation_and_the_latest_message() -> None:
    jev = FakeJev()
    history = [Turn("user", "hi"), Turn("assistant", "Hello, how are you?")]
    await screen_user_message(jev, history, "not great")
    assert jev.states == [
        "CONVERSATION SO FAR:\n"
        "USER: hi\n"
        "ASSISTANT: Hello, how are you?\n\n"
        "LATEST USER MESSAGE:\n"
        "not great"
    ]


def test_first_message_has_an_empty_conversation() -> None:
    state = user_check_state([], "hello")
    assert state == "CONVERSATION SO FAR: (none)\n\nLATEST USER MESSAGE:\nhello"


def test_transcript_keeps_only_the_most_recent_turns() -> None:
    history = [Turn("user", f"message {index}") for index in range(WINDOW_TURNS + 2)]
    state = user_check_state(history, "latest")
    assert "message 1\n" not in state
    assert "USER: message 2\n" in state
    assert state.count("USER: message") == WINDOW_TURNS


def test_conversation_turns_keep_only_what_was_said() -> None:
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[SystemPromptPart("instructions"), UserPromptPart("I feel low")]
        ),
        ModelResponse(
            parts=[
                ToolCallPart("recall_memory", {}),
                TextPart("I'm "),
                TextPart("here."),
            ]
        ),
        ModelRequest(parts=[ToolReturnPart("recall_memory", "nothing")]),
        ModelResponse(parts=[ToolCallPart("recall_memory", {})]),
        ModelRequest(
            parts=[
                UserPromptPart(["look at this", ImageUrl("https://example.com/a.png")])
            ]
        ),
        ModelRequest(parts=[UserPromptPart([ImageUrl("https://example.com/b.png")])]),
        ModelRequest(
            parts=[UserPromptPart([TextContent("first"), "second", CachePoint()])]
        ),
    ]
    assert conversation_turns(messages) == [
        Turn("user", "I feel low"),
        Turn("assistant", "I'm here."),
        Turn("user", "look at this"),
        Turn("user", "first\nsecond"),
    ]
