"""Turns Jev can't screen: who replies, what they're told, and the recheck."""

from collections.abc import AsyncIterator

from fastapi.testclient import TestClient
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel

from opencouch_agent.agent.companion import UNSCREENED_INSTRUCTIONS
from opencouch_agent.safety import FailureReason, RiskLevel
from opencouch_agent.safety.audit import InMemoryCrisisAuditLog
from opencouch_agent.safety.risk import UnscreenedTurn
from opencouch_agent.sessions import InMemorySessionStore
from tests.ag_ui_requests import reply_text, run_input, user_message
from tests.apps import CRISIS_REPLY, build_app
from tests.fake_jev import ScriptedJev

COMPANION_REPLY = "I'm here with you."
MISSED = "i'm going to take all my pills tonight"
CRISIS_READING = {"intent": 0.9, "ideation": 0.8, "latest_adds_risk": 0.9}


class _Companion:
    """A companion model that records the instructions of each run."""

    def __init__(self) -> None:
        self.instructions: list[str | None] = []

    async def reply(self, _: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        self.instructions.append(info.instructions)
        yield COMPANION_REPLY


def _client(
    store: InMemorySessionStore,
    jev: ScriptedJev,
    companion: _Companion,
    audit: InMemoryCrisisAuditLog | None = None,
) -> TestClient:
    model = FunctionModel(stream_function=companion.reply)
    return TestClient(build_app(store, model, jev=jev, audit=audit))


def _reply(client: TestClient, text: str, message_id: str) -> str:
    response = client.post("/api/agent", json=run_input(user_message(text, message_id)))
    assert response.status_code == 200
    return reply_text(response)


async def test_below_elevated_the_companion_answers_told_screening_did_not_run() -> (
    None
):
    store = InMemorySessionStore()
    companion = _Companion()
    client = _client(store, ScriptedJev(failing=frozenset({MISSED})), companion)

    reply = _reply(client, MISSED, "m1")

    assert reply == COMPANION_REPLY
    [instructions] = companion.instructions
    assert instructions is not None
    assert UNSCREENED_INSTRUCTIONS in instructions
    risk = (await store.load("thread-1")).risk
    assert risk.level is RiskLevel.NONE
    assert risk.unscreened == (UnscreenedTurn(start=0, count=1),)


async def test_a_screened_turn_gets_no_screening_note() -> None:
    companion = _Companion()
    client = _client(InMemorySessionStore(), ScriptedJev(), companion)

    _reply(client, "work has been busy", "m1")

    [instructions] = companion.instructions
    assert instructions is not None
    assert UNSCREENED_INSTRUCTIONS not in instructions


async def test_the_note_stays_while_a_missed_turn_waits_for_its_recheck() -> None:
    companion = _Companion()
    client = _client(
        InMemorySessionStore(), ScriptedJev(failing=frozenset({MISSED})), companion
    )
    _reply(client, MISSED, "m1")

    _reply(client, "anyway", "m2")

    assert len(companion.instructions) == 2
    assert all(
        UNSCREENED_INSTRUCTIONS in (instructions or "")
        for instructions in companion.instructions
    )


async def test_once_jev_is_back_a_missed_crisis_goes_to_the_crisis_responder() -> None:
    store = InMemorySessionStore()
    audit = InMemoryCrisisAuditLog()
    jev = ScriptedJev({MISSED: CRISIS_READING}, failing=frozenset({MISSED}))
    client = _client(store, jev, _Companion(), audit)
    _reply(client, MISSED, "m1")
    jev.failing.clear()

    reply = _reply(client, "ok", "m2")

    assert reply == CRISIS_REPLY
    risk = (await store.load("thread-1")).risk
    assert risk.level is RiskLevel.CRISIS
    assert risk.unscreened == ()
    failed, rechecked = audit.entries
    assert failed.screening_failure is FailureReason.ERROR
    assert rechecked.level_after is RiskLevel.CRISIS
    assert rechecked.reason.startswith("recheck of a missed turn: ")
