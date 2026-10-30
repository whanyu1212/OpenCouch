"""The check-in is a server offer, never a client instruction to lower risk."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta, tzinfo
from typing import Any, Self

import pytest
from fastapi.testclient import TestClient
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from opencouch_agent.api.identity import LOCAL_USER_ID
from opencouch_agent.safety.follow_up import (
    CHECK_IN_GUIDANCE,
    CHECK_IN_TEXT,
    RESOURCE_PIN_TIME,
    RETURN_AFTER,
    RETURN_CHECK_IN,
)
from opencouch_agent.safety.policy import RiskLevel
from opencouch_agent.safety.risk import RiskState, UnscreenedTurn
from opencouch_agent.sessions import InMemorySessionStore
from tests.ag_ui_requests import (
    event_types,
    reply_text,
    run_input,
    state_snapshots,
    user_message,
)
from tests.apps import build_app
from tests.fake_jev import ScriptedJev

AT = datetime(2026, 10, 30, 12, tzinfo=UTC)
READY = RiskState(
    level=RiskLevel.CRISIS,
    peak=RiskLevel.CRISIS,
    score=0.1,
    calm_turns=4,
    last_risk_at=AT - timedelta(minutes=15),
    last_turn_at=AT,
    crisis_types=frozenset({"suicide_self_harm"}),
    crisis_resources=("sg-sos", "sg-emergency"),
)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[datetime]:
    times = [AT]

    class Clock(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> Self:
            return cls.fromtimestamp(times[0].timestamp(), UTC)

    monkeypatch.setattr("opencouch_agent.turn.datetime", Clock)
    monkeypatch.setattr("opencouch_agent.api.history.datetime", Clock)
    return times


async def _seed(store: InMemorySessionStore, risk: RiskState = READY) -> None:
    assert await store.claim_thread("thread-1", LOCAL_USER_ID)
    assert await store.claim_turn("thread-1", "seed", timedelta(minutes=1))
    messages = [ModelRequest(parts=[UserPromptPart("earlier turn")])]
    assert await store.record_user_turn("thread-1", "seed", messages, risk)
    await store.release_turn("thread-1", "seed")


def _tap(
    offer: str, answer: str = "safe", text: str = CHECK_IN_TEXT["safe"]
) -> dict[str, Any]:
    request = run_input(user_message(text))
    request["forwardedProps"] = {
        "crisis_check_in": {"offer_id": offer, "answer": answer}
    }
    return request


async def test_offer_is_recorded_before_reply_and_safe_tap_steps_down_with_a_pin(
    clock: list[datetime],
) -> None:
    store = InMemorySessionStore()
    await _seed(store)
    seen: list[str] = []

    async def reply(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert (await store.load("thread-1")).risk.check_in_offer is not None
        seen.append(info.instructions or "")
        return ModelResponse(parts=[TextPart("Tap the answer that fits.")])

    jev = ScriptedJev()
    client = TestClient(
        build_app(
            store,
            TestModel(custom_output_text="We can keep talking."),
            crisis=FunctionModel(reply),
            jev=jev,
        )
    )
    response = client.post(
        "/api/agent", json=run_input(user_message("My sister is here."))
    )
    offer = state_snapshots(response.text)[0]["crisis_check_in"]
    assert offer
    assert CHECK_IN_GUIDANCE in seen[0]
    assert (
        client.get("/api/threads/thread-1/messages").json()["crisis_check_in"] == offer
    )

    response = client.post("/api/agent", json=_tap(offer))
    risk = (await store.load("thread-1")).risk
    assert risk.level is RiskLevel.ELEVATED
    assert risk.check_in_offer is None
    assert risk.resources_pinned_until == AT + RESOURCE_PIN_TIME
    assert jev.screened[-1] == CHECK_IN_TEXT["safe"]
    assert reply_text(response) == "We can keep talking."
    snapshot = state_snapshots(response.text)[0]
    assert snapshot["crisis_resources"] == ["sg-sos", "sg-emergency"]
    assert snapshot["get_help_resources"] == snapshot["crisis_resources"]
    assert snapshot["crisis_check_in"] is None
    types = event_types(response.text)
    assert types.index("STATE_SNAPSHOT") < types.index("TEXT_MESSAGE_START")

    clock[0] += timedelta(hours=1)
    client.post("/api/agent", json=run_input(user_message("Can we talk?")))
    assert (
        await store.load("thread-1")
    ).risk.resources_pinned_until == risk.resources_pinned_until
    clock[0] = AT + RESOURCE_PIN_TIME
    history = client.get("/api/threads/thread-1/messages").json()
    assert history["crisis_resources"] == []
    assert history["get_help_resources"] == ["sg-sos", "sg-emergency"]
    response = client.post("/api/agent", json=run_input(user_message("Still here")))
    assert state_snapshots(response.text)[0]["crisis_resources"] == []


@pytest.mark.parametrize(
    "kind", ["typed", "forged", "stale", "wrong_text", "too_early", "client_state"]
)
async def test_only_the_matching_server_offer_can_step_down(
    kind: str, clock: list[datetime]
) -> None:
    store = InMemorySessionStore()
    initial = replace(READY, check_in_offer="offered")
    if kind == "too_early":
        initial = replace(initial, score=1.0, calm_turns=0)
    await _seed(store, initial)
    client = TestClient(build_app(store, TestModel()))
    request = _tap("offered")
    if kind == "typed":
        request["forwardedProps"] = {}
    elif kind == "forged":
        request = _tap("invented")
    elif kind == "stale":
        client.post("/api/agent", json=run_input(user_message("Another tab sent this")))
    elif kind == "wrong_text":
        request = _tap("offered", text="Something else")
    elif kind == "client_state":
        request["forwardedProps"] = {}
        request["state"] = {
            "crisis_check_in": "offered",
            "check_in": "safe",
            "risk": {"level": "none"},
        }
    response = client.post("/api/agent", json=request)
    assert response.status_code == 200
    assert (await store.load("thread-1")).risk.level is RiskLevel.CRISIS


@pytest.mark.parametrize("answer", ["not_sure", "need_help"])
@pytest.mark.parametrize("failed", [False, True])
async def test_other_taps_keep_crisis_and_help_now_prioritises_emergency(
    answer: str, failed: bool, clock: list[datetime]
) -> None:
    store = InMemorySessionStore()
    await _seed(store, replace(READY, check_in_offer="offer"))
    text = "I'm not sure" if answer == "not_sure" else "I need help now"
    jev = ScriptedJev(failing=frozenset({text}) if failed else frozenset())
    client = TestClient(build_app(store, TestModel(), jev=jev))
    response = client.post("/api/agent", json=_tap("offer", answer, text))
    risk = (await store.load("thread-1")).risk
    assert risk.level is RiskLevel.CRISIS
    assert risk.check_in_offer != "offer"
    assert jev.screened == [text]
    if answer == "need_help":
        assert (
            state_snapshots(response.text)[0]["crisis_resources"][0] == "sg-emergency"
        )
        assert risk.last_risk_at == AT
    if failed:
        assert risk.unscreened
        assert risk.check_in_offer is None


@pytest.mark.parametrize(
    "reason",
    ["current_failed", "missed_failed", "missed_risky", "current_risky", "abandoned"],
)
async def test_unknown_or_new_risk_cannot_be_cleared_by_a_safe_tap(
    reason: str, clock: list[datetime]
) -> None:
    store = InMemorySessionStore()
    risk = replace(READY, check_in_offer="offer")
    readings = {}
    failing: set[str] = set()
    if reason == "current_failed":
        failing.add(CHECK_IN_TEXT["safe"])
    elif reason == "current_risky":
        readings[CHECK_IN_TEXT["safe"]] = {"intent": 0.95, "latest_adds_risk": 0.9}
    else:
        marker = UnscreenedTurn(
            start=0, count=1, failed_rechecks=2 if reason == "abandoned" else 0
        )
        risk = replace(risk, unscreened=(marker,))
        if reason == "missed_risky":
            readings["earlier turn"] = {"intent": 0.95, "latest_adds_risk": 0.9}
        else:
            failing.add("earlier turn")
    await _seed(store, risk)
    client = TestClient(
        build_app(
            store, TestModel(), jev=ScriptedJev(readings, failing=frozenset(failing))
        )
    )
    response = client.post("/api/agent", json=_tap("offer"))
    assert response.status_code == 200
    assert (await store.load("thread-1")).risk.level is RiskLevel.CRISIS


@pytest.mark.parametrize("fails", [False, True])
async def test_return_after_silence_starts_with_check_in_even_on_model_failure(
    fails: bool, clock: list[datetime]
) -> None:
    store = InMemorySessionStore()
    await _seed(store, replace(READY, check_in_offer="old"))
    clock[0] += RETURN_AFTER
    seen: list[str] = []

    def reply(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info.instructions or "")
        if fails:
            raise RuntimeError("unavailable")
        return ModelResponse(parts=[TextPart("We can keep talking.")])

    client = TestClient(build_app(store, TestModel(), crisis=FunctionModel(reply)))
    assert (
        client.get("/api/threads/thread-1/messages").json()["crisis_check_in"] is None
    )
    response = client.post("/api/agent", json=_tap("old"))
    assert reply_text(response).startswith(f"{RETURN_CHECK_IN}\n\n")
    assert reply_text(response).count("?") == 1
    assert "without repeating" in seen[0]
    assert (await store.load("thread-1")).risk.level is RiskLevel.CRISIS
    assert state_snapshots(response.text)[0]["crisis_check_in"] not in (None, "old")
    history = client.get("/api/threads/thread-1/messages").json()
    assert history["messages"][-1]["content"] == reply_text(response)


@pytest.mark.parametrize(
    "submission",
    [
        {"answer": "safe"},
        {"offer_id": "x", "answer": "yes"},
        "safe",
        {"offer_id": "x" * 65, "answer": "safe"},
    ],
)
def test_malformed_tap_is_rejected_before_screening(submission: object) -> None:
    jev = ScriptedJev()
    client = TestClient(build_app(InMemorySessionStore(), TestModel(), jev=jev))
    request = run_input(user_message(CHECK_IN_TEXT["safe"]))
    request["forwardedProps"] = {"crisis_check_in": submission}
    assert client.post("/api/agent", json=request).status_code == 422
    assert jev.screened == []


@pytest.mark.parametrize("fails", [False, True])
async def test_help_now_after_silence_keeps_emergency_guidance_first(
    fails: bool, clock: list[datetime]
) -> None:
    store = InMemorySessionStore()
    await _seed(store, replace(READY, check_in_offer="old"))
    clock[0] += RETURN_AFTER

    def reply(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert "## Right now" in (info.instructions or "")
        assert "server will begin" not in (info.instructions or "")
        if fails:
            raise RuntimeError("unavailable")
        return ModelResponse(parts=[TextPart("Call 995 now. Are you alone?")])

    client = TestClient(build_app(store, TestModel(), crisis=FunctionModel(reply)))
    response = client.post(
        "/api/agent", json=_tap("old", "need_help", "I need help now")
    )
    text = reply_text(response)
    assert not text.startswith(RETURN_CHECK_IN)
    assert "995" in text.split(".")[0]
    assert state_snapshots(response.text)[0]["crisis_resources"][0] == "sg-emergency"
    assert (await store.load("thread-1")).risk.level is RiskLevel.CRISIS
