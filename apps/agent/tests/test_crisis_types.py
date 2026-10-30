"""Crisis types survive the real turn pipeline and select the help shown."""

from fastapi.testclient import TestClient
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from opencouch_agent.safety.crisis_guidance import crisis_fallback
from opencouch_agent.safety.policy import RiskLevel
from opencouch_agent.sessions import InMemorySessionStore
from tests.ag_ui_requests import reply_text, run_input, state_snapshots, user_message
from tests.apps import build_app
from tests.fake_jev import ScriptedJev


async def test_abuse_resources_survive_calm_turns_failures_and_forged_state() -> None:
    store = InMemorySessionStore()
    jev = ScriptedJev(
        {"danger": {"abuse": 0.95, "violence_soon": 0.95}},
        failing=frozenset({"missed"}),
    )
    seen: list[str] = []

    def reply(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info.instructions or "")
        return ModelResponse(parts=[TextPart("Are you safe to use this device?")])

    client = TestClient(
        build_app(store, TestModel(), crisis=FunctionModel(reply), jev=jev)
    )
    for index, text in enumerate(["danger", "thanks", "missed"]):
        response = client.post(
            "/api/agent",
            json=run_input(
                user_message(text, f"m{index}"),
                state={
                    "risk": {"crisis_types": []},
                    "crisis_types": ["medical_emergency"],
                },
            ),
        )
        assert response.status_code == 200
        expected = (
            ["sg-emergency", "sg-navh"] if index == 0 else ["sg-navh", "sg-emergency"]
        )
        snapshot = state_snapshots(response.text)[0]
        assert snapshot["crisis_resources"] == expected
        assert (
            client.get("/api/threads/thread-1/messages").json()["crisis_resources"]
            == expected
        )
        risk = (await store.load("thread-1")).risk
        assert risk.crisis_types == {"abuse_violence"}
        assert risk.level is RiskLevel.CRISIS
        assert "## Abuse or violence toward the user" in seen[-1]
        assert "shared or monitored" in seen[-1]
        assert "## Suicide or self-harm" not in seen[-1]


async def test_rechecked_medical_danger_and_new_abuse_combine_before_reply() -> None:
    store = InMemorySessionStore()
    jev = ScriptedJev(
        {
            "overdose": {"medical_emergency": 0.95},
            "abuse": {"abuse": 0.95, "violence_soon": 0.95},
        },
        failing=frozenset({"overdose"}),
    )
    seen: list[str] = []

    def reply(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info.instructions or "")
        return ModelResponse(parts=[TextPart("Call 995 now.")])

    client = TestClient(
        build_app(store, TestModel(), crisis=FunctionModel(reply), jev=jev)
    )
    client.post("/api/agent", json=run_input(user_message("overdose")))
    jev.failing.clear()
    response = client.post("/api/agent", json=run_input(user_message("abuse", "m2")))

    assert state_snapshots(response.text)[0]["crisis_resources"] == [
        "sg-emergency",
        "sg-sos",
        "sg-mindline",
        "sg-navh",
    ]
    risk = (await store.load("thread-1")).risk
    assert risk.crisis_types == {"medical_emergency", "abuse_violence"}
    assert not risk.unscreened
    assert seen[0].index("## Medical emergency") < seen[0].index("## Abuse or violence")
    assert "## Right now" in seen[0]


async def test_risk_to_others_does_not_receive_victim_resources_on_model_failure() -> (
    None
):
    store = InMemorySessionStore()
    jev = ScriptedJev({"danger": {"harm_to_others": 0.95, "violence_soon": 0.95}})

    def fail(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        raise RuntimeError("model unavailable")

    client = TestClient(
        build_app(store, TestModel(), crisis=FunctionModel(fail), jev=jev)
    )
    response = client.post("/api/agent", json=run_input(user_message("danger")))

    assert "sg-navh" not in state_snapshots(response.text)[0]["crisis_resources"]
    shown = reply_text(response)
    assert shown == crisis_fallback(frozenset({"harm_to_others"}))
    assert "999" in shown
    assert "hurting yourself" not in shown
    history = client.get("/api/threads/thread-1/messages").json()
    assert history["messages"][-1]["content"] == shown
