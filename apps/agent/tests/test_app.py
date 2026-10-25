import json

from fastapi.testclient import TestClient

from opencouch_agent.config import Settings
from opencouch_agent.main import create_app
from tests.ag_ui_requests import reply_text, run_input, user_message


def test_health(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_agent_streams_ag_ui_events(client: TestClient) -> None:
    response = client.post(
        "/api/agent",
        json=run_input(user_message("Rough day today.")),
        headers={"accept": "text/event-stream"},
    )

    assert response.status_code == 200
    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    types = [event["type"] for event in events]
    assert types[0] == "RUN_STARTED"
    assert types[-1] == "RUN_FINISHED"
    assert reply_text(response) == "I'm here with you."


def test_a_local_run_with_the_fakes_serves_turns() -> None:
    app = create_app(settings=Settings(model="test", fake_jev=True, fake_auth=True))

    with TestClient(app) as client:
        response = client.post("/api/agent", json=run_input(user_message("hi")))

    assert response.status_code == 200
    assert "RUN_FINISHED" in response.text


def test_crisis_resources_are_public_with_links_and_no_reviewer(
    client: TestClient,
) -> None:
    response = client.get("/api/crisis-resources")

    assert response.status_code == 200
    resources = response.json()["resources"]
    assert [r["key"] for r in resources] == [
        "sg-sos",
        "sg-mindline",
        "sg-navh",
        "sg-emergency",
    ]
    assert resources[0]["channels"][0]["url"].startswith("https://wa.me/6591511767")
    assert all("verified_by" not in r for r in resources)
