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


def test_a_local_run_with_the_fake_jev_serves_turns() -> None:
    app = create_app(settings=Settings(model="test", fake_jev=True))

    with TestClient(app) as client:
        response = client.post("/api/agent", json=run_input(user_message("hi")))

    assert response.status_code == 200
    assert "RUN_FINISHED" in response.text
