import json

from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_agent_streams_ag_ui_events(client: TestClient) -> None:
    run_input = {
        "threadId": "thread-1",
        "runId": "run-1",
        "state": {},
        "messages": [{"id": "msg-1", "role": "user", "content": "Rough day today."}],
        "tools": [],
        "context": [],
        "forwardedProps": {},
    }

    response = client.post(
        "/api/agent", json=run_input, headers={"accept": "text/event-stream"}
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
    text = "".join(e["delta"] for e in events if e["type"] == "TEXT_MESSAGE_CONTENT")
    assert text == "I'm here with you."
