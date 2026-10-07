import pytest
from fastapi.testclient import TestClient
from pydantic_ai.models.test import TestModel

from opencouch_agent.agent import build_companion_agent
from opencouch_agent.config import Settings
from opencouch_agent.main import create_app
from opencouch_agent.sessions import InMemorySessionStore


@pytest.fixture
def store() -> InMemorySessionStore:
    return InMemorySessionStore()


@pytest.fixture
def client(store: InMemorySessionStore) -> TestClient:
    agent = build_companion_agent(TestModel(custom_output_text="I'm here with you."))
    return TestClient(create_app(settings=Settings(), agent=agent, store=store))
