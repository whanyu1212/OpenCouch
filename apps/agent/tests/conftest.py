import pytest
from fastapi.testclient import TestClient
from pydantic_ai.models.test import TestModel

from opencouch_agent.sessions import InMemorySessionStore
from tests.apps import build_app


@pytest.fixture
def store() -> InMemorySessionStore:
    return InMemorySessionStore()


@pytest.fixture
def client(store: InMemorySessionStore) -> TestClient:
    return TestClient(
        build_app(store, TestModel(custom_output_text="I'm here with you."))
    )
