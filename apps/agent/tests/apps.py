"""Builds the app for tests: test models, a scripted Jev, an in-memory store."""

from fastapi import FastAPI
from pydantic_ai.models import Model
from pydantic_ai.models.test import TestModel

from opencouch_agent.agent import build_companion_agent
from opencouch_agent.config import Settings
from opencouch_agent.main import create_app
from opencouch_agent.safety.crisis import build_crisis_agent
from opencouch_agent.sessions import SessionStore
from tests.fake_jev import ScriptedJev

CRISIS_REPLY = "Are you safe right now?"


def build_app(
    store: SessionStore,
    companion: Model,
    *,
    crisis: Model | None = None,
    jev: ScriptedJev | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """The app with a test model for each agent and a calm Jev by default."""
    return create_app(
        settings=settings or Settings(),
        agent=build_companion_agent(companion),
        crisis_agent=build_crisis_agent(
            crisis or TestModel(custom_output_text=CRISIS_REPLY)
        ),
        jev=jev or ScriptedJev(),
        store=store,
    )
