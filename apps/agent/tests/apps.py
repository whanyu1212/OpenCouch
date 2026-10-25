"""Builds the app for tests: test models, a scripted Jev, an in-memory store."""

from fastapi import FastAPI
from pydantic_ai.models import Model
from pydantic_ai.models.test import TestModel

from opencouch_agent.agent import build_companion_agent
from opencouch_agent.api.identity import FakeUserVerifier, UserVerifier
from opencouch_agent.beta.access_requests import AccessRequests
from opencouch_agent.beta.invites import BetaInvites
from opencouch_agent.config import Settings
from opencouch_agent.main import create_app
from opencouch_agent.safety.audit import CrisisAuditLog
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
    audit: CrisisAuditLog | None = None,
    verifier: UserVerifier | None = None,
    invites: BetaInvites | None = None,
    access_requests: AccessRequests | None = None,
) -> FastAPI:
    """The app with test models, a calm Jev, and any caller signed in and invited."""
    return create_app(
        settings=settings or Settings(),
        agent=build_companion_agent(companion),
        crisis_agent=build_crisis_agent(
            crisis or TestModel(custom_output_text=CRISIS_REPLY)
        ),
        jev=jev or ScriptedJev(),
        store=store,
        audit=audit,
        verifier=verifier or FakeUserVerifier(),
        invites=invites or BetaInvites(everyone=True),
        access_requests=access_requests,
    )
