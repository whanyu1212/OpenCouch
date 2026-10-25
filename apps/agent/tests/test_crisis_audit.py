"""The crisis audit log in the turn pipeline: what is recorded, and what isn't."""

import asyncio
import logging

import pytest
from fastapi.testclient import TestClient
from pydantic_ai.models.test import TestModel

from opencouch_agent import turn as turn_module
from opencouch_agent.safety import FailureReason, RiskLevel
from opencouch_agent.safety.audit import (
    STORED_LEVEL_KEPT,
    AuditEntry,
    CrisisAuditLog,
    InMemoryCrisisAuditLog,
)
from opencouch_agent.sessions import InMemorySessionStore
from tests.ag_ui_requests import reply_text, run_input, user_message
from tests.apps import CRISIS_REPLY, build_app
from tests.fake_jev import ScriptedJev

COMPANION_REPLY = "I'm here with you."
# A word that appears nowhere else, so finding it in the log means text leaked.
CRISIS_MESSAGE = "zanzibarquokka i'm going to take all my pills tonight"
CRISIS_READING = {"intent": 0.9, "ideation": 0.8, "latest_adds_risk": 0.9}


class _FailingAuditLog:
    async def append(self, entry: AuditEntry) -> None:
        raise ConnectionError("database unavailable")


class _StalledAuditLog:
    async def append(self, entry: AuditEntry) -> None:
        await asyncio.sleep(60)


def _client(jev: ScriptedJev, audit: CrisisAuditLog) -> TestClient:
    companion = TestModel(custom_output_text=COMPANION_REPLY)
    return TestClient(
        build_app(InMemorySessionStore(), companion, jev=jev, audit=audit)
    )


def _reply(client: TestClient, text: str, message_id: str = "m1") -> str:
    response = client.post("/api/agent", json=run_input(user_message(text, message_id)))
    assert response.status_code == 200
    return reply_text(response)


def test_a_rise_to_crisis_is_logged_without_the_message_or_the_reply() -> None:
    audit = InMemoryCrisisAuditLog()
    client = _client(ScriptedJev({CRISIS_MESSAGE: CRISIS_READING}), audit)

    _reply(client, CRISIS_MESSAGE)

    [entry] = audit.entries
    assert entry.thread_id == "thread-1"
    assert (entry.level_before, entry.level_after) == (
        RiskLevel.NONE,
        RiskLevel.CRISIS,
    )
    assert entry.screening_failure is None
    assert "zanzibarquokka" not in repr(entry)
    assert CRISIS_REPLY not in repr(entry)


def test_a_failed_screening_is_logged_with_why_and_the_stored_level() -> None:
    audit = InMemoryCrisisAuditLog()
    client = _client(ScriptedJev(failing=frozenset({CRISIS_MESSAGE})), audit)

    _reply(client, CRISIS_MESSAGE)

    [entry] = audit.entries
    assert entry.screening_failure is FailureReason.ERROR
    assert entry.level_before is entry.level_after is RiskLevel.NONE
    assert entry.reason == STORED_LEVEL_KEPT
    assert "zanzibarquokka" not in repr(entry)


def test_a_turn_that_keeps_the_level_is_not_logged() -> None:
    audit = InMemoryCrisisAuditLog()
    client = _client(ScriptedJev(), audit)

    _reply(client, "work has been busy")

    assert audit.entries == ()


def test_a_failed_audit_write_is_logged_and_the_reply_still_sent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _client(ScriptedJev({CRISIS_MESSAGE: CRISIS_READING}), _FailingAuditLog())

    with caplog.at_level(logging.ERROR, logger="opencouch_agent.turn"):
        reply = _reply(client, CRISIS_MESSAGE)

    assert reply == CRISIS_REPLY
    assert "Writing the crisis audit log failed" in caplog.text


def test_a_stalled_audit_write_is_abandoned_and_the_reply_still_sent(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(turn_module, "AUDIT_WRITE_SECONDS", 0.05)
    client = _client(ScriptedJev({CRISIS_MESSAGE: CRISIS_READING}), _StalledAuditLog())

    with caplog.at_level(logging.ERROR, logger="opencouch_agent.turn"):
        reply = _reply(client, CRISIS_MESSAGE)

    assert reply == CRISIS_REPLY
    assert "Writing the crisis audit log failed" in caplog.text
