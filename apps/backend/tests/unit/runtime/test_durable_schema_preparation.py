"""Every durable-backend implementation exposes schema preparation.

The runtime prepares each owned backend at startup, so a backend missing
``ensure_schema`` would only fail on the path that backend serves. These
tests assert the hook exists everywhere and that ephemeral backends keep
it credential-free.
"""

from __future__ import annotations

import pytest

from agent.audit.crisis_log import InMemoryCrisisLogBackend, NullCrisisLogBackend
from agent.audit.postgres_crisis_log import PostgresCrisisLogBackend
from agent.feedback.postgres_session_feedback import PostgresSessionFeedbackBackend
from agent.feedback.session_feedback import (
    InMemorySessionFeedbackBackend,
    NullSessionFeedbackBackend,
)
from agent.memory.store import OpenCouchMemoryStore
from agent.memory.store.postgres import PostgresMemoryStore

pytestmark = pytest.mark.asyncio

_UNREACHABLE_DSN = "postgresql://unused:unused@127.0.0.1:1/unused"


async def test_every_backend_implementation_exposes_ensure_schema() -> None:
    """Durable and ephemeral backends alike implement the preparation hook."""

    backends = [
        OpenCouchMemoryStore,
        PostgresMemoryStore,
        InMemoryCrisisLogBackend,
        NullCrisisLogBackend,
        PostgresCrisisLogBackend,
        InMemorySessionFeedbackBackend,
        NullSessionFeedbackBackend,
        PostgresSessionFeedbackBackend,
    ]

    missing = [
        backend.__name__
        for backend in backends
        if not hasattr(backend, "ensure_schema")
    ]
    assert missing == []


async def test_ephemeral_backends_prepare_without_connecting() -> None:
    """In-memory and null backends prepare without touching a network."""

    memory_store = OpenCouchMemoryStore()
    crisis_backend = InMemoryCrisisLogBackend()
    feedback_backend = InMemorySessionFeedbackBackend()
    null_crisis = NullCrisisLogBackend()
    null_feedback = NullSessionFeedbackBackend()
    try:
        await memory_store.ensure_schema()
        await crisis_backend.ensure_schema()
        await feedback_backend.ensure_schema()
        await null_crisis.ensure_schema()
        await null_feedback.ensure_schema()

        # Preparation must leave the backends usable rather than consuming them.
        assert await memory_store.arecord_count(("owner", "semantic")) == 0
    finally:
        await memory_store.aclose()
        await crisis_backend.aclose()
        await feedback_backend.aclose()


async def test_ephemeral_backends_reject_preparation_after_close() -> None:
    """Preparing a closed ephemeral backend fails rather than silently passing."""

    memory_store = OpenCouchMemoryStore()
    await memory_store.aclose()
    with pytest.raises(RuntimeError):
        await memory_store.ensure_schema()

    crisis_backend = InMemoryCrisisLogBackend()
    await crisis_backend.aclose()
    with pytest.raises(RuntimeError):
        await crisis_backend.ensure_schema()


async def test_durable_backends_surface_connection_failures_at_preparation() -> None:
    """Preparation reports an unreachable database instead of deferring it.

    Surfacing the failure at startup is the point of the hook: the
    alternative is discovering it inside a bounded safety-capture timeout.
    """

    store = PostgresMemoryStore(_UNREACHABLE_DSN)
    try:
        with pytest.raises(Exception):
            await store.ensure_schema()
    finally:
        await store.aclose()
