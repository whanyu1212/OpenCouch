"""Incognito runtimes reject dependencies that could persist or call out.

Runtime-owned backend selection picks ephemeral backends for incognito, but
injected dependencies take precedence over that choice. These tests pin the
fail-closed contract: an injected dependency must declare itself ephemeral,
so an implementation the runtime has never seen is rejected rather than
assumed safe.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent.audit.crisis_log import InMemoryCrisisLogBackend, NullCrisisLogBackend
from agent.audit.postgres_crisis_log import PostgresCrisisLogBackend
from agent.feedback.postgres_session_feedback import PostgresSessionFeedbackBackend
from agent.feedback.session_feedback import (
    InMemorySessionFeedbackBackend,
    NullSessionFeedbackBackend,
)
from agent.memory.modes import MemoryMode, is_ephemeral_capable
from agent.memory.providers.embeddings import NullEmbeddingProvider
from agent.memory.store import OpenCouchMemoryStore
from agent.memory.store.postgres import PostgresMemoryStore
from agent.runtime.backends import (
    create_crisis_log_backend,
    create_embedding_provider,
    create_memory_store,
    create_session_feedback_backend,
)
from agent.runtime.resources import build_runtime_resources

_DURABLE_DSN = "postgresql://user:password@db.example.invalid:5432/prod"


class _NetworkEmbeddingProvider:
    """Stand-in for a provider that calls a remote embedding service."""

    model_name = "remote-model"

    async def aembed(
        self, texts: list[str], task_type: str | None = None
    ) -> list[None]:
        return [None] * len(texts)

    async def awarmup(self) -> None:
        return None


class _OptedOutStore(OpenCouchMemoryStore):
    """Ephemeral by implementation but explicitly declining incognito use."""

    supports_incognito: bool = False


def test_in_repo_ephemeral_backends_declare_the_capability() -> None:
    """Every ephemeral implementation opts in; durable ones do not."""

    ephemeral = [
        OpenCouchMemoryStore(),
        InMemoryCrisisLogBackend(),
        NullCrisisLogBackend(),
        InMemorySessionFeedbackBackend(),
        NullSessionFeedbackBackend(),
        NullEmbeddingProvider(),
    ]
    durable = [
        PostgresMemoryStore(_DURABLE_DSN),
        PostgresCrisisLogBackend(_DURABLE_DSN),
        PostgresSessionFeedbackBackend(_DURABLE_DSN),
        _NetworkEmbeddingProvider(),
    ]

    assert [
        type(dep).__name__ for dep in ephemeral if not is_ephemeral_capable(dep)
    ] == []
    assert [type(dep).__name__ for dep in durable if is_ephemeral_capable(dep)] == []


def test_incognito_rejects_durable_memory_store() -> None:
    with pytest.raises(ValueError, match="Incognito runtimes cannot use"):
        create_memory_store(
            memory_store=PostgresMemoryStore(_DURABLE_DSN),
            memory_backend="memory",
            memory_database_url=None,
            memory_mode=MemoryMode.INCOGNITO,
        )


def test_incognito_rejects_durable_crisis_log_backend() -> None:
    with pytest.raises(ValueError, match="crisis-log backend"):
        create_crisis_log_backend(
            crisis_log_backend=PostgresCrisisLogBackend(_DURABLE_DSN),
            crisis_log_persistence_backend="memory",
            crisis_log_database_url=None,
            memory_mode=MemoryMode.INCOGNITO,
        )


def test_incognito_rejects_durable_session_feedback_backend() -> None:
    with pytest.raises(ValueError, match="session-feedback backend"):
        create_session_feedback_backend(
            session_feedback_backend=PostgresSessionFeedbackBackend(_DURABLE_DSN),
            session_feedback_persistence_backend="memory",
            session_feedback_database_url=None,
            memory_mode=MemoryMode.INCOGNITO,
        )


def test_incognito_rejects_network_embedding_provider() -> None:
    with pytest.raises(ValueError, match="embedding provider"):
        create_embedding_provider(
            memory_mode=MemoryMode.INCOGNITO,
            embedding_provider=_NetworkEmbeddingProvider(),
        )


def test_incognito_rejects_unmarked_dependency() -> None:
    """An unrecognized implementation is rejected rather than assumed safe."""

    class _UnknownStore:
        """Implements nothing the runtime recognizes as ephemeral."""

    with pytest.raises(ValueError, match="_UnknownStore"):
        create_memory_store(
            memory_store=_UnknownStore(),  # type: ignore[arg-type]
            memory_backend="memory",
            memory_database_url=None,
            memory_mode=MemoryMode.INCOGNITO,
        )


def test_incognito_rejects_explicitly_opted_out_dependency() -> None:
    """Setting the marker to False is as meaningful as never setting it."""

    with pytest.raises(ValueError, match="_OptedOutStore"):
        create_memory_store(
            memory_store=_OptedOutStore(),
            memory_backend="memory",
            memory_database_url=None,
            memory_mode=MemoryMode.INCOGNITO,
        )


def test_instance_attribute_cannot_grant_the_capability() -> None:
    """The declaration must be on the class, not stamped onto an instance.

    Reading the marker off the instance would let a stray
    ``store.supports_incognito = True`` anywhere in caller code silently
    disable a privacy control, with no class-level declaration to review.
    """

    store = PostgresMemoryStore(_DURABLE_DSN)
    store.supports_incognito = True  # type: ignore[attr-defined]

    assert not is_ephemeral_capable(store)
    with pytest.raises(ValueError, match="PostgresMemoryStore"):
        create_memory_store(
            memory_store=store,
            memory_backend="memory",
            memory_database_url=None,
            memory_mode=MemoryMode.INCOGNITO,
        )


@pytest.mark.parametrize("truthy_value", [1, "yes", ["non-empty"]])
def test_truthy_marker_values_do_not_grant_the_capability(
    truthy_value: Any,
) -> None:
    """Only exactly ``True`` opts in, so a mistyped value cannot grant it."""

    class _TruthyStore(PostgresMemoryStore):
        supports_incognito = truthy_value

    with pytest.raises(ValueError, match="_TruthyStore"):
        create_memory_store(
            memory_store=_TruthyStore(_DURABLE_DSN),
            memory_backend="memory",
            memory_database_url=None,
            memory_mode=MemoryMode.INCOGNITO,
        )


def test_incognito_accepts_declared_ephemeral_dependencies() -> None:
    """Legitimate in-memory overrides keep working under incognito."""

    store = create_memory_store(
        memory_store=OpenCouchMemoryStore(),
        memory_backend="memory",
        memory_database_url=None,
        memory_mode=MemoryMode.INCOGNITO,
    )
    crisis_backend = create_crisis_log_backend(
        crisis_log_backend=InMemoryCrisisLogBackend(),
        crisis_log_persistence_backend="memory",
        crisis_log_database_url=None,
        memory_mode=MemoryMode.INCOGNITO,
    )
    feedback_backend = create_session_feedback_backend(
        session_feedback_backend=InMemorySessionFeedbackBackend(),
        session_feedback_persistence_backend="memory",
        session_feedback_database_url=None,
        memory_mode=MemoryMode.INCOGNITO,
    )

    assert isinstance(store, OpenCouchMemoryStore)
    assert isinstance(crisis_backend, InMemoryCrisisLogBackend)
    assert isinstance(feedback_backend, InMemorySessionFeedbackBackend)


@pytest.mark.parametrize("memory_mode", [MemoryMode.LOCAL, MemoryMode.SYNCED])
def test_persistent_modes_still_accept_durable_overrides(
    memory_mode: MemoryMode,
) -> None:
    """Only incognito is restricted; durable modes are unaffected."""

    store = create_memory_store(
        memory_store=PostgresMemoryStore(_DURABLE_DSN),
        memory_backend="postgres",
        memory_database_url=_DURABLE_DSN,
        memory_mode=memory_mode,
    )

    assert isinstance(store, PostgresMemoryStore)


def _build_incognito_resources(**overrides: Any) -> Any:
    """Build incognito runtime resources with the given dependency overrides."""

    return build_runtime_resources(
        memory_mode=MemoryMode.INCOGNITO,
        text_session_sqlite_path=None,
        thread_persistence_backend="memory",
        thread_database_url=None,
        text_session_backend="disabled",
        text_session_database_url=None,
        text_session_create_tables=True,
        text_session_history_limit=None,
        memory_store=overrides.get("memory_store"),
        memory_backend="postgres",
        memory_database_url=None,
        crisis_log_backend=overrides.get("crisis_log_backend"),
        crisis_log_persistence_backend="postgres",
        crisis_log_database_url=None,
        session_feedback_backend=overrides.get("session_feedback_backend"),
        session_feedback_persistence_backend="postgres",
        session_feedback_database_url=None,
        embedding_provider=overrides.get("embedding_provider"),
    )


@pytest.mark.parametrize(
    ("override_name", "dependency_factory"),
    [
        ("memory_store", lambda: PostgresMemoryStore(_DURABLE_DSN)),
        ("crisis_log_backend", lambda: PostgresCrisisLogBackend(_DURABLE_DSN)),
        (
            "session_feedback_backend",
            lambda: PostgresSessionFeedbackBackend(_DURABLE_DSN),
        ),
        ("embedding_provider", _NetworkEmbeddingProvider),
    ],
)
def test_incognito_runtime_construction_rejects_each_override_class(
    override_name: str,
    dependency_factory: Any,
) -> None:
    """Every dependency override class is covered end to end."""

    with pytest.raises(ValueError, match="Incognito runtimes cannot use"):
        _build_incognito_resources(**{override_name: dependency_factory()})


def test_incognito_runtime_never_holds_a_durable_dependency() -> None:
    """A default incognito runtime resolves to ephemeral backends only."""

    resources = _build_incognito_resources()

    durable_types = (
        PostgresMemoryStore,
        PostgresCrisisLogBackend,
        PostgresSessionFeedbackBackend,
    )
    for dependency in (
        resources.memory_store,
        resources.crisis_log_backend,
        resources.session_feedback_backend,
        resources.embedding_provider,
    ):
        assert not isinstance(dependency, durable_types)
        assert is_ephemeral_capable(dependency)
