from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException

from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.audit.postgres_crisis_log import PostgresCrisisLogBackend
from agent.feedback.postgres_session_feedback import PostgresSessionFeedbackBackend
from agent.feedback.session_feedback import InMemorySessionFeedbackBackend
from agent.memory.modes import MemoryMode
from api import dependencies
from api.dependencies import (
    get_runtime,
    get_runtime_for_memory_mode,
    get_runtime_selection,
    parse_api_memory_mode,
)
from api.models import ApiMemoryMode
from config import Settings


def test_build_runtime_rejects_persistent_legacy_sqlite() -> None:
    with pytest.raises(ValueError, match="SQLite crisis-audit and session-feedback"):
        dependencies._build_runtime(
            memory_mode=MemoryMode.LOCAL,
            settings=Settings(
                persistence_backend="sqlite",
                allow_legacy_sqlite=True,
                text_session_backend="disabled",
            ),
            llm_client=None,
        )


def test_build_runtime_keeps_incognito_credential_free_under_legacy_setting() -> None:
    runtime = dependencies._build_runtime(
        memory_mode=MemoryMode.INCOGNITO,
        settings=Settings(
            persistence_backend="sqlite",
            allow_legacy_sqlite=True,
            text_session_backend="disabled",
        ),
        llm_client=None,
    )

    assert isinstance(runtime.crisis_log_backend, InMemoryCrisisLogBackend)
    assert isinstance(runtime.session_feedback_backend, InMemorySessionFeedbackBackend)


def test_build_runtime_uses_postgres_for_persistent_application_stores() -> None:
    runtime = dependencies._build_runtime(
        memory_mode=MemoryMode.LOCAL,
        settings=Settings(
            persistence_backend="postgres",
            memory_database_url="postgresql://unused/opencouch",
            text_session_backend="disabled",
        ),
        llm_client=None,
    )

    assert isinstance(runtime.crisis_log_backend, PostgresCrisisLogBackend)
    assert isinstance(runtime.session_feedback_backend, PostgresSessionFeedbackBackend)


def test_parse_api_memory_mode_defaults_when_unset() -> None:
    assert (
        parse_api_memory_mode(None, default=ApiMemoryMode.PERSISTENT)
        is ApiMemoryMode.PERSISTENT
    )


def test_parse_api_memory_mode_accepts_persistent() -> None:
    assert (
        parse_api_memory_mode("persistent", default=ApiMemoryMode.INCOGNITO)
        is ApiMemoryMode.PERSISTENT
    )


def test_parse_api_memory_mode_accepts_incognito() -> None:
    assert (
        parse_api_memory_mode("incognito", default=ApiMemoryMode.PERSISTENT)
        is ApiMemoryMode.INCOGNITO
    )


def test_parse_api_memory_mode_normalizes_whitespace_and_case() -> None:
    assert (
        parse_api_memory_mode("  InCoGnItO  ", default=ApiMemoryMode.PERSISTENT)
        is ApiMemoryMode.INCOGNITO
    )


@pytest.mark.parametrize("value", ["guest", "local", "synced", ""])
def test_parse_api_memory_mode_rejects_stale_or_internal_values(value: str) -> None:
    with pytest.raises(ValueError, match="Invalid OPENCOUCH_MEMORY_MODE"):
        parse_api_memory_mode(value, default=ApiMemoryMode.PERSISTENT)


def test_get_runtime_selection_returns_requested_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistent_runtime = object()
    incognito_runtime = object()
    monkeypatch.setattr(
        dependencies,
        "_runtimes",
        {
            ApiMemoryMode.PERSISTENT: persistent_runtime,
            ApiMemoryMode.INCOGNITO: incognito_runtime,
        },
    )
    monkeypatch.setattr(
        dependencies,
        "_default_memory_mode",
        ApiMemoryMode.PERSISTENT,
    )

    selection = get_runtime_selection(ApiMemoryMode.PERSISTENT)
    assert selection.memory_mode is ApiMemoryMode.PERSISTENT
    assert selection.runtime is persistent_runtime
    assert get_runtime_for_memory_mode("incognito") is incognito_runtime


def test_get_runtime_selection_uses_configured_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistent_runtime = object()
    incognito_runtime = object()
    monkeypatch.setattr(
        dependencies,
        "_runtimes",
        {
            ApiMemoryMode.PERSISTENT: persistent_runtime,
            ApiMemoryMode.INCOGNITO: incognito_runtime,
        },
    )
    monkeypatch.setattr(
        dependencies,
        "_default_memory_mode",
        ApiMemoryMode.INCOGNITO,
    )

    selection = get_runtime_selection(None)
    assert selection.memory_mode is ApiMemoryMode.INCOGNITO
    assert selection.runtime is incognito_runtime
    assert get_runtime() is incognito_runtime


def test_get_runtime_selection_errors_before_lifespan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dependencies, "_runtimes", {})

    with pytest.raises(RuntimeError, match="Agent runtime not initialized"):
        get_runtime_selection(ApiMemoryMode.PERSISTENT)


def test_get_runtime_selection_reports_unavailable_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dependencies,
        "_runtimes",
        {ApiMemoryMode.INCOGNITO: object()},
    )

    with pytest.raises(HTTPException) as exc_info:
        get_runtime_selection(ApiMemoryMode.PERSISTENT)

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == {
        "code": "memory_mode_unavailable",
        "message": "Memory mode 'persistent' is unavailable for this server configuration.",
    }


class _FakeRuntime:
    def __init__(self, memory_mode: MemoryMode) -> None:
        self.memory_mode = memory_mode
        self.entered = False
        self.exited = False
        self.finalized = False

    async def __aenter__(self) -> _FakeRuntime:
        self.entered = True
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.exited = True

    async def finalize_active_sessions(self, *, llm_client: object | None) -> None:
        self.finalized = True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("env_value", "expected_default", "expected_modes"),
    [
        (
            None,
            ApiMemoryMode.PERSISTENT,
            {ApiMemoryMode.PERSISTENT, ApiMemoryMode.INCOGNITO},
        ),
        (
            "persistent",
            ApiMemoryMode.PERSISTENT,
            {ApiMemoryMode.PERSISTENT, ApiMemoryMode.INCOGNITO},
        ),
        ("incognito", ApiMemoryMode.INCOGNITO, {ApiMemoryMode.INCOGNITO}),
    ],
)
async def test_lifespan_initializes_available_runtimes_and_default_mode(
    env_value: str | None,
    expected_default: ApiMemoryMode,
    expected_modes: set[ApiMemoryMode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built_runtimes: list[_FakeRuntime] = []

    def fake_build_runtime(
        *,
        memory_mode: MemoryMode,
        settings: Settings,
        llm_client: object | None,
    ) -> _FakeRuntime:
        runtime = _FakeRuntime(memory_mode)
        built_runtimes.append(runtime)
        return runtime

    monkeypatch.setenv("OPENCOUCH_MEMORY_MODE", env_value or "")
    if env_value is None:
        monkeypatch.delenv("OPENCOUCH_MEMORY_MODE", raising=False)
    monkeypatch.setattr(
        dependencies, "create_configured_control_llm_client", lambda: None
    )
    monkeypatch.setattr(
        dependencies,
        "create_configured_response_llm_clients",
        lambda: {"fast": None, "quality": None},
    )
    monkeypatch.setattr(dependencies, "get_settings", Settings)
    monkeypatch.setattr(dependencies, "_build_runtime", fake_build_runtime)

    async with dependencies.lifespan(FastAPI()):
        assert dependencies._default_memory_mode is expected_default
        assert set(dependencies._runtimes) == expected_modes
        if ApiMemoryMode.PERSISTENT in expected_modes:
            assert (
                dependencies._runtimes[ApiMemoryMode.PERSISTENT].memory_mode
                is MemoryMode.LOCAL
            )
        assert (
            dependencies._runtimes[ApiMemoryMode.INCOGNITO].memory_mode
            is MemoryMode.INCOGNITO
        )
        assert all(runtime.entered for runtime in built_runtimes)

    assert all(runtime.finalized for runtime in built_runtimes)
    assert all(runtime.exited for runtime in built_runtimes)
    assert dependencies._runtimes == {}
    assert dependencies._default_memory_mode is ApiMemoryMode.PERSISTENT


@pytest.mark.asyncio
async def test_lifespan_incognito_does_not_require_postgres(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENCOUCH_MEMORY_MODE", "incognito")
    monkeypatch.setattr(
        dependencies, "create_configured_control_llm_client", lambda: None
    )
    monkeypatch.setattr(
        dependencies,
        "create_configured_response_llm_clients",
        lambda: {"fast": None, "quality": None},
    )
    monkeypatch.setattr(
        dependencies,
        "get_settings",
        lambda: Settings(
            persistence_backend="postgres",
            memory_database_url=None,
            text_session_backend="disabled",
        ),
    )

    async with dependencies.lifespan(FastAPI()):
        runtime = dependencies._runtimes[ApiMemoryMode.INCOGNITO]
        assert set(dependencies._runtimes) == {ApiMemoryMode.INCOGNITO}
        assert isinstance(runtime.crisis_log_backend, InMemoryCrisisLogBackend)
        assert isinstance(
            runtime.session_feedback_backend,
            InMemorySessionFeedbackBackend,
        )


@pytest.mark.asyncio
async def test_lifespan_rejects_stale_guest_memory_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENCOUCH_MEMORY_MODE", "guest")
    monkeypatch.setattr(
        dependencies, "create_configured_control_llm_client", lambda: None
    )
    monkeypatch.setattr(
        dependencies,
        "create_configured_response_llm_clients",
        lambda: {"fast": None, "quality": None},
    )
    monkeypatch.setattr(dependencies, "get_settings", Settings)

    with pytest.raises(ValueError, match="Invalid OPENCOUCH_MEMORY_MODE"):
        async with dependencies.lifespan(FastAPI()):
            pass
