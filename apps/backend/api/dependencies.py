"""FastAPI dependency injection for the agent runtime.

The ``PersistentAgentRuntime`` is an async context manager that owns
configured persistence backends, an embedding provider, and the runtime
session store. It must be opened once at startup and closed at
shutdown, not per request. FastAPI's lifespan protocol handles this.

Usage in ``main.py``::

    from api.dependencies import lifespan

    app = FastAPI(lifespan=lifespan)

Then in route handlers::

    from api.dependencies import (
        get_llm_client,
        get_response_llm_clients,
        get_runtime,
    )

    @router.post("/chat")
    async def chat(
        body: ChatRequest,
        runtime: PersistentAgentRuntime = Depends(get_runtime),
        llm_client: BaseLLMClient | None = Depends(get_llm_client),
        response_llm_clients = Depends(get_response_llm_clients),
    ):
        ...

The runtime registry and LLM clients are singletons. Requests share
the runtime selected for their memory mode. This is safe because
``PersistentAgentRuntime`` serializes agent turns per thread_id via
the persistent session store, and the LLM clients are stateless
(each call is independent).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI

from agent.memory.modes import MemoryMode
from api.models import ApiMemoryMode
from agent.runtime import (
    DEFAULT_CRISIS_LOG_DB_PATH,
    DEFAULT_MEMORY_DB_PATH,
    DEFAULT_THREAD_DB_PATH,
    PersistentAgentRuntime,
    RuntimeDependencies,
    RuntimePersistenceConfig,
    RuntimeStoragePaths,
)
from config import (
    ResponseModelTier,
    Settings,
    create_configured_control_llm_client,
    create_configured_response_llm_clients,
    get_settings,
)
from llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


# Module-level singletons populated by the lifespan handler and
# read by the Depends() callables below. Using module-level state
# (rather than app.state) keeps the dependency signatures simple
# and avoids importing FastAPI in every route module.
_runtimes: dict[ApiMemoryMode, PersistentAgentRuntime] = {}
_default_memory_mode = ApiMemoryMode.PERSISTENT
_llm_client: BaseLLMClient | None = None
_response_llm_clients: dict[ResponseModelTier, BaseLLMClient | None] = {
    "fast": None,
    "quality": None,
}


@dataclass(frozen=True)
class ApiRuntimeSelection:
    """Resolved API memory mode and its shared runtime."""

    memory_mode: ApiMemoryMode
    runtime: PersistentAgentRuntime


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001
    """Open the agent runtime on startup, close on shutdown.

    The runtime opens its configured persistence backends and resolves
    the embedding provider. On shutdown, all connections are closed
    cleanly.

    The LLM client is resolved separately because it's stateless
    and doesn't need lifecycle management. We resolve it once
    at startup so route handlers don't pay the resolution cost
    per-request.

    Args:
        app: FastAPI app receiving the lifespan hook.

    Yields:
        None while the application is serving requests.

    Returns:
        Async lifespan context manager.
    """

    global _default_memory_mode, _llm_client, _response_llm_clients, _runtimes  # noqa: PLW0603

    # Resolve clients once so request handlers do not pay setup cost.
    # Missing API keys leave clients as None, keeping deterministic paths available.
    try:
        _llm_client = create_configured_control_llm_client()
    except Exception:
        _llm_client = None
    try:
        response_clients = create_configured_response_llm_clients()
        _response_llm_clients = {
            "fast": response_clients["fast"],
            "quality": response_clients["quality"],
        }
    except Exception:
        _response_llm_clients = {"fast": None, "quality": None}

    settings = get_settings()
    _default_memory_mode = parse_api_memory_mode(
        os.getenv("OPENCOUCH_MEMORY_MODE"),
        default=ApiMemoryMode.PERSISTENT,
    )
    _runtimes = {
        ApiMemoryMode.PERSISTENT: _build_runtime(
            memory_mode=MemoryMode.LOCAL,
            settings=settings,
            llm_client=_llm_client,
        ),
        ApiMemoryMode.INCOGNITO: _build_runtime(
            memory_mode=MemoryMode.INCOGNITO,
            settings=settings,
            llm_client=_llm_client,
        ),
    }

    async with AsyncExitStack() as stack:
        for runtime in _runtimes.values():
            await stack.enter_async_context(runtime)
        try:
            yield
        finally:
            for runtime in _runtimes.values():
                try:
                    await runtime.finalize_active_sessions(llm_client=_llm_client)
                except Exception:
                    logger.warning(
                        "api lifespan shutdown: failed to finalize active sessions",
                        exc_info=True,
                    )

    _runtimes = {}
    _default_memory_mode = ApiMemoryMode.PERSISTENT
    _llm_client = None
    _response_llm_clients = {"fast": None, "quality": None}


def parse_api_memory_mode(
    value: str | ApiMemoryMode | None,
    *,
    default: ApiMemoryMode,
) -> ApiMemoryMode:
    """Parse the configured default API memory mode."""

    if value is None:
        return default
    try:
        return ApiMemoryMode(str(value).strip().lower())
    except ValueError as exc:
        valid_values = ", ".join(mode.value for mode in ApiMemoryMode)
        raise ValueError(
            f"Invalid OPENCOUCH_MEMORY_MODE={value!r}; expected one of: {valid_values}"
        ) from exc


def _build_runtime(
    *,
    memory_mode: MemoryMode,
    settings: Settings,
    llm_client: BaseLLMClient | None,
) -> PersistentAgentRuntime:
    """Construct a runtime with API persistence settings."""

    return PersistentAgentRuntime(
        storage_paths=RuntimeStoragePaths(
            sqlite_path=str(DEFAULT_THREAD_DB_PATH),
            memory_sqlite_path=str(DEFAULT_MEMORY_DB_PATH),
            crisis_log_sqlite_path=str(DEFAULT_CRISIS_LOG_DB_PATH),
        ),
        persistence_config=RuntimePersistenceConfig.for_shared_backend(
            memory_mode=memory_mode,
            persistence_backend=settings.persistence_backend,
            database_url=settings.memory_database_url,
            text_session_backend=settings.text_session_backend,
            text_session_database_url=settings.text_session_database_url,
        ),
        dependencies=RuntimeDependencies(
            default_llm_client=llm_client,
        ),
    )


def resolve_api_memory_mode(
    memory_mode: ApiMemoryMode | str | None = None,
) -> ApiMemoryMode:
    """Resolve an API memory mode against the configured backend default."""

    return parse_api_memory_mode(memory_mode, default=_default_memory_mode)


def get_runtime_for_memory_mode(
    memory_mode: ApiMemoryMode | str | None = None,
) -> PersistentAgentRuntime:
    """Return the shared runtime for a request's resolved API memory mode."""

    return get_runtime_selection(memory_mode).runtime


def get_runtime_selection(
    memory_mode: ApiMemoryMode | str | None = None,
) -> ApiRuntimeSelection:
    """Return a request's resolved API memory mode and matching runtime."""

    resolved_mode = resolve_api_memory_mode(memory_mode)
    runtime = _runtimes.get(resolved_mode)
    if runtime is None:
        raise RuntimeError(
            "Agent runtime not initialized. "
            "Ensure the lifespan handler is configured on the FastAPI app."
        )
    return ApiRuntimeSelection(memory_mode=resolved_mode, runtime=runtime)


def get_runtime() -> PersistentAgentRuntime:
    """FastAPI dependency that returns the shared runtime instance.

    Works with both HTTP and WebSocket endpoints (no Request
    parameter needed). The runtime is a module-level singleton
    populated by the lifespan handler).

    Returns:
        The initialized shared runtime.

    Raises:
        RuntimeError: If the FastAPI lifespan hook has not initialized
            the runtime yet.
    """

    return get_runtime_for_memory_mode(None)


def get_llm_client() -> BaseLLMClient | None:
    """FastAPI dependency that returns the shared LLM client.

    Returns None when no LLM provider is configured (deterministic
    mode).

    Returns:
        The configured control-plane LLM client, or None.
    """

    return _llm_client


def get_response_llm_clients() -> dict[ResponseModelTier, BaseLLMClient | None]:
    """FastAPI dependency that returns the shared response-tier clients.

    Returns:
        Mapping from response model tier to the configured client for
        that tier, or None when unavailable.
    """

    return _response_llm_clients
