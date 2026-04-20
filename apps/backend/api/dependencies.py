"""FastAPI dependency injection for the agent runtime.

The ``PersistentAgentRuntime`` is an async context manager that owns
SQLite connections, an embedding provider, and the LangGraph
checkpointer. It must be opened once at startup and closed at
shutdown — not per-request. FastAPI's lifespan protocol handles this.

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

The runtime and LLM client are singletons — every request shares
the same instance. This is safe because ``PersistentAgentRuntime``
serializes graph invocations per thread_id via the LangGraph
checkpointer, and the LLM clients are stateless (each call is
independent).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agent.memory.modes import MemoryMode
from agent.persistence import (
    DEFAULT_CRISIS_LOG_DB_PATH,
    DEFAULT_MEMORY_DB_PATH,
    DEFAULT_THREAD_DB_PATH,
    PersistentAgentRuntime,
)
from core.config import (
    ResponseModelTier,
    create_configured_control_llm_client,
    create_configured_response_llm_clients,
)
from services.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


# Module-level singletons populated by the lifespan handler and
# read by the Depends() callables below. Using module-level state
# (rather than app.state) keeps the dependency signatures simple
# and avoids importing FastAPI in every route module.
_runtime: PersistentAgentRuntime | None = None
_llm_client: BaseLLMClient | None = None
_response_llm_clients: dict[ResponseModelTier, BaseLLMClient | None] = {
    "fast": None,
    "quality": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001
    """Open the agent runtime on startup, close on shutdown.

    The runtime opens its SQLite connections (thread checkpointer,
    memory store, crisis log) and resolves the embedding provider.
    On shutdown, all connections are closed cleanly.

    The LLM client is resolved separately because it's stateless
    and doesn't need lifecycle management — but we resolve it once
    at startup so route handlers don't pay the resolution cost
    per-request.
    """

    global _runtime, _llm_client, _response_llm_clients  # noqa: PLW0603

    # Resolve LLM client — same logic as the CLI's resolve_llm_client.
    # Returns None if no API key is configured (deterministic mode).
    try:
        _llm_client = create_configured_control_llm_client()
    except Exception:
        _llm_client = None
    try:
        _response_llm_clients = create_configured_response_llm_clients()
    except Exception:
        _response_llm_clients = {"fast": None, "quality": None}

    # Resolve memory mode from environment. Default to LOCAL for the
    # API server (persistent storage). The CLI has an interactive
    # prompt for this; the API server just reads the env var.
    import os

    memory_mode_str = os.getenv("OPENCOUCH_MEMORY_MODE", "persistent")
    memory_mode = (
        MemoryMode.INCOGNITO if memory_mode_str == "guest" else MemoryMode.LOCAL
    )

    _runtime = PersistentAgentRuntime(
        sqlite_path=str(DEFAULT_THREAD_DB_PATH),
        memory_sqlite_path=str(DEFAULT_MEMORY_DB_PATH),
        crisis_log_sqlite_path=str(DEFAULT_CRISIS_LOG_DB_PATH),
        memory_mode=memory_mode,
        default_llm_client=_llm_client,
    )
    async with _runtime:
        try:
            yield
        finally:
            try:
                await _runtime.finalize_active_sessions(llm_client=_llm_client)
            except Exception:
                logger.warning(
                    "api lifespan shutdown: failed to finalize active sessions",
                    exc_info=True,
                )

    _runtime = None
    _llm_client = None
    _response_llm_clients = {"fast": None, "quality": None}


def get_runtime() -> PersistentAgentRuntime:
    """FastAPI dependency that returns the shared runtime instance.

    Works with both HTTP and WebSocket endpoints (no Request
    parameter needed — the runtime is a module-level singleton
    populated by the lifespan handler).
    """

    if _runtime is None:
        raise RuntimeError(
            "Agent runtime not initialized. "
            "Ensure the lifespan handler is configured on the FastAPI app."
        )
    return _runtime


def get_llm_client() -> BaseLLMClient | None:
    """FastAPI dependency that returns the shared LLM client.

    Returns None when no LLM provider is configured (deterministic
    mode).
    """

    return _llm_client


def get_response_llm_clients() -> dict[ResponseModelTier, BaseLLMClient | None]:
    """FastAPI dependency that returns the shared response-tier clients."""

    return _response_llm_clients
