"""FastAPI dependency injection for the agent runtime.

The ``PersistentAgentRuntime`` is an async context manager that owns
SQLite connections, an embedding provider, and the LangGraph
checkpointer. It must be opened once at startup and closed at
shutdown — not per-request. FastAPI's lifespan protocol handles this.

Usage in ``main.py``::

    from api.dependencies import lifespan

    app = FastAPI(lifespan=lifespan)

Then in route handlers::

    from api.dependencies import get_runtime, get_llm_client

    @router.post("/chat")
    async def chat(
        body: ChatRequest,
        runtime: PersistentAgentRuntime = Depends(get_runtime),
        llm_client: BaseLLMClient | None = Depends(get_llm_client),
    ):
        ...

The runtime and LLM client are singletons — every request shares
the same instance. This is safe because ``PersistentAgentRuntime``
serializes graph invocations per thread_id via the LangGraph
checkpointer, and the LLM clients are stateless (each call is
independent).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from agent.memory.modes import MemoryMode
from agent.persistence import (
    DEFAULT_CRISIS_LOG_DB_PATH,
    DEFAULT_MEMORY_DB_PATH,
    DEFAULT_THREAD_DB_PATH,
    PersistentAgentRuntime,
)
from core.config import create_configured_llm_client
from services.llm.base import BaseLLMClient


# Module-level singletons populated by the lifespan handler and
# read by the Depends() callables below. Using module-level state
# (rather than app.state) keeps the dependency signatures simple
# and avoids importing FastAPI in every route module.
_runtime: PersistentAgentRuntime | None = None
_llm_client: BaseLLMClient | None = None


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

    global _runtime, _llm_client  # noqa: PLW0603

    # Resolve LLM client — same logic as the CLI's resolve_llm_client.
    # Returns None if no API key is configured (deterministic mode).
    try:
        _llm_client = create_configured_llm_client()
    except Exception:
        _llm_client = None

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
    )
    async with _runtime:
        yield

    _runtime = None
    _llm_client = None


def get_runtime(request: Request) -> PersistentAgentRuntime:  # noqa: ARG001
    """FastAPI dependency that returns the shared runtime instance.

    Raises ``RuntimeError`` if called before the lifespan handler
    has opened the runtime (which shouldn't happen in normal
    operation because FastAPI runs the lifespan before accepting
    requests).
    """

    if _runtime is None:
        raise RuntimeError(
            "Agent runtime not initialized. "
            "Ensure the lifespan handler is configured on the FastAPI app."
        )
    return _runtime


def get_llm_client(request: Request) -> BaseLLMClient | None:  # noqa: ARG001
    """FastAPI dependency that returns the shared LLM client.

    Returns None when no LLM provider is configured (deterministic
    mode). Route handlers that require an LLM client should check
    for None and return a 503 if the operation can't proceed without
    one.
    """

    return _llm_client
