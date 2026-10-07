"""FastAPI app exposing the companion agent over AG-UI."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from psycopg_pool import AsyncConnectionPool
from pydantic_ai import Agent
from starlette.requests import Request
from starlette.responses import Response

from opencouch_agent.agent import CompanionDeps, build_companion_agent
from opencouch_agent.config import Settings, get_settings
from opencouch_agent.sessions import InMemorySessionStore, SessionStore
from opencouch_agent.sessions.postgres import PostgresSessionStore
from opencouch_agent.turn import run_turn

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    agent: Agent[CompanionDeps, str] | None = None,
    store: SessionStore | None = None,
) -> FastAPI:
    """Create the FastAPI app.

    Args:
        settings: Service settings. Defaults to settings loaded from the
            environment.
        agent: The companion agent to serve. Tests pass one backed by a test
            model; by default it is built from `settings.model`.
        store: The session store. Tests pass one directly; by default it is
            Postgres when `settings.database_url` is set, otherwise in-memory.

    Returns:
        The configured app.
    """
    settings = settings or get_settings()
    companion = agent or build_companion_agent(settings.model)
    session_store: SessionStore = store or InMemorySessionStore()
    pool: AsyncConnectionPool | None = None
    turn_timeout = timedelta(seconds=settings.turn_timeout_seconds)

    if store is None and settings.database_url:
        pool = AsyncConnectionPool(settings.database_url, open=False)
        session_store = PostgresSessionStore(pool)
    elif store is None:
        logger.warning("OPENCOUCH_DATABASE_URL is unset; using in-memory sessions.")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if pool is None:
            yield
            return
        await pool.open()
        try:
            await PostgresSessionStore(pool).create_schema()
            yield
        finally:
            await pool.close()

    app = FastAPI(title="OpenCouch Agent", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/agent")
    async def run_agent(request: Request) -> Response:
        return await run_turn(
            request,
            agent=companion,
            store=session_store,
            turn_timeout=turn_timeout,
        )

    return app


def run() -> None:
    """Serve the app with uvicorn, using host and port from settings."""
    settings = get_settings()
    uvicorn.run(
        "opencouch_agent.main:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
    )
