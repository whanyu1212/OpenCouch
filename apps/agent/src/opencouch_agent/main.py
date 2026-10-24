"""FastAPI app exposing the companion agent over AG-UI."""

import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import timedelta

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from psycopg_pool import AsyncConnectionPool
from pydantic_ai import Agent
from starlette.requests import Request
from starlette.responses import Response

from opencouch_agent.agent import CompanionDeps, CompanionState, build_companion_agent
from opencouch_agent.api.body_limit import BodySizeLimit
from opencouch_agent.api.history import (
    ConversationHistory,
    shown_history,
    wait_for_turn,
)
from opencouch_agent.config import Settings, get_settings
from opencouch_agent.safety import JevClient
from opencouch_agent.safety.crisis import CrisisResponder, build_crisis_agent
from opencouch_agent.safety.crisis_resources import CrisisCatalog, load_shipped_catalog
from opencouch_agent.safety.jev import TypeSafeJevClient, build_jev_client
from opencouch_agent.sessions import InMemorySessionStore, SessionStore
from opencouch_agent.sessions.postgres import PostgresSessionStore
from opencouch_agent.turn import (
    CANCEL_GRACE_SECONDS,
    MAX_REQUEST_BYTES,
    TurnAgents,
    run_turn,
)

logger = logging.getLogger(__name__)

# Long enough for a disconnected turn to stop and save its cut-off reply, and
# short enough to fit in the web client's 10-second limit on the history load.
HISTORY_WAIT = timedelta(seconds=CANCEL_GRACE_SECONDS + 1)


def create_app(
    settings: Settings | None = None,
    agent: Agent[CompanionDeps, str] | None = None,
    store: SessionStore | None = None,
    crisis_agent: Agent[CompanionDeps, str] | None = None,
    jev: JevClient | None = None,
) -> FastAPI:
    """Create the FastAPI app.

    Args:
        settings: Service settings. Defaults to settings loaded from the
            environment.
        agent: The companion agent to serve. Tests pass one backed by a test
            model; by default it is built from `settings.model`.
        store: The session store. Tests pass one directly; by default it is
            Postgres when `settings.database_url` is set, otherwise in-memory.
        crisis_agent: The crisis responder. Tests pass one backed by a test
            model; by default it is built from `settings.model`.
        jev: The Jev client for the safety gate. Tests pass a fake; by default
            it is built from the settings, which fails without a key.

    Returns:
        The configured app.

    Raises:
        ValueError: If `jev` isn't given and the settings can't build one.
        pydantic.ValidationError: If the crisis resource catalog is invalid.
    """
    settings = settings or get_settings()
    # Safety: load the catalog up front, so a bad entry stops startup instead
    # of failing in the middle of a crisis.
    crisis_catalog = load_shipped_catalog()
    agents = TurnAgents(
        companion=agent or build_companion_agent(settings.model),
        crisis=CrisisResponder.from_catalog(
            crisis_agent or build_crisis_agent(settings.model), crisis_catalog
        ),
    )
    jev_client = jev or _jev_from_settings(settings)
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
        async with AsyncExitStack() as stack:
            # Close only a client this app built; a caller's is theirs.
            if jev is None and isinstance(jev_client, TypeSafeJevClient):
                stack.push_async_callback(jev_client.aclose)
            if pool is not None:
                await pool.open()
                stack.push_async_callback(pool.close)
                await PostgresSessionStore(pool).create_schema()
            yield

    app = FastAPI(title="OpenCouch Agent", lifespan=lifespan)
    # Added before CORS, so CORS wraps it and a 413 still reaches the browser.
    app.add_middleware(BodySizeLimit, max_bytes=MAX_REQUEST_BYTES)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    # Public, with no sign-in: someone in crisis must never hit a sign-up wall
    # before they see where to get help.
    @app.get("/api/crisis-resources")
    async def crisis_resources() -> CrisisCatalog:
        return crisis_catalog

    # Only what was said; tool calls, state and risk stay on the server.
    # Unauthenticated until HY-221 ties each thread to its owner, so only for
    # the password-protected demo (HY-352).
    @app.get("/api/threads/{thread_id}/messages")
    async def thread_messages(thread_id: str) -> ConversationHistory:
        await wait_for_turn(session_store, thread_id, HISTORY_WAIT)
        session = await session_store.load(thread_id)
        state = CompanionState.model_validate(session.state)
        return shown_history(session.history, crisis_resources=state.crisis_resources)

    @app.post("/api/agent")
    async def run_agent(request: Request) -> Response:
        return await run_turn(
            request,
            agents=agents,
            jev=jev_client,
            store=session_store,
            turn_timeout=turn_timeout,
        )

    return app


def _jev_from_settings(settings: Settings) -> JevClient:
    api_key = settings.ai_gateway_api_key
    return build_jev_client(
        api_key=api_key.get_secret_value() if api_key else None,
        fake=settings.fake_jev,
        model=settings.model,
    )


def run() -> None:
    """Serve the app with uvicorn, using host and port from settings."""
    settings = get_settings()
    uvicorn.run(
        "opencouch_agent.main:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
    )
