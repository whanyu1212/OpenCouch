"""FastAPI app exposing the companion agent over AG-UI."""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from datetime import timedelta
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from psycopg_pool import AsyncConnectionPool
from pydantic_ai import Agent
from starlette.requests import Request
from starlette.responses import Response

from opencouch_agent.agent import CompanionDeps, build_companion_agent
from opencouch_agent.api.access import record_access_request
from opencouch_agent.api.body_limit import BodySizeLimit
from opencouch_agent.api.history import ConversationHistory, thread_history
from opencouch_agent.api.identity import (
    UserVerifier,
    build_user_verifier,
    signed_in_user,
)
from opencouch_agent.beta.access_requests import (
    AccessRequestBody,
    AccessRequestReceived,
    AccessRequests,
    InMemoryAccessRequests,
)
from opencouch_agent.beta.invites import BetaInvites, build_beta_invites
from opencouch_agent.beta.postgres import PostgresAccessRequests
from opencouch_agent.config import Settings, get_settings
from opencouch_agent.program.postgres import PostgresRecordStore
from opencouch_agent.safety import JevClient
from opencouch_agent.safety.audit import CrisisAuditLog, InMemoryCrisisAuditLog
from opencouch_agent.safety.crisis import CrisisResponder, build_crisis_agent
from opencouch_agent.safety.crisis_resources import CrisisCatalog, load_shipped_catalog
from opencouch_agent.safety.jev import TypeSafeJevClient, build_jev_client
from opencouch_agent.safety.postgres_audit import PostgresCrisisAuditLog
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
    audit: CrisisAuditLog | None = None,
    verifier: UserVerifier | None = None,
    invites: BetaInvites | None = None,
    access_requests: AccessRequests | None = None,
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
        audit: The crisis audit log. Tests pass one directly; by default it
            shares the default store's Postgres database, otherwise in-memory.
        verifier: Checks who is calling. Tests pass a fake; by default it
            checks Neon Auth tokens, which fails without a Neon Auth URL.
        invites: Who may use the coach. Tests pass their own; by default
            they come from `OPENCOUCH_BETA_INVITES`, and nobody without it.
        access_requests: Where requests for early access go. Tests pass one;
            by default it shares the default store's Postgres database.

    Returns:
        The configured app.

    Raises:
        ValueError: If `jev` or `verifier` isn't given and the settings can't
            build one.
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
    # Checked before the route runs, so an unknown caller never claims a
    # thread or reaches Jev.
    signed_in = signed_in_user(
        verifier or _verifier_from_settings(settings),
        invites or _invites_from_settings(settings),
    )
    session_store: SessionStore = store or InMemorySessionStore()
    pool: AsyncConnectionPool | None = None
    turn_timeout = timedelta(seconds=settings.turn_timeout_seconds)

    if store is None and settings.database_url:
        # No prepared statements: a pooled connection (Neon's, through PgBouncer)
        # can hand each transaction to a different server session.
        pool = AsyncConnectionPool(
            settings.database_url, open=False, kwargs={"prepare_threshold": None}
        )
        session_store = PostgresSessionStore(pool)
    elif store is None:
        logger.warning("No database URL is set; using in-memory sessions.")
    audit_log = audit or _default_audit_log(pool)
    requests = access_requests or _default_access_requests(pool)

    # Close only a client this app built; a caller's is theirs.
    owned_jev = jev_client if jev is None else None
    app = FastAPI(title="OpenCouch Agent", lifespan=_lifespan(pool, owned_jev))
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
    @app.get("/api/threads/{thread_id}/messages")
    async def thread_messages(
        thread_id: str, user_id: Annotated[str, Depends(signed_in)]
    ) -> ConversationHistory:
        return await thread_history(
            session_store, thread_id, user_id, wait=HISTORY_WAIT
        )

    # Public, like the crisis catalog: it's how someone without an invite asks
    # for one. Every valid request gets the same reply, so whether an address
    # is already invited or has asked before stays private.
    @app.post("/api/access-requests", status_code=202)
    async def request_access(body: AccessRequestBody) -> AccessRequestReceived:
        await record_access_request(requests, body.email)
        return AccessRequestReceived()

    @app.post("/api/agent")
    async def run_agent(
        request: Request, user_id: Annotated[str, Depends(signed_in)]
    ) -> Response:
        return await run_turn(
            request,
            user_id=user_id,
            agents=agents,
            jev=jev_client,
            store=session_store,
            audit=audit_log,
            turn_timeout=turn_timeout,
        )

    return app


def _default_audit_log(pool: AsyncConnectionPool | None) -> CrisisAuditLog:
    if pool is None:
        return InMemoryCrisisAuditLog()
    return PostgresCrisisAuditLog(pool)


def _verifier_from_settings(settings: Settings) -> UserVerifier:
    return build_user_verifier(
        base_url=settings.neon_auth_base_url,
        fake=settings.fake_auth,
        model=settings.model,
    )


def _lifespan(
    pool: AsyncConnectionPool | None, owned_jev: JevClient | None
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Open the pool and create the tables on startup; close what the app opened."""

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            if isinstance(owned_jev, TypeSafeJevClient):
                stack.push_async_callback(owned_jev.aclose)
            if pool is not None:
                await pool.open()
                stack.push_async_callback(pool.close)
                await PostgresSessionStore(pool).create_schema()
                await PostgresCrisisAuditLog(pool).create_schema()
                await PostgresAccessRequests(pool).create_schema()
                await PostgresRecordStore(pool).create_schema()
            yield

    return lifespan


def _default_access_requests(pool: AsyncConnectionPool | None) -> AccessRequests:
    if pool is None:
        return InMemoryAccessRequests()
    return PostgresAccessRequests(pool)


def _invites_from_settings(settings: Settings) -> BetaInvites:
    return build_beta_invites(
        invites=settings.beta_invites,
        everyone=settings.beta_open,
        model=settings.model,
    )


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
