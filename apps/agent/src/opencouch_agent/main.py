"""FastAPI app exposing the companion agent over AG-UI."""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic_ai import Agent
from pydantic_ai.ui.ag_ui import AGUIAdapter
from starlette.requests import Request
from starlette.responses import Response

from opencouch_agent.agent import CompanionDeps, build_companion_agent
from opencouch_agent.config import Settings, get_settings


def create_app(
    settings: Settings | None = None,
    agent: Agent[CompanionDeps, str] | None = None,
) -> FastAPI:
    """Create the FastAPI app.

    Args:
        settings: Service settings. Defaults to settings loaded from the
            environment.
        agent: The companion agent to serve. Tests pass one backed by a test
            model; by default it is built from `settings.model`.

    Returns:
        The configured app.
    """
    settings = settings or get_settings()
    companion = agent or build_companion_agent(settings.model)

    app = FastAPI(title="OpenCouch Agent")
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
        return await AGUIAdapter.dispatch_request(
            request, agent=companion, deps=CompanionDeps()
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
