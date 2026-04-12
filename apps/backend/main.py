from fastapi import FastAPI

from api.dependencies import lifespan
from api.router import api_router


def create_app() -> FastAPI:
    """Create the configured FastAPI application.

    The lifespan handler opens a ``PersistentAgentRuntime`` on startup
    (with SQLite connections, embedding provider, and LLM client
    resolution) and closes it on shutdown. Route handlers access the
    runtime via ``Depends(get_runtime)``.

    Returns:
        The configured backend ASGI application.
    """

    app = FastAPI(
        title="OpenCouch Backend",
        version="0.2.0",
        description="API service for OpenCouch — mental health support agent.",
        lifespan=lifespan,
    )
    app.include_router(api_router, prefix="/api")
    return app


app = create_app()
