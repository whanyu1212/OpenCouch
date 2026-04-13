from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

    # CORS: allow the frontend dev server and any localhost port
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:3001",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api")
    return app


app = create_app()
