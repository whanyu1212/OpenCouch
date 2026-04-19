import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import lifespan
from api.router import api_router


def _cors_origins() -> tuple[list[str], bool]:
    """Resolve allowed CORS origins from environment configuration.

    Returns:
        A tuple of allowed origins and whether credentials should be allowed.
    """

    default_origins = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ]
    configured = os.getenv("OPENCOUCH_CORS_ORIGINS", "").strip()
    if configured == "*":
        return ["*"], False

    extra_origins = [
        origin.strip() for origin in configured.split(",") if origin.strip()
    ]
    origins = list(dict.fromkeys([*default_origins, *extra_origins]))
    return origins, True


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

    cors_origins, allow_credentials = _cors_origins()

    # CORS defaults to local development origins and can be extended
    # in deployed environments with OPENCOUCH_CORS_ORIGINS.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api")
    return app


app = create_app()
