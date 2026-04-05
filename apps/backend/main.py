from fastapi import FastAPI

from api.router import api_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="OpenCouch Backend",
        version="0.1.0",
        description="API service for OpenCouch.",
    )
    app.include_router(api_router, prefix="/api")
    return app


app = create_app()
