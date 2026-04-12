from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def healthcheck() -> dict[str, str]:
    """Return a minimal health response for the backend.

    Returns:
        A simple status payload indicating the service is running.
    """

    return {"status": "ok"}
