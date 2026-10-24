"""A cap on request body size, enforced as the body arrives (HY-347)."""

from fastapi import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

TOO_LARGE_MESSAGE = "The request is too large."


class BodySizeLimit:
    """Refuse a request body over `max_bytes` with 413.

    A body that declares a larger Content-Length is refused before any of it
    is read. One sent without a length, as a proxy may stream it, is counted
    as it arrives, so it is never buffered past the cap.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Pass the request on, with its body capped."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = _declared_length(scope)
        if declared is not None and declared > self.max_bytes:
            response = JSONResponse({"detail": TOO_LARGE_MESSAGE}, status_code=413)
            await response(scope, receive, send)
            return
        received = 0

        async def counted_receive() -> Message:
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > self.max_bytes:
                # Raised where the endpoint reads the body, so FastAPI turns it
                # into a 413 like any other HTTPException.
                raise HTTPException(status_code=413, detail=TOO_LARGE_MESSAGE)
            return message

        await self.app(scope, counted_receive, send)


def _declared_length(scope: Scope) -> int | None:
    for name, value in scope["headers"]:
        # bytes.isdigit() is ASCII-only, so int() can always parse what it passes.
        if name == b"content-length" and value.isdigit():
            return int(value)
    return None
