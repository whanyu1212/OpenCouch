"""Cloud Run-friendly backend entrypoint.

This module starts the FastAPI app with a host and port that match
container runtime environments such as Cloud Run.
"""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    """Run the backend server with environment-driven host and port.

    Returns:
        None.
    """

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    log_level = os.getenv("LOG_LEVEL", "info")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()
