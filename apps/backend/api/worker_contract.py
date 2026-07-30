"""Single-worker deployment contract enforcement.

OpenCouch serves from one process. ``ThreadLockManager`` holds
``dict[str, asyncio.Lock]`` and binds itself to one OS thread and event loop,
so every "take the thread lock" guarantee in the runtime is process-local:
session finalization, feedback atomicity, and active-session mutation all
serialize within a worker and not across workers.

Most of that fails loudly under multiple workers — the lock manager raises.
The exception is the durable active-session mutation marker, which two
processes can both claim, leaving runtime state last-writer-wins with no
error. A silent corruption path is worse than a refused boot, so startup
rejects a multi-worker configuration outright rather than serving unsafely.
"""

from __future__ import annotations

import os

#: Environment variables that conventionally carry a worker count.
WORKER_COUNT_ENV_VARS: tuple[str, ...] = (
    "WEB_CONCURRENCY",
    "UVICORN_WORKERS",
    "GUNICORN_WORKERS",
    "OPENCOUCH_WORKERS",
)

_CONTRACT_MESSAGE = (
    "OpenCouch supports a single worker process only. "
    "{source} requests {count} workers.\n"
    "Runtime mutual exclusion is process-local (see "
    "agent/runtime/session/lock.py), so multiple workers would interleave "
    "active-session mutations and lose session state without raising.\n"
    "Run one worker per process and scale with a single replica, or remove "
    "the worker-count setting."
)


class MultiWorkerConfigurationError(RuntimeError):
    """Raised when startup detects an unsupported multi-worker configuration."""


def _parse_worker_count(raw: str | None) -> int | None:
    """Return a worker count from an environment value, if it holds one.

    Args:
        raw (str | None): Raw environment value.

    Returns:
        int | None: Parsed count, or ``None`` when absent or unparsable. An
            unparsable value is left to the server that owns it rather than
            failing startup on a string this module does not define.
    """

    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    try:
        return int(stripped)
    except ValueError:
        return None


def detect_configured_worker_count() -> tuple[str, int] | None:
    """Return the first environment-declared worker count above one.

    Returns:
        tuple[str, int] | None: The variable name and count when a
            multi-worker configuration is declared, otherwise ``None``.
    """

    for name in WORKER_COUNT_ENV_VARS:
        count = _parse_worker_count(os.getenv(name))
        if count is not None and count > 1:
            return name, count
    return None


def enforce_single_worker_contract() -> None:
    """Reject startup when the environment declares multiple workers.

    Returns:
        None: Returns when the configuration is supported.

    Raises:
        MultiWorkerConfigurationError: If a worker count above one is set.
    """

    detected = detect_configured_worker_count()
    if detected is None:
        return
    source, count = detected
    raise MultiWorkerConfigurationError(
        _CONTRACT_MESSAGE.format(source=source, count=count)
    )


__all__ = [
    "MultiWorkerConfigurationError",
    "WORKER_COUNT_ENV_VARS",
    "detect_configured_worker_count",
    "enforce_single_worker_contract",
]
