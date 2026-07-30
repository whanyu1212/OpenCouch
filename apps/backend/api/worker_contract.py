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

import multiprocessing
import os
import sys

#: Environment variables that conventionally carry a worker count.
WORKER_COUNT_ENV_VARS: tuple[str, ...] = (
    "WEB_CONCURRENCY",
    "UVICORN_WORKERS",
    "GUNICORN_WORKERS",
    "OPENCOUCH_WORKERS",
)

#: Command-line flags that carry a worker count.
WORKER_COUNT_CLI_FLAGS: tuple[str, ...] = ("--workers", "-w")

_CONTRACT_MESSAGE = (
    "OpenCouch supports a single worker process only. "
    "{source} requests {count} workers.\n"
    "Runtime mutual exclusion is process-local (see "
    "agent/runtime/session/lock.py), so multiple workers would interleave "
    "active-session mutations and lose session state without raising.\n"
    "Run one worker per process and scale with a single replica, or remove "
    "the worker-count setting."
)

_SPAWNED_WORKER_MESSAGE = (
    "OpenCouch supports a single worker process only. This process was "
    "spawned as a server worker child ({process_name}).\n"
    "Runtime mutual exclusion is process-local (see "
    "agent/runtime/session/lock.py), so multiple workers would interleave "
    "active-session mutations and lose session state without raising.\n"
    "Run one worker per process and scale with a single replica."
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


def _detect_cli_worker_count(argv: list[str] | None = None) -> tuple[str, int] | None:
    """Return a worker count passed on the command line, if above one.

    ``uvicorn main:app --workers 2`` is the standard way to request workers
    and exports nothing to the environment, so an environment-only check
    would miss the most common multi-worker configuration.

    Args:
        argv (list[str] | None): Argument vector to scan. Defaults to
            ``sys.argv``.

    Returns:
        tuple[str, int] | None: The flag and count when a multi-worker
            configuration is declared, otherwise ``None``.
    """

    args = sys.argv if argv is None else argv
    for index, arg in enumerate(args):
        for flag in WORKER_COUNT_CLI_FLAGS:
            if arg == flag:
                following = args[index + 1] if index + 1 < len(args) else None
                count = _parse_worker_count(following)
            elif arg.startswith(f"{flag}="):
                count = _parse_worker_count(arg.split("=", 1)[1])
            else:
                continue
            if count is not None and count > 1:
                return flag, count
    return None


def detect_configured_worker_count() -> tuple[str, int] | None:
    """Return the first declared worker count above one.

    Checks the command line before the environment, because an explicit
    ``--workers`` value overrides the ``WEB_CONCURRENCY`` default it would
    otherwise fall back to.

    Returns:
        tuple[str, int] | None: The source and count when a multi-worker
            configuration is declared, otherwise ``None``.
    """

    cli_detection = _detect_cli_worker_count()
    if cli_detection is not None:
        return cli_detection

    for name in WORKER_COUNT_ENV_VARS:
        count = _parse_worker_count(os.getenv(name))
        if count is not None and count > 1:
            return name, count
    return None


def is_spawned_worker_process() -> bool:
    """Return whether this process is a server-spawned worker child.

    Uvicorn's multiprocess supervisor starts each worker with
    ``multiprocessing.Process``, so a child sees a non-main process name even
    when it can observe neither the parent's argv nor an environment flag.
    This is the backstop that catches worker configurations supplied by a
    route this module does not enumerate.

    Returns:
        bool: ``True`` when running inside a spawned worker child.
    """

    return multiprocessing.current_process().name != "MainProcess"


def enforce_single_worker_contract() -> None:
    """Reject startup when the environment declares multiple workers.

    Returns:
        None: Returns when the configuration is supported.

    Raises:
        MultiWorkerConfigurationError: If a worker count above one is set.
    """

    detected = detect_configured_worker_count()
    if detected is not None:
        source, count = detected
        raise MultiWorkerConfigurationError(
            _CONTRACT_MESSAGE.format(source=source, count=count)
        )

    if is_spawned_worker_process():
        raise MultiWorkerConfigurationError(
            _SPAWNED_WORKER_MESSAGE.format(
                process_name=multiprocessing.current_process().name
            )
        )


__all__ = [
    "MultiWorkerConfigurationError",
    "WORKER_COUNT_CLI_FLAGS",
    "WORKER_COUNT_ENV_VARS",
    "detect_configured_worker_count",
    "enforce_single_worker_contract",
    "is_spawned_worker_process",
]
