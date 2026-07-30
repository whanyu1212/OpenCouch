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

Detection combines explicit signals, because no single one covers every route:

- an explicit ``--workers``/``-w`` count on the command line, which uvicorn
  never exports to the environment;
- the conventional worker-count environment variables;
- ``GUNICORN_CMD_ARGS``, which gunicorn parses as additional CLI arguments;
- gunicorn's resolved configuration instances, which are the in-process
  evidence of a config-file worker count.

Process identity is deliberately *not* the general backstop: it catches
spawn-based children (uvicorn) but a ``fork``-based prefork worker inherits
``MainProcess`` and ``parent_process() is None``, so it is invisible to
introspection. Detection is therefore best-effort against unenumerated
supervisors, and the deployment contract in the backend README remains the
authority.
"""

from __future__ import annotations

import gc
import multiprocessing
import os
import shlex
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

#: Flags that put the server in auto-reload mode, which excludes workers.
RELOAD_CLI_FLAGS: tuple[str, ...] = ("--reload",)

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
    """Return any worker count passed on the command line.

    ``uvicorn main:app --workers 2`` is the standard way to request workers
    and exports nothing to the environment, so an environment-only check
    would miss the most common multi-worker configuration.

    The count is returned whatever its value, including ``1``. An explicit
    CLI value overrides the ``WEB_CONCURRENCY`` default, so the caller needs
    to see ``--workers 1`` in order to stop consulting the environment.

    Args:
        argv (list[str] | None): Argument vector to scan. Defaults to
            ``sys.argv``.

    Returns:
        tuple[str, int] | None: The flag and count when one is declared,
            otherwise ``None``.
    """

    args = sys.argv if argv is None else argv
    detected: tuple[str, int] | None = None
    for index, arg in enumerate(args):
        for flag in WORKER_COUNT_CLI_FLAGS:
            if arg == flag:
                following = args[index + 1] if index + 1 < len(args) else None
                count = _parse_worker_count(following)
            elif arg.startswith(f"{flag}="):
                count = _parse_worker_count(arg.split("=", 1)[1])
            else:
                continue
            if count is not None:
                detected = (flag, count)
    return detected


def detect_configured_worker_count() -> tuple[str, int] | None:
    """Return a declared worker count above one, if any.

    An explicit command-line value wins outright: uvicorn documents
    ``--workers`` as defaulting to ``$WEB_CONCURRENCY``, so ``--workers 1``
    alongside ``WEB_CONCURRENCY=2`` runs one worker and must be allowed.
    The environment is consulted only when the command line declares nothing.

    Returns:
        tuple[str, int] | None: The source and count when a multi-worker
            configuration is declared, otherwise ``None``.
    """

    cli_detection = _detect_cli_worker_count()
    if cli_detection is not None:
        flag, count = cli_detection
        return (flag, count) if count > 1 else None

    for name in WORKER_COUNT_ENV_VARS:
        count = _parse_worker_count(os.getenv(name))
        if count is not None and count > 1:
            return name, count
    return None


def is_reload_child(argv: list[str] | None = None) -> bool:
    """Return whether this process is the auto-reload child.

    ``--reload`` also runs the application in a spawned child, and uvicorn
    names reload and worker children identically (``SpawnProcess-N``), so the
    process name cannot tell them apart. A spawned child inherits the
    parent's argv, and uvicorn rejects ``--reload`` together with
    ``--workers``, so the reload flag identifies a single-application child.

    Args:
        argv (list[str] | None): Argument vector to scan. Defaults to
            ``sys.argv``.

    Returns:
        bool: ``True`` when the process was started in auto-reload mode.
    """

    args = sys.argv if argv is None else argv
    return any(
        arg == flag or arg.startswith(f"{flag}=")
        for arg in args
        for flag in RELOAD_CLI_FLAGS
    )


def detect_gunicorn_worker_count() -> tuple[str, int] | None:
    """Return a worker count gunicorn resolved or declared.

    Gunicorn parses this variable as additional command-line arguments
    (``Config.get_cmd_args_from_env``), so a count set there never reaches
    ``sys.argv`` and is invisible to the argv scan.

    Gunicorn config-file values are available on ``Config`` instances, not the
    ``Arbiter`` class. A forked worker keeps those instances in memory, so the
    guard scans live objects for the resolved worker count when gunicorn is
    importable.

    Returns:
        tuple[str, int] | None: The source and count when gunicorn declares
            more than one worker, otherwise ``None``.
    """

    if _detect_cli_worker_count() is None:
        raw = os.getenv("GUNICORN_CMD_ARGS")
        if raw:
            try:
                args = shlex.split(raw)
            except ValueError:
                # Unbalanced quoting is gunicorn's to reject, not this guard's.
                return None
            detection = _detect_cli_worker_count(args)
            if detection is not None:
                _, count = detection
                return ("GUNICORN_CMD_ARGS", count) if count > 1 else None

    try:
        from gunicorn.config import Config  # noqa: PLC0415
    except Exception:
        return None

    for obj in gc.get_objects():
        try:
            if not isinstance(obj, Config):
                continue
            workers = getattr(obj, "workers", None)
        except Exception:
            continue
        if isinstance(workers, int) and workers > 1:
            return "gunicorn configuration", workers
    return None


def is_spawned_worker_process() -> bool:
    """Return whether this process is a spawn-based server worker child.

    Uvicorn's multiprocess supervisor starts each worker with
    ``multiprocessing.Process``, so a child sees a non-main process name even
    when it can observe neither an environment flag nor an explicit count.

    Auto-reload children are excluded: they are also spawned and share the
    same process name, but run exactly one application instance.

    This covers spawn-based supervisors only. A ``fork``-based prefork worker
    (gunicorn) keeps the parent's ``MainProcess`` identity, so it is
    unreachable from process introspection and is handled by
    :func:`detect_gunicorn_worker_count` instead.

    Returns:
        bool: ``True`` when running inside a spawned worker child.
    """

    if multiprocessing.current_process().name == "MainProcess":
        return False
    return not is_reload_child()


def enforce_single_worker_contract() -> None:
    """Reject startup when the environment declares multiple workers.

    Returns:
        None: Returns when the configuration is supported.

    Raises:
        MultiWorkerConfigurationError: If a worker count above one is set.
    """

    detected = detect_configured_worker_count() or detect_gunicorn_worker_count()
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
    "RELOAD_CLI_FLAGS",
    "WORKER_COUNT_CLI_FLAGS",
    "WORKER_COUNT_ENV_VARS",
    "detect_configured_worker_count",
    "detect_gunicorn_worker_count",
    "enforce_single_worker_contract",
    "is_reload_child",
    "is_spawned_worker_process",
]
