"""Standalone channel gateway entry points."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import logging
import os
import signal
from pathlib import Path
from types import FrameType
from typing import TextIO

from telegram.ext import Application

from agent.memory.modes import MemoryMode
from agent.persistence import (
    DEFAULT_CRISIS_LOG_DB_PATH,
    DEFAULT_MEMORY_DB_PATH,
    DEFAULT_THREAD_DB_PATH,
    PersistentAgentRuntime,
)
from channels.telegram import (
    TelegramChannel,
    TelegramConfigurationError,
    TelegramGatewayConfig,
    build_telegram_application,
    build_telegram_session_registry,
    is_rotated_telegram_thread_id,
)
from config import (
    create_configured_control_llm_client,
    create_configured_response_llm_client,
    get_settings,
    load_runtime_env,
)

logger = logging.getLogger(__name__)


class GatewayLockError(RuntimeError):
    """Raised when the Telegram gateway advisory lock is already held."""


class TelegramGatewayLock:
    """Process-scoped advisory lock for one Telegram gateway store."""

    def __init__(self, path: str | Path) -> None:
        """Initialize a Telegram gateway lock.

        Args:
            path: Lockfile path.
        """

        self.path = Path(path)
        self._file: TextIO | None = None

    def acquire(self) -> None:
        """Acquire the advisory lock.

        Raises:
            GatewayLockError: If another process already holds the lock.
        """

        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
        lock_file: TextIO | None = None
        try:
            lock_file = os.fdopen(fd, "r+", encoding="utf-8")
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            if lock_file is None:
                os.close(fd)
            else:
                lock_file.close()
            raise GatewayLockError(
                f"another Telegram gateway already holds the lock at {self.path}"
            ) from exc
        except Exception:
            if lock_file is None:
                os.close(fd)
            else:
                lock_file.close()
            raise

        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"{os.getpid()}\n")
        lock_file.flush()
        self._file = lock_file

    def release(self) -> None:
        """Release the advisory lock if held.

        Returns:
            None.
        """

        if self._file is None:
            return
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
            self._file = None

    def __enter__(self) -> TelegramGatewayLock:
        """Acquire and return this lock.

        Returns:
            Acquired lock instance.
        """

        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Release the lock on context exit.

        Args:
            exc_type: Active exception type, if any.
            exc: Active exception instance, if any.
            tb: Active traceback, if any.

        Returns:
            None.
        """

        self.release()


def telegram_gateway_lock_path(
    sqlite_path: str | Path = DEFAULT_THREAD_DB_PATH,
) -> Path:
    """Return the lockfile path scoped to the runtime store directory.

    Args:
        sqlite_path: Thread checkpoint SQLite path.

    Returns:
        Lockfile path next to the configured thread database.
    """

    path = Path(sqlite_path)
    if path == Path(":memory:"):
        raise TelegramConfigurationError(
            "Telegram gateway requires persistent storage and cannot use :memory:."
        )
    return path.expanduser().parent / "telegram_gateway.lock"


def resolve_telegram_memory_mode(raw: str | None = None) -> MemoryMode:
    """Resolve and validate Telegram gateway memory mode.

    Args:
        raw: Optional raw mode value. Reads `OPENCOUCH_MEMORY_MODE` when omitted.

    Returns:
        Durable memory mode for the Telegram gateway.

    Raises:
        TelegramConfigurationError: If incognito/guest or an unknown mode is
            requested.
    """

    value = raw if raw is not None else os.getenv("OPENCOUCH_MEMORY_MODE", "")
    normalized = value.strip().lower() or "persistent"
    if normalized in {"guest", "incognito"}:
        raise TelegramConfigurationError(
            "Telegram gateway requires persistent memory; "
            "OPENCOUCH_MEMORY_MODE=guest/incognito would lose memory on restart."
        )
    if normalized in {"persistent", "local"}:
        return MemoryMode.LOCAL
    if normalized == "synced":
        return MemoryMode.SYNCED
    raise TelegramConfigurationError(
        f"Unsupported OPENCOUCH_MEMORY_MODE for Telegram gateway: {value!r}."
    )


async def run_telegram_gateway(
    *,
    config: TelegramGatewayConfig | None = None,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Run the standalone Telegram gateway until stopped.

    Args:
        config: Optional validated settings, mostly for tests.
        stop_event: Optional event used to stop polling in tests.

    Returns:
        None.
    """

    load_runtime_env()
    config = config or TelegramGatewayConfig.from_env()
    settings = get_settings()
    memory_mode = resolve_telegram_memory_mode()
    llm_client = create_configured_control_llm_client()
    response_llm_client = create_configured_response_llm_client(
        config.response_model_tier
    )

    lock_path = telegram_gateway_lock_path(DEFAULT_THREAD_DB_PATH)
    with TelegramGatewayLock(lock_path):
        runtime = PersistentAgentRuntime(
            sqlite_path=str(DEFAULT_THREAD_DB_PATH),
            memory_backend=settings.persistence_backend,
            memory_database_url=settings.memory_database_url,
            thread_persistence_backend=settings.persistence_backend,
            thread_database_url=settings.memory_database_url,
            crisis_log_persistence_backend=settings.persistence_backend,
            crisis_log_database_url=settings.memory_database_url,
            session_feedback_persistence_backend=settings.persistence_backend,
            session_feedback_database_url=settings.memory_database_url,
            memory_sqlite_path=str(DEFAULT_MEMORY_DB_PATH),
            crisis_log_sqlite_path=str(DEFAULT_CRISIS_LOG_DB_PATH),
            memory_mode=memory_mode,
            default_llm_client=llm_client,
            finalize_active_sessions_on_close=(not config.thread_rotation_enabled),
            auto_finalize_excluded=(
                is_rotated_telegram_thread_id
                if config.thread_rotation_enabled
                else None
            ),
        )
        async with runtime:
            session_registry = (
                build_telegram_session_registry(
                    backend=settings.persistence_backend,
                    sqlite_path=config.session_registry_sqlite_path,
                    database_url=settings.memory_database_url,
                )
                if config.thread_rotation_enabled
                else None
            )
            channel = TelegramChannel(
                config=config,
                runtime=runtime,
                llm_client=llm_client,
                response_llm_client=response_llm_client,
                session_registry=session_registry,
            )
            await channel.start()
            try:
                application = build_telegram_application(
                    config=config,
                    channel=channel,
                )
                await run_telegram_application(
                    application,
                    drop_pending_updates=config.drop_pending_updates,
                    stop_event=stop_event,
                )
            finally:
                await channel.stop()


async def run_telegram_application(
    application: Application,
    *,
    drop_pending_updates: bool,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Run a Telegram application with explicit async lifecycle control.

    Args:
        application: Configured Telegram application.
        drop_pending_updates: Whether Telegram should discard queued updates on
            startup.
        stop_event: Optional external stop event.

    Returns:
        None.
    """

    updater = application.updater
    if updater is None:
        raise RuntimeError("Telegram application was built without an updater.")

    event = stop_event or _stop_event_from_signals()
    initialized = False
    polling_started = False
    app_started = False
    try:
        await application.initialize()
        initialized = True
        await updater.start_polling(
            allowed_updates=["message"],
            drop_pending_updates=drop_pending_updates,
        )
        polling_started = True
        await application.start()
        app_started = True
        logger.info(
            "telegram gateway started; drop_pending_updates=%s",
            drop_pending_updates,
        )
        await event.wait()
    finally:
        if polling_started:
            await updater.stop()
        if app_started:
            await application.stop()
        if initialized:
            await application.shutdown()


def _stop_event_from_signals() -> asyncio.Event:
    """Create an asyncio event wired to SIGINT and SIGTERM.

    Returns:
        Stop event set by process termination signals.
    """

    event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def request_stop(signum: int, frame: FrameType | None = None) -> None:  # noqa: ARG001
        logger.info("received signal %s; stopping Telegram gateway", signum)
        event.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, request_stop, signum)
        except (NotImplementedError, RuntimeError, ValueError):
            try:
                signal.signal(signum, request_stop)
            except (RuntimeError, ValueError) as exc:
                raise RuntimeError(
                    "failed to register Telegram gateway shutdown signal "
                    "handlers; pass an explicit stop_event when running outside "
                    "the main thread"
                ) from exc
    return event


def main(argv: list[str] | None = None) -> int:
    """Run a channel gateway from the command line.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code.
    """

    parser = argparse.ArgumentParser(description="Run an OpenCouch channel gateway.")
    parser.add_argument("channel", choices=["telegram"])
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.channel == "telegram":
            asyncio.run(run_telegram_gateway())
    except (TelegramConfigurationError, GatewayLockError) as exc:
        logger.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
