"""Startup enforces the single-worker deployment contract.

Runtime mutual exclusion is process-local, and the durable active-session
mutation marker is the one seam that would interleave silently rather than
raising. Refusing to boot converts that into a deploy-time error.
"""

from __future__ import annotations

import pytest

from api import worker_contract
from api.worker_contract import (
    WORKER_COUNT_ENV_VARS,
    MultiWorkerConfigurationError,
    detect_configured_worker_count,
    enforce_single_worker_contract,
    is_spawned_worker_process,
)


@pytest.fixture(autouse=True)
def _clear_worker_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start each test from an environment with no worker count set."""

    for name in WORKER_COUNT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("GUNICORN_CMD_ARGS", raising=False)


def test_unset_worker_count_is_allowed() -> None:
    assert detect_configured_worker_count() is None
    enforce_single_worker_contract()


@pytest.mark.parametrize("env_var", WORKER_COUNT_ENV_VARS)
def test_single_worker_is_allowed(
    env_var: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(env_var, "1")

    assert detect_configured_worker_count() is None
    enforce_single_worker_contract()


@pytest.mark.parametrize("env_var", WORKER_COUNT_ENV_VARS)
def test_multiple_workers_refuse_to_boot(
    env_var: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every recognized worker-count variable is enforced."""

    monkeypatch.setenv(env_var, "4")

    assert detect_configured_worker_count() == (env_var, 4)
    with pytest.raises(MultiWorkerConfigurationError) as excinfo:
        enforce_single_worker_contract()

    message = str(excinfo.value)
    assert env_var in message
    assert "single worker" in message


@pytest.mark.parametrize("raw_value", ["", "   ", "auto", "not-a-number"])
def test_unparsable_worker_count_does_not_block_startup(
    raw_value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A value this module cannot interpret is left to the server that owns it.

    Failing startup on an unrecognized string would make the guard a source
    of outages rather than a safety net.
    """

    monkeypatch.setenv("WEB_CONCURRENCY", raw_value)

    assert detect_configured_worker_count() is None
    enforce_single_worker_contract()


def test_zero_or_negative_worker_count_is_not_treated_as_multi_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a count above one is unsafe; the server resolves the rest."""

    monkeypatch.setenv("WEB_CONCURRENCY", "0")
    enforce_single_worker_contract()

    monkeypatch.setenv("WEB_CONCURRENCY", "-1")
    enforce_single_worker_contract()


def test_first_declared_multi_worker_variable_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The message names a variable the operator actually set."""

    monkeypatch.setenv("WEB_CONCURRENCY", "1")
    monkeypatch.setenv("GUNICORN_WORKERS", "8")

    detected = detect_configured_worker_count()

    assert detected == ("GUNICORN_WORKERS", 8)


# ─── Command-line detection ──────────────────────────────────────────
#
# ``uvicorn main:app --workers 2`` is the standard way to request workers and
# exports nothing to the environment, so an environment-only guard would miss
# the most common multi-worker configuration entirely.


@pytest.mark.parametrize(
    "argv",
    [
        ["uvicorn", "main:app", "--workers", "2"],
        ["uvicorn", "main:app", "--workers=4"],
        ["uvicorn", "main:app", "-w", "3"],
        ["uvicorn", "main:app", "-w=5"],
        ["gunicorn", "-w", "8", "main:app"],
    ],
)
def test_command_line_worker_count_is_detected(
    argv: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(worker_contract.sys, "argv", argv)

    detected = detect_configured_worker_count()

    assert detected is not None
    flag, count = detected
    assert flag in {"--workers", "-w"}
    assert count > 1

    with pytest.raises(MultiWorkerConfigurationError, match="single worker"):
        enforce_single_worker_contract()


@pytest.mark.parametrize(
    "argv",
    [
        ["uvicorn", "main:app"],
        ["uvicorn", "main:app", "--workers", "1"],
        ["uvicorn", "main:app", "--workers=1"],
        ["uvicorn", "main:app", "--reload"],
        # A trailing flag with no value must not raise IndexError.
        ["uvicorn", "main:app", "--workers"],
        # Unrelated flags whose values look like counts.
        ["uvicorn", "main:app", "--port", "8080"],
    ],
)
def test_single_or_absent_command_line_worker_count_is_allowed(
    argv: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(worker_contract.sys, "argv", argv)

    assert detect_configured_worker_count() is None
    enforce_single_worker_contract()


def test_command_line_takes_precedence_over_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit ``--workers`` overrides the ``WEB_CONCURRENCY`` default."""

    monkeypatch.setenv("WEB_CONCURRENCY", "1")
    monkeypatch.setattr(
        worker_contract.sys, "argv", ["uvicorn", "main:app", "--workers", "6"]
    )

    assert detect_configured_worker_count() == ("--workers", 6)


@pytest.mark.parametrize(
    "argv",
    [
        ["uvicorn", "main:app", "--workers", "1"],
        ["uvicorn", "main:app", "--workers=1"],
        ["uvicorn", "main:app", "-w", "1"],
    ],
)
def test_explicit_one_worker_overrides_a_multi_worker_environment(
    argv: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--workers 1`` beside ``WEB_CONCURRENCY=2`` runs one worker.

    Uvicorn documents ``--workers`` as defaulting to ``$WEB_CONCURRENCY``, so
    an explicit CLI value replaces it. Falling through to the environment
    would reject a deployment that is actually single-worker.
    """

    monkeypatch.setenv("WEB_CONCURRENCY", "2")
    monkeypatch.setattr(worker_contract.sys, "argv", argv)

    assert detect_configured_worker_count() is None
    enforce_single_worker_contract()


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["uvicorn", "main:app", "--workers", "4", "--workers", "1"], None),
        (["uvicorn", "main:app", "--workers", "1", "--workers", "4"], 4),
        (["uvicorn", "main:app", "-w", "8", "--workers=1"], None),
    ],
)
def test_last_repeated_command_line_worker_count_wins(
    argv: list[str], expected: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Click resolves repeated non-multiple options to the last occurrence."""

    monkeypatch.setenv("WEB_CONCURRENCY", "2")
    monkeypatch.setattr(worker_contract.sys, "argv", argv)

    detected = detect_configured_worker_count()
    if expected is None:
        assert detected is None
        enforce_single_worker_contract()
    else:
        assert detected == ("--workers", expected)
        with pytest.raises(MultiWorkerConfigurationError, match="--workers"):
            enforce_single_worker_contract()


# ─── Spawned-worker backstop ─────────────────────────────────────────


def test_main_process_is_not_treated_as_a_spawned_worker() -> None:
    assert is_spawned_worker_process() is False


def test_reload_child_is_allowed_to_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--reload`` also spawns a child, and it must not be rejected.

    Uvicorn names reload and worker children identically
    (``SpawnProcess-N``), so the process name alone cannot distinguish them.
    Rejecting the reload child would break the repository's default
    development paths: ``compose.yml`` and the Dockerfile dev target both
    pass ``--reload``.
    """

    class _ReloadChildProcess:
        name = "SpawnProcess-1"

    monkeypatch.setattr(
        worker_contract.multiprocessing,
        "current_process",
        lambda: _ReloadChildProcess(),
    )
    monkeypatch.setattr(
        worker_contract.sys, "argv", ["uvicorn", "main:app", "--reload"]
    )

    assert worker_contract.is_reload_child() is True
    assert is_spawned_worker_process() is False
    enforce_single_worker_contract()


def test_reload_child_ignores_web_concurrency_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Uvicorn ignores ``WEB_CONCURRENCY`` when auto-reload is enabled."""

    class _ReloadChildProcess:
        name = "SpawnProcess-1"

    monkeypatch.setenv("WEB_CONCURRENCY", "4")
    monkeypatch.setattr(
        worker_contract.multiprocessing,
        "current_process",
        lambda: _ReloadChildProcess(),
    )
    monkeypatch.setattr(
        worker_contract.sys, "argv", ["uvicorn", "main:app", "--reload"]
    )

    assert detect_configured_worker_count() is None
    enforce_single_worker_contract()


def test_reload_child_with_explicit_worker_count_is_still_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit worker count wins even alongside a reload flag.

    Uvicorn rejects the combination itself, but the guard should not become
    a way to opt out of the contract by adding ``--reload``.
    """

    class _ReloadChildProcess:
        name = "SpawnProcess-1"

    monkeypatch.setattr(
        worker_contract.multiprocessing,
        "current_process",
        lambda: _ReloadChildProcess(),
    )
    monkeypatch.setattr(
        worker_contract.sys,
        "argv",
        ["uvicorn", "main:app", "--reload", "--workers", "4"],
    )

    with pytest.raises(MultiWorkerConfigurationError, match="--workers"):
        enforce_single_worker_contract()


def test_spawned_worker_child_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker child rejects startup even without argv or env evidence.

    Uvicorn's supervisor spawns workers with ``multiprocessing.Process``, so
    a child can detect its own status when it can see neither the parent's
    command line nor an environment flag.
    """

    class _WorkerChildProcess:
        name = "SpawnProcess-1"

    monkeypatch.setattr(
        worker_contract.multiprocessing,
        "current_process",
        lambda: _WorkerChildProcess(),
    )
    # No reload flag: this child belongs to a multi-worker supervisor.
    monkeypatch.setattr(worker_contract.sys, "argv", ["uvicorn", "main:app"])

    assert is_spawned_worker_process() is True
    with pytest.raises(MultiWorkerConfigurationError, match="SpawnProcess-1"):
        enforce_single_worker_contract()


# ─── Gunicorn prefork detection ──────────────────────────────────────
#
# Gunicorn forks its workers, so a child keeps ``MainProcess`` and
# ``parent_process()`` returns ``None`` — no process-identity check can see
# one. ``GUNICORN_CMD_ARGS`` and its resolved configuration cover routes that
# are invisible to argv and the conventional worker-count environment scan.


def test_gunicorn_absent_is_not_treated_as_multi_worker() -> None:
    """The check is inert when gunicorn is not the server."""

    assert worker_contract.detect_gunicorn_worker_count() is None
    enforce_single_worker_contract()


def _install_fake_gunicorn_config(
    monkeypatch: pytest.MonkeyPatch, *, workers: object
) -> None:
    """Register a minimal fake gunicorn exposing a resolved config instance."""

    import sys as real_sys
    import types

    gunicorn_module = types.ModuleType("gunicorn")
    gunicorn_module.__path__ = []  # type: ignore[attr-defined]
    config_module = types.ModuleType("gunicorn.config")

    class _Config:
        pass

    cfg = _Config()
    cfg.workers = workers  # type: ignore[attr-defined]
    config_module.Config = _Config  # type: ignore[attr-defined]

    monkeypatch.setitem(real_sys.modules, "gunicorn", gunicorn_module)
    monkeypatch.setitem(real_sys.modules, "gunicorn.config", config_module)
    monkeypatch.setattr(worker_contract.gc, "get_objects", lambda: [cfg])


def test_gunicorn_cmd_args_worker_count_is_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GUNICORN_CMD_ARGS", "--workers 4")
    monkeypatch.setattr(worker_contract.sys, "argv", ["gunicorn", "main:app"])

    assert worker_contract.detect_gunicorn_worker_count() == ("GUNICORN_CMD_ARGS", 4)
    with pytest.raises(MultiWorkerConfigurationError, match="GUNICORN_CMD_ARGS"):
        enforce_single_worker_contract()


def test_gunicorn_cmd_args_honors_last_repeated_worker_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GUNICORN_CMD_ARGS", "--workers 4 --workers 1")
    monkeypatch.setattr(worker_contract.sys, "argv", ["gunicorn", "main:app"])

    assert worker_contract.detect_gunicorn_worker_count() is None
    enforce_single_worker_contract()


def test_explicit_cli_overrides_gunicorn_cmd_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GUNICORN_CMD_ARGS", "--workers 4")
    monkeypatch.setattr(
        worker_contract.sys, "argv", ["gunicorn", "--workers", "1", "main:app"]
    )

    assert detect_configured_worker_count() is None
    assert worker_contract.detect_gunicorn_worker_count() is None
    enforce_single_worker_contract()


def test_gunicorn_config_file_worker_count_is_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A config-file worker count is invisible to argv and env scans."""

    _install_fake_gunicorn_config(monkeypatch, workers=4)
    monkeypatch.setattr(worker_contract.sys, "argv", ["gunicorn", "main:app"])

    assert worker_contract.detect_gunicorn_worker_count() == (
        "gunicorn configuration",
        4,
    )
    with pytest.raises(MultiWorkerConfigurationError, match="gunicorn"):
        enforce_single_worker_contract()


def test_gunicorn_single_worker_configuration_is_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_gunicorn_config(monkeypatch, workers=1)
    monkeypatch.setattr(worker_contract.sys, "argv", ["gunicorn", "main:app"])

    assert worker_contract.detect_gunicorn_worker_count() is None
    enforce_single_worker_contract()


def test_gunicorn_unreadable_worker_count_does_not_block_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-integer count is left to gunicorn rather than failing startup."""

    _install_fake_gunicorn_config(monkeypatch, workers=None)
    monkeypatch.setattr(worker_contract.sys, "argv", ["gunicorn", "main:app"])

    assert worker_contract.detect_gunicorn_worker_count() is None
    enforce_single_worker_contract()
