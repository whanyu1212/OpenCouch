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
