"""Startup enforces the single-worker deployment contract.

Runtime mutual exclusion is process-local, and the durable active-session
mutation marker is the one seam that would interleave silently rather than
raising. Refusing to boot converts that into a deploy-time error.
"""

from __future__ import annotations

import pytest

from api.worker_contract import (
    WORKER_COUNT_ENV_VARS,
    MultiWorkerConfigurationError,
    detect_configured_worker_count,
    enforce_single_worker_contract,
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
