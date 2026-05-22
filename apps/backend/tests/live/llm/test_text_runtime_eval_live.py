"""Opt-in live OpenAITextRuntime eval smoke tests.

These tests wrap ``eval/runners/run_live_text_runtime_eval.py`` so pytest has
coverage for full live runtime paths without making normal live classifier tests
more expensive. They require explicit runtime-eval flags in addition to API keys.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm.factory import create_llm_client

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.runners import run_live_text_runtime_eval as live_eval  # noqa: E402
from eval.runners.helpers.judge import make_judge_client  # noqa: E402


def _has_live_openai_runtime_eval_env() -> bool:
    return (
        bool(os.getenv("OPENAI_API_KEY"))
        and os.getenv("RUN_LIVE_OPENAI_RUNTIME_EVALS") == "1"
    )


def _has_live_openai_trajectory_eval_env() -> bool:
    return (
        bool(os.getenv("OPENAI_API_KEY"))
        and os.getenv("RUN_LIVE_OPENAI_TRAJECTORY_EVALS") == "1"
    )


def _has_live_openai_trajectory_judge_eval_env() -> bool:
    return (
        bool(os.getenv("OPENAI_API_KEY"))
        and os.getenv("RUN_LIVE_OPENAI_TRAJECTORY_JUDGE_EVALS") == "1"
    )


def _trajectory_judge_model() -> str | None:
    model = os.getenv("OPENCOUCH_LIVE_TRAJECTORY_JUDGE_MODEL")
    if model is None:
        return None
    stripped = model.strip()
    return stripped or None


def _trajectory_judge_min_score() -> int:
    return int(os.getenv("OPENCOUCH_LIVE_TRAJECTORY_JUDGE_MIN_SCORE", "4"))


def _trajectory_judge_samples() -> int:
    return int(os.getenv("OPENCOUCH_LIVE_TRAJECTORY_JUDGE_SAMPLES", "1"))


def test_live_openai_trajectory_judge_eval_env_requires_dedicated_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("RUN_LIVE_OPENAI_TRAJECTORY_EVALS", "1")
    monkeypatch.delenv("RUN_LIVE_OPENAI_TRAJECTORY_JUDGE_EVALS", raising=False)

    assert _has_live_openai_trajectory_judge_eval_env() is False

    monkeypatch.setenv("RUN_LIVE_OPENAI_TRAJECTORY_JUDGE_EVALS", "1")
    assert _has_live_openai_trajectory_judge_eval_env() is True


def test_live_openai_trajectory_judge_policy_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENCOUCH_LIVE_TRAJECTORY_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("OPENCOUCH_LIVE_TRAJECTORY_JUDGE_MIN_SCORE", raising=False)
    monkeypatch.delenv("OPENCOUCH_LIVE_TRAJECTORY_JUDGE_SAMPLES", raising=False)

    assert _trajectory_judge_model() is None
    assert _trajectory_judge_min_score() == 4
    assert _trajectory_judge_samples() == 1


def test_live_openai_trajectory_judge_policy_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENCOUCH_LIVE_TRAJECTORY_JUDGE_MODEL", "gpt-5.4")
    monkeypatch.setenv("OPENCOUCH_LIVE_TRAJECTORY_JUDGE_MIN_SCORE", "5")
    monkeypatch.setenv("OPENCOUCH_LIVE_TRAJECTORY_JUDGE_SAMPLES", "3")

    assert _trajectory_judge_model() == "gpt-5.4"
    assert _trajectory_judge_min_score() == 5
    assert _trajectory_judge_samples() == 3


@pytest.mark.asyncio
async def test_run_live_suite_threads_judge_client_and_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live_client = object()
    judge_client = object()
    case = object()
    calls: list[dict[str, object]] = []
    monkeypatch.setenv("OPENCOUCH_LIVE_TRAJECTORY_JUDGE_MIN_SCORE", "5")
    monkeypatch.setenv("OPENCOUCH_LIVE_TRAJECTORY_JUDGE_MODEL", "gpt-5.4")
    monkeypatch.setenv("OPENCOUCH_LIVE_TRAJECTORY_JUDGE_SAMPLES", "3")
    monkeypatch.setattr(
        sys.modules[__name__], "create_llm_client", lambda provider: live_client
    )
    monkeypatch.setattr(
        sys.modules[__name__],
        "make_judge_client",
        lambda *, provider, model: judge_client,
    )
    monkeypatch.setattr(
        live_eval,
        "_dataset_paths_for_suite",
        lambda suite: (Path(f"{suite}.jsonl"),),
    )
    monkeypatch.setattr(live_eval, "_load_cases_from_paths", lambda paths: [case])
    monkeypatch.setattr(live_eval, "_select_cases", lambda cases, **kwargs: cases)

    async def fake_run_case_samples(case_arg, **kwargs):
        calls.append({"case": case_arg, **kwargs})
        return SimpleNamespace(id="case-1", failures=[], passed=True)

    monkeypatch.setattr(live_eval, "_run_case_samples", fake_run_case_samples)

    failures = await _run_live_suite("trajectories", judge=True)

    assert failures == {}
    assert calls == [
        {
            "case": case,
            "live_client": live_client,
            "judge_client": judge_client,
            "min_judge_score": 5,
            "openai_agent_model": live_eval.DEFAULT_OPENAI_MODEL,
            "samples": 3,
        }
    ]


async def _run_live_suite(
    suite: str,
    *,
    judge: bool = False,
) -> dict[str, list[str]]:
    live_client = create_llm_client(provider="openai")
    judge_client = (
        make_judge_client(provider="openai", model=_trajectory_judge_model())
        if judge
        else None
    )
    min_judge_score = _trajectory_judge_min_score()
    cases = live_eval._select_cases(
        live_eval._load_cases_from_paths(live_eval._dataset_paths_for_suite(suite)),
        case_ids=None,
        provider="openai",
    )

    results = [
        await live_eval._run_case_samples(
            case,
            live_client=live_client,
            judge_client=judge_client,
            min_judge_score=min_judge_score,
            openai_agent_model=live_eval.DEFAULT_OPENAI_MODEL,
            samples=_trajectory_judge_samples() if judge else 1,
        )
        for case in cases
    ]
    return {result.id: result.failures for result in results if not result.passed}


@pytest.mark.asyncio
@pytest.mark.live_api
@pytest.mark.skipif(
    not _has_live_openai_runtime_eval_env(),
    reason="Set OPENAI_API_KEY and RUN_LIVE_OPENAI_RUNTIME_EVALS=1 to run.",
)
async def test_live_openai_text_runtime_smoke_eval_cases() -> None:
    failures = await _run_live_suite("smoke")

    assert failures == {}


@pytest.mark.asyncio
@pytest.mark.live_api
@pytest.mark.skipif(
    not _has_live_openai_trajectory_eval_env(),
    reason="Set OPENAI_API_KEY and RUN_LIVE_OPENAI_TRAJECTORY_EVALS=1 to run.",
)
async def test_live_openai_text_runtime_trajectory_eval_cases() -> None:
    failures = await _run_live_suite("trajectories")

    assert failures == {}


@pytest.mark.asyncio
@pytest.mark.live_api
@pytest.mark.skipif(
    not _has_live_openai_trajectory_judge_eval_env(),
    reason=("Set OPENAI_API_KEY and RUN_LIVE_OPENAI_TRAJECTORY_JUDGE_EVALS=1 to run."),
)
async def test_live_openai_text_runtime_trajectory_judge_eval_cases() -> None:
    failures = await _run_live_suite("trajectories", judge=True)

    assert failures == {}
