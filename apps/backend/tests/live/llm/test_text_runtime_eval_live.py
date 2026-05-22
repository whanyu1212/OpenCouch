"""Opt-in live OpenAITextRuntime eval smoke tests.

These tests wrap ``eval/runners/run_live_text_runtime_eval.py`` so pytest has
coverage for full live runtime paths without making normal live classifier tests
more expensive. They require explicit runtime-eval flags in addition to API keys.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from llm.factory import create_llm_client

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.runners import run_live_text_runtime_eval as live_eval  # noqa: E402


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


async def _run_live_suite(suite: str) -> dict[str, list[str]]:
    live_client = create_llm_client(provider="openai")
    cases = live_eval._select_cases(
        live_eval._load_cases_from_paths(live_eval._dataset_paths_for_suite(suite)),
        case_ids=None,
        provider="openai",
    )

    results = [
        await live_eval._run_case(
            case,
            live_client=live_client,
            judge_client=None,
            min_judge_score=4,
            openai_agent_model=live_eval.DEFAULT_OPENAI_MODEL,
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
