"""Runner for crisis gate evaluation."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Literal

BACKEND_ROOT = Path(__file__).resolve().parents[2] / "apps" / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agent.graph import build_initial_state
from agent.models import AgentInput
from agent.nodes.crisis_gate import (
    assess_crisis_risk_deterministically,
    assess_crisis_risk_with_llm,
    detect_crisis_override,
    normalize_crisis_assessment,
)
from core.config import create_configured_llm_client
from services.llm.base import BaseLLMClient

DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "datasets" / "crisis_detection_v1.json"
)
EvalMode = Literal["auto", "deterministic", "hybrid"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run crisis gate evaluation.")
    parser.add_argument(
        "--mode",
        choices=["auto", "deterministic", "hybrid"],
        default="auto",
        help=(
            "Evaluation mode. 'auto' uses the configured LLM client when available "
            "and falls back to deterministic mode otherwise."
        ),
    )
    return parser


def _load_cases() -> list[dict]:
    return json.loads(DATASET_PATH.read_text())


def _resolve_llm_client(mode: EvalMode) -> tuple[BaseLLMClient | None, str]:
    if mode == "deterministic":
        return None, "deterministic"

    if mode == "hybrid":
        return create_configured_llm_client(), "hybrid"

    try:
        return create_configured_llm_client(), "hybrid"
    except Exception:
        return None, "deterministic"


async def _evaluate_case(
    case: dict, llm_client: BaseLLMClient | None
) -> tuple[bool, str | None]:
    state = build_initial_state(
        AgentInput(
            message=case["message"],
            history=case["history"],
        ),
        include_input_history=True,
    )

    # Drive the classifier helpers directly so the eval mirrors the node's
    # internal decision tree without depending on its Command/Runtime wrapping.
    override = detect_crisis_override(state)
    if override is not None:
        _, override_assessment = override
        assessment = normalize_crisis_assessment(override_assessment)
    else:
        deterministic = assess_crisis_risk_deterministically(state)
        if deterministic.level >= 2 or llm_client is None:
            assessment = normalize_crisis_assessment(deterministic)
        else:
            try:
                llm_assessment = await assess_crisis_risk_with_llm(
                    state, llm_client=llm_client
                )
                assessment = normalize_crisis_assessment(llm_assessment)
            except Exception:
                assessment = normalize_crisis_assessment(deterministic)

    matched = (
        assessment.level == case["expected_level"]
        and assessment.needs_crisis_response == case["expected_needs_crisis_response"]
        and assessment.needs_clarification == case["expected_needs_clarification"]
    )

    if matched:
        return True, None

    detail = (
        f"FAIL {case['id']}: got level={assessment.level}, "
        f"needs_crisis_response={assessment.needs_crisis_response}, "
        f"needs_clarification={assessment.needs_clarification}"
    )
    return False, detail


async def _run(mode: EvalMode) -> int:
    cases = _load_cases()
    llm_client, resolved_mode = _resolve_llm_client(mode)
    failures = 0

    print(f"Running crisis gate eval in {resolved_mode} mode on {len(cases)} case(s).")

    for case in cases:
        passed, detail = await _evaluate_case(case, llm_client=llm_client)
        if not passed and detail is not None:
            failures += 1
            print(detail)

    if failures:
        print(f"\n{failures} crisis eval case(s) failed.")
        return 1

    print(f"All {len(cases)} crisis eval case(s) passed.")
    return 0


def main() -> int:
    """Run the crisis gate evaluation runner.

    Returns:
        Process exit code for the evaluation run.
    """

    args = _build_parser().parse_args()
    return asyncio.run(_run(args.mode))


if __name__ == "__main__":
    raise SystemExit(main())
