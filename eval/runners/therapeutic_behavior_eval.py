"""Evaluate therapeutic routing and state-change behavior."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from eval.runners.base import (
    BaseEvaluator,
    EvalResult,
    build_base_arg_parser,
    run_evaluator_cli,
)
from eval.runners.therapeutic_common import (
    TherapeuticEvalCase,
    build_live_therapeutic_llms,
    build_scripted_llm,
    grade_therapeutic_output,
    invoke_therapeutic_branch,
    parse_therapeutic_case,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATASET = _REPO_ROOT / "eval" / "datasets" / "therapeutic" / "behavior_v1.json"


class TherapeuticBehaviorEvaluator(BaseEvaluator[TherapeuticEvalCase]):
    """Run therapeutic behavior checks against scripted or live LLMs."""

    def __init__(self, *, dataset_path: str | Path, mode: str) -> None:
        super().__init__(
            dataset_path=dataset_path,
            name=f"therapeutic_behavior_{mode}",
        )
        self.mode = mode

    def parse_case(self, raw_case: Any) -> TherapeuticEvalCase:
        """Parse one therapeutic behavior case.

        Args:
            raw_case (Any): Raw JSON case.

        Returns:
            TherapeuticEvalCase: Parsed case.
        """

        return parse_therapeutic_case(raw_case)

    def case_id(self, case: TherapeuticEvalCase, index: int) -> str:
        """Return the dataset id for one therapeutic case.

        Args:
            case (TherapeuticEvalCase): Parsed case.
            index (int): Zero-based position.

        Returns:
            str: Stable case id.
        """

        return case.id

    async def run_case(self, case: TherapeuticEvalCase) -> EvalResult:
        """Run and grade one therapeutic behavior case.

        Args:
            case (TherapeuticEvalCase): Parsed case.

        Returns:
            EvalResult: Case result.
        """

        if self.mode == "scripted":
            llm = build_scripted_llm(case)
            output = await invoke_therapeutic_branch(
                case,
                llm_client=llm,
                response_llm=llm,
            )
        else:
            control_llm, response_llm = build_live_therapeutic_llms()
            output = await invoke_therapeutic_branch(
                case,
                llm_client=control_llm,
                response_llm=response_llm,
            )

        failures = grade_therapeutic_output(case, output)
        return EvalResult(
            case_id=case.id,
            passed=not failures,
            score=1.0 if not failures else 0.0,
            details={
                "description": case.description,
                "mode": self.mode,
                "failures": failures,
                "output": output,
            },
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = build_base_arg_parser("Evaluate therapeutic routing behavior.")
    parser.set_defaults(dataset=_DEFAULT_DATASET)
    parser.add_argument(
        "--mode",
        choices=("scripted", "live"),
        default="scripted",
        help="scripted avoids provider calls; live uses configured LLM clients.",
    )
    return parser


def main() -> int:
    """Run the therapeutic behavior evaluator CLI.

    Returns:
        int: Shell exit code.
    """

    return run_evaluator_cli(
        lambda args: TherapeuticBehaviorEvaluator(
            dataset_path=args.dataset,
            mode=args.mode,
        ),
        parser=_build_parser(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
