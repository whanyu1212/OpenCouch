"""Evaluate the therapeutic subgraph contract."""

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
    build_scripted_llm,
    grade_therapeutic_output,
    invoke_therapeutic_subgraph,
    parse_therapeutic_case,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATASET = _REPO_ROOT / "eval" / "datasets" / "therapeutic" / "contract_v1.json"


class TherapeuticContractEvaluator(BaseEvaluator[TherapeuticEvalCase]):
    """Run deterministic therapeutic subgraph contract checks."""

    def __init__(self, *, dataset_path: str | Path) -> None:
        super().__init__(
            dataset_path=dataset_path,
            name="therapeutic_contract",
        )

    def parse_case(self, raw_case: Any) -> TherapeuticEvalCase:
        """Parse one therapeutic contract case.

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
        """Run and grade one therapeutic contract case.

        Args:
            case (TherapeuticEvalCase): Parsed case.

        Returns:
            EvalResult: Case result.
        """

        expected_error = case.expected.get("error_contains")
        try:
            llm = build_scripted_llm(case) if case.scripted else None
            output = await invoke_therapeutic_subgraph(
                case,
                llm_client=llm,
                response_llm=llm,
            )
        except Exception as exc:  # noqa: BLE001 - expected errors are eval outcomes
            if expected_error and str(expected_error).casefold() in str(exc).casefold():
                return EvalResult(
                    case_id=case.id,
                    passed=True,
                    score=1.0,
                    details={
                        "description": case.description,
                        "expected_error": str(expected_error),
                        "actual_error": f"{type(exc).__name__}: {exc}",
                    },
                )
            raise

        failures = grade_therapeutic_output(case, output)
        if expected_error:
            failures.append(f"expected error containing {expected_error!r}")
        return EvalResult(
            case_id=case.id,
            passed=not failures,
            score=1.0 if not failures else 0.0,
            details={
                "description": case.description,
                "failures": failures,
                "output": output,
            },
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = build_base_arg_parser("Evaluate the therapeutic subgraph contract.")
    parser.set_defaults(dataset=_DEFAULT_DATASET)
    return parser


def main() -> int:
    """Run the therapeutic contract evaluator CLI.

    Returns:
        int: Shell exit code.
    """

    return run_evaluator_cli(
        lambda args: TherapeuticContractEvaluator(dataset_path=args.dataset),
        parser=_build_parser(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
