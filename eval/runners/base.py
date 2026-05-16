"""Shared evaluator primitives.

The base evaluator owns dataset loading, per-case timing, exception capture,
summary aggregation, and a small JSON output helper. Domain evaluators should
own case schemas, model setup, application calls, and grading rules.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar, cast


CaseT = TypeVar("CaseT")


@dataclass(frozen=True)
class EvalResult:
    """Result for one evaluation case."""

    case_id: str
    passed: bool
    score: float | None = None
    duration_ms: float = 0.0
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalSummary:
    """Aggregated result for one evaluator run."""

    evaluator: str
    dataset_path: str
    total: int
    passed: int
    failed: int
    errored: int
    duration_ms: float
    results: list[EvalResult]

    @property
    def success_rate(self) -> float:
        """Return the pass ratio across all cases."""

        if self.total == 0:
            return 0.0
        return self.passed / self.total

    @property
    def exit_code(self) -> int:
        """Return a shell exit code for the summary."""

        return 0 if self.failed == 0 and self.errored == 0 else 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize the summary to plain JSON-compatible data."""

        return {
            **asdict(self),
            "success_rate": self.success_rate,
        }


class BaseEvaluator(Generic[CaseT]):
    """Base class for dataset-backed async evaluators."""

    def __init__(
        self,
        *,
        dataset_path: str | Path,
        name: str | None = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            dataset_path (str | Path): JSON dataset path.
            name (str | None): Optional display name. Defaults to the class name.

        Returns:
            None.
        """

        self.dataset_path = Path(dataset_path)
        self.name = name or self.__class__.__name__

    def load_cases(self) -> list[CaseT]:
        """Load and parse cases from the dataset file.

        Returns:
            list[CaseT]: Parsed cases for the evaluator.
        """

        raw = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            cases = raw
        elif isinstance(raw, Mapping) and isinstance(raw.get("cases"), list):
            cases = raw["cases"]
        else:
            raise ValueError(
                f"{self.dataset_path} must contain a JSON list or an object "
                "with a 'cases' list."
            )
        return [self.parse_case(item) for item in cases]

    def parse_case(self, raw_case: Any) -> CaseT:
        """Parse one raw dataset item.

        Args:
            raw_case (Any): Raw JSON value from the dataset.

        Returns:
            CaseT: Parsed evaluator-specific case.
        """

        return cast(CaseT, raw_case)

    def case_id(self, case: CaseT, index: int) -> str:
        """Return a stable identifier for a case.

        Args:
            case (CaseT): Parsed case.
            index (int): Zero-based case position.

        Returns:
            str: Stable case id for result reporting.
        """

        if isinstance(case, Mapping):
            value = case.get("id")
            if value:
                return str(value)
        return f"case_{index + 1}"

    async def run_case(self, case: CaseT) -> EvalResult:
        """Evaluate one case.

        Args:
            case (CaseT): Parsed case.

        Returns:
            EvalResult: Case result.
        """

        raise NotImplementedError

    async def run(self) -> EvalSummary:
        """Run all cases and return an aggregate summary.

        Returns:
            EvalSummary: Aggregate run result.
        """

        cases = self.load_cases()
        started = time.perf_counter()
        results: list[EvalResult] = []

        for index, case in enumerate(cases):
            case_started = time.perf_counter()
            try:
                result = await self.run_case(case)
            except Exception as exc:  # noqa: BLE001 - evals must report all cases
                result = EvalResult(
                    case_id=self.case_id(case, index),
                    passed=False,
                    duration_ms=_elapsed_ms(case_started),
                    error=f"{type(exc).__name__}: {exc}",
                    details={"traceback": traceback.format_exc()},
                )
            else:
                result = _with_duration(result, duration_ms=_elapsed_ms(case_started))
            results.append(result)

        errored = sum(1 for result in results if result.error is not None)
        passed = sum(1 for result in results if result.passed)
        failed = len(results) - passed - errored

        return EvalSummary(
            evaluator=self.name,
            dataset_path=str(self.dataset_path),
            total=len(results),
            passed=passed,
            failed=failed,
            errored=errored,
            duration_ms=_elapsed_ms(started),
            results=results,
        )


def print_summary(summary: EvalSummary, *, use_rich: bool = True) -> None:
    """Print a compact human-readable summary.

    Args:
        summary (EvalSummary): Evaluator run summary.
        use_rich (bool): Whether to use Rich formatting when available.

    Returns:
        None.
    """

    if use_rich and _print_rich_summary(summary):
        return

    print(
        f"{summary.evaluator}: {summary.passed}/{summary.total} passed "
        f"({summary.success_rate:.1%}) in {summary.duration_ms:.1f}ms"
    )
    for result in summary.results:
        status = "PASS" if result.passed else "FAIL"
        if result.error:
            status = "ERROR"
        suffix = f" - {result.error}" if result.error else ""
        print(f"  {status} {result.case_id} ({result.duration_ms:.1f}ms){suffix}")


def build_base_arg_parser(description: str) -> argparse.ArgumentParser:
    """Build common CLI arguments for eval runners.

    Args:
        description (str): CLI description.

    Returns:
        argparse.ArgumentParser: Parser with shared flags.
    """

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to the JSON dataset.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path to write the full JSON summary.",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Disable Rich formatting and print plain text.",
    )
    return parser


def run_evaluator_cli(
    evaluator_factory: Callable[[argparse.Namespace], BaseEvaluator[Any]],
    *,
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
) -> int:
    """Run an evaluator from a CLI entrypoint.

    Args:
        evaluator_factory (Callable[[argparse.Namespace], BaseEvaluator[Any]]):
            Factory that builds the domain evaluator from parsed arguments.
        parser (argparse.ArgumentParser): Parser for shared and domain flags.
        argv (Sequence[str] | None): Optional argv override for tests.

    Returns:
        int: Shell exit code.
    """

    args = parser.parse_args(argv)
    summary = asyncio.run(evaluator_factory(args).run())
    print_summary(summary, use_rich=not args.plain)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(summary.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

    return summary.exit_code


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _with_duration(result: EvalResult, *, duration_ms: float) -> EvalResult:
    return EvalResult(
        case_id=result.case_id,
        passed=result.passed,
        score=result.score,
        duration_ms=duration_ms,
        error=result.error,
        details=result.details,
    )


def _print_rich_summary(summary: EvalSummary) -> bool:
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
    except ImportError:
        return False

    status = "PASS" if summary.exit_code == 0 else "FAIL"
    status_style = "bold green" if summary.exit_code == 0 else "bold red"
    console = Console()
    console.print(
        Panel.fit(
            (
                f"[{status_style}]{status}[/{status_style}] {summary.evaluator}\n"
                f"{summary.passed}/{summary.total} passed "
                f"({summary.success_rate:.1%}) in {summary.duration_ms:.1f}ms"
            ),
            title="OpenCouch Eval",
            border_style="green" if summary.exit_code == 0 else "red",
        )
    )

    table = Table(show_header=True, header_style="bold")
    table.add_column("Status", no_wrap=True)
    table.add_column("Case")
    table.add_column("Score", justify="right", no_wrap=True)
    table.add_column("Duration", justify="right", no_wrap=True)
    table.add_column("Error")

    for result in summary.results:
        if result.error:
            case_status = "[red]ERROR[/red]"
        elif result.passed:
            case_status = "[green]PASS[/green]"
        else:
            case_status = "[red]FAIL[/red]"
        score = "" if result.score is None else f"{result.score:.2f}"
        table.add_row(
            case_status,
            result.case_id,
            score,
            f"{result.duration_ms:.1f}ms",
            result.error or "",
        )

    console.print(table)
    return True
