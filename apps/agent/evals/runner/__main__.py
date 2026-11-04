"""CLI entry point to run a multi-turn eval suite or re-score a saved run (HY-239).

Run a suite on the test model (CI default):
`uv run python -m evals.runner --suite evals/cases/coach.yaml --model test`

Run on the real model (opt-in, 3 reps by default):
`uv run --env-file .env python -m evals.runner --suite evals/cases/coach.yaml`
`--model openai:gpt-6.1-sol`

Re-score a saved run offline without calling the model again:
`uv run python -m evals.runner --rescore evals/results/<run-dir>`
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from evals.crisis.record import BATCH_ATTEMPT_TIMEOUT_SECONDS, BATCH_RETRY
from evals.runner.cases import Split
from evals.runner.suite import evaluate_saved_run, run_suite
from opencouch_agent.safety import TypeSafeJevClient


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--suite", type=Path, help="path to a YAML suite file")
    mode.add_argument("--rescore", type=Path, help="saved run directory to re-score")
    parser.add_argument("--split", choices=["dev", "holdout"], help="filter by split")
    parser.add_argument(
        "--model",
        default="test",
        help="model string (default: test; opt-in live: e.g. openai:gpt-6.1-sol)",
    )
    parser.add_argument("--reps", type=int, default=3, help="repetitions per case")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, help="custom run directory")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="do not write results.jsonl / errors.jsonl to evals/results/",
    )
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    """Run or re-score a suite and return exit status (0 if all passed, 1 otherwise)."""
    args = parse_args(argv)
    if args.rescore is not None:
        outcome = await evaluate_saved_run(args.rescore)
        print(outcome.report.render(include_reasons=True))
        return 0 if outcome.all_passed else 1

    owned_jev = _live_jev(args.model)
    try:
        split: Split | None = args.split
        outcome = await run_suite(
            args.suite,
            split=split,
            requested_model=args.model,
            jev=owned_jev,
            reps=args.reps,
            max_concurrency=args.concurrency,
            output_dir=args.output_dir,
            save=not args.no_save,
        )
    finally:
        if owned_jev is not None:
            await owned_jev.aclose()

    print(outcome.report.render(include_reasons=True))
    if outcome.run_dir is not None:
        print(
            f"saved {len(outcome.results)} result(s), "
            f"{len(outcome.errors)} error(s) -> {outcome.run_dir}"
        )
    return 0 if outcome.all_passed else 1


def _live_jev(model: str) -> TypeSafeJevClient | None:
    if model == "test":
        return None
    api_key = os.environ.get("AI_GATEWAY_API_KEY")
    if not api_key:
        raise SystemExit(
            "AI_GATEWAY_API_KEY is required when running against a real model."
        )
    return TypeSafeJevClient(
        api_key=api_key,
        retry=BATCH_RETRY,
        attempt_timeout_seconds=BATCH_ATTEMPT_TIMEOUT_SECONDS,
    )


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
