"""Opt-in LLM judging: `uv run --env-file ../../.env python -m evals.judge`."""

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from evals.judge.assess import load_rubric
from evals.judge.client import JudgeClient
from evals.judge.evaluator import SavedQualityEvaluator
from evals.judge.jobs import JudgeJob, load_jobs, sanity_jobs
from evals.judge.models import Route
from evals.judge.providers import build_routes
from evals.judge.run import judge_jobs
from evals.judge.summary import sanity_passed
from evals.runner.runs import load_run_meta
from evals.runner.suite import default_evaluators, evaluate_saved_run


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse explicitly opt-in paid judge commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", type=Path, help="saved coach run to judge")
    mode.add_argument(
        "--sanity", action="store_true", help="judge known negatives/control"
    )
    parser.add_argument(
        "--compare", type=Path, help="second saved run; judge both orders"
    )
    parser.add_argument(
        "--route", choices=["anthropic", "gateway", "openai"], default="anthropic"
    )
    parser.add_argument("--openai-model", default="gpt-6.1-sol")
    parser.add_argument(
        "--reps", type=int, default=1, help="judge reps per reply; use 2 for variance"
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--limit", type=int, help="maximum source replies, before repetitions"
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.compare is not None and args.run is None:
        parser.error("--compare requires --run")
    if (
        args.reps < 1
        or args.concurrency < 1
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("reps, concurrency and limit must be positive")
    return args


def _inputs(args: argparse.Namespace) -> tuple[list[JudgeJob], list[JudgeJob] | None]:
    jobs = sanity_jobs() if args.sanity else load_jobs(args.run)
    alternatives = None
    if args.compare is not None:
        if (
            load_run_meta(args.run).suite_digest
            != load_run_meta(args.compare).suite_digest
        ):
            raise ValueError("comparison runs must use the same suite and split")
        alternatives = load_jobs(args.compare)
    if args.limit is not None:
        jobs = jobs[: args.limit]
    if not jobs:
        raise ValueError("no saved replies to judge")
    return jobs, alternatives


async def main(argv: list[str] | None = None) -> int:
    """Save verdicts and advisory metrics; errors/failed sanity checks exit nonzero."""
    args = parse_args(argv)
    jobs, alternatives = _inputs(args)
    route: Route = args.route
    client = JudgeClient(
        build_routes(route, openai_model=args.openai_model), system_prompt=load_rubric()
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output_dir or Path("evals/results") / f"judge-{stamp}"
    outcome = await judge_jobs(
        jobs,
        client,
        output,
        reps=args.reps,
        concurrency=args.concurrency,
        alternatives=alternatives,
    )
    if args.run is not None and args.compare is None:
        report = await evaluate_saved_run(
            args.run,
            evaluators=[
                *default_evaluators(),
                SavedQualityEvaluator(outcome.judgments),
            ],
        )
        (output / "report.txt").write_text(report.report.render(include_reasons=True))
    summary = json.loads((output / "summary.json").read_text())
    print(json.dumps(summary, indent=2))
    print(f"Saved judge run -> {output}")
    if outcome.errors:
        return 1
    return int(args.sanity and not sanity_passed(outcome.judgments))


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
