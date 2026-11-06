"""Report calibration agreement offline, or explicitly run the paid judge."""

import argparse
import asyncio
import json
from pathlib import Path

from evals.calibration.agreement import agreement
from evals.calibration.models import load_labels
from evals.judge.assess import load_rubric, rubric_digest
from evals.judge.client import JudgeClient
from evals.judge.providers import build_routes
from evals.judge.run import judge_jobs


async def main(argv: list[str] | None = None) -> int:
    """Keep partial review reports non-passing and paid requests explicitly opt-in."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--report", type=Path, help="existing judge output directory")
    mode.add_argument("--judge", type=Path, help="NEW paid judge output directory")
    args = parser.parse_args(argv)
    labels = load_labels(args.labels)
    if args.report is not None:
        report = agreement(labels, args.report)
        print(json.dumps(report.model_dump(), indent=2))
        return 0 if report.meets_target else 1
    if any(item.verdict is None for item in labels.items) or not labels.labeller:
        parser.error("Complete blind first-pass labels before running the judge")
    if labels.rubric_digest != rubric_digest():
        parser.error("Labels use a different rubric; review before judging")
    # Calibrate Opus only; a different-model fallback would mix two judges.
    routes = [route for route in build_routes("anthropic") if route.route != "openai"]
    client = JudgeClient(routes, system_prompt=load_rubric())
    outcome = await judge_jobs(labels.jobs(), client, args.judge, reps=2)
    print(f"Saved judge run -> {outcome.output_dir}")
    return int(bool(outcome.errors))


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
