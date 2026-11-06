"""Report or compare saved evals offline; never generate or judge new replies."""

import argparse
import sys
from pathlib import Path

from evals.baseline.compare import comparison, render
from evals.baseline.models import Prices, Snapshot
from evals.baseline.report import build_report

DEFAULT_BASELINE = Path(__file__).parent / "snapshots" / "coach-dev-2026-11-05.json"


def main(argv: list[str] | None = None) -> int:
    """Return 0 for success, 1 for regressions/incompleteness, 2 for invalid input."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("report", "compare"))
    parser.add_argument("run", type=Path, help="Saved coach run directory")
    parser.add_argument("--judge", type=Path, help="Matching saved quality judge run")
    parser.add_argument(
        "--suite", type=Path, help="Original YAML for an older recording"
    )
    parser.add_argument("--prices", type=Path, help="Explicit JSON token-rate snapshot")
    parser.add_argument(
        "--output", type=Path, help="Write a transcript-free JSON snapshot"
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    args = parser.parse_args(argv)
    try:
        prices = (
            Prices.model_validate_json(args.prices.read_text()) if args.prices else None
        )
        snapshot = build_report(
            args.run, judge=args.judge, suite=args.suite, prices=prices
        )
        if args.command == "compare":
            baseline = Snapshot.model_validate_json(args.baseline.read_text())
            text, regressed = comparison(baseline, snapshot)
        else:
            text, regressed = render(snapshot), False
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(snapshot.model_dump_json(indent=2) + "\n")
    except (ValueError, OSError) as error:
        print(f"Baseline {args.command} failed: {error}", file=sys.stderr)
        return 2
    print(text, end="")
    return int(regressed)


if __name__ == "__main__":
    sys.exit(main())
