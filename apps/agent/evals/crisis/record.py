"""Record Jev's answers for a set of cases, once per question-set version.

Each case is sent exactly as the live gate would send a first message:
`user_check_state` with an empty history, and the current question set.
Answers are stored raw, keyed by question name, so the scorer can re-score
them under any policy without calling Jev again.

    uv run --env-file .env python -m evals.crisis.record --split validation --limit 20
    uv run --env-file .env python -m evals.crisis.record --resume <run dir>

A run directory holds `meta.json`, `recorded.jsonl` (one row per answered
case) and `errors.jsonl` (cases that failed this run). `--resume` skips cases
already recorded, which covers both a crash and retrying errors. Recorded
times use the batch client below, not live settings, so they aren't live
latency.
"""

import argparse
import asyncio
import hashlib
import json
import os
import time
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from typesafe_sdk import RetryPolicy

from evals.crisis.cases import DATA_DIR, Case, bhh_path, load_cases
from opencouch_agent.safety import (
    USER_TURN_QUESTIONS,
    JevClient,
    TypeSafeJevClient,
    user_turn_questions,
)
from opencouch_agent.safety.screening import user_check_state

RESULTS_DIR: Final = Path(__file__).parents[1] / "results" / "crisis"

# Batch runs can wait: respect Retry-After, retry timeouts, allow long texts.
BATCH_RETRY: Final = RetryPolicy(max_retries=4, backoff_initial=1.0, backoff_max=30.0)
BATCH_ATTEMPT_TIMEOUT_SECONDS: Final = 30.0


def question_set_version() -> str:
    """A short hash of the question text; a recording is only valid for one."""
    text = json.dumps(dict(sorted(USER_TURN_QUESTIONS.items())))
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON Lines file, or nothing if it doesn't exist."""
    if not path.exists():
        return []
    with path.open() as lines:
        return [json.loads(line) for line in lines if line.strip()]


async def record_case(client: JevClient, case: Case) -> dict[str, Any]:
    """Ask Jev about one case; the result row says whether it worked."""
    start = time.perf_counter()
    try:
        state = user_check_state([], case.text)
        probabilities = await client.ask(state, user_turn_questions())
    except Exception as error:
        # Keep the case out of the recording; errors are never scored as NONE.
        return {"id": case.id, "error": type(error).__name__}
    elapsed_ms = round((time.perf_counter() - start) * 1000)
    return {**asdict(case), "probabilities": probabilities, "ms": elapsed_ms}


async def record_cases(
    client: JevClient, cases: Sequence[Case], run_dir: Path, concurrency: int
) -> tuple[int, int]:
    """Record every case not yet in `run_dir`, writing rows as they finish.

    Returns:
        How many cases were recorded and how many failed in this run.
    """
    done = {row["id"] for row in read_jsonl(run_dir / "recorded.jsonl")}
    pending = [case for case in cases if case.id not in done]
    limit = asyncio.Semaphore(concurrency)

    async def bounded(case: Case) -> dict[str, Any]:
        async with limit:
            return await record_case(client, case)

    recorded = failed = 0
    with (
        (run_dir / "recorded.jsonl").open("a") as recorded_file,
        (run_dir / "errors.jsonl").open("w") as errors_file,
    ):
        for finished in asyncio.as_completed([bounded(case) for case in pending]):
            row = await finished
            target = errors_file if "error" in row else recorded_file
            target.write(json.dumps(row) + "\n")
            target.flush()
            failed += "error" in row
            recorded += "error" not in row
    return recorded, failed


def new_run_dir(name: str) -> Path:
    """Create a run directory named for the cases, question set and time."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RESULTS_DIR / f"{name}-q{question_set_version()}-{stamp}"
    run_dir.mkdir(parents=True)
    return run_dir


def write_meta(run_dir: Path, name: str, cases_path: Path) -> None:
    """Record what the run was made from, so the scorer can check it."""
    meta: dict[str, Any] = {
        "cases": name,
        "cases_path": str(cases_path),
        "question_set_version": question_set_version(),
        "started_at": datetime.now(UTC).isoformat(),
    }
    dataset_meta = DATA_DIR / "bhh.meta.json"
    if name.startswith("bhh") and dataset_meta.exists():
        meta["dataset"] = json.loads(dataset_meta.read_text())
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--split", choices=["validation", "test"], help="BHH split")
    source.add_argument("--cases", type=Path, help="a cases JSON Lines file")
    source.add_argument("--resume", type=Path, help="a run directory to finish")
    parser.add_argument("--limit", type=int, help="record only the first N cases")
    parser.add_argument("--concurrency", type=int, default=4)
    return parser.parse_args()


def prepare_run(args: argparse.Namespace) -> tuple[Path, list[Case]]:
    """Pick or create the run directory and load its cases."""
    if args.resume:
        meta = json.loads((args.resume / "meta.json").read_text())
        if meta["question_set_version"] != question_set_version():
            raise SystemExit("The questions changed since this run; start a new one.")
        return args.resume, load_cases(Path(meta["cases_path"]))
    cases_path = bhh_path(args.split) if args.split else args.cases
    name = f"bhh-{args.split}" if args.split else cases_path.stem
    run_dir = new_run_dir(name)
    write_meta(run_dir, name, cases_path)
    return run_dir, load_cases(cases_path)


async def main() -> None:
    """Record a run and report where it went."""
    args = parse_args()
    run_dir, cases = prepare_run(args)
    client = TypeSafeJevClient(
        api_key=os.environ["AI_GATEWAY_API_KEY"],
        retry=BATCH_RETRY,
        attempt_timeout_seconds=BATCH_ATTEMPT_TIMEOUT_SECONDS,
    )
    try:
        recorded, failed = await record_cases(
            client, cases[: args.limit], run_dir, args.concurrency
        )
    finally:
        await client.aclose()
    print(f"recorded {recorded}, failed {failed} -> {run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
