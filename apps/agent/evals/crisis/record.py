"""Record Jev's answers for a set of cases, once per Jev input version.

Each case is sent exactly as the live gate would send a first message:
`user_check_state` with an empty history, and the current question set.
Answers are stored raw, keyed by question name, so the scorer can re-score
them under any policy without calling Jev again.

    uv run --env-file .env python -m evals.crisis.record --split validation --limit 20
    uv run --env-file .env python -m evals.crisis.record --resume <run dir>

See `runs.py` for what a run directory holds. `--resume` skips cases already
recorded, which covers both a crash and retrying errors. Recorded times use
the batch client below, not live settings, so they aren't live latency.
"""

import argparse
import asyncio
import os
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from typesafe_sdk import RetryPolicy

from evals.crisis.cases import BHH_META_PATH, Case, bhh_path, load_cases
from evals.crisis.runs import (
    ERRORS_FILE,
    RECORDED_FILE,
    DatasetInfo,
    ErrorRow,
    InputVersionMismatchError,
    RecordedRow,
    RunMeta,
    check_input_version,
    jev_input_version,
    load_meta,
    read_recorded,
    write_meta,
)
from opencouch_agent.safety import JevClient, TypeSafeJevClient, user_turn_questions
from opencouch_agent.safety.screening import user_check_state

AGENT_DIR: Final = Path(__file__).parents[2]
RESULTS_DIR: Final = Path(__file__).parents[1] / "results" / "crisis"

# Batch runs can wait: respect Retry-After, retry timeouts, allow long texts.
BATCH_RETRY: Final = RetryPolicy(max_retries=4, backoff_initial=1.0, backoff_max=30.0)
BATCH_ATTEMPT_TIMEOUT_SECONDS: Final = 30.0


async def record_case(client: JevClient, case: Case) -> RecordedRow | ErrorRow:
    """Ask Jev about one case; an error row if it didn't answer."""
    start = time.perf_counter()
    try:
        state = user_check_state([], case.text)
        probabilities = await client.ask(state, user_turn_questions())
    except Exception as error:
        # Any failure is kept out of the recording, so it's never scored as NONE.
        return ErrorRow(id=case.id, error=type(error).__name__)
    elapsed_ms = round((time.perf_counter() - start) * 1000)
    return RecordedRow(
        id=case.id,
        text=case.text,
        label=case.label,
        probabilities=probabilities,
        ms=elapsed_ms,
    )


async def record_cases(
    client: JevClient, cases: Sequence[Case], run_dir: Path, concurrency: int
) -> tuple[int, int]:
    """Record every case not yet in `run_dir`, writing rows as they finish.

    Returns:
        How many cases were recorded and how many failed in this run.
    """
    done = {row.id for row in read_recorded(run_dir)}
    pending = [case for case in cases if case.id not in done]
    limit = asyncio.Semaphore(concurrency)

    async def bounded(case: Case) -> RecordedRow | ErrorRow:
        async with limit:
            return await record_case(client, case)

    recorded = failed = 0
    with (
        (run_dir / RECORDED_FILE).open("a") as recorded_file,
        (run_dir / ERRORS_FILE).open("w") as errors_file,
    ):
        for finished in asyncio.as_completed([bounded(case) for case in pending]):
            row = await finished
            if isinstance(row, ErrorRow):
                failed += 1
                target = errors_file
            else:
                recorded += 1
                target = recorded_file
            # Written as each case finishes, so a crash loses nothing.
            target.write(row.model_dump_json() + "\n")
            target.flush()
    return recorded, failed


def new_run_dir(name: str) -> Path:
    """Create a run directory named for the cases, Jev input and time."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RESULTS_DIR / f"{name}-q{jev_input_version()}-{stamp}"
    run_dir.mkdir(parents=True)
    return run_dir


def _portable(path: Path) -> str:
    # Relative to apps/agent when inside it, so committed runs carry no local
    # paths; absolute otherwise. Either way `--resume` works from any directory.
    resolved = path.resolve()
    if resolved.is_relative_to(AGENT_DIR):
        return str(resolved.relative_to(AGENT_DIR))
    return str(resolved)


def _resolve(stored: str) -> Path:
    path = Path(stored)
    return path if path.is_absolute() else AGENT_DIR / path


def build_meta(name: str, cases_path: Path, *, is_bhh: bool) -> RunMeta:
    """Describe a new run, with the dataset revision when the cases are BHH."""
    dataset = None
    if is_bhh and BHH_META_PATH.exists():
        dataset = DatasetInfo.model_validate_json(BHH_META_PATH.read_text())
    return RunMeta(
        cases=name,
        cases_path=_portable(cases_path),
        jev_input_version=jev_input_version(),
        started_at=datetime.now(UTC).isoformat(),
        dataset=dataset,
    )


def prepare_run(
    *, split: str | None, cases_path: Path | None, resume: Path | None
) -> tuple[Path, list[Case]]:
    """Pick or create the run directory and load its cases.

    Raises:
        InputVersionMismatchError: If resuming a run recorded with another Jev input.
    """
    if resume:
        meta = load_meta(resume)
        check_input_version(meta, resume)
        return resume, load_cases(_resolve(meta.cases_path))
    path = bhh_path(split) if split else cases_path
    if path is None:
        raise ValueError("Give a BHH split, a cases file or a run to resume.")
    name = f"bhh-{split}" if split else path.stem
    run_dir = new_run_dir(name)
    write_meta(run_dir, build_meta(name, path, is_bhh=split is not None))
    return run_dir, load_cases(path)


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


async def main() -> None:
    """Record a run and report where it went."""
    args = parse_args()
    try:
        run_dir, cases = prepare_run(
            split=args.split, cases_path=args.cases, resume=args.resume
        )
    except InputVersionMismatchError as error:
        raise SystemExit(str(error)) from error
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
