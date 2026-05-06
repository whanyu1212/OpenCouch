"""Runner for guided-exercise selection evaluation.

Grades the production guided-exercise start path on a hand-curated dataset of
message -> selected exercise / option-list outcomes.

Usage:
    # Regex / pending-choice fallback path only. LLM-tier misses are reported
    # but do not fail the process.
    python eval/runners/exercise_selection_eval.py --mode deterministic

    # LLM-primary path via the configured provider.
    python eval/runners/exercise_selection_eval.py --mode hybrid

    # Auto-detect: hybrid if a provider is configured, else deterministic.
    python eval/runners/exercise_selection_eval.py --mode auto  # default
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Literal, cast

BACKEND_ROOT = Path(__file__).resolve().parents[2] / "apps" / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.memory.modes import MemoryMode
from agent.memory.store import OpenCouchMemoryStore
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.therapeutic.guided_exercise import run_guided_exercise_response_node
from core.config import create_configured_llm_client
from services.base import BaseLLMClient

DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "datasets" / "exercise_selection_v1.json"
)

EvalMode = Literal["auto", "deterministic", "hybrid"]
SelectionTier = Literal["deterministic", "llm"]


class _MockRuntime:
    """Minimal runtime that exposes the guided-exercise node dependencies."""

    def __init__(self, *, llm_client: BaseLLMClient | None) -> None:
        self.context = WorkflowContext(
            llm_client=llm_client,
            memory_store=OpenCouchMemoryStore(),
            crisis_log_backend=InMemoryCrisisLogBackend(),
            memory_mode=MemoryMode.INCOGNITO,
        )


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        Configured argument parser.
    """

    parser = argparse.ArgumentParser(
        description="Run guided-exercise selection evaluation."
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "deterministic", "hybrid"],
        default="auto",
        help=(
            "Evaluation mode. 'auto' uses the configured LLM client when "
            "available and falls back to deterministic mode otherwise."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help=f"Dataset JSON path. Default: {DATASET_PATH}",
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Optional single case id to run.",
    )
    return parser


def _load_cases(path: Path) -> list[dict[str, Any]]:
    """Load the eval dataset.

    Args:
        path: Dataset JSON path.

    Returns:
        List of case dictionaries.
    """

    return json.loads(path.read_text())


def _resolve_llm_client(mode: EvalMode) -> tuple[BaseLLMClient | None, str]:
    """Return the LLM client plus resolved mode label.

    Args:
        mode: Requested eval mode.

    Returns:
        Tuple of optional LLM client and resolved mode label.
    """

    if mode == "deterministic":
        return None, "deterministic"

    if mode == "hybrid":
        return create_configured_llm_client(), "hybrid"

    try:
        return create_configured_llm_client(), "hybrid"
    except Exception:
        return None, "deterministic"


def _build_state(case: dict[str, Any]) -> AgentState:
    """Build the partial AgentState read by the exercise node.

    Args:
        case: Dataset case.

    Returns:
        Cast AgentState dictionary for the node.
    """

    state: dict[str, Any] = {
        "message": case["message"],
        "history": case.get("history", []),
        "session_progress": {"turn_count": 1},
        "exercise_state": case.get("exercise_state", {}),
    }
    return cast(AgentState, state)


def _actual_outcome(delta: dict[str, Any]) -> tuple[str, str | None, list[str]]:
    """Extract the selection outcome from a guided-exercise node delta.

    Args:
        delta: State delta returned by the guided-exercise node.

    Returns:
        Tuple of outcome kind, selected exercise id, and offered option ids.
    """

    exercise_state = delta.get("exercise_state", {}) or {}
    selected = exercise_state.get("exercise_type")
    options = list(exercise_state.get("exercise_selection_options") or [])
    if selected is not None:
        return "selected", str(selected), options
    if len(options) >= 2:
        return "options", None, options
    return "unknown", None, options


def _evaluate_expected(
    case: dict[str, Any],
    *,
    outcome: str,
    selected: str | None,
    options: list[str],
) -> str | None:
    """Return a failure detail if the actual outcome misses expectations.

    Args:
        case: Dataset case.
        outcome: Actual outcome kind.
        selected: Actual selected exercise id, if any.
        options: Actual offered option ids, if any.

    Returns:
        Failure detail, or None when the case passes.
    """

    expected_outcome = case["expected_outcome"]
    if outcome != expected_outcome:
        return f"got outcome={outcome}, expected {expected_outcome}"

    if expected_outcome == "selected":
        expected_exercise = case["expected_exercise"]
        if selected != expected_exercise:
            return f"got exercise={selected}, expected {expected_exercise}"
        return None

    required_options = case.get("expected_options_include") or []
    missing = [option for option in required_options if option not in options]
    if missing:
        return f"options missing required id(s): {missing!r}; got {options!r}"

    include_any = case.get("expected_options_include_any") or []
    if include_any and not any(option in options for option in include_any):
        return f"options missing any of {include_any!r}; got {options!r}"

    excluded = case.get("expected_options_exclude") or []
    present_excluded = [option for option in excluded if option in options]
    if present_excluded:
        return f"options included excluded id(s): {present_excluded!r}"

    return None


async def _evaluate_case(
    case: dict[str, Any],
    *,
    llm_client: BaseLLMClient | None,
) -> tuple[bool, str, str | None]:
    """Run one case through the guided-exercise node.

    Args:
        case: Dataset case.
        llm_client: Optional LLM client for hybrid mode.

    Returns:
        Tuple of pass flag, actual outcome label, and optional failure detail.
    """

    runtime = _MockRuntime(llm_client=llm_client)
    state = _build_state(case)
    delta = await run_guided_exercise_response_node(
        state,
        runtime,  # type: ignore[arg-type]
    )
    outcome, selected, options = _actual_outcome(delta)
    failure = _evaluate_expected(
        case,
        outcome=outcome,
        selected=selected,
        options=options,
    )
    if failure is None:
        return True, outcome, None

    detail = (
        f"FAIL [{case.get('selection_tier', '?')}] {case['id']}: {failure}. "
        f"message={case['message']!r}"
    )
    return False, outcome, detail


async def _run(
    mode: EvalMode,
    dataset_path: Path,
    case_id: str | None,
) -> int:
    """Drive the full eval suite.

    Args:
        mode: Requested evaluation mode.
        dataset_path: Dataset JSON path.
        case_id: Optional single case id.

    Returns:
        Process exit code.
    """

    cases = _load_cases(dataset_path)
    if case_id is not None:
        cases = [case for case in cases if case["id"] == case_id]
        if not cases:
            print(f"No exercise selection eval case found for id={case_id!r}.")
            return 1

    llm_client, resolved_mode = _resolve_llm_client(mode)
    print(
        f"Running exercise selection eval in {resolved_mode} mode "
        f"on {len(cases)} case(s) from {dataset_path.name}."
    )
    print()

    by_tier: dict[str, dict[str, int]] = {
        "deterministic": {"total": 0, "passed": 0},
        "llm": {"total": 0, "passed": 0},
    }
    failures: list[str] = []

    for case in cases:
        tier: SelectionTier = case.get("selection_tier", "deterministic")
        if tier not in by_tier:
            by_tier[tier] = {"total": 0, "passed": 0}
        by_tier[tier]["total"] += 1

        passed, _actual, detail = await _evaluate_case(
            case,
            llm_client=llm_client,
        )
        if passed:
            by_tier[tier]["passed"] += 1
        elif detail is not None:
            failures.append(detail)

    for tier_name, counts in sorted(by_tier.items()):
        if counts["total"] == 0:
            continue
        print(f"  {tier_name:13s} {counts['passed']:2d}/{counts['total']:2d} passed")

    overall_total = sum(counts["total"] for counts in by_tier.values())
    overall_passed = sum(counts["passed"] for counts in by_tier.values())
    print()
    print(f"Overall: {overall_passed}/{overall_total} passed")

    if failures:
        print()
        print("Failures:")
        for detail in failures:
            print(f"  {detail}")

    if resolved_mode == "deterministic":
        deterministic = by_tier.get("deterministic", {"total": 0, "passed": 0})
        blocking_failed = deterministic["total"] - deterministic["passed"]
        if blocking_failed > 0:
            print()
            print(
                f"{blocking_failed} deterministic-tier failure(s) — these should "
                "pass without an LLM client."
            )
            return 1

        llm_tier = by_tier.get("llm", {"total": 0, "passed": 0})
        if llm_tier["total"] > 0:
            print()
            print(
                f"Note: {llm_tier['total'] - llm_tier['passed']} llm-tier case(s) "
                "did not pass in deterministic mode. This is expected; run with "
                "--mode hybrid to grade the LLM-primary selector."
            )
        return 0

    if overall_passed < overall_total:
        return 1
    return 0


def main() -> int:
    """Run the exercise selection eval command.

    Returns:
        Process exit code.
    """

    args = _build_parser().parse_args()
    return asyncio.run(_run(args.mode, args.dataset, args.case))


if __name__ == "__main__":
    raise SystemExit(main())
