"""Runner for memory-control routing evaluation.

Grades the text memory-control gate on a hand-curated dataset of message ->
memory-control / pass-through outcomes.

Usage:
    # Hard deterministic routes only. LLM-tier misses are reported but do not
    # fail the process.
    python eval/runners/memory_control_routing_eval.py --mode deterministic

    # LLM-primary middle path via the configured provider.
    python eval/runners/memory_control_routing_eval.py --mode hybrid

    # Auto-detect: hybrid if a provider is configured, else deterministic.
    python eval/runners/memory_control_routing_eval.py --mode auto  # default
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
from agent.nodes.memory_control_gate import run_memory_control_gate_node
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from core.config import create_configured_llm_client
from services.base import BaseLLMClient

DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "datasets" / "memory_control_routing_v1.json"
)

EvalMode = Literal["auto", "deterministic", "hybrid"]
RoutingTier = Literal["deterministic", "llm"]


class _MockRuntime:
    """Minimal runtime that exposes the memory-control gate dependencies."""

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
        description="Run memory-control routing evaluation."
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
    """Build the partial AgentState read by the memory-control gate.

    Args:
        case: Dataset case.

    Returns:
        Cast AgentState dictionary for the gate.
    """

    state: dict[str, Any] = {
        "message": case["message"],
        "history": case.get("history", []),
        "working_memory": case.get("working_memory", []),
        "memory_control": case.get("memory_control", {}),
    }
    return cast(AgentState, state)


def _actual_outcome(command: Any) -> tuple[str, dict[str, Any]]:
    """Extract the routing outcome and memory-control action.

    Args:
        command: Command returned by the memory-control gate.

    Returns:
        Tuple of outcome label and action dictionary.
    """

    update = cast(dict[str, Any], command.update or {})
    if command.goto == "memory_control_node":
        memory_control = cast(dict[str, Any], update.get("memory_control", {}) or {})
        action = memory_control.get("action", {}) or update.get(
            "memory_control_action", {}
        )
        return "memory_control", dict(action or {})
    if command.goto == "grounded_lookup_gate_node":
        return "pass_through", {}
    return f"unknown({command.goto})", {}


def _check_text_contains(
    value: str,
    *,
    required: list[str],
    include_any: list[str],
) -> str | None:
    """Return failure detail when text misses required terms.

    Args:
        value: Text to inspect.
        required: Terms that must all be present.
        include_any: Terms where at least one must be present.

    Returns:
        Failure detail, or None when the text satisfies expectations.
    """

    lowered = value.lower()
    missing = [term for term in required if term.lower() not in lowered]
    if missing:
        return f"missing required term(s): {missing!r}; got {value!r}"

    if include_any and not any(term.lower() in lowered for term in include_any):
        return f"missing any of {include_any!r}; got {value!r}"

    return None


def _evaluate_expected(
    case: dict[str, Any],
    *,
    outcome: str,
    action: dict[str, Any],
) -> str | None:
    """Return a failure detail if the actual outcome misses expectations.

    Args:
        case: Dataset case.
        outcome: Actual outcome label.
        action: Actual memory-control action.

    Returns:
        Failure detail, or None when the case passes.
    """

    expected_outcome = case["expected_outcome"]
    if outcome != expected_outcome:
        return f"got outcome={outcome}, expected {expected_outcome}"

    if expected_outcome != "memory_control":
        return None

    expected_action_type = case["expected_action_type"]
    if action.get("type") != expected_action_type:
        return f"got action={action.get('type')}, expected {expected_action_type}"

    if "expected_enabled" in case and action.get("enabled") != case["expected_enabled"]:
        return (
            f"got enabled={action.get('enabled')}, expected {case['expected_enabled']}"
        )

    if (
        "expected_target_kind" in case
        and action.get("target_kind") != case["expected_target_kind"]
    ):
        return (
            f"got target_kind={action.get('target_kind')}, "
            f"expected {case['expected_target_kind']}"
        )

    if expected_action_type == "forget_by_query":
        failure = _check_text_contains(
            str(action.get("query", "")),
            required=case.get("expected_query_contains") or [],
            include_any=case.get("expected_query_contains_any") or [],
        )
        if failure is not None:
            return f"query {failure}"

    if expected_action_type == "save_preference":
        failure = _check_text_contains(
            str(action.get("rule_text", "")),
            required=case.get("expected_rule_contains") or [],
            include_any=case.get("expected_rule_contains_any") or [],
        )
        if failure is not None:
            return f"rule_text {failure}"

    return None


async def _evaluate_case(
    case: dict[str, Any],
    *,
    llm_client: BaseLLMClient | None,
) -> tuple[bool, str, str | None]:
    """Run one case through the memory-control gate.

    Args:
        case: Dataset case.
        llm_client: Optional LLM client for hybrid mode.

    Returns:
        Tuple of pass flag, actual outcome label, and optional failure detail.
    """

    runtime = _MockRuntime(llm_client=llm_client)
    state = _build_state(case)
    command = await run_memory_control_gate_node(
        state,
        runtime,  # type: ignore[arg-type]
    )
    outcome, action = _actual_outcome(command)
    failure = _evaluate_expected(case, outcome=outcome, action=action)
    if failure is None:
        return True, outcome, None

    detail = (
        f"FAIL [{case.get('routing_tier', '?')}] {case['id']}: {failure}. "
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
            print(f"No memory-control routing eval case found for id={case_id!r}.")
            return 1

    llm_client, resolved_mode = _resolve_llm_client(mode)
    print(
        f"Running memory-control routing eval in {resolved_mode} mode "
        f"on {len(cases)} case(s) from {dataset_path.name}."
    )
    print()

    by_tier: dict[str, dict[str, int]] = {
        "deterministic": {"total": 0, "passed": 0},
        "llm": {"total": 0, "passed": 0},
    }
    failures: list[str] = []

    for case in cases:
        tier: RoutingTier = case.get("routing_tier", "deterministic")
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
                f"{blocking_failed} deterministic-tier failure(s) - these should "
                "pass without an LLM client."
            )
            return 1

        llm_tier = by_tier.get("llm", {"total": 0, "passed": 0})
        if llm_tier["total"] > 0:
            print()
            print(
                f"Note: {llm_tier['total'] - llm_tier['passed']} llm-tier case(s) "
                "did not pass in deterministic mode. This is expected; run with "
                "--mode hybrid to grade the LLM-primary memory-control classifier."
            )
        return 0

    if overall_passed < overall_total:
        return 1
    return 0


def main() -> int:
    """Run the memory-control routing eval command.

    Returns:
        Process exit code.
    """

    args = _build_parser().parse_args()
    return asyncio.run(_run(args.mode, args.dataset, args.case))


if __name__ == "__main__":
    raise SystemExit(main())
