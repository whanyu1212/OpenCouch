"""Runner for therapeutic dispatcher routing evaluation.

Grades the dispatcher's accuracy on a hand-curated dataset of message
→ expected-response-style pairs. Each case is labeled with a ``dispatch_tier``
(``regex`` or ``llm``) so the runner can report tier-scoped accuracy
separately — regex cases should always pass regardless of LLM
availability, while LLM cases only pass when the LLM classifier is
running.

Usage:
    # Regex-only path (no LLM calls)
    python eval/runners/therapeutic_routing_eval.py --mode deterministic

    # Hybrid path (LLM dispatcher when available)
    python eval/runners/therapeutic_routing_eval.py --mode hybrid

    # Auto-detect: hybrid if a provider is configured, else deterministic
    python eval/runners/therapeutic_routing_eval.py --mode auto  # default

The runner drives the real ``run_therapeutic_dispatch_node`` via a
minimal mock runtime, so the eval exercises exactly the same code path
as production (including the regex fast-path bypass). This differs from
``crisis_gate_eval.py`` which bypasses the node and drives the
classifier helpers directly — the therapeutic dispatcher's internal
decision tree (regex → LLM → fallback) IS the thing being graded, so
the node is the right entry point.
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
from agent.therapeutic.dispatcher import (
    CLARIFYING_NODE,
    CLOSING_NODE,
    GUIDED_EXERCISE_NODE,
    PSYCHOEDUCATION_NODE,
    REFLECTIVE_NODE,
    SUPPORTIVE_NODE,
    run_therapeutic_dispatch_node,
)
from core.config import create_configured_llm_client
from services.llm.base import BaseLLMClient

DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "datasets" / "therapeutic_routing_v0.json"
)

EvalMode = Literal["auto", "deterministic", "hybrid"]
DispatchTier = Literal["regex", "llm"]

# Map subgraph node name to logical response-style name for clean reporting.
_NODE_TO_RESPONSE_STYLE = {
    SUPPORTIVE_NODE: "supportive",
    REFLECTIVE_NODE: "reflective",
    CLARIFYING_NODE: "clarifying",
    PSYCHOEDUCATION_NODE: "psychoeducation",
    CLOSING_NODE: "closing",
    GUIDED_EXERCISE_NODE: "guided_exercise",
}


class _MockRuntime:
    """Minimal runtime stand-in that only exposes ``.context``.

    The dispatcher reads ``llm_client`` from context; everything else
    is left as None/placeholder. Matches the pattern used in the
    therapeutic routing unit tests.
    """

    def __init__(self, *, llm_client: BaseLLMClient | None) -> None:
        self.context = WorkflowContext(
            llm_client=llm_client,
            memory_store=OpenCouchMemoryStore(),
            crisis_log_backend=InMemoryCrisisLogBackend(),
            memory_mode=MemoryMode.LOCAL,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run therapeutic dispatcher routing evaluation."
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
    return parser


def _load_cases(path: Path) -> list[dict[str, Any]]:
    """Load the eval dataset from disk."""

    return json.loads(path.read_text())


def _resolve_llm_client(mode: EvalMode) -> tuple[BaseLLMClient | None, str]:
    """Return the LLM client + resolved mode label.

    - deterministic: always returns (None, "deterministic")
    - hybrid: returns (client, "hybrid") — raises if no client is configured
    - auto: tries hybrid first, falls back to deterministic on any error
    """

    if mode == "deterministic":
        return None, "deterministic"

    if mode == "hybrid":
        return create_configured_llm_client(), "hybrid"

    # auto
    try:
        return create_configured_llm_client(), "hybrid"
    except Exception:
        return None, "deterministic"


def _build_state(case: dict[str, Any]) -> AgentState:
    """Return a partial AgentState for the dispatcher node to read.

    The dispatcher only reads ``message``, ``history``, and
    ``working_memory`` so the rest of the AgentState schema can be
    omitted. We cast to AgentState to satisfy the type checker.
    """

    state: Any = {
        "message": case["message"],
        "history": case.get("history", []),
        "working_memory": case.get("working_memory", []),
    }
    return cast(AgentState, state)


async def _evaluate_case(
    case: dict[str, Any],
    llm_client: BaseLLMClient | None,
) -> tuple[bool, str, str | None]:
    """Run one case through the dispatcher and compare to the expected style.

    Returns:
        ``(passed, actual_style, failure_detail)``. ``failure_detail`` is
        ``None`` on a passing case, or a human-readable explanation on a
        failing case.
    """

    runtime = _MockRuntime(llm_client=llm_client)
    state = _build_state(case)
    cmd = await run_therapeutic_dispatch_node(state, runtime)  # type: ignore[arg-type]

    update = cast(dict[str, Any], cmd.update or {})
    actual_style = str(
        update.get("response_style")
        or _NODE_TO_RESPONSE_STYLE.get(cmd.goto, f"unknown({cmd.goto})")
    )
    expected_style = case.get("expected_response_style", case.get("expected_mode"))

    if actual_style == expected_style:
        return True, actual_style, None

    tier = case.get("dispatch_tier", "?")
    detail = (
        f"FAIL [{tier}] {case['id']}: "
        f"got {actual_style}, expected {expected_style}. "
        f"message={case['message']!r}"
    )
    return False, actual_style, detail


async def _run(mode: EvalMode, dataset_path: Path) -> int:
    """Drive the full eval and return a process exit code."""

    cases = _load_cases(dataset_path)
    llm_client, resolved_mode = _resolve_llm_client(mode)

    print(
        f"Running therapeutic routing eval in {resolved_mode} mode "
        f"on {len(cases)} case(s) from {dataset_path.name}."
    )
    print()

    # Tier-scoped accounting so regex and llm cases can be reported separately.
    by_tier: dict[str, dict[str, int]] = {
        "regex": {"total": 0, "passed": 0},
        "llm": {"total": 0, "passed": 0},
    }
    failures: list[str] = []

    for case in cases:
        tier = case.get("dispatch_tier", "regex")
        if tier not in by_tier:
            by_tier[tier] = {"total": 0, "passed": 0}
        by_tier[tier]["total"] += 1

        passed, actual, detail = await _evaluate_case(case, llm_client=llm_client)
        if passed:
            by_tier[tier]["passed"] += 1
        elif detail is not None:
            failures.append(detail)

    # Report
    for tier_name, counts in sorted(by_tier.items()):
        if counts["total"] == 0:
            continue
        print(f"  {tier_name:10s} {counts['passed']:2d}/{counts['total']:2d} passed")

    overall_total = sum(c["total"] for c in by_tier.values())
    overall_passed = sum(c["passed"] for c in by_tier.values())
    print()
    print(f"Overall: {overall_passed}/{overall_total} passed")

    if failures:
        print()
        print("Failures:")
        for detail in failures:
            print(f"  {detail}")

    # In deterministic mode, LLM-tier failures are EXPECTED (regex can't
    # handle those cases). Only regex-tier failures count against the
    # exit code. In hybrid mode, any failure counts.
    if resolved_mode == "deterministic":
        blocking_failures = by_tier.get("regex", {"total": 0, "passed": 0})
        blocking_failed = blocking_failures["total"] - blocking_failures["passed"]
        if blocking_failed > 0:
            print()
            print(
                f"{blocking_failed} regex-tier failure(s) — these should pass "
                f"regardless of LLM availability."
            )
            return 1
        llm_tier = by_tier.get("llm", {"total": 0, "passed": 0})
        if llm_tier["total"] > 0:
            print()
            print(
                f"Note: {llm_tier['total'] - llm_tier['passed']} llm-tier case(s) "
                f"did not pass in deterministic mode. This is expected — those "
                f"cases require the LLM classifier. Run with --mode hybrid to "
                f"grade them."
            )
        return 0

    # hybrid mode: every failure counts
    if overall_passed < overall_total:
        return 1
    return 0


def main() -> int:
    """Entry point for the therapeutic routing eval runner."""

    args = _build_parser().parse_args()
    return asyncio.run(_run(args.mode, args.dataset))


if __name__ == "__main__":
    raise SystemExit(main())
