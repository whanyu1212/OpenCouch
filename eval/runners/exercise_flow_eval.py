"""Runner for exercise flow (state transition) evaluation.

Grades the guided exercise node's state transitions on a hand-curated
dataset of (exercise_type, step, message) → expected transition. This
drives the real run_guided_exercise_response_node() with a mock runtime
(no LLM), so it tests the deterministic classifier and state machine.

Transitions:
- "advance": exercise_step incremented, exercise_type unchanged
- "hold": no progress change (or progress unchanged)
- "stuck": no progress change (stuck path, rephrase offered)
- "exit": exercise_type and exercise_step cleared to None
- "complete": exercise_type and exercise_step cleared (last step)

Usage:
    python eval/runners/exercise_flow_eval.py
"""

# ruff: noqa: E402

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, cast

BACKEND_ROOT = Path(__file__).resolve().parents[2] / "apps" / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.memory.modes import MemoryMode
from agent.memory.store import OpenCouchMemoryStore
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.therapeutic.guided_exercise import run_guided_exercise_response_node

DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "datasets" / "exercise_flow_v1.json"
)


class _MockRuntime:
    """Minimal runtime with no LLM — tests deterministic paths only."""

    def __init__(self) -> None:
        self.context = WorkflowContext(
            llm_client=None,
            memory_store=OpenCouchMemoryStore(),
            crisis_log_backend=InMemoryCrisisLogBackend(),
            memory_mode=MemoryMode.INCOGNITO,
        )


def _load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


def _classify_transition(
    case: dict[str, Any],
    delta: dict[str, Any],
) -> str:
    """Determine the actual transition from the delta output.

    Returns one of: "advance", "hold", "stuck", "exit", "complete".
    """

    progress = delta.get("progress", {})
    exercise_type = progress.get("exercise_type", "MISSING")
    exercise_step = progress.get("exercise_step", "MISSING")

    # If exercise_type is None → cleared (exit or complete)
    if exercise_type is None and exercise_step is None:
        # Distinguish exit from complete: complete happens on the last step
        from agent.therapeutic.guided_exercise import _EXERCISE_REGISTRY

        orig_type = case["exercise_type"]
        orig_step = case["exercise_step"]
        steps = _EXERCISE_REGISTRY.get(orig_type, ())
        if orig_step == len(steps) - 1:
            return "complete"
        return "exit"

    # If step advanced
    if exercise_step == case["exercise_step"] + 1:
        return "advance"

    # If no progress key at all, or step unchanged → hold or stuck
    # Distinguish by response text (stuck offers rephrase)
    response_text = delta.get("response", {}).get("text", "").lower()
    if "make it smaller" in response_text or "simpler" in response_text:
        return "stuck"

    return "hold"


async def _evaluate_case(
    case: dict[str, Any],
) -> tuple[bool, str, str | None]:
    """Run one case and compare to expected transition."""

    runtime = _MockRuntime()
    state: Any = {
        "message": case["message"],
        "history": [],
        "progress": {
            "exercise_type": case["exercise_type"],
            "exercise_step": case["exercise_step"],
            "turn_count": 2,
        },
        "response": {},
        "routing": {},
    }

    delta = await run_guided_exercise_response_node(
        cast(AgentState, state),
        runtime,  # type: ignore[arg-type]
    )

    actual = _classify_transition(case, delta)
    expected = case["expected_transition"]

    if actual == expected:
        return True, actual, None

    detail = (
        f"FAIL {case['id']}: got {actual}, expected {expected}. "
        f"message={case['message']!r} "
        f"exercise={case['exercise_type']}@step{case['exercise_step']}"
    )
    return False, actual, detail


async def _run() -> int:
    cases = _load_cases(DATASET_PATH)

    print(f"Running exercise flow eval on {len(cases)} case(s).")
    print()

    passed = 0
    failures: list[str] = []

    # Group by transition type for reporting
    by_transition: dict[str, dict[str, int]] = {}

    for case in cases:
        expected = case["expected_transition"]
        if expected not in by_transition:
            by_transition[expected] = {"total": 0, "passed": 0}
        by_transition[expected]["total"] += 1

        ok, actual, detail = await _evaluate_case(case)
        if ok:
            passed += 1
            by_transition[expected]["passed"] += 1
        elif detail is not None:
            failures.append(detail)

    # Report by transition type
    for transition, counts in sorted(by_transition.items()):
        print(f"  {transition:10s} {counts['passed']:2d}/{counts['total']:2d} passed")

    print()
    print(f"Overall: {passed}/{len(cases)} passed")

    if failures:
        print()
        print("Failures:")
        for detail in failures:
            print(f"  {detail}")
        return 1

    print()
    print("All cases passed.")
    return 0


def main() -> int:
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
