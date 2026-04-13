"""Runner for exercise memory (completion fact writing) evaluation.

Tests that exercise completions write semantic facts and that exits,
stuck, hold, and incognito mode do NOT write. Deterministic — no LLM.

Usage:
    python eval/runners/exercise_memory_eval.py
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

from agent.state import AgentState
from agent.therapeutic.guided_exercise import run_guided_exercise_response_node

DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "datasets" / "exercise_memory_v1.json"
)


class _RecordingMemoryStore:
    """In-memory store that records aput calls."""

    def __init__(self) -> None:
        self.writes: list[dict[str, Any]] = []

    async def aput(
        self,
        namespace: Any,
        key: str,
        value: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        self.writes.append({"namespace": namespace, "key": key, "value": value})


class _MockRuntime:
    def __init__(self, memory_store: Any, memory_mode: str) -> None:
        self.context = {
            "llm_client": None,
            "memory_store": memory_store,
            "crisis_log_backend": None,
            "memory_mode": memory_mode,
        }


def _load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


async def _evaluate_case(case: dict[str, Any]) -> tuple[bool, str | None]:
    store = _RecordingMemoryStore()
    runtime = _MockRuntime(
        memory_store=store,
        memory_mode=case["memory_mode"],
    )
    state: Any = {
        "message": case["message"],
        "history": [],
        "user_id": "eval-user",
        "session_id": "eval-session",
        "progress": {
            "exercise_type": case["exercise_type"],
            "exercise_step": case["exercise_step"],
            "turn_count": 5,
        },
        "response": {},
        "routing": {},
    }

    await run_guided_exercise_response_node(
        cast(AgentState, state),
        runtime,  # type: ignore[arg-type]
    )

    fact_written = len(store.writes) > 0
    expected = case["expected_fact_written"]

    if fact_written != expected:
        detail = (
            f"FAIL {case['id']}: fact_written={fact_written}, "
            f"expected={expected}. message={case['message']!r}"
        )
        return False, detail

    # If a fact was written, check its content
    if fact_written and expected:
        fact = store.writes[0]["value"]
        if fact["category"] != case["expected_category"]:
            return False, (
                f"FAIL {case['id']}: category={fact['category']}, "
                f"expected={case['expected_category']}"
            )
        obj_id = fact.get("object", {}).get("identifier", "")
        if case["expected_object_contains"] not in obj_id:
            return False, (
                f"FAIL {case['id']}: object.identifier={obj_id!r}, "
                f"expected to contain {case['expected_object_contains']!r}"
            )

    return True, None


async def _run() -> int:
    cases = _load_cases(DATASET_PATH)
    print(f"Running exercise memory eval on {len(cases)} case(s).")
    print()

    passed = 0
    failures: list[str] = []

    for case in cases:
        ok, detail = await _evaluate_case(case)
        if ok:
            passed += 1
        elif detail:
            failures.append(detail)

    # Report by expected outcome
    write_cases = [c for c in cases if c["expected_fact_written"]]
    no_write_cases = [c for c in cases if not c["expected_fact_written"]]
    write_passed = sum(
        1 for c in write_cases if not any(c["id"] in f for f in failures)
    )
    no_write_passed = sum(
        1 for c in no_write_cases if not any(c["id"] in f for f in failures)
    )

    print(f"  write      {write_passed:2d}/{len(write_cases):2d} passed")
    print(f"  no_write   {no_write_passed:2d}/{len(no_write_cases):2d} passed")
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
