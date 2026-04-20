"""Runner for the deterministic phase-1 memory write policy eval.

Tests that the candidate-builder + policy pair routes cases into the
expected action bucket:

- ``commit_now``
- ``commit_at_session_end``
- ``require_repetition``
- ``drop``

Usage:
    python eval/runners/memory_write_policy_eval.py
"""

# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2] / "apps" / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agent.memory.candidates import build_procedural_candidate, build_semantic_candidate
from agent.memory.models import EntityRef, MemoryWrite, ProceduralRuleDraft
from agent.memory.write_policy import (
    decide_procedural_candidate,
    decide_semantic_candidate,
)

DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "datasets" / "memory_write_policy_v1.json"
)


def _load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


def _semantic_action(case: dict[str, Any]) -> str:
    candidate = build_semantic_candidate(
        MemoryWrite(
            category=case["category"],
            subject=EntityRef(type="User", identifier="eval-user"),
            predicate=case["predicate"],
            object=EntityRef(
                type=case["object_type"],
                identifier=case["object_identifier"],
            ),
            evidence_quote=case["evidence_quote"],
            confidence="high",
            source_session_id="eval-session",
            source_turn_index=2,
        ),
        message=case["message"],
    )
    return decide_semantic_candidate(candidate).action


def _procedural_action(case: dict[str, Any]) -> str:
    candidate = build_procedural_candidate(
        ProceduralRuleDraft(
            rule=case["rule"],
            evidence=case["evidence"],
        ),
        message=case["message"],
        session_id="eval-session",
        turn_index=2,
    )
    return decide_procedural_candidate(candidate).action


def _evaluate_case(case: dict[str, Any]) -> tuple[bool, str | None]:
    if case["layer"] == "semantic":
        actual = _semantic_action(case)
    else:
        actual = _procedural_action(case)

    expected = case["expected_action"]
    if actual != expected:
        return (
            False,
            f"FAIL {case['id']}: actual={actual}, expected={expected}",
        )
    return True, None


def main() -> int:
    cases = _load_cases(DATASET_PATH)
    print(f"Running memory write policy eval on {len(cases)} case(s).")
    print()

    passed = 0
    failures: list[str] = []

    for case in cases:
        ok, detail = _evaluate_case(case)
        if ok:
            passed += 1
        elif detail:
            failures.append(detail)

    for action in (
        "commit_now",
        "commit_at_session_end",
        "require_repetition",
        "drop",
    ):
        action_cases = [c for c in cases if c["expected_action"] == action]
        action_passed = sum(
            1
            for c in action_cases
            if not any(c["id"] in failure for failure in failures)
        )
        print(f"  {action:22s} {action_passed:2d}/{len(action_cases):2d} passed")

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


if __name__ == "__main__":
    raise SystemExit(main())
