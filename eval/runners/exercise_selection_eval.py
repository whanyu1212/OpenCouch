"""Runner for exercise selection evaluation.

Grades the keyword-based exercise selector (_select_exercise) on a
hand-curated dataset of message → expected-exercise pairs. This is
a purely deterministic eval — no LLM required.

Usage:
    python eval/runners/exercise_selection_eval.py
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

from agent.therapeutic.guided_exercise import _select_exercise

DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "datasets" / "exercise_selection_v1.json"
)


def _load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


def main() -> int:
    cases = _load_cases(DATASET_PATH)

    print(f"Running exercise selection eval on {len(cases)} case(s).")
    print()

    passed = 0
    failures: list[str] = []

    for case in cases:
        actual = _select_exercise(case["message"])
        expected = case["expected_exercise"]

        if actual == expected:
            passed += 1
        else:
            failures.append(
                f"FAIL {case['id']}: got {actual}, expected {expected}. "
                f"message={case['message']!r}"
            )

    print(f"  {passed}/{len(cases)} passed")

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
