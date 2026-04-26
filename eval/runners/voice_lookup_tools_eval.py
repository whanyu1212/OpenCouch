"""Deterministic eval for LiveKit voice lookup tools.

This runner exercises the tool contract directly rather than relying on LLM
tool selection. It covers:

- explicit grounded factual lookup
- location-aware crisis resource lookup
- fail-closed behavior when search is unavailable

Usage:
    python eval/runners/voice_lookup_tools_eval.py
"""

# ruff: noqa: E402

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2] / "apps" / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from voice.livekit.session_data import SessionData
from voice.livekit.tools import (
    answer_grounded_factual_lookup,
    provide_crisis_resources,
)

DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "datasets" / "voice_lookup_tools_v1.json"
)

logging.getLogger("agent.tools.web_search").setLevel(logging.CRITICAL)


class FakeLookupLLM:
    """Deterministic text client for search-backed voice tools."""

    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        """Return the next scripted lookup response.

        Args:
            prompt: Prompt sent by the lookup helper.
            system_instruction: Optional system instruction.
            use_search: Whether provider-native search was requested.

        Returns:
            Scripted text response.
        """

        self.calls.append(
            {
                "prompt": prompt,
                "system_instruction": system_instruction,
                "use_search": use_search,
            }
        )
        if not self.responses:
            raise AssertionError("No scripted lookup response left.")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _load_cases(path: Path) -> list[dict[str, Any]]:
    """Load voice lookup eval cases.

    Args:
        path: Dataset path.

    Returns:
        Parsed eval cases.
    """

    return json.loads(path.read_text())


def _parse_responses(raw_responses: list[Any]) -> list[str | Exception]:
    """Convert dataset response fixtures into fake LLM responses.

    Args:
        raw_responses: JSON response fixtures.

    Returns:
        Scripted text or exception responses.
    """

    responses: list[str | Exception] = []
    for response in raw_responses:
        if isinstance(response, dict) and "raises" in response:
            responses.append(RuntimeError(str(response["raises"])))
        else:
            responses.append(str(response))
    return responses


async def _run_tool(context: SimpleNamespace, step: dict[str, Any]) -> str:
    """Run one voice lookup tool step.

    Args:
        context: Fake LiveKit ``RunContext`` with userdata.
        step: Dataset step payload.

    Returns:
        Tool result string.
    """

    tool = step["tool"]
    args = step.get("args", {})
    if tool == "answer_grounded_factual_lookup":
        return await answer_grounded_factual_lookup(context, **args)
    if tool == "provide_crisis_resources":
        return await provide_crisis_resources(context, **args)
    raise ValueError(f"Unknown tool: {tool}")


def _check_step(
    *,
    case_id: str,
    step_index: int,
    step: dict[str, Any],
    result: str,
    lookup_llm: FakeLookupLLM | None,
) -> list[str]:
    """Return assertion failures for one eval step.

    Args:
        case_id: Current case id.
        step_index: 1-based step number.
        step: Dataset step payload.
        result: Tool result string.
        lookup_llm: Fake search-capable text client, if configured.

    Returns:
        Failure messages for this step.
    """

    failures: list[str] = []
    prefix = f"{case_id} step {step_index}"

    for expected in step.get("expect_contains", []):
        if expected not in result:
            failures.append(f"{prefix}: missing text {expected!r}; result={result!r}")

    expected_search = step.get("expect_use_search")
    if expected_search is not None:
        actual_search = (
            [bool(call["use_search"]) for call in lookup_llm.calls]
            if lookup_llm is not None
            else []
        )
        if actual_search != expected_search:
            failures.append(
                f"{prefix}: use_search={actual_search}, expected={expected_search}"
            )

    return failures


async def _evaluate_case(case: dict[str, Any]) -> tuple[bool, list[str]]:
    """Evaluate one voice lookup case.

    Args:
        case: Dataset case payload.

    Returns:
        ``(passed, failures)`` for the case.
    """

    setup = case.get("setup", {})
    lookup_llm = None
    if setup.get("llm_available", True):
        lookup_llm = FakeLookupLLM(_parse_responses(setup.get("llm_responses", [])))

    userdata = SessionData(
        user_id="voice-lookup-eval-user",
        thread_id="voice-lookup-eval",
        llm_client=lookup_llm,
    )
    context = SimpleNamespace(userdata=userdata)

    failures: list[str] = []
    for step_index, step in enumerate(case["steps"], start=1):
        result = await _run_tool(context, step)
        failures.extend(
            _check_step(
                case_id=case["id"],
                step_index=step_index,
                step=step,
                result=result,
                lookup_llm=lookup_llm,
            )
        )

    return not failures, failures


async def _amain() -> int:
    """Run the voice lookup eval.

    Returns:
        Process exit code.
    """

    cases = _load_cases(DATASET_PATH)
    print(f"Running voice lookup-tools eval on {len(cases)} case(s).")
    print()

    passed = 0
    failures: list[str] = []
    for case in cases:
        ok, case_failures = await _evaluate_case(case)
        if ok:
            passed += 1
            print(f"  PASS {case['id']}")
        else:
            print(f"  FAIL {case['id']}")
            failures.extend(case_failures)

    print()
    print(f"Overall: {passed}/{len(cases)} passed")

    if failures:
        print()
        print("Failures:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print()
    print("All cases passed.")
    return 0


def main() -> int:
    """Run the async eval entrypoint.

    Returns:
        Process exit code.
    """

    return asyncio.run(_amain())


if __name__ == "__main__":
    raise SystemExit(main())
