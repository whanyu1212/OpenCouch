"""Deterministic eval for LiveKit voice therapeutic process control.

This runner checks the non-LLM contract that shapes the LiveKit voice
agent's therapeutic posture:

- detailed venting should not remain purely passive
- permission for active guidance should persist across follow-up turns
- explicit "just listen" requests should pause active guidance
- action requests should route to small conversational next steps, not
  automatic formal exercises

Usage:
    python eval/runners/voice_therapeutic_process_eval.py
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

from livekit.agents import ChatContext

from voice.livekit.agent import (
    _assess_therapeutic_process_state,
    _build_therapeutic_process_guidance,
    _compose_therapeutic_agent_instructions,
    _therapeutic_agent_kind_for_state,
)
from voice.livekit.session_data import TherapeuticProcessState
from voice.realtime import build_voice_system_prompt

DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "voice_therapeutic_process_v1.json"
)


def _load_cases(path: Path) -> list[dict[str, Any]]:
    """Load voice therapeutic-process eval cases.

    Args:
        path: Dataset path.

    Returns:
        Parsed eval cases.
    """

    return json.loads(path.read_text())


def _previous_state(raw: dict[str, Any]) -> TherapeuticProcessState:
    """Build prior process state from dataset fields.

    Args:
        raw: Previous-state fixture.

    Returns:
        Therapeutic process state.
    """

    return TherapeuticProcessState(
        session_intent=raw.get("session_intent", "vent"),
        guidance_permission=raw.get("guidance_permission", "unknown"),
        process_stage=raw.get("process_stage", "hold"),
    )


def _chat_context(case: dict[str, Any]) -> ChatContext:
    """Build chat context from optional prior assistant text.

    Args:
        case: Eval case payload.

    Returns:
        Chat context for the process-state classifier.
    """

    chat_ctx = ChatContext()
    assistant_text = case.get("assistant_text")
    if assistant_text:
        chat_ctx.add_message(role="assistant", content=assistant_text)
    return chat_ctx


def _check_static_prompt_contract() -> list[str]:
    """Check the static anti-passivity prompt contract.

    Returns:
        Failure messages.
    """

    failures: list[str] = []
    prompt = build_voice_system_prompt()
    for expected in (
        "Be actively collaborative",
        "move one small step forward",
        "do not become a passive echo",
        "Active support does not mean jumping to a structured exercise",
        "Do not introduce grounding, breathing, or other structured exercises",
    ):
        if expected not in prompt:
            failures.append(f"voice prompt missing {expected!r}")

    hold_prompt = _compose_therapeutic_agent_instructions(
        base_instructions="base",
        agent_kind="hold_space",
    )
    technique_prompt = _compose_therapeutic_agent_instructions(
        base_instructions="base",
        agent_kind="technique",
    )
    if "do not simply echo the user" not in hold_prompt:
        failures.append("hold_space prompt does not discourage passive echoing")
    if "conversational micro-steps before formal exercises" not in technique_prompt:
        failures.append("technique prompt does not prefer micro-steps")

    return failures


def _evaluate_case(case: dict[str, Any]) -> tuple[bool, list[str]]:
    """Evaluate one therapeutic-process case.

    Args:
        case: Eval case payload.

    Returns:
        ``(passed, failures)`` for the case.
    """

    state = _assess_therapeutic_process_state(
        turn_ctx=_chat_context(case),
        user_text=case["user_text"],
        previous=_previous_state(case.get("previous", {})),
    )
    guidance = _build_therapeutic_process_guidance(state)
    agent_kind = _therapeutic_agent_kind_for_state(state)

    failures: list[str] = []
    case_id = case["id"]

    if "expect_session_intent" in case and (
        state.session_intent != case["expect_session_intent"]
    ):
        failures.append(
            f"{case_id}: session_intent={state.session_intent}, expected={case['expect_session_intent']}"
        )

    if "expect_guidance_permission" in case and (
        state.guidance_permission != case["expect_guidance_permission"]
    ):
        failures.append(
            f"{case_id}: guidance_permission={state.guidance_permission}, expected={case['expect_guidance_permission']}"
        )

    if state.process_stage not in case.get("expect_process_stage_in", []):
        failures.append(
            f"{case_id}: process_stage={state.process_stage}, expected one of {case['expect_process_stage_in']}"
        )

    if agent_kind not in case.get("expect_agent_kind_in", []):
        failures.append(
            f"{case_id}: agent_kind={agent_kind}, expected one of {case['expect_agent_kind_in']}"
        )

    if case.get("expect_hot_thought") and not state.formulation.hot_thought:
        failures.append(f"{case_id}: expected hot_thought to be populated")

    expected_guidance = case.get("expect_guidance_contains_any", [])
    if expected_guidance and not any(item in guidance for item in expected_guidance):
        failures.append(
            f"{case_id}: guidance missing any of {expected_guidance}; guidance={guidance!r}"
        )

    return not failures, failures


def main() -> int:
    """Run the deterministic voice therapeutic-process eval.

    Returns:
        Process exit code.
    """

    cases = _load_cases(DATASET_PATH)
    print(f"Running voice therapeutic-process eval on {len(cases)} case(s).")
    print()

    failures = _check_static_prompt_contract()
    if failures:
        print("  FAIL static_prompt_contract")
    else:
        print("  PASS static_prompt_contract")

    passed = 0
    for case in cases:
        ok, case_failures = _evaluate_case(case)
        if ok:
            passed += 1
            print(f"  PASS {case['id']}")
        else:
            print(f"  FAIL {case['id']}")
            failures.extend(case_failures)

    print()
    print(f"Overall: {passed}/{len(cases)} cases passed")

    if failures:
        print()
        print("Failures:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
