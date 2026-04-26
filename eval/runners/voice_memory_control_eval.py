"""Deterministic eval for LiveKit voice memory-control tools.

This runner exercises the tool contract directly rather than relying on
LLM tool selection. It verifies the durable behavior that must remain
stable once the Realtime model decides to call a memory-control tool:

- list saved memory
- report memory status
- toggle proactive recall
- prepare and confirm deletion
- cancel deletion
- no-op in incognito mode

Usage:
    python eval/runners/voice_memory_control_eval.py
"""

# ruff: noqa: E402

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2] / "apps" / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agent.memory.modes import MemoryMode
from agent.memory.procedural import (
    aadd_procedural_rule,
    aget_procedural_profile,
    aset_proactive_recall,
    build_procedural_rule,
)
from agent.memory.store import OpenCouchMemoryStore
from voice.livekit.session_data import SessionData
from voice.livekit.tools import (
    cancel_memory_deletion,
    confirm_memory_deletion,
    prepare_memory_deletion,
    set_proactive_memory_recall,
    show_memory_status,
    show_saved_memory,
)

DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "datasets" / "voice_memory_control_v1.json"
)


def _load_cases(path: Path) -> list[dict[str, Any]]:
    """Load voice memory-control eval cases.

    Args:
        path: Dataset path.

    Returns:
        Parsed eval cases.
    """

    return json.loads(path.read_text())


async def _seed_case(
    store: OpenCouchMemoryStore,
    *,
    user_id: str,
    setup: dict[str, Any],
) -> None:
    """Seed memory records for one eval case.

    Args:
        store: Store to seed.
        user_id: Owner namespace.
        setup: Case setup payload.

    Returns:
        None: Mutates the supplied store.
    """

    for index, fact in enumerate(setup.get("facts", []), start=1):
        await store.aput(
            (user_id, "semantic"),
            f"fact-{index}",
            {
                "evidence_quote": fact,
                "source": "voice_memory_control_eval",
                "thread_id": "voice-memory-eval",
            },
        )

    for rule_text in setup.get("rules", []):
        await aadd_procedural_rule(
            store,
            user_id=user_id,
            rule=build_procedural_rule(
                rule_text=rule_text,
                evidence=[rule_text],
            ),
        )

    if setup.get("recall_enabled") is not None:
        await aset_proactive_recall(
            store,
            user_id=user_id,
            enabled=bool(setup["recall_enabled"]),
        )


async def _run_tool(context: SimpleNamespace, step: dict[str, Any]) -> str:
    """Run one voice memory-control tool step.

    Args:
        context: Fake LiveKit ``RunContext`` with userdata.
        step: Dataset step payload.

    Returns:
        Tool result string.
    """

    tool = step["tool"]
    args = step.get("args", {})
    if tool == "show_saved_memory":
        return await show_saved_memory(context)
    if tool == "show_memory_status":
        return await show_memory_status(context)
    if tool == "set_proactive_memory_recall":
        return await set_proactive_memory_recall(context, **args)
    if tool == "prepare_memory_deletion":
        return await prepare_memory_deletion(context, **args)
    if tool == "confirm_memory_deletion":
        return await confirm_memory_deletion(context)
    if tool == "cancel_memory_deletion":
        return await cancel_memory_deletion(context)
    raise ValueError(f"Unknown tool: {tool}")


async def _check_step(
    *,
    case_id: str,
    step_index: int,
    step: dict[str, Any],
    result: str,
    store: OpenCouchMemoryStore,
    userdata: SessionData,
    interruptions: list[str],
) -> list[str]:
    """Return assertion failures for one eval step.

    Args:
        case_id: Current case id.
        step_index: 1-based step number.
        step: Dataset step payload.
        result: Tool result string.
        store: Memory store after the step.
        userdata: Session userdata after the step.
        interruptions: Recorded disallow-interruptions calls.

    Returns:
        Failure messages for this step.
    """

    failures: list[str] = []
    prefix = f"{case_id} step {step_index}"

    for expected in step.get("expect_contains", []):
        if expected not in result:
            failures.append(f"{prefix}: missing text {expected!r}; result={result!r}")

    if "expect_pending_delete" in step:
        actual = userdata.pending_memory_delete is not None
        if actual is not bool(step["expect_pending_delete"]):
            failures.append(
                f"{prefix}: pending_delete={actual}, expected={step['expect_pending_delete']}"
            )

    if "expect_semantic_count" in step:
        actual_count = await store.arecord_count((userdata.user_id, "semantic"))
        if actual_count != step["expect_semantic_count"]:
            failures.append(
                f"{prefix}: semantic_count={actual_count}, expected={step['expect_semantic_count']}"
            )

    if "expect_rule_count" in step:
        profile = await aget_procedural_profile(store, user_id=userdata.user_id)
        if len(profile.rules) != step["expect_rule_count"]:
            failures.append(
                f"{prefix}: rule_count={len(profile.rules)}, expected={step['expect_rule_count']}"
            )

    if "expect_recall_enabled" in step:
        profile = await aget_procedural_profile(store, user_id=userdata.user_id)
        expected_recall = bool(step["expect_recall_enabled"])
        if profile.proactive_recall_enabled is not expected_recall:
            failures.append(
                f"{prefix}: stored_recall={profile.proactive_recall_enabled}, expected={expected_recall}"
            )
        if userdata.proactive_recall_enabled is not expected_recall:
            failures.append(
                f"{prefix}: session_recall={userdata.proactive_recall_enabled}, expected={expected_recall}"
            )

    if "expect_interruptions_blocked" in step:
        expected_blocks = int(step["expect_interruptions_blocked"])
        if len(interruptions) != expected_blocks:
            failures.append(
                f"{prefix}: interruptions_blocked={len(interruptions)}, expected={expected_blocks}"
            )

    return failures


async def _evaluate_case(case: dict[str, Any]) -> tuple[bool, list[str]]:
    """Evaluate one voice memory-control case.

    Args:
        case: Dataset case payload.

    Returns:
        ``(passed, failures)`` for the case.
    """

    user_id = "voice-memory-eval-user"
    store = OpenCouchMemoryStore()
    setup = case.get("setup", {})
    await _seed_case(store, user_id=user_id, setup=setup)

    mode = (
        MemoryMode.INCOGNITO if setup.get("mode") == "incognito" else MemoryMode.LOCAL
    )
    userdata = SessionData(
        user_id=user_id,
        thread_id="voice-memory-eval",
        memory_store=store,
        memory_mode=mode,
        proactive_recall_enabled=bool(setup.get("recall_enabled", False)),
    )
    interruptions: list[str] = []
    context = SimpleNamespace(
        userdata=userdata,
        disallow_interruptions=lambda: interruptions.append("blocked"),
    )

    failures: list[str] = []
    for step_index, step in enumerate(case["steps"], start=1):
        result = await _run_tool(context, step)
        failures.extend(
            await _check_step(
                case_id=case["id"],
                step_index=step_index,
                step=step,
                result=result,
                store=store,
                userdata=userdata,
                interruptions=interruptions,
            )
        )

    return not failures, failures


async def _amain() -> int:
    """Run the voice memory-control eval.

    Returns:
        Process exit code.
    """

    cases = _load_cases(DATASET_PATH)
    print(f"Running voice memory-control eval on {len(cases)} case(s).")
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
