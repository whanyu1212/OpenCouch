"""Runner for session trajectory evaluation.

Replays realistic multi-turn conversations through the real persisted
runtime, normalizes each turn into a stable record, and grades sparse
checkpoint assertions from the dataset.

Supports two dataset schemas:

- **Inline expect** (``session_trajectory_v1.json``): each turn has an
  ``expect`` block graded immediately after that turn.
- **Checkpoint** (``session_trajectory_long_v1.json``): sparse
  ``checkpoints`` graded at specific turn numbers.  Final expectations
  use the key ``final_expectations`` instead of ``final_expect``.

Usage:
    python eval/runners/session_trajectory_eval.py --mode deterministic
    python eval/runners/session_trajectory_eval.py --mode hybrid
    python eval/runners/session_trajectory_eval.py --mode auto  # default
    python eval/runners/session_trajectory_eval.py --mode hybrid \\
        --dataset eval/datasets/session_trajectory_long_v1.json
    python eval/runners/session_trajectory_eval.py --mode hybrid \\
        --dataset eval/datasets/session_trajectory_long_v1.json \\
        --case out_of_scope_boundary_and_recovery_with_closing
    python eval/runners/session_trajectory_eval.py --mode hybrid \\
        --dataset eval/datasets/session_trajectory_memory_v1.json
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

BACKEND_ROOT = Path(__file__).resolve().parents[2] / "apps" / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agent.memory.modes import MemoryMode
from agent.memory.models import SemanticFact, StoredSessionArc
from agent.memory.procedural import aget_procedural_profile
from agent.memory.store import OpenCouchMemoryStore
from agent.persistence import PersistentAgentRuntime
from core.config import create_configured_llm_client
from services.llm.base import BaseLLMClient

DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "datasets" / "session_trajectory_v1.json"
)

EvalMode = Literal["auto", "deterministic", "hybrid"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run session trajectory evaluation.")
    parser.add_argument(
        "--mode",
        choices=["auto", "deterministic", "hybrid"],
        default="auto",
        help=(
            "Evaluation mode. 'auto' uses the configured LLM client when available "
            "and falls back to deterministic mode otherwise."
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
        help="Run only the case with this id. Useful for debugging a single scenario.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Print per-case and per-turn progress details, including checkpoint "
            "evaluations and running pass/fail status."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            "Number of cases to run concurrently. Each case uses its own "
            "thread/session so they are independent. Higher values speed up "
            "hybrid runs significantly (5 LLM calls per turn). Default: 1 "
            "(sequential). Recommended: 4-8 for hybrid runs."
        ),
    )
    return parser


def _log(verbose: bool, message: str) -> None:
    """Print a progress message when verbose logging is enabled."""

    if verbose:
        print(message)


def _load_cases(path: Path) -> list[dict[str, Any]]:
    """Load the eval dataset from disk."""

    return json.loads(path.read_text())


def _resolve_llm_client(mode: EvalMode) -> tuple[BaseLLMClient | None, str]:
    """Return the LLM client plus resolved mode label."""

    if mode == "deterministic":
        return None, "deterministic"

    if mode == "hybrid":
        return create_configured_llm_client(), "hybrid"

    try:
        return create_configured_llm_client(), "hybrid"
    except Exception:
        return None, "deterministic"


def _case_supports_mode(case: dict[str, Any], resolved_mode: str) -> bool:
    """Return whether a case should run in the given mode.

    Cases with an explicit ``supported_modes`` field are filtered by it.
    Cases without the field (long trajectory dataset) are treated as
    hybrid-only — they require an LLM client for meaningful assertions.
    """

    supported = case.get("supported_modes")
    if supported is not None:
        return resolved_mode in supported
    return resolved_mode == "hybrid"


# ── Helpers ──────────────────────────────────────────────────────────────


def _contains_any(text: str | None, expected: list[str]) -> bool:
    """Return whether text contains at least one expected substring."""

    low = (text or "").lower()
    return any(item.lower() in low for item in expected)


def _contains_none(text: str | None, forbidden: list[str]) -> bool:
    """Return whether text contains none of the forbidden substrings."""

    low = (text or "").lower()
    return not any(item.lower() in low for item in forbidden)


def _contains_substring(text: str | None, expected: str) -> bool:
    """Return whether text contains a single expected substring."""

    return expected.lower() in (text or "").lower()


def _contains_any_in_field(
    items: list[dict[str, Any]],
    field: str,
    expected: list[str],
) -> bool:
    """Return whether any serialized item field contains any expected substring."""

    lowered_expected = [item.lower() for item in expected]
    for entry in items:
        value = str(entry.get(field, "") or "").lower()
        if any(item in value for item in lowered_expected):
            return True
    return False


# ── Memory snapshotting ───────────────────────────────────────────────────


def _empty_memory_snapshot() -> dict[str, list[dict[str, Any]]]:
    """Return an empty normalized memory snapshot."""

    return {
        "semantic_facts": [],
        "procedural_rules": [],
        "episodic_arcs": [],
    }


def _serialize_semantic_fact(fact: SemanticFact) -> dict[str, Any]:
    """Normalize a stored semantic fact into an eval-friendly shape."""

    return {
        "id": fact.id,
        "category": "semantic",
        "semantic_category": fact.category,
        "predicate": fact.predicate,
        "subject_identifier": fact.subject.identifier,
        "object_identifier": fact.object.identifier,
        "evidence_quote": fact.evidence_quote,
        "source_turn_index": fact.source_turn_index,
    }


def _serialize_procedural_rule(rule: Any) -> dict[str, Any]:
    """Normalize a stored procedural rule into an eval-friendly shape."""

    return {
        "category": "procedural",
        "rule": rule.rule,
        "evidence": list(rule.evidence),
        "confidence": rule.confidence,
        "added_at": rule.added_at,
        "source": rule.source,
    }


def _serialize_episodic_arc(arc: StoredSessionArc) -> dict[str, Any]:
    """Normalize a stored episodic arc into an eval-friendly shape."""

    return {
        "id": arc.id,
        "category": "episodic",
        "summary": arc.summary,
        "primary_themes": list(arc.primary_themes),
        "open_loops": list(arc.open_loops),
        "resolved_threads": list(arc.resolved_threads),
    }


async def _snapshot_memory_state(
    store: Any,
    *,
    owner_id: str,
) -> dict[str, list[dict[str, Any]]]:
    """Read the concrete memory store state for this owner."""

    semantic_records = await store.asearch(
        (owner_id, "semantic"), query=None, limit=1000
    )
    episodic_records = await store.asearch(
        (owner_id, "episodic"), query=None, limit=1000
    )
    procedural_profile = await aget_procedural_profile(store, user_id=owner_id)

    semantic_facts = [
        _serialize_semantic_fact(SemanticFact.model_validate(record.value))
        for record in semantic_records
    ]
    semantic_facts.sort(key=lambda fact: fact["id"])

    episodic_arcs = [
        _serialize_episodic_arc(StoredSessionArc.model_validate(record.value))
        for record in episodic_records
    ]
    episodic_arcs.sort(key=lambda arc: arc["id"])

    procedural_rules = [
        _serialize_procedural_rule(rule) for rule in procedural_profile.rules
    ]

    return {
        "semantic_facts": semantic_facts,
        "procedural_rules": procedural_rules,
        "episodic_arcs": episodic_arcs,
    }


def _diff_memory_state(
    before: dict[str, list[dict[str, Any]]],
    after: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Return normalized write events and count deltas between two snapshots."""

    before_semantic_ids = {fact["id"] for fact in before["semantic_facts"]}
    semantic_writes = [
        fact
        for fact in after["semantic_facts"]
        if fact["id"] not in before_semantic_ids
    ]

    before_procedural_rules = before["procedural_rules"]
    after_procedural_rules = after["procedural_rules"]
    if (
        len(after_procedural_rules) >= len(before_procedural_rules)
        and after_procedural_rules[: len(before_procedural_rules)]
        == before_procedural_rules
    ):
        procedural_writes = after_procedural_rules[len(before_procedural_rules) :]
    else:
        before_rule_keys = {
            (rule["rule"], rule["added_at"], rule["source"])
            for rule in before_procedural_rules
        }
        procedural_writes = [
            rule
            for rule in after_procedural_rules
            if (rule["rule"], rule["added_at"], rule["source"]) not in before_rule_keys
        ]

    before_episodic_ids = {arc["id"] for arc in before["episodic_arcs"]}
    episodic_writes = [
        arc for arc in after["episodic_arcs"] if arc["id"] not in before_episodic_ids
    ]

    memory_writes = [*semantic_writes, *procedural_writes, *episodic_writes]
    return {
        "memory_writes": memory_writes,
        "semantic_fact_count_delta": len(semantic_writes),
        "procedural_rule_count_delta": len(procedural_writes),
        "episodic_arc_count_delta": len(episodic_writes),
    }


# ── Normalization ────────────────────────────────────────────────────────


def _normalize_turn_record(
    turn_index: int,
    user_text: str,
    result: Any,
    *,
    memory_snapshot: dict[str, list[dict[str, Any]]],
    memory_delta: dict[str, Any],
) -> dict[str, Any]:
    """Normalize one runtime turn into the stable eval record shape."""

    output = result.output
    state = result.state or {}
    progress = state.get("progress", {}) or {}
    response = state.get("response", {}) or {}
    routing = state.get("routing", {}) or {}

    return {
        "turn_index": turn_index,
        "user_text": user_text,
        "assistant_text": output.response_text,
        "mode": output.mode,
        "response_type": output.response_type.value
        if output.response_type is not None
        else None,
        "crisis_level": output.crisis.level,
        "needs_crisis_response": output.crisis.needs_crisis_response,
        "needs_clarification": output.crisis.needs_clarification,
        "response_guidance": response.get("guidance"),
        "session_intent": progress.get("intent"),
        "session_intent_source": progress.get("intent_source"),
        "session_stage": progress.get("stage"),
        "modality": routing.get("modality") or output.modality,
        "exercise_active": progress.get("exercise_type") is not None,
        "exercise_type": progress.get("exercise_type"),
        "exercise_step": progress.get("exercise_step"),
        "memory_writes": list(memory_delta["memory_writes"]),
        "semantic_fact_count_delta": memory_delta["semantic_fact_count_delta"],
        "procedural_rule_count_delta": memory_delta["procedural_rule_count_delta"],
        "episodic_arc_count_delta": memory_delta["episodic_arc_count_delta"],
        "semantic_facts_total": len(memory_snapshot["semantic_facts"]),
        "procedural_rules_total": len(memory_snapshot["procedural_rules"]),
        "episodic_arcs_total": len(memory_snapshot["episodic_arcs"]),
        "semantic_facts": list(memory_snapshot["semantic_facts"]),
        "procedural_rules": list(memory_snapshot["procedural_rules"]),
        "episodic_arcs": list(memory_snapshot["episodic_arcs"]),
        "raw_observations": {
            "mode_type": str(output.mode_type)
            if output.mode_type is not None
            else None,
            "mode_source": output.mode_source,
            "modality": output.modality,
            "diagnostics": output.diagnostics,
            "memory_snapshot": memory_snapshot,
        },
    }


def _normalize_final_record(
    stored_arc: Any,
    last_turn_record: dict[str, Any] | None,
    total_memory_writes: int,
    *,
    end_session_run: bool,
    memory_snapshot: dict[str, list[dict[str, Any]]],
    session_end_delta: dict[str, Any],
) -> dict[str, Any]:
    """Normalize the final eval artifact into the stable record shape.

    Carries forward key fields from the last turn so final assertions
    can grade session-end state (mode, stage, intent, modality, response
    text) without needing a separate replay.
    """

    last = last_turn_record or {}

    return {
        "summary_text": None if stored_arc is None else stored_arc.summary,
        "summary_expected": stored_arc is not None,
        "end_session_run": end_session_run,
        "open_loops": [] if stored_arc is None else list(stored_arc.open_loops),
        "resolved_threads": []
        if stored_arc is None
        else list(stored_arc.resolved_threads),
        "exercise_active": last.get("exercise_active", False),
        "exercise_type": last.get("exercise_type"),
        "memory_writes_total": total_memory_writes,
        "session_end_memory_writes": list(session_end_delta["memory_writes"]),
        "session_end_semantic_fact_count_delta": session_end_delta[
            "semantic_fact_count_delta"
        ],
        "session_end_procedural_rule_count_delta": session_end_delta[
            "procedural_rule_count_delta"
        ],
        "session_end_episodic_arc_count_delta": session_end_delta[
            "episodic_arc_count_delta"
        ],
        "semantic_facts_total": len(memory_snapshot["semantic_facts"]),
        "procedural_rules_total": len(memory_snapshot["procedural_rules"]),
        "episodic_arcs_total": len(memory_snapshot["episodic_arcs"]),
        "semantic_facts": list(memory_snapshot["semantic_facts"]),
        "procedural_rules": list(memory_snapshot["procedural_rules"]),
        "episodic_arcs": list(memory_snapshot["episodic_arcs"]),
        # Carry forward from last turn for final grading.
        "session_intent": last.get("session_intent"),
        "session_stage": last.get("session_stage"),
        "mode": last.get("mode"),
        "modality": last.get("modality"),
        "response_guidance": last.get("response_guidance"),
        "assistant_text": last.get("assistant_text"),
        "needs_clarification": last.get("needs_clarification"),
        "needs_crisis_response": last.get("needs_crisis_response"),
    }


# ── Turn-level grading ───────────────────────────────────────────────────


def _check_turn_expectation(
    case_id: str,
    turn_number: int,
    record: dict[str, Any],
    expect: dict[str, Any],
) -> list[str]:
    """Return a list of failure messages for one turn expectation block."""

    failures: list[str] = []

    # ── Mode ─────────────────────────────────────────────────────────
    if "mode_in" in expect and record["mode"] not in expect["mode_in"]:
        failures.append(
            f"FAIL [{case_id}] turn {turn_number}: expected mode_in={expect['mode_in']!r}, "
            f"got mode={record['mode']!r}. user={record['user_text']!r}"
        )

    if "allowed_modes" in expect and record["mode"] not in expect["allowed_modes"]:
        failures.append(
            f"FAIL [{case_id}] turn {turn_number}: expected allowed_modes="
            f"{expect['allowed_modes']!r}, got mode={record['mode']!r}. "
            f"user={record['user_text']!r}"
        )

    # ── Crisis ───────────────────────────────────────────────────────
    if (
        "crisis_level_in" in expect
        and record["crisis_level"] not in expect["crisis_level_in"]
    ):
        failures.append(
            f"FAIL [{case_id}] turn {turn_number}: expected crisis_level_in="
            f"{expect['crisis_level_in']!r}, got crisis_level={record['crisis_level']!r}."
        )

    if (
        "needs_crisis_response" in expect
        and record["needs_crisis_response"] != expect["needs_crisis_response"]
    ):
        failures.append(
            f"FAIL [{case_id}] turn {turn_number}: expected needs_crisis_response="
            f"{expect['needs_crisis_response']!r}, got {record['needs_crisis_response']!r}."
        )

    if (
        "needs_clarification" in expect
        and record["needs_clarification"] != expect["needs_clarification"]
    ):
        failures.append(
            f"FAIL [{case_id}] turn {turn_number}: expected needs_clarification="
            f"{expect['needs_clarification']!r}, got {record['needs_clarification']!r}."
        )

    # ── Session intent ───────────────────────────────────────────────
    if (
        "session_intent" in expect
        and record["session_intent"] != expect["session_intent"]
    ):
        failures.append(
            f"FAIL [{case_id}] turn {turn_number}: expected session_intent="
            f"{expect['session_intent']!r}, got {record['session_intent']!r}."
        )

    if (
        "allowed_session_intents" in expect
        and record["session_intent"] not in expect["allowed_session_intents"]
    ):
        failures.append(
            f"FAIL [{case_id}] turn {turn_number}: expected session_intent in "
            f"{expect['allowed_session_intents']!r}, got {record['session_intent']!r}."
        )

    if (
        "session_intent_source" in expect
        and record["session_intent_source"] != expect["session_intent_source"]
    ):
        failures.append(
            f"FAIL [{case_id}] turn {turn_number}: expected session_intent_source="
            f"{expect['session_intent_source']!r}, got {record['session_intent_source']!r}."
        )

    # ── Session stage ────────────────────────────────────────────────
    if "session_stage" in expect and record["session_stage"] != expect["session_stage"]:
        failures.append(
            f"FAIL [{case_id}] turn {turn_number}: expected session_stage="
            f"{expect['session_stage']!r}, got {record['session_stage']!r}."
        )

    if (
        "allowed_session_stages" in expect
        and record["session_stage"] not in expect["allowed_session_stages"]
    ):
        failures.append(
            f"FAIL [{case_id}] turn {turn_number}: expected session_stage in "
            f"{expect['allowed_session_stages']!r}, got {record['session_stage']!r}."
        )

    # ── Modality ─────────────────────────────────────────────────────
    if (
        "required_modalities" in expect
        and record["modality"] not in expect["required_modalities"]
    ):
        failures.append(
            f"FAIL [{case_id}] turn {turn_number}: expected modality in "
            f"{expect['required_modalities']!r}, got {record['modality']!r}."
        )

    # ── Response type ────────────────────────────────────────────────
    if "response_type" in expect and record["response_type"] != expect["response_type"]:
        failures.append(
            f"FAIL [{case_id}] turn {turn_number}: expected response_type="
            f"{expect['response_type']!r}, got {record['response_type']!r}."
        )

    # ── Response guidance ────────────────────────────────────────────
    if "response_guidance_contains_any" in expect and not _contains_any(
        record["response_guidance"],
        expect["response_guidance_contains_any"],
    ):
        failures.append(
            f"FAIL [{case_id}] turn {turn_number}: expected response_guidance to contain "
            f"one of {expect['response_guidance_contains_any']!r}, "
            f"got {record['response_guidance']!r}."
        )

    if "response_guidance_contains" in expect and not _contains_substring(
        record["response_guidance"],
        expect["response_guidance_contains"],
    ):
        failures.append(
            f"FAIL [{case_id}] turn {turn_number}: expected response_guidance to contain "
            f"{expect['response_guidance_contains']!r}, "
            f"got {record['response_guidance']!r}."
        )

    # ── Response text ────────────────────────────────────────────────
    if "response_contains_any" in expect and not _contains_any(
        record["assistant_text"],
        expect["response_contains_any"],
    ):
        failures.append(
            f"FAIL [{case_id}] turn {turn_number}: expected assistant response to contain "
            f"one of {expect['response_contains_any']!r}, got {record['assistant_text']!r}."
        )

    if "response_not_contains" in expect and not _contains_none(
        record["assistant_text"],
        expect["response_not_contains"],
    ):
        failures.append(
            f"FAIL [{case_id}] turn {turn_number}: assistant response contained a forbidden "
            f"substring from {expect['response_not_contains']!r}. "
            f"assistant={record['assistant_text']!r}"
        )

    # ── Exercise state ───────────────────────────────────────────────
    if (
        "exercise_type_in" in expect
        and record["exercise_type"] not in expect["exercise_type_in"]
    ):
        failures.append(
            f"FAIL [{case_id}] turn {turn_number}: expected exercise_type_in="
            f"{expect['exercise_type_in']!r}, got exercise_type={record['exercise_type']!r}."
        )

    if (
        "exercise_step_in" in expect
        and record["exercise_step"] not in expect["exercise_step_in"]
    ):
        failures.append(
            f"FAIL [{case_id}] turn {turn_number}: expected exercise_step_in="
            f"{expect['exercise_step_in']!r}, got exercise_step={record['exercise_step']!r}."
        )

    if (
        "exercise_active" in expect
        and record["exercise_active"] != expect["exercise_active"]
    ):
        failures.append(
            f"FAIL [{case_id}] turn {turn_number}: expected exercise_active="
            f"{expect['exercise_active']!r}, got {record['exercise_active']!r}."
        )

    # ── Memory writes ────────────────────────────────────────────────
    if "memory_write_expected" in expect:
        actual_write = len(record["memory_writes"]) > 0
        if actual_write != expect["memory_write_expected"]:
            failures.append(
                f"FAIL [{case_id}] turn {turn_number}: expected memory_write_expected="
                f"{expect['memory_write_expected']!r}, got {actual_write!r}."
            )

    if "memory_write_category_in" in expect:
        categories = {write.get("category") for write in record["memory_writes"]}
        if not any(
            category in expect["memory_write_category_in"] for category in categories
        ):
            failures.append(
                f"FAIL [{case_id}] turn {turn_number}: expected memory write category in "
                f"{expect['memory_write_category_in']!r}, got categories={sorted(categories)!r}."
            )

    if (
        "semantic_fact_count_delta" in expect
        and record["semantic_fact_count_delta"] != expect["semantic_fact_count_delta"]
    ):
        failures.append(
            f"FAIL [{case_id}] turn {turn_number}: expected semantic_fact_count_delta="
            f"{expect['semantic_fact_count_delta']!r}, got "
            f"{record['semantic_fact_count_delta']!r}."
        )

    if (
        "procedural_rule_count_delta" in expect
        and record["procedural_rule_count_delta"]
        != expect["procedural_rule_count_delta"]
    ):
        failures.append(
            f"FAIL [{case_id}] turn {turn_number}: expected procedural_rule_count_delta="
            f"{expect['procedural_rule_count_delta']!r}, got "
            f"{record['procedural_rule_count_delta']!r}."
        )

    if (
        "episodic_arc_count_delta" in expect
        and record["episodic_arc_count_delta"] != expect["episodic_arc_count_delta"]
    ):
        failures.append(
            f"FAIL [{case_id}] turn {turn_number}: expected episodic_arc_count_delta="
            f"{expect['episodic_arc_count_delta']!r}, got "
            f"{record['episodic_arc_count_delta']!r}."
        )

    if (
        "semantic_facts_total" in expect
        and record["semantic_facts_total"] != expect["semantic_facts_total"]
    ):
        failures.append(
            f"FAIL [{case_id}] turn {turn_number}: expected semantic_facts_total="
            f"{expect['semantic_facts_total']!r}, got {record['semantic_facts_total']!r}."
        )

    if (
        "procedural_rules_total" in expect
        and record["procedural_rules_total"] != expect["procedural_rules_total"]
    ):
        failures.append(
            f"FAIL [{case_id}] turn {turn_number}: expected procedural_rules_total="
            f"{expect['procedural_rules_total']!r}, got "
            f"{record['procedural_rules_total']!r}."
        )

    if (
        "episodic_arcs_total" in expect
        and record["episodic_arcs_total"] != expect["episodic_arcs_total"]
    ):
        failures.append(
            f"FAIL [{case_id}] turn {turn_number}: expected episodic_arcs_total="
            f"{expect['episodic_arcs_total']!r}, got {record['episodic_arcs_total']!r}."
        )

    semantic_turn_writes = [
        write
        for write in record["memory_writes"]
        if write.get("category") == "semantic"
    ]
    procedural_turn_writes = [
        write
        for write in record["memory_writes"]
        if write.get("category") == "procedural"
    ]

    if "semantic_fact_object_contains_any" in expect and not _contains_any_in_field(
        semantic_turn_writes,
        "object_identifier",
        expect["semantic_fact_object_contains_any"],
    ):
        failures.append(
            f"FAIL [{case_id}] turn {turn_number}: expected semantic write object to "
            f"contain one of {expect['semantic_fact_object_contains_any']!r}, got "
            f"{[write.get('object_identifier') for write in semantic_turn_writes]!r}."
        )

    if "semantic_evidence_contains_any" in expect and not _contains_any_in_field(
        semantic_turn_writes,
        "evidence_quote",
        expect["semantic_evidence_contains_any"],
    ):
        failures.append(
            f"FAIL [{case_id}] turn {turn_number}: expected semantic write evidence to "
            f"contain one of {expect['semantic_evidence_contains_any']!r}."
        )

    if "procedural_rule_contains_any" in expect and not _contains_any_in_field(
        procedural_turn_writes,
        "rule",
        expect["procedural_rule_contains_any"],
    ):
        failures.append(
            f"FAIL [{case_id}] turn {turn_number}: expected procedural rule write to "
            f"contain one of {expect['procedural_rule_contains_any']!r}, got "
            f"{[write.get('rule') for write in procedural_turn_writes]!r}."
        )

    return failures


# ── Final grading ────────────────────────────────────────────────────────


def _check_final_expectation(
    case_id: str,
    record: dict[str, Any],
    expect: dict[str, Any],
) -> list[str]:
    """Return a list of failure messages for final expectations."""

    failures: list[str] = []

    # ── Summary checks ───────────────────────────────────────────────
    summary_checks_requested = (
        any(
            key in expect
            for key in (
                "summary_contains_any",
                "summary_not_contains",
                "open_loops_nonempty",
                "resolved_threads_nonempty",
            )
        )
        or expect.get("summary_expected") is True
    )

    if summary_checks_requested and not record["end_session_run"]:
        failures.append(
            f"FAIL [{case_id}] final: summary expectations were provided, but run_end_session=false."
        )
        return failures

    if (
        "summary_expected" in expect
        and record["summary_expected"] != expect["summary_expected"]
    ):
        failures.append(
            f"FAIL [{case_id}] final: expected summary_expected="
            f"{expect['summary_expected']!r}, got {record['summary_expected']!r}."
        )

    if "summary_contains_any" in expect and not _contains_any(
        record["summary_text"],
        expect["summary_contains_any"],
    ):
        failures.append(
            f"FAIL [{case_id}] final: expected summary_text to contain one of "
            f"{expect['summary_contains_any']!r}, got {record['summary_text']!r}."
        )

    if "summary_not_contains" in expect and not _contains_none(
        record["summary_text"],
        expect["summary_not_contains"],
    ):
        failures.append(
            f"FAIL [{case_id}] final: summary_text contained a forbidden substring from "
            f"{expect['summary_not_contains']!r}. summary={record['summary_text']!r}"
        )

    if expect.get("open_loops_nonempty") and not record["open_loops"]:
        failures.append(
            f"FAIL [{case_id}] final: expected open_loops_nonempty=true, got open_loops=[]."
        )

    if expect.get("resolved_threads_nonempty") and not record["resolved_threads"]:
        failures.append(
            f"FAIL [{case_id}] final: expected resolved_threads_nonempty=true, "
            "got resolved_threads=[]."
        )

    # ── Memory writes ────────────────────────────────────────────────
    if "memory_write_expected" in expect:
        actual_write = int(record["memory_writes_total"] or 0) > 0
        if actual_write != expect["memory_write_expected"]:
            failures.append(
                f"FAIL [{case_id}] final: expected memory_write_expected="
                f"{expect['memory_write_expected']!r}, got {actual_write!r}."
            )

    if "session_end_memory_write_expected" in expect:
        actual_write = len(record["session_end_memory_writes"]) > 0
        if actual_write != expect["session_end_memory_write_expected"]:
            failures.append(
                f"FAIL [{case_id}] final: expected session_end_memory_write_expected="
                f"{expect['session_end_memory_write_expected']!r}, got {actual_write!r}."
            )

    if "session_end_memory_write_category_in" in expect:
        categories = {
            write.get("category") for write in record["session_end_memory_writes"]
        }
        if not any(
            category in expect["session_end_memory_write_category_in"]
            for category in categories
        ):
            failures.append(
                f"FAIL [{case_id}] final: expected session-end memory write category in "
                f"{expect['session_end_memory_write_category_in']!r}, got "
                f"categories={sorted(categories)!r}."
            )

    if (
        "session_end_semantic_fact_count_delta" in expect
        and record["session_end_semantic_fact_count_delta"]
        != expect["session_end_semantic_fact_count_delta"]
    ):
        failures.append(
            f"FAIL [{case_id}] final: expected session_end_semantic_fact_count_delta="
            f"{expect['session_end_semantic_fact_count_delta']!r}, got "
            f"{record['session_end_semantic_fact_count_delta']!r}."
        )

    if (
        "session_end_procedural_rule_count_delta" in expect
        and record["session_end_procedural_rule_count_delta"]
        != expect["session_end_procedural_rule_count_delta"]
    ):
        failures.append(
            f"FAIL [{case_id}] final: expected session_end_procedural_rule_count_delta="
            f"{expect['session_end_procedural_rule_count_delta']!r}, got "
            f"{record['session_end_procedural_rule_count_delta']!r}."
        )

    if (
        "session_end_episodic_arc_count_delta" in expect
        and record["session_end_episodic_arc_count_delta"]
        != expect["session_end_episodic_arc_count_delta"]
    ):
        failures.append(
            f"FAIL [{case_id}] final: expected session_end_episodic_arc_count_delta="
            f"{expect['session_end_episodic_arc_count_delta']!r}, got "
            f"{record['session_end_episodic_arc_count_delta']!r}."
        )

    if (
        "semantic_facts_total" in expect
        and record["semantic_facts_total"] != expect["semantic_facts_total"]
    ):
        failures.append(
            f"FAIL [{case_id}] final: expected semantic_facts_total="
            f"{expect['semantic_facts_total']!r}, got {record['semantic_facts_total']!r}."
        )

    if (
        "procedural_rules_total" in expect
        and record["procedural_rules_total"] != expect["procedural_rules_total"]
    ):
        failures.append(
            f"FAIL [{case_id}] final: expected procedural_rules_total="
            f"{expect['procedural_rules_total']!r}, got "
            f"{record['procedural_rules_total']!r}."
        )

    if (
        "episodic_arcs_total" in expect
        and record["episodic_arcs_total"] != expect["episodic_arcs_total"]
    ):
        failures.append(
            f"FAIL [{case_id}] final: expected episodic_arcs_total="
            f"{expect['episodic_arcs_total']!r}, got {record['episodic_arcs_total']!r}."
        )

    if "semantic_fact_object_contains_any" in expect and not _contains_any_in_field(
        record["semantic_facts"],
        "object_identifier",
        expect["semantic_fact_object_contains_any"],
    ):
        failures.append(
            f"FAIL [{case_id}] final: expected semantic fact object to contain one of "
            f"{expect['semantic_fact_object_contains_any']!r}, got "
            f"{[fact.get('object_identifier') for fact in record['semantic_facts']]!r}."
        )

    if "procedural_rule_contains_any" in expect and not _contains_any_in_field(
        record["procedural_rules"],
        "rule",
        expect["procedural_rule_contains_any"],
    ):
        failures.append(
            f"FAIL [{case_id}] final: expected procedural rule to contain one of "
            f"{expect['procedural_rule_contains_any']!r}, got "
            f"{[rule.get('rule') for rule in record['procedural_rules']]!r}."
        )

    # ── Exercise state ───────────────────────────────────────────────
    if (
        "exercise_active" in expect
        and record["exercise_active"] != expect["exercise_active"]
    ):
        failures.append(
            f"FAIL [{case_id}] final: expected exercise_active="
            f"{expect['exercise_active']!r}, got {record['exercise_active']!r}."
        )

    if (
        "exercise_type_in" in expect
        and record["exercise_type"] not in expect["exercise_type_in"]
    ):
        failures.append(
            f"FAIL [{case_id}] final: expected exercise_type_in="
            f"{expect['exercise_type_in']!r}, got {record['exercise_type']!r}."
        )

    # ── Session intent (from last turn) ──────────────────────────────
    if (
        "session_intent" in expect
        and record["session_intent"] != expect["session_intent"]
    ):
        failures.append(
            f"FAIL [{case_id}] final: expected session_intent="
            f"{expect['session_intent']!r}, got {record['session_intent']!r}."
        )

    if (
        "allowed_session_intents" in expect
        and record["session_intent"] not in expect["allowed_session_intents"]
    ):
        failures.append(
            f"FAIL [{case_id}] final: expected session_intent in "
            f"{expect['allowed_session_intents']!r}, got {record['session_intent']!r}."
        )

    # ── Session stage (from last turn) ───────────────────────────────
    if "session_stage" in expect and record["session_stage"] != expect["session_stage"]:
        failures.append(
            f"FAIL [{case_id}] final: expected session_stage="
            f"{expect['session_stage']!r}, got {record['session_stage']!r}."
        )

    # ── Mode (from last turn) ────────────────────────────────────────
    if "allowed_modes" in expect and record["mode"] not in expect["allowed_modes"]:
        failures.append(
            f"FAIL [{case_id}] final: expected mode in "
            f"{expect['allowed_modes']!r}, got {record['mode']!r}."
        )

    # ── Modality (from last turn) ────────────────────────────────────
    if (
        "required_modalities" in expect
        and record["modality"] not in expect["required_modalities"]
    ):
        failures.append(
            f"FAIL [{case_id}] final: expected modality in "
            f"{expect['required_modalities']!r}, got {record['modality']!r}."
        )

    # ── Response guidance (from last turn) ───────────────────────────
    if "response_guidance_contains" in expect and not _contains_substring(
        record["response_guidance"],
        expect["response_guidance_contains"],
    ):
        failures.append(
            f"FAIL [{case_id}] final: expected response_guidance to contain "
            f"{expect['response_guidance_contains']!r}, "
            f"got {record['response_guidance']!r}."
        )

    # ── Crisis signals (from last turn) ──────────────────────────────
    if (
        "needs_clarification" in expect
        and record["needs_clarification"] != expect["needs_clarification"]
    ):
        failures.append(
            f"FAIL [{case_id}] final: expected needs_clarification="
            f"{expect['needs_clarification']!r}, got {record['needs_clarification']!r}."
        )

    if (
        "needs_crisis_response" in expect
        and record["needs_crisis_response"] != expect["needs_crisis_response"]
    ):
        failures.append(
            f"FAIL [{case_id}] final: expected needs_crisis_response="
            f"{expect['needs_crisis_response']!r}, got {record['needs_crisis_response']!r}."
        )

    # ── Response text quality (from last turn) ───────────────────────
    if "max_question_marks" in expect:
        count = (record.get("assistant_text") or "").count("?")
        if count > expect["max_question_marks"]:
            failures.append(
                f"FAIL [{case_id}] final: expected max_question_marks="
                f"{expect['max_question_marks']!r}, got {count}."
            )

    if "must_not_include_any" in expect and not _contains_none(
        record.get("assistant_text"),
        expect["must_not_include_any"],
    ):
        failures.append(
            f"FAIL [{case_id}] final: last response contained a forbidden "
            f"substring from {expect['must_not_include_any']!r}."
        )

    return failures


# ── Case execution ───────────────────────────────────────────────────────


async def _run_case(
    case: dict[str, Any],
    *,
    resolved_mode: str,
    llm_client: BaseLLMClient | None,
    verbose: bool = False,
    case_index: int | None = None,
    total_cases: int | None = None,
) -> tuple[bool, list[str]]:
    """Replay one session trajectory case and grade its expectations."""

    if not _case_supports_mode(case, resolved_mode):
        return True, []

    failures: list[str] = []
    total_memory_writes = 0
    last_turn_record: dict[str, Any] | None = None
    run_end_session = bool(case.get("run_end_session", False))

    # Build a checkpoint map for the long-trajectory schema.
    # {1-indexed turn number → expect dict}
    checkpoint_map: dict[int, dict[str, Any]] = {}
    uses_checkpoints = "checkpoints" in case
    if uses_checkpoints:
        for cp in case["checkpoints"]:
            checkpoint_map[cp["turn"]] = cp["expect"]

    turns = case.get("turns", [])
    case_prefix = (
        f"[case {case_index}/{total_cases}] "
        if case_index is not None and total_cases is not None
        else ""
    )
    _log(
        verbose,
        f"{case_prefix}Starting {case['id']} ({len(turns)} turn(s), "
        f"checkpoints={len(checkpoint_map)}, end_session={run_end_session})",
    )

    with tempfile.TemporaryDirectory(prefix="opencouch-session-eval-") as tmpdir:
        sqlite_path = Path(tmpdir) / "threads.sqlite3"
        async with PersistentAgentRuntime(
            sqlite_path=sqlite_path,
            memory_store=OpenCouchMemoryStore(),
            memory_mode=MemoryMode.LOCAL,
        ) as runtime:
            thread_id = f"eval-{case['id']}-{uuid4().hex[:8]}"
            user_id = f"eval-user-{case['id']}"
            prior_memory_snapshot = await _snapshot_memory_state(
                runtime.memory_store,
                owner_id=user_id,
            )

            for index, turn in enumerate(turns):
                user_text = turn["user"] if isinstance(turn, dict) else turn
                turn_number = index + 1  # 1-indexed
                preview = user_text.strip().replace("\n", " ")
                if len(preview) > 100:
                    preview = f"{preview[:97]}..."
                checkpoint_due = uses_checkpoints and turn_number in checkpoint_map
                _log(
                    verbose,
                    f"  -> turn {turn_number}/{len(turns)} input={preview!r}"
                    + (" [checkpoint]" if checkpoint_due else ""),
                )

                result = await runtime.run_turn(
                    thread_id=thread_id,
                    message=user_text,
                    user_id=user_id,
                    llm_client=llm_client,
                )
                current_memory_snapshot = await _snapshot_memory_state(
                    runtime.memory_store,
                    owner_id=user_id,
                )
                memory_delta = _diff_memory_state(
                    prior_memory_snapshot,
                    current_memory_snapshot,
                )
                record = _normalize_turn_record(
                    index,
                    user_text,
                    result,
                    memory_snapshot=current_memory_snapshot,
                    memory_delta=memory_delta,
                )
                last_turn_record = record
                total_memory_writes += len(record["memory_writes"])
                prior_memory_snapshot = current_memory_snapshot

                _log(
                    verbose,
                    "     "
                    f"mode={record['mode']}, intent={record['session_intent']}, "
                    f"stage={record['session_stage']}, modality={record['modality']}, "
                    f"writes={len(record['memory_writes'])}",
                )

                if uses_checkpoints:
                    if checkpoint_due:
                        checkpoint_failures = _check_turn_expectation(
                            case["id"],
                            turn_number,
                            record,
                            checkpoint_map[turn_number],
                        )
                        failures.extend(checkpoint_failures)
                        _log(
                            verbose,
                            f"     checkpoint turn {turn_number}: "
                            f"{'PASS' if not checkpoint_failures else f'FAIL ({len(checkpoint_failures)})'}",
                        )
                else:
                    turn_failures = _check_turn_expectation(
                        case["id"],
                        turn_number,
                        record,
                        turn.get("expect", {}) if isinstance(turn, dict) else {},
                    )
                    failures.extend(turn_failures)
                    _log(
                        verbose,
                        f"     expectation turn {turn_number}: "
                        f"{'PASS' if not turn_failures else f'FAIL ({len(turn_failures)})'}",
                    )

            stored_arc = None
            final_memory_snapshot = prior_memory_snapshot
            session_end_delta = _diff_memory_state(
                prior_memory_snapshot,
                prior_memory_snapshot,
            )
            if run_end_session:
                _log(verbose, "  -> ending session and collecting final summary")
                stored_arc = await runtime.end_session(thread_id, llm_client=llm_client)
                final_memory_snapshot = await _snapshot_memory_state(
                    runtime.memory_store,
                    owner_id=user_id,
                )
                session_end_delta = _diff_memory_state(
                    prior_memory_snapshot,
                    final_memory_snapshot,
                )
                total_memory_writes += len(session_end_delta["memory_writes"])

            final_record = _normalize_final_record(
                stored_arc,
                last_turn_record,
                total_memory_writes,
                end_session_run=run_end_session,
                memory_snapshot=final_memory_snapshot,
                session_end_delta=session_end_delta,
            )

            final_expect = case.get("final_expectations") or case.get(
                "final_expect", {}
            )
            final_failures = _check_final_expectation(
                case["id"],
                final_record,
                final_expect,
            )
            failures.extend(final_failures)
            _log(
                verbose,
                f"  -> final expectations: "
                f"{'PASS' if not final_failures else f'FAIL ({len(final_failures)})'}",
            )

    _log(
        verbose,
        f"{case_prefix}Finished {case['id']}: "
        f"{'PASS' if not failures else f'FAIL ({len(failures)})'}",
    )
    return len(failures) == 0, failures


# ── Main driver ──────────────────────────────────────────────────────────


async def _run(
    mode: EvalMode,
    dataset_path: Path,
    *,
    case_filter: str | None,
    verbose: bool = False,
    concurrency: int = 1,
) -> int:
    """Drive the session trajectory eval and return a process exit code."""

    cases = _load_cases(dataset_path)
    llm_client, resolved_mode = _resolve_llm_client(mode)

    if case_filter:
        cases = [c for c in cases if c["id"] == case_filter]
        if not cases:
            print(f"No case found with id={case_filter!r} in {dataset_path.name}.")
            return 1

    runnable_cases = [
        case for case in cases if _case_supports_mode(case, resolved_mode)
    ]

    print(
        f"Running session trajectory eval in {resolved_mode} mode on "
        f"{len(runnable_cases)} runnable case(s) from {dataset_path.name}."
    )
    if concurrency > 1:
        print(f"Concurrency: {concurrency} cases in parallel.")
    if verbose:
        print("Verbose progress logging enabled.")
    print()

    if concurrency <= 1:
        # Sequential execution — simpler output ordering.
        results: list[tuple[bool, list[str]]] = []
        for case_index, case in enumerate(runnable_cases, start=1):
            result = await _run_case(
                case,
                resolved_mode=resolved_mode,
                llm_client=llm_client,
                verbose=verbose,
                case_index=case_index,
                total_cases=len(runnable_cases),
            )
            results.append(result)
    else:
        # Concurrent execution bounded by a semaphore.
        semaphore = asyncio.Semaphore(concurrency)

        async def _bounded_run(
            case: dict[str, Any], case_index: int
        ) -> tuple[bool, list[str]]:
            async with semaphore:
                return await _run_case(
                    case,
                    resolved_mode=resolved_mode,
                    llm_client=llm_client,
                    verbose=verbose,
                    case_index=case_index,
                    total_cases=len(runnable_cases),
                )

        results = await asyncio.gather(
            *[
                _bounded_run(case, idx)
                for idx, case in enumerate(runnable_cases, start=1)
            ]
        )

    passed = sum(1 for ok, _ in results if ok)
    failures: list[str] = []
    for _, case_failures in results:
        failures.extend(case_failures)

    print(f"Overall: {passed}/{len(runnable_cases)} passed")

    if failures:
        print()
        print("Failures:")
        for detail in failures:
            print(f"  {detail}")
        return 1

    print()
    print("All runnable cases passed.")
    return 0


def main() -> int:
    """Entry point for the session trajectory eval runner."""

    args = _build_parser().parse_args()
    return asyncio.run(
        _run(
            args.mode,
            args.dataset,
            case_filter=args.case,
            verbose=args.verbose,
            concurrency=args.concurrency,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
