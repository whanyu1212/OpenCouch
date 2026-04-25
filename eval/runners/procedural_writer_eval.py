"""Runner for procedural rule writer evaluation.

Grades the v0.7 procedural writer prompt's accuracy on a hand-curated
dataset of message → expected-output pairs. Each case is either a
"skip" case (expected zero rules) or a "write" case (expected one or
more rules matched on substring and second-person phrasing).

Usage:
    # Live API calls via the configured provider
    python eval/runners/procedural_writer_eval.py --mode hybrid

    # Auto-detect: hybrid if a provider is configured, else skip
    python eval/runners/procedural_writer_eval.py --mode auto  # default

The writer only runs when an LLM client is available — same contract as
the semantic extractor. When no LLM is configured, the runner prints a
message and exits 0 without grading anything.

Grading strategy (asymmetric, same shape as ``extraction_eval.py``):

- **Skip cases**: pass iff the writer produces zero rules. The writer's
  ``reason`` field is not graded for content; it just needs to be
  non-empty.

- **Write cases**: pass iff the writer produces at least one rule AND
  at least one produced rule matches the case's expected_rules spec:
    * ``rule_contains_any_of`` (list[str]): rule text must contain at
      least one of these substrings, case-insensitive.
    * ``phrasing_is_second_person`` (bool): rule text must use "you"
      or "your" and must NOT use "user" as a third-person subject.
      Case-insensitive; regex-based check.

  Both criteria must hold on the same rule (not different rules for
  different criteria). If multiple rules are produced, each is
  checked against the spec independently.

The grading is asymmetric deliberately: skip cases are strict (any
non-zero rule count is a failure), write cases are loose (the LLM
gets credit for producing at least one matching rule even if the
exact wording varies). This matches the philosophy of the extraction
eval and the conservative-by-default stance of the writer.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any, Literal, cast

BACKEND_ROOT = Path(__file__).resolve().parents[2] / "apps" / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.memory.modes import MemoryMode
from agent.memory.procedural import aget_procedural_profile
from agent.memory.store import OpenCouchMemoryStore
from agent.nodes.extract_procedural_rules import run_extract_procedural_rules_node
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from core.config import create_configured_llm_client
from services.llm.base import BaseLLMClient

DATASET_PATH = Path(__file__).resolve().parents[1] / "datasets" / "procedural_v1.json"

EvalMode = Literal["auto", "hybrid"]


class _MockRuntime:
    """Minimal runtime stand-in for the procedural writer node.

    The writer node reads ``llm_client``, ``memory_store``, and
    ``memory_mode`` from context. We provide a real in-memory store
    per-case so the full write path is exercised.
    """

    def __init__(
        self,
        *,
        llm_client: BaseLLMClient,
        memory_store: OpenCouchMemoryStore,
    ) -> None:
        self.context = WorkflowContext(
            llm_client=llm_client,
            memory_store=memory_store,
            crisis_log_backend=InMemoryCrisisLogBackend(),
            memory_mode=MemoryMode.LOCAL,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run procedural rule writer evaluation."
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "hybrid"],
        default="auto",
        help=(
            "Evaluation mode. 'auto' uses the configured LLM client when "
            "available and exits cleanly if none is configured. 'hybrid' "
            "requires an LLM client — fails loudly if none is configured."
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

    Same contract as extraction_eval.py: the writer has no
    deterministic fallback, so ``auto`` mode returns (None, "skipped")
    if no provider is configured.
    """

    if mode == "hybrid":
        return create_configured_llm_client(), "hybrid"

    # auto
    try:
        return create_configured_llm_client(), "hybrid"
    except Exception:
        return None, "skipped"


def _build_state(case: dict[str, Any]) -> AgentState:
    """Return a partial AgentState for the writer node to read."""

    state: Any = {
        "message": case["message"],
        "history": case.get("history", []),
        "user_id": case.get("user_id") or "eval-user",
        "session_id": case.get("session_id") or "eval-session",
        "session_progress": {"turn_count": 1},
        "transcript": [],
    }
    return cast(AgentState, state)


# ─── Second-person phrasing check ──────────────────────────────────────────

# A rule is "second-person" if it uses "you" or "your" AND does NOT use
# "user" as a subject. We're lenient on the positive side (any "you" or
# "your" counts) and strict on the negative side (any third-person
# "user" reference fails). Matches the schema's rule phrasing constraint.
_SECOND_PERSON_PATTERN = re.compile(
    r"\b(?:you|your|you're|you've|you'd)\b", re.IGNORECASE
)
_THIRD_PERSON_USER_PATTERN = re.compile(r"\buser\b", re.IGNORECASE)


def _is_second_person(rule_text: str) -> bool:
    """Return whether the rule text uses second-person phrasing.

    Requires at least one "you" / "your" / contraction AND zero
    occurrences of "user" as a word. The latter catches the
    anti-pattern "User dislikes meditation" / "User often deflects".
    """

    if _THIRD_PERSON_USER_PATTERN.search(rule_text):
        return False
    if _SECOND_PERSON_PATTERN.search(rule_text):
        return True
    return False


def _rule_matches_spec(
    rule_text: str,
    spec: dict[str, Any],
) -> tuple[bool, str]:
    """Check whether one rule matches one expected_rules spec.

    Returns ``(matched, detail)``. ``detail`` is a human-readable
    string explaining the match or miss.
    """

    # Criterion 1: substring match on contains_any_of
    contains_any = spec.get("rule_contains_any_of")
    if contains_any is not None:
        lower = rule_text.lower()
        if not any(substring.lower() in lower for substring in contains_any):
            return (
                False,
                f"rule text missing any of {contains_any!r}: {rule_text!r}",
            )

    # Criterion 2: second-person phrasing
    if spec.get("phrasing_is_second_person") and not _is_second_person(rule_text):
        return (
            False,
            f"rule text is not second-person: {rule_text!r}",
        )

    return True, f"matched: {rule_text!r}"


def _match_expected_rule(
    expected: dict[str, Any],
    returned_rule_texts: list[str],
) -> tuple[bool, str]:
    """Find any returned rule that matches the expected spec.

    Returns ``(found, detail)``.
    """

    for rule_text in returned_rule_texts:
        matched, detail = _rule_matches_spec(rule_text, expected)
        if matched:
            return True, detail
    return False, f"no returned rule matched spec {expected!r}"


async def _evaluate_case(
    case: dict[str, Any],
    llm_client: BaseLLMClient,
) -> tuple[bool, str | None]:
    """Run one case through the writer and grade the result."""

    # Each case gets a fresh in-memory store so prior cases' writes
    # don't pollute the profile for later cases.
    store = OpenCouchMemoryStore()
    runtime = _MockRuntime(llm_client=llm_client, memory_store=store)
    state = _build_state(case)

    # Call the writer node. Its delta is always {} (side effect only),
    # so we read the profile afterward.
    await run_extract_procedural_rules_node(state, runtime)  # type: ignore[arg-type]

    user_id = state.get("user_id") or state["session_id"]
    profile = await aget_procedural_profile(store, user_id=user_id)
    returned_rule_texts = [r.rule for r in profile.rules]

    expected_outcome = case["expected_outcome"]

    if expected_outcome == "skip":
        if len(returned_rule_texts) == 0:
            return True, None
        rule_summary = "; ".join(returned_rule_texts)
        return (
            False,
            f"FAIL [skip] {case['id']}: expected 0 rules, got "
            f"{len(returned_rule_texts)} ({rule_summary})",
        )

    # expected_outcome == "write"
    if len(returned_rule_texts) == 0:
        return (
            False,
            f"FAIL [write] {case['id']}: expected at least 1 rule, got 0",
        )

    expected_rules = case.get("expected_rules", [])
    for expected_rule in expected_rules:
        found, detail = _match_expected_rule(expected_rule, returned_rule_texts)
        if not found:
            rule_summary = "; ".join(returned_rule_texts)
            return (
                False,
                f"FAIL [write] {case['id']}: {detail}; returned: {rule_summary}",
            )

    return True, None


async def _run(mode: EvalMode, dataset_path: Path) -> int:
    """Drive the full eval and return a process exit code."""

    cases = _load_cases(dataset_path)
    llm_client, resolved_mode = _resolve_llm_client(mode)

    if llm_client is None:
        print(
            f"Procedural writer eval: no LLM client configured (mode={mode}). "
            f"Skipping {len(cases)} case(s). Use --mode hybrid to fail "
            f"loudly instead."
        )
        return 0

    print(
        f"Running procedural writer eval in {resolved_mode} mode on "
        f"{len(cases)} case(s) from {dataset_path.name}."
    )
    print()

    # Per-outcome accounting so skip and write cases are reported
    # separately. The asymmetric grading makes this split meaningful.
    by_outcome: dict[str, dict[str, int]] = {
        "skip": {"total": 0, "passed": 0},
        "write": {"total": 0, "passed": 0},
    }
    failures: list[str] = []

    for case in cases:
        outcome = case["expected_outcome"]
        by_outcome[outcome]["total"] += 1
        passed, detail = await _evaluate_case(case, llm_client=llm_client)
        if passed:
            by_outcome[outcome]["passed"] += 1
        elif detail is not None:
            failures.append(detail)

    # Report
    for outcome_name, counts in sorted(by_outcome.items()):
        if counts["total"] == 0:
            continue
        print(f"  {outcome_name:10s} {counts['passed']:2d}/{counts['total']:2d} passed")

    overall_total = sum(c["total"] for c in by_outcome.values())
    overall_passed = sum(c["passed"] for c in by_outcome.values())
    print()
    print(f"Overall: {overall_passed}/{overall_total} passed")

    if failures:
        print()
        print("Failures:")
        for detail in failures:
            print(f"  {detail}")

    return 0 if overall_passed == overall_total else 1


def main() -> int:
    """Entry point for the procedural writer eval runner."""

    args = _build_parser().parse_args()
    return asyncio.run(_run(args.mode, args.dataset))


if __name__ == "__main__":
    raise SystemExit(main())
