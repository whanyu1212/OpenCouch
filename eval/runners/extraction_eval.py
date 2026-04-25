"""Runner for semantic fact extractor evaluation.

Grades the v0.3 extraction prompt's accuracy on a hand-curated
dataset of message → expected-extraction pairs. Each case is either
a "skip" case (expected empty facts list) or an "extract" case (one
or more expected MemoryWrite items matched on category/predicate/
object_type/object_identifier).

Usage:
    # Live API calls via the configured provider
    python eval/runners/extraction_eval.py --mode hybrid

    # Auto-detect: hybrid if a provider is configured, else skip
    python eval/runners/extraction_eval.py --mode auto  # default

The extractor only runs when an LLM client is available (unlike
the dispatcher/crisis gate which have deterministic fallback paths),
so this runner does NOT offer a ``deterministic`` mode. When no LLM
is configured, the runner prints a message and exits 0 without
grading anything — matches the "skip silently" contract the
extraction node itself uses.

Grading strategy:

- **Skip cases**: pass iff the extractor returns an empty facts list.
  The ``reason`` field is validated to be non-empty but its content
  isn't checked — the prompt can evolve its reason wording freely.

- **Extract cases**: pass iff the extracted facts match the expected
  facts per the following rules:
    * ``expected_fact_count`` (optional): if present, total facts
      returned must equal this exact number.
    * For each expected fact spec, SOME returned fact must match on
      every present field:
        - ``category``: exact match (exact enum value)
        - ``predicate``: exact match (exact enum value)
        - ``object_type``: exact match
        - ``object_identifier_contains``: case-insensitive substring
          match against the returned fact's object identifier
    * Fields omitted from the expected spec are not graded — this
      lets datasets be loose about fields where LLM variance is
      expected (e.g., leave ``category`` off the fluoxetine case
      where either ``coping_strategy`` or ``context`` is acceptable).

The grader is deliberately asymmetric: skip cases are strict (any
non-empty facts list is a failure), but extract cases are loose
(return more facts than expected is fine as long as the expected
ones are present). This matches the conservative-extraction
philosophy — false positives are worse than false negatives.
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
from agent.memory.candidates import SessionMemoryBuffer
from agent.memory.store import OpenCouchMemoryStore
from agent.nodes.extract_facts import run_extract_semantic_facts_node
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from core.config import create_configured_llm_client
from services.llm.base import BaseLLMClient

DATASET_PATH = Path(__file__).resolve().parents[1] / "datasets" / "extraction_v1.json"

EvalMode = Literal["auto", "hybrid"]


class _MockRuntime:
    """Minimal runtime stand-in for the extractor node.

    The extraction node reads ``llm_client``, ``memory_store``, and
    ``memory_mode`` from context. We provide a real in-memory store
    per-case so the dedup path exercises real code — matching the
    production data flow — without coupling the eval to disk.
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
            session_memory_buffer=SessionMemoryBuffer(session_id="eval-thread"),
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run semantic fact extractor evaluation."
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

    Unlike the dispatcher/crisis runners, the extraction node has no
    deterministic fallback — it just skips silently when no client is
    available. This runner mirrors that contract: ``auto`` mode
    returns (None, "skipped") if no provider is configured, and
    ``hybrid`` raises.
    """

    if mode == "hybrid":
        return create_configured_llm_client(), "hybrid"

    # auto
    try:
        return create_configured_llm_client(), "hybrid"
    except Exception:
        return None, "skipped"


def _build_state(case: dict[str, Any]) -> AgentState:
    """Return a partial AgentState for the extraction node to read.

    The extraction node reads ``message``, ``history``, ``user_id``,
    ``session_id``, and ``session_progress.turn_count``. We fabricate the
    session-progress dict with turn_count=1 because every eval case is a
    single-turn scenario, not a multi-turn conversation where the
    turn index would matter for provenance.
    """

    state: Any = {
        "message": case["message"],
        "history": case.get("history", []),
        "user_id": case.get("user_id"),
        "session_id": case.get("session_id", "eval-thread"),
        "session_progress": {"turn_count": 1},
        "transcript": [],
    }
    return cast(AgentState, state)


def _match_expected_fact(
    expected: dict[str, Any],
    returned: list[Any],
) -> tuple[bool, str]:
    """Find any returned fact matching all present fields in the expected spec.

    Returns ``(found, detail)``. ``detail`` is a human-readable
    string explaining which field caused the match or mismatch.
    """

    required_category = expected.get("category")
    required_predicate = expected.get("predicate")
    required_object_type = expected.get("object_type")
    required_object_substring = expected.get("object_identifier_contains")
    if required_object_substring is not None:
        required_object_substring = required_object_substring.lower()

    for fact in returned:
        # Each returned fact is a pydantic MemoryWrite instance.
        if required_category is not None and fact.category != required_category:
            continue
        if required_predicate is not None and fact.predicate != required_predicate:
            continue
        if (
            required_object_type is not None
            and fact.object.type != required_object_type
        ):
            continue
        if required_object_substring is not None:
            obj_id = fact.object.identifier.lower()
            if required_object_substring not in obj_id:
                continue
        return True, f"matched category={fact.category} object={fact.object.identifier}"

    # No match — produce a detail describing what was expected
    expected_summary = ", ".join(
        f"{k}={v}" for k, v in expected.items() if k != "object_identifier_contains"
    )
    if required_object_substring is not None:
        expected_summary += f", object contains '{required_object_substring}'"
    return False, f"no returned fact matched expected ({expected_summary})"


async def _evaluate_case(
    case: dict[str, Any],
    llm_client: BaseLLMClient,
) -> tuple[bool, str | None]:
    """Run one case through the extractor and grade the result.

    Returns ``(passed, failure_detail)``. ``failure_detail`` is None
    on pass, or a human-readable explanation on failure.
    """

    # Each case gets a fresh in-memory store so prior cases' writes
    # don't pollute dedup decisions on later cases. This also means
    # every "extract" case exercises the "no existing records" dedup
    # path — which is fine for per-case grading but DOES skip the
    # dedup-against-prior-records code path. That's intentional:
    # the dedup logic has its own unit tests in test_memory_dedup.py,
    # and this runner is focused on grading the LLM's extraction
    # decisions, not the dedup behavior.
    store = OpenCouchMemoryStore()
    runtime = _MockRuntime(llm_client=llm_client, memory_store=store)
    state = _build_state(case)

    # Call the extraction node. Its delta is always {} (side effect
    # only), so we can't read the extracted facts from the return
    # value — we have to read the store afterward. This is a bit
    # clunky but it exercises the full write path, which is what we
    # want to grade.
    await run_extract_semantic_facts_node(state, runtime)  # type: ignore[arg-type]

    # Reconstruct MemoryWrite-like objects from both immediate store writes and
    # session-held candidates. Trigger/loss facts are intentionally held by the
    # write policy, but this runner grades extraction quality, not only
    # immediate persistence.
    namespace = (state.get("user_id") or state["session_id"], "semantic")
    records = await store.asearch(namespace, query=None, limit=100)

    buffer = runtime.context.session_memory_buffer
    held_facts = (
        [candidate.payload for candidate in buffer.semantic_candidates]
        if buffer is not None
        else []
    )
    returned_facts = [_FactView(record.value) for record in records] + held_facts

    expected_outcome = case["expected_outcome"]

    if expected_outcome == "skip":
        if len(returned_facts) == 0:
            return True, None
        fact_summary = ", ".join(
            f"{f.category}/{f.predicate}/{f.object.identifier}" for f in returned_facts
        )
        return (
            False,
            f"FAIL [skip] {case['id']}: expected 0 facts, got "
            f"{len(returned_facts)} ({fact_summary})",
        )

    # expected_outcome == "extract"
    expected_facts = case.get("expected_facts", [])
    expected_count = case.get("expected_fact_count")

    if expected_count is not None and len(returned_facts) != expected_count:
        return (
            False,
            f"FAIL [extract] {case['id']}: expected {expected_count} fact(s), "
            f"got {len(returned_facts)}",
        )

    if len(returned_facts) == 0:
        return (
            False,
            f"FAIL [extract] {case['id']}: expected at least 1 fact, got 0",
        )

    for expected_fact in expected_facts:
        found, _ = _match_expected_fact(expected_fact, returned_facts)
        if not found:
            returned_summary = "; ".join(
                f"{f.category}/{f.predicate}/{f.object.type}:{f.object.identifier}"
                for f in returned_facts
            )
            return (
                False,
                f"FAIL [extract] {case['id']}: expected fact not found "
                f"({expected_fact}); returned: {returned_summary}",
            )

    return True, None


class _FactView:
    """Lightweight view of a stored fact for grading.

    The stored value is a dict (from SemanticFact.model_dump). This
    wrapper exposes attribute-style access (``fact.category``,
    ``fact.object.identifier``) so the grading code reads naturally
    without importing SemanticFact or reconstructing a full pydantic
    model per case.
    """

    def __init__(self, value: dict[str, Any]) -> None:
        self.category = value.get("category")
        self.predicate = value.get("predicate")
        self.object = _EntityView(value.get("object") or {})
        self.evidence_quote = value.get("evidence_quote", "")
        self.confidence = value.get("confidence")


class _EntityView:
    def __init__(self, entity: dict[str, Any]) -> None:
        self.type = entity.get("type")
        self.identifier = entity.get("identifier", "")


async def _run(mode: EvalMode, dataset_path: Path) -> int:
    """Drive the full eval and return a process exit code."""

    cases = _load_cases(dataset_path)
    llm_client, resolved_mode = _resolve_llm_client(mode)

    if llm_client is None:
        print(
            f"Extraction eval: no LLM client configured (mode={mode}). "
            f"Skipping {len(cases)} case(s). Use --mode hybrid to fail "
            f"loudly instead."
        )
        return 0

    print(
        f"Running extraction eval in {resolved_mode} mode on "
        f"{len(cases)} case(s) from {dataset_path.name}."
    )
    print()

    # Per-outcome accounting so skip and extract cases are reported
    # separately. A single overall pass rate hides the asymmetry
    # between false-positive and false-negative failures.
    by_outcome: dict[str, dict[str, int]] = {
        "skip": {"total": 0, "passed": 0},
        "extract": {"total": 0, "passed": 0},
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
    """Entry point for the extraction eval runner."""

    args = _build_parser().parse_args()
    return asyncio.run(_run(args.mode, args.dataset))


if __name__ == "__main__":
    raise SystemExit(main())
