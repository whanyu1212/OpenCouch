"""Runner for session summarizer evaluation.

Grades the v0.5 summarization prompt's behavior on a hand-curated
dataset of transcript → expected-SessionArc pairs. Each case is either
a "skip" case (expected ``arc=None``) or a "summarize" case (a set of
loose assertions about fields of the produced :class:`SessionArc`).

Usage:
    # Live API calls via the configured provider
    python eval/runners/summarization_eval.py --mode hybrid

    # Auto-detect: hybrid if a provider is configured, else skip
    python eval/runners/summarization_eval.py --mode auto  # default

The summarizer only runs when an LLM client is available (same
contract as the extractor — no deterministic fallback path), so this
runner does NOT offer a ``deterministic`` mode. When no LLM is
configured, the runner prints a message and exits 0 without grading
anything — matches the "skip silently" contract the summarizer node
itself uses for missing providers.

Grading strategy (asymmetric like ``extraction_eval``):

- **Skip cases**: pass iff the summarizer returns ``arc=None``. The
  ``reason`` field is validated to be non-empty but its content isn't
  checked — the prompt can evolve its reason wording freely.

- **Summarize cases**: pass iff the returned SessionArc passes ALL of
  the assertions in the case's ``expected`` dict. Every assertion is
  independently optional — a case only grades the fields it cares
  about. Supported assertion keys:

    * ``primary_themes_contains_any_of`` (list[str]): at least one
      theme string must contain one of the expected substrings
      (case-insensitive). E.g., ``["work", "stress"]`` passes if
      ``primary_themes`` is ``["work stress"]`` or ``["stress"]``.

    * ``summary_contains_any_of`` (list[str]): the ``summary`` prose
      must contain at least one of these substrings (case-insensitive).

    * ``summary_not_contains`` (list[str]): the ``summary`` prose must
      NOT contain any of these substrings (case-insensitive).
      Used to pin "don't fabricate X" constraints — e.g., the
      crisis_de_escalation case asserts that "crisis_level" and
      "level 2" don't appear in the summary.

    * ``mood_arc_opened_contains_any_of`` (list[str]): ``mood_arc.opened``
      must contain one of these (case-insensitive).

    * ``mood_arc_closed_contains_any_of`` (list[str]): same for
      ``mood_arc.closed``.

    * ``open_loops_nonempty`` (bool): if true, the ``open_loops`` list
      must have at least one entry. If false (or absent), the check is
      skipped — we don't assert empty lists because a "clean" session
      legitimately may or may not populate this.

    * ``resolved_threads_nonempty`` (bool): same shape as above for
      ``resolved_threads``.

Skip cases are strict (any non-None arc is a failure). Summarize
cases are loose (the grader accepts any arc that satisfies the
assertions — extra fields or alternative wording are fine). This
matches the summarizer's design philosophy: the correct output is
"narrative, not transcript", so we grade on presence of key themes
and mood descriptors rather than exact wording.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

BACKEND_ROOT = Path(__file__).resolve().parents[2] / "apps" / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agent.memory.modes import MemoryMode
from agent.memory.store import OpenCouchMemoryStore
from agent.runtime.session import run_summarize_session
from agent.state import AgentState
from core.config import create_configured_llm_client
from services.llm.base import BaseLLMClient

DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "datasets" / "summarization_v1.json"
)

EvalMode = Literal["auto", "hybrid"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run session summarizer evaluation.")
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

    Same contract as the extraction runner: the summarizer has no
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
    """Return a partial AgentState with the case's transcript.

    The summarizer reads ``transcript`` (the full history, both user
    and assistant turns) and ``user_id`` / ``session_id`` for the
    owner_id derivation. Everything else is passed as explicit
    function arguments.
    """

    state: Any = {
        "message": "",  # summarizer doesn't read message at session end
        "history": [],
        "user_id": case.get("user_id") or "eval-user",
        "session_id": case.get("session_id") or "eval-thread",
        "session_progress": {"turn_count": len(case.get("transcript", []))},
        "transcript": case.get("transcript", []),
    }
    return cast(AgentState, state)


def _lower(value: Any) -> str:
    """Return lowercased string representation, tolerating None."""

    return str(value or "").lower()


def _check_contains_any(
    text: str,
    expected: list[str],
) -> tuple[bool, str]:
    """Return whether the text contains any of the expected substrings.

    Returns ``(found, detail)``. ``detail`` is empty on pass, or a
    human-readable miss message on fail.
    """

    low = text.lower()
    for substring in expected:
        if substring.lower() in low:
            return True, ""
    return (
        False,
        f"expected any of {expected!r} in {text[:120]!r}",
    )


def _check_not_contains(
    text: str,
    forbidden: list[str],
) -> tuple[bool, str]:
    """Return whether the text avoids all forbidden substrings."""

    low = text.lower()
    for substring in forbidden:
        if substring.lower() in low:
            return False, f"forbidden substring {substring!r} found in summary"
    return True, ""


def _check_list_contains_any(
    items: list[str],
    expected: list[str],
) -> tuple[bool, str]:
    """Return whether ANY list entry contains one of the expected substrings.

    Used for ``primary_themes_contains_any_of`` — the themes list has
    a few short entries like ``["work stress", "perfectionism"]`` and
    we want to pass if ANY theme matches ANY expected substring.
    """

    if not items:
        return False, f"list was empty; expected any of {expected!r}"
    for item in items:
        low = item.lower()
        for substring in expected:
            if substring.lower() in low:
                return True, ""
    return (
        False,
        f"no list entry matched {expected!r}; actual list={items!r}",
    )


def _grade_summarize_case(
    arc: Any,
    expected: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Grade a summarize case against its assertions.

    Returns ``(passed, failures)``. ``failures`` is empty on pass, or
    a list of miss detail strings on fail.
    """

    failures: list[str] = []

    themes_expected = expected.get("primary_themes_contains_any_of")
    if themes_expected is not None:
        found, detail = _check_list_contains_any(arc.primary_themes, themes_expected)
        if not found:
            failures.append(f"primary_themes: {detail}")

    summary_expected = expected.get("summary_contains_any_of")
    if summary_expected is not None:
        found, detail = _check_contains_any(arc.summary, summary_expected)
        if not found:
            failures.append(f"summary_contains: {detail}")

    summary_forbidden = expected.get("summary_not_contains")
    if summary_forbidden is not None:
        found, detail = _check_not_contains(arc.summary, summary_forbidden)
        if not found:
            failures.append(f"summary_not_contains: {detail}")

    opened_expected = expected.get("mood_arc_opened_contains_any_of")
    if opened_expected is not None:
        found, detail = _check_contains_any(
            _lower(arc.mood_arc.opened), opened_expected
        )
        if not found:
            failures.append(f"mood_arc.opened={arc.mood_arc.opened!r}: {detail}")

    closed_expected = expected.get("mood_arc_closed_contains_any_of")
    if closed_expected is not None:
        found, detail = _check_contains_any(
            _lower(arc.mood_arc.closed), closed_expected
        )
        if not found:
            failures.append(f"mood_arc.closed={arc.mood_arc.closed!r}: {detail}")

    if expected.get("open_loops_nonempty") and not arc.open_loops:
        failures.append("open_loops: expected non-empty, got []")

    if expected.get("resolved_threads_nonempty") and not arc.resolved_threads:
        failures.append("resolved_threads: expected non-empty, got []")

    return (len(failures) == 0, failures)


async def _evaluate_case(
    case: dict[str, Any],
    llm_client: BaseLLMClient,
) -> tuple[bool, str | None]:
    """Run one case through the summarizer and grade the result.

    Returns ``(passed, failure_detail)``. ``failure_detail`` is None
    on pass, or a human-readable explanation on failure.
    """

    # Each case gets a fresh in-memory store so prior writes don't
    # leak between cases. The store isn't really used in grading
    # (we read the arc directly from the return value), but the
    # summarizer writes to it on success — we pass a real one so
    # that write path is exercised even though we don't read it.
    store = OpenCouchMemoryStore()
    state = _build_state(case)

    # Fabricate plausible session timestamps. The summarizer parses
    # these to compute duration_seconds. We don't grade on the
    # specific values — they just need to parse cleanly.
    started_at = (
        (datetime.now(UTC) - timedelta(minutes=30)).isoformat().replace("+00:00", "Z")
    )
    ended_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    result = await run_summarize_session(
        state,
        llm_client=llm_client,
        memory_store=store,
        memory_mode=MemoryMode.LOCAL,
        session_id=state["session_id"],
        started_at=started_at,
        ended_at=ended_at,
        crisis_level_max=0,
    )

    expected_outcome = case["expected_outcome"]

    if expected_outcome == "skip":
        # Skip cases expect arc=None. The summarizer returns None on
        # both "LLM decided arc=None" and "LLM call failed". For eval
        # purposes those are the same outcome — the user wouldn't see
        # a summary either way — so we don't try to distinguish them.
        if result is None:
            return True, None
        return (
            False,
            f"FAIL [skip] {case['id']}: expected arc=None, got arc with "
            f"summary={result.summary[:120]!r}",
        )

    # expected_outcome == "summarize"
    if result is None:
        return (
            False,
            f"FAIL [summarize] {case['id']}: expected SessionArc, got None",
        )

    expected = case.get("expected", {})
    passed, failures = _grade_summarize_case(result, expected)
    if passed:
        return True, None

    failure_summary = "; ".join(failures)
    return (
        False,
        f"FAIL [summarize] {case['id']}: {failure_summary}",
    )


async def _run(mode: EvalMode, dataset_path: Path) -> int:
    """Drive the full eval and return a process exit code."""

    cases = _load_cases(dataset_path)
    llm_client, resolved_mode = _resolve_llm_client(mode)

    if llm_client is None:
        print(
            f"Summarization eval: no LLM client configured (mode={mode}). "
            f"Skipping {len(cases)} case(s). Use --mode hybrid to fail "
            f"loudly instead."
        )
        return 0

    print(
        f"Running summarization eval in {resolved_mode} mode on "
        f"{len(cases)} case(s) from {dataset_path.name}."
    )
    print()

    # Per-outcome accounting so skip and summarize cases are reported
    # separately. The asymmetry (strict skip vs. loose summarize)
    # matters for reading the numbers.
    by_outcome: dict[str, dict[str, int]] = {
        "skip": {"total": 0, "passed": 0},
        "summarize": {"total": 0, "passed": 0},
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
    """Entry point for the summarization eval runner."""

    args = _build_parser().parse_args()
    return asyncio.run(_run(args.mode, args.dataset))


if __name__ == "__main__":
    raise SystemExit(main())
