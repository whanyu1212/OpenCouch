"""Profile post-finalize extraction latency to inform Opportunity #5/#6.

This runner answers a single question: **how much wall-clock would
background extraction (#5) actually save?**

It runs a representative mix of turns through the production runtime
(``PersistentAgentRuntime``), captures the post-finalize diagnostics
(``extract_facts_ms``, ``extract_procedural_ms``, ``post_finalize_ms``,
``turn_total_ms``), and prints the distribution. The output is a
printout, not a pass/fail — pytest is for correctness, this script
is for evidence-driven roadmap decisions.

Usage:
    # Default: 10 cases sampled from extraction_v1 + procedural_v1
    python eval/runners/extraction_latency_profile.py

    # Larger sample
    python eval/runners/extraction_latency_profile.py --sample 30

    # Use a specific dataset
    python eval/runners/extraction_latency_profile.py \\
        --dataset eval/datasets/procedural_v1.json --sample 18

Decision rule (printed at the bottom of every report):
    - post_finalize_ms median <  ~50ms → no headroom; skip #5/#6.
    - post_finalize_ms median 50-300ms → marginal; skip unless dashboard
        signal forces it.
    - post_finalize_ms median > ~300ms → real latency to recover; do the
        de-risking pass and consider implementation.

Per-extractor numbers also tell you whether to do BOTH or just ONE:
    - If extract_facts_ms ≫ extract_procedural_ms, only background-ify
        the semantic extractor.
    - If both are comparable, do them together (#5 implies #6 anyway).

Output is intentionally LLM-provider-stamped. Numbers from openai vs.
gemini vs. local are not directly comparable; always read the provider
line at the top of the report.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2] / "apps" / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agent.memory.modes import MemoryMode
from agent.persistence import PersistentAgentRuntime
from config import create_configured_llm_client, get_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASETS = [
    REPO_ROOT / "eval" / "datasets" / "extraction_v1.json",
    REPO_ROOT / "eval" / "datasets" / "procedural_v1.json",
]


@dataclass(slots=True)
class TurnSample:
    """One profiled turn's latency breakdown."""

    case_id: str
    turn_total_ms: float
    post_finalize_ms: float
    extract_facts_ms: float
    extract_procedural_ms: float


def _load_cases(paths: list[Path]) -> list[dict[str, Any]]:
    """Load and concatenate cases from one or more dataset files.

    Args:
        paths: Dataset JSON files to load.

    Returns:
        Combined list of case dicts; each must have ``id`` and ``message``.
    """

    cases: list[dict[str, Any]] = []
    for path in paths:
        loaded = json.loads(path.read_text())
        for case in loaded:
            case.setdefault("_source", path.name)
        cases.extend(loaded)
    return cases


def _percentile(values: list[float], pct: float) -> float:
    """Return the ``pct`` percentile of ``values`` using nearest-rank.

    Args:
        values: Sample values; need not be sorted.
        pct: Percentile in [0, 100].

    Returns:
        The percentile value, or ``0.0`` when ``values`` is empty.
    """

    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round(pct / 100.0 * (len(ordered) - 1)))))
    return ordered[k]


def _format_distribution(label: str, values: list[float]) -> str:
    """Render a single metric's median/p95/min/max as a one-line report."""

    if not values:
        return f"  {label:<25}  no samples"
    return (
        f"  {label:<25}  "
        f"median={statistics.median(values):>7.1f}ms  "
        f"p95={_percentile(values, 95):>7.1f}ms  "
        f"min={min(values):>7.1f}ms  "
        f"max={max(values):>7.1f}ms  "
        f"n={len(values)}"
    )


async def _profile_case(
    runtime: PersistentAgentRuntime,
    case: dict[str, Any],
    *,
    thread_id: str,
    llm_client: Any,
) -> TurnSample | None:
    """Run one case through the runtime and collect its latency sample.

    Args:
        runtime: Live runtime instance.
        case: Dataset case dict (must have ``id`` and ``message``).
        thread_id: Per-case thread id so prior-turn checkpoints don't
            interfere across cases.
        llm_client: LLM client to drive the turn. Passed via ``run_turn``'s
            per-turn parameter (NOT the runtime's ``default_llm_client``,
            which is used only by background sweeps). Without this the
            extractors short-circuit with ``"skipped: no llm_client"``.

    Returns:
        ``TurnSample`` carrying the four diagnostic timings, or ``None``
        when the turn produced no diagnostics (defensive — should not
        happen on the therapeutic path).
    """

    result = await runtime.run_turn(
        thread_id=thread_id,
        message=str(case["message"]),
        llm_client=llm_client,
    )
    diag = result.output.diagnostics or {}

    if "post_finalize_ms" not in diag or "turn_total_ms" not in diag:
        return None

    return TurnSample(
        case_id=str(case["id"]),
        turn_total_ms=float(diag["turn_total_ms"]),
        post_finalize_ms=float(diag["post_finalize_ms"]),
        extract_facts_ms=float(diag.get("extract_facts_ms", 0.0)),
        extract_procedural_ms=float(diag.get("extract_procedural_ms", 0.0)),
    )


async def _run_profile(
    cases: list[dict[str, Any]],
    *,
    sample_size: int,
    seed: int,
) -> list[TurnSample]:
    """Sample N cases, run each through a fresh runtime, return all samples.

    A fresh ``PersistentAgentRuntime`` per profile run keeps memory state
    isolated from prior runs so retrieval-store size doesn't drift across
    iterations and skew load_memory timings (which would in turn shift
    the post_finalize fraction).

    Args:
        cases: Pool of dataset cases.
        sample_size: Number of cases to profile.
        seed: RNG seed for reproducible sampling.

    Returns:
        Per-turn ``TurnSample`` records.
    """

    rng = random.Random(seed)
    chosen = rng.sample(cases, min(sample_size, len(cases)))

    llm_client = create_configured_llm_client()

    samples: list[TurnSample] = []
    async with PersistentAgentRuntime(
        sqlite_path=":memory:",
        memory_sqlite_path=":memory:",
        crisis_log_sqlite_path=":memory:",
        memory_mode=MemoryMode.LOCAL,
        default_llm_client=llm_client,
    ) as runtime:
        for index, case in enumerate(chosen):
            try:
                sample = await _profile_case(
                    runtime,
                    case,
                    thread_id=f"profile-{index}",
                    llm_client=llm_client,
                )
            except Exception as exc:  # pragma: no cover — best-effort profiling
                print(f"  [skip] case {case.get('id')!r} raised: {exc}")
                continue
            if sample is not None:
                samples.append(sample)
                print(
                    f"  [{index + 1:>3}/{len(chosen)}] {sample.case_id:<45}  "
                    f"post_finalize={sample.post_finalize_ms:>6.1f}ms  "
                    f"facts={sample.extract_facts_ms:>6.1f}ms  "
                    f"proc={sample.extract_procedural_ms:>6.1f}ms"
                )
    return samples


def _print_report(samples: list[TurnSample]) -> None:
    """Render the distribution + decision rule summary."""

    if not samples:
        print("\nNo samples collected. Re-run with a configured LLM provider.")
        return

    print("\n" + "─" * 72)
    print("Distribution")
    print("─" * 72)
    print(_format_distribution("turn_total_ms", [s.turn_total_ms for s in samples]))
    print(
        _format_distribution(
            "  post_finalize_ms", [s.post_finalize_ms for s in samples]
        )
    )
    print(
        _format_distribution(
            "    extract_facts_ms", [s.extract_facts_ms for s in samples]
        )
    )
    print(
        _format_distribution(
            "    extract_procedural_ms",
            [s.extract_procedural_ms for s in samples],
        )
    )

    median_post = statistics.median(s.post_finalize_ms for s in samples)
    median_facts = statistics.median(s.extract_facts_ms for s in samples)
    median_proc = statistics.median(s.extract_procedural_ms for s in samples)

    print("\n" + "─" * 72)
    print("Decision rule for Opportunity #5/#6 (background extraction)")
    print("─" * 72)
    if median_post < 50:
        verdict = (
            "SKIP. Median post_finalize_ms < 50ms — extraction is already "
            "near-instant. Background-ifying it would add lifecycle complexity "
            "for negligible savings."
        )
    elif median_post < 300:
        verdict = (
            "MARGINAL. Median post_finalize_ms is in the 50-300ms band. The "
            "savings are real but small relative to the contract churn (#5 "
            "needs 200+ LOC across runtime, shutdown, tests). Skip unless a "
            "production dashboard signal makes p95 the priority."
        )
    else:
        verdict = (
            "WORTH IT. Median post_finalize_ms > 300ms. Background extraction "
            "would meaningfully reduce end-to-end turn duration. Run the "
            "de-risking pass before implementation (test surface enumeration, "
            "DoneEvent consumer audit, mutation-token lifecycle sketch)."
        )
    print(f"  Verdict: {verdict}")

    if median_post >= 50 and median_facts > 1.5 * max(median_proc, 1.0):
        print(
            "\n  Note: extract_facts_ms is much larger than "
            "extract_procedural_ms — if you proceed, only the semantic "
            "extractor is worth backgrounding. Procedural can stay "
            "synchronous."
        )
    elif median_post >= 50 and median_proc > 1.5 * max(median_facts, 1.0):
        print(
            "\n  Note: extract_procedural_ms dominates — only the procedural "
            "extractor is worth backgrounding."
        )

    print("─" * 72)


def main() -> int:
    """Entry point.

    Returns:
        ``0`` on success, ``1`` when no LLM client is configured.
    """

    parser = argparse.ArgumentParser(
        description="Profile post-finalize extraction latency for #5/#6."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        action="append",
        default=None,
        help=(
            "Dataset JSON file. Pass multiple times to mix datasets. "
            f"Defaults to: {[p.name for p in DEFAULT_DATASETS]}."
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=10,
        help="Number of cases to profile. Default: 10.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for reproducible sampling. Default: 0.",
    )
    args = parser.parse_args()

    datasets = args.dataset or DEFAULT_DATASETS
    cases = _load_cases(datasets)
    print(f"Loaded {len(cases)} case(s) from: {[Path(p).name for p in datasets]}")

    try:
        settings = get_settings()
    except Exception as exc:
        print(f"Failed to load runtime settings: {exc}")
        return 1
    print(
        f"LLM provider: {settings.llm_provider} | "
        f"model: {settings.openai_model if settings.llm_provider == 'openai' else settings.gemini_model}"
    )
    print(f"Sampling {args.sample} case(s) (seed={args.seed}):")
    print()

    try:
        samples = asyncio.run(
            _run_profile(cases, sample_size=args.sample, seed=args.seed)
        )
    except Exception as exc:
        print(f"\nProfile run failed: {exc}")
        return 1

    _print_report(samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
