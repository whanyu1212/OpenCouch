"""Runner for retrieval quality evaluation (v0.8.1).

Grades the store's retrieval path against a hand-curated dataset of
``(seed_records, query, expected_keys)`` cases, comparing three
scorers side-by-side:

1. **token_recall** — the v0.3.1 scorer via ``store.asearch(query=...)``.
   Deterministic, literal token overlap, threshold 0.33.
2. **pure_embedding** — cosine similarity only, via a direct scan
   through the in-memory store's records. No token-recall contribution.
3. **hybrid_rrf** — the v0.8.1 path via ``store.asearch_similar(...)``
   with both scorers combined via Reciprocal Rank Fusion.

Unlike the other eval runners (dispatcher, extractor, crisis, summarizer)
this one does NOT invoke any LLM structured-output call. It seeds a
per-case in-memory store with pre-built records, runs the query through
all three scorers, and grades on simple set-membership:

- **recall@1**: is the expected key returned as the top-1 result?
- **recall@5**: is the expected key returned anywhere in the top-5?

Negative cases (``expected_keys == []``) are graded inversely:
return fewer than 1 result = pass, return anything = fail.

Usage:
    # Run with embeddings (requires GEMINI_API_KEY or GOOGLE_API_KEY)
    python eval/runners/retrieval_eval.py --mode hybrid

    # Auto-detect: runs all three scorers if a provider is configured,
    # otherwise runs only token_recall and reports "embeddings skipped"
    python eval/runners/retrieval_eval.py --mode auto  # default

    # Token-recall only (useful when debugging token-recall regressions
    # or running in environments without network access)
    python eval/runners/retrieval_eval.py --mode token-only

Why this runner is important: the whole point of v0.8.1's hybrid
retrieval is that it beats pure token-recall on stemming, synonyms,
paraphrase, and low-signal queries, while matching or beating pure
embedding on proper-noun and short-query cases. Without this runner,
"hybrid is the right choice" is theoretical. With it, the comparison
matrix shows which scorer wins which case and why. Dogfood
observations that find a retrieval failure should land as a new case
in ``retrieval_v1.json`` before the fix, so the improvement is
regression-pinned.

Reporting shape:

    Category       token_recall    pure_embedding    hybrid_rrf
    ───────────────────────────────────────────────────────────
    stemming       0/3  / 0/3      3/3  / 3/3        3/3  / 3/3
    synonyms       ...

Two numbers per cell are recall@1 / recall@5.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Literal

BACKEND_ROOT = Path(__file__).resolve().parents[2] / "apps" / "backend"
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Load .env files so API keys (OPENAI_API_KEY, GEMINI_API_KEY) are
# available to the embedding provider. The main app does this via
# core/config.py, but eval runners import providers directly.
from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env", override=False)
load_dotenv(BACKEND_ROOT / ".env", override=False)
load_dotenv(BACKEND_ROOT / ".env.local", override=False)

from agent.memory.embeddings import (
    EmbeddingProvider,
    NullEmbeddingProvider,
    create_configured_embedding_provider,
)
from agent.memory.retrieval import (
    EMBEDDING_MATCH_THRESHOLD,
    ScoredRecord,
    cosine_similarity,
)
from agent.memory.store import OpenCouchMemoryStore

DATASET_PATH = Path(__file__).resolve().parents[1] / "datasets" / "retrieval_v1.json"

EvalMode = Literal["auto", "hybrid", "token-only"]
ScorerName = Literal["token_recall", "pure_embedding", "hybrid_rrf"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run retrieval quality evaluation (v0.8.1)."
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "hybrid", "token-only"],
        default="auto",
        help=(
            "Evaluation mode. 'auto' runs all three scorers when an "
            "embedding provider is configured and falls back to "
            "token-recall only otherwise. 'hybrid' requires an embedding "
            "provider and fails loudly if none is configured. 'token-only' "
            "grades only the v0.3.1 token-recall path — useful when "
            "debugging token-recall regressions or running offline."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help=f"Dataset JSON path. Default: {DATASET_PATH}",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-case outcomes, not just the summary matrix.",
    )
    return parser


def _load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


async def _seed_store(
    store: OpenCouchMemoryStore,
    *,
    owner_id: str,
    records: list[dict[str, Any]],
    embedding_provider: EmbeddingProvider | None,
) -> None:
    """Populate the store with the case's seed records.

    When an embedding provider is configured, each seed record is
    embedded via the provider's ``aembed`` (with
    ``task_type="RETRIEVAL_DOCUMENT"``, matching production
    write-path semantics) and the embedding is attached to the
    record via ``aput(embedding=...)``. When no provider is
    configured, records are written without embeddings and only
    the token-recall path can return them.

    The namespace is always ``(owner_id, "semantic")`` — this
    eval doesn't exercise episodic retrieval (the episodic path
    has additional catch-up logic that would muddy the scorer
    comparison). A future episodic retrieval eval could live in
    ``retrieval_episodic_v1.json``.
    """

    namespace = (owner_id, "semantic")

    # Collect quotes in record-list order for the batch embedding call.
    quotes = [r["value"].get("evidence_quote", "") for r in records]

    embeddings: list[list[float] | None] = [None] * len(quotes)
    embedding_model: str | None = None
    if embedding_provider is not None and not isinstance(
        embedding_provider, NullEmbeddingProvider
    ):
        embeddings = await embedding_provider.aembed(
            quotes,
            task_type="RETRIEVAL_DOCUMENT",
        )
        embedding_model = embedding_provider.model_name

    for record, embedding in zip(records, embeddings, strict=True):
        await store.aput(
            namespace,
            key=record["key"],
            value=record["value"],
            embedding=embedding,
            embedding_model=(embedding_model if embedding is not None else None),
        )


async def _score_token_recall(
    store: OpenCouchMemoryStore,
    *,
    namespace: tuple[str, ...],
    query: str,
    limit: int,
) -> list[str]:
    """Return record keys ranked by v0.3.1 token-recall (via asearch)."""

    records = await store.asearch(namespace, query=query, limit=limit)
    return [r.key for r in records]


async def _score_pure_embedding(
    store: OpenCouchMemoryStore,
    *,
    namespace: tuple[str, ...],
    query_embedding: list[float] | None,
    embedding_model: str | None,
    limit: int,
) -> list[str]:
    """Return record keys ranked by cosine similarity only.

    Implemented as a direct scan through the in-memory store's
    records rather than going through ``asearch_similar``, because
    ``asearch_similar`` always runs the token-recall side too and
    we want the pure-embedding ranking in isolation for the
    comparison matrix. The scan matches the logic inside
    ``OpenCouchMemoryStore.asearch_similar`` but omits the lexical
    pass — essentially "what would hybrid RRF look like if the
    lexical list were empty?"

    When ``query_embedding`` is None (no provider), returns an
    empty list — the runner treats this as "scorer not available"
    rather than as a recall failure.
    """

    if query_embedding is None:
        return []

    bucket = store._buckets.get(namespace)  # type: ignore[attr-defined]
    if bucket is None:
        return []

    scored: list[ScoredRecord] = []
    for insertion_index, record in enumerate(bucket.records.values()):
        if record.embedding is None:
            continue
        if (
            embedding_model is not None
            and record.embedding_model is not None
            and record.embedding_model != embedding_model
        ):
            continue
        if len(record.embedding) != len(query_embedding):
            continue
        sim = cosine_similarity(query_embedding, record.embedding)
        if sim >= EMBEDDING_MATCH_THRESHOLD:
            scored.append(
                ScoredRecord(
                    record=record,
                    score=sim,
                    insertion_index=insertion_index,
                )
            )

    scored.sort(key=lambda sr: (-sr.score, sr.insertion_index))
    return [sr.record.key for sr in scored[:limit]]


async def _score_hybrid_rrf(
    store: OpenCouchMemoryStore,
    *,
    namespace: tuple[str, ...],
    query: str,
    query_embedding: list[float] | None,
    embedding_model: str | None,
    limit: int,
) -> list[str]:
    """Return record keys ranked by v0.8.1 hybrid RRF (via asearch_similar)."""

    records = await store.asearch_similar(
        namespace,
        query_text=query,
        query_embedding=query_embedding,
        embedding_model=embedding_model,
        limit=limit,
    )
    return [r.key for r in records]


def _grade_ranking(
    returned_keys: list[str],
    *,
    expected_keys: list[str],
) -> tuple[bool, bool]:
    """Return ``(recall_at_1, recall_at_5)`` for a ranked list.

    Positive cases: ``expected_keys`` is non-empty. recall@1 is True
    iff the top-1 result is in ``expected_keys``; recall@5 is True
    iff any key in the top-5 is in ``expected_keys``.

    Negative cases: ``expected_keys`` is empty. Both recall metrics
    are True iff the returned list is empty — returning anything is
    a false positive.

    The recall@1 metric is stricter and captures "does the best
    scorer put the right thing at the top?" The recall@5 metric is
    looser and captures "is the right thing at least in the
    context window?" which matters because load_memory_node uses
    limit=5.
    """

    if not expected_keys:
        # Negative case: any result is a failure.
        empty = len(returned_keys) == 0
        return empty, empty

    top1_hit = len(returned_keys) >= 1 and returned_keys[0] in expected_keys
    top5_hit = any(key in expected_keys for key in returned_keys[:5])
    return top1_hit, top5_hit


def _resolve_provider(mode: EvalMode) -> tuple[EmbeddingProvider | None, str]:
    """Resolve the embedding provider based on the eval mode.

    Returns ``(provider, resolved_mode_label)``.

    - ``hybrid`` → require a real provider; raise otherwise.
    - ``token-only`` → always return None regardless of env.
    - ``auto`` → use a real provider if configured, else None.

    ``NullEmbeddingProvider`` is treated the same as None for the
    purposes of mode resolution: if no real provider is available,
    we fall back to token-only grading and label the run accordingly.
    """

    if mode == "token-only":
        return None, "token-only"

    if mode == "hybrid":
        provider = create_configured_embedding_provider()
        if isinstance(provider, NullEmbeddingProvider):
            raise RuntimeError(
                "Retrieval eval --mode hybrid requires a real embedding "
                "provider. Set OPENAI_API_KEY (preferred) or GEMINI_API_KEY."
            )
        return provider, "hybrid"

    # auto
    provider = create_configured_embedding_provider()
    if isinstance(provider, NullEmbeddingProvider):
        return None, "auto (no provider — token-only)"
    return provider, "auto (hybrid)"


async def _run(mode: EvalMode, dataset_path: Path, verbose: bool) -> int:
    cases = _load_cases(dataset_path)
    provider, resolved_mode = _resolve_provider(mode)

    run_embedding_scorers = provider is not None

    print(
        f"Running retrieval eval in {resolved_mode} mode on "
        f"{len(cases)} case(s) from {dataset_path.name}."
    )
    if not run_embedding_scorers:
        print(
            "  (No embedding provider available — grading token_recall only. "
            "Set OPENAI_API_KEY (preferred) or GEMINI_API_KEY for hybrid grading.)"
        )
    print()

    # Grid: category → scorer → (pass@1 count, pass@5 count, total)
    scorers: list[ScorerName] = (
        ["token_recall", "pure_embedding", "hybrid_rrf"]
        if run_embedding_scorers
        else ["token_recall"]
    )
    grid: dict[str, dict[ScorerName, dict[str, int]]] = {}
    per_case_detail: list[
        tuple[str, str, dict[ScorerName, tuple[bool, bool, list[str]]]]
    ] = []
    total_by_scorer: dict[ScorerName, dict[str, int]] = {
        s: {"r1": 0, "r5": 0, "total": 0} for s in scorers
    }

    for case in cases:
        owner_id = f"eval-{case['id']}"
        namespace = (owner_id, "semantic")
        store = OpenCouchMemoryStore()
        await _seed_store(
            store,
            owner_id=owner_id,
            records=case["seed_records"],
            embedding_provider=provider,
        )

        # Compute the query embedding once per case.
        query_embedding: list[float] | None = None
        embedding_model: str | None = None
        if run_embedding_scorers and provider is not None:
            embeddings = await provider.aembed(
                [case["query"]],
                task_type="RETRIEVAL_QUERY",
            )
            query_embedding = embeddings[0] if embeddings else None
            if query_embedding is not None:
                embedding_model = provider.model_name

        # Run each scorer
        scorer_results: dict[ScorerName, tuple[bool, bool, list[str]]] = {}

        token_keys = await _score_token_recall(
            store,
            namespace=namespace,
            query=case["query"],
            limit=5,
        )
        r1, r5 = _grade_ranking(token_keys, expected_keys=case["expected_keys"])
        scorer_results["token_recall"] = (r1, r5, token_keys)

        if run_embedding_scorers:
            pure_keys = await _score_pure_embedding(
                store,
                namespace=namespace,
                query_embedding=query_embedding,
                embedding_model=embedding_model,
                limit=5,
            )
            r1, r5 = _grade_ranking(pure_keys, expected_keys=case["expected_keys"])
            scorer_results["pure_embedding"] = (r1, r5, pure_keys)

            hybrid_keys = await _score_hybrid_rrf(
                store,
                namespace=namespace,
                query=case["query"],
                query_embedding=query_embedding,
                embedding_model=embedding_model,
                limit=5,
            )
            r1, r5 = _grade_ranking(hybrid_keys, expected_keys=case["expected_keys"])
            scorer_results["hybrid_rrf"] = (r1, r5, hybrid_keys)

        # Accumulate per-category counts
        category = case["category"]
        if category not in grid:
            grid[category] = {s: {"r1": 0, "r5": 0, "total": 0} for s in scorers}
        for scorer, (hit1, hit5, _keys) in scorer_results.items():
            grid[category][scorer]["total"] += 1
            total_by_scorer[scorer]["total"] += 1
            if hit1:
                grid[category][scorer]["r1"] += 1
                total_by_scorer[scorer]["r1"] += 1
            if hit5:
                grid[category][scorer]["r5"] += 1
                total_by_scorer[scorer]["r5"] += 1

        per_case_detail.append((case["id"], category, scorer_results))

        await store.aclose()

    # ── Reporting ─────────────────────────────────────────────────────
    _print_matrix(grid, total_by_scorer, scorers)
    if verbose:
        _print_case_details(per_case_detail, scorers)
    _print_disagreements(per_case_detail, scorers)

    # Return nonzero exit only if token_recall regressed on something
    # the dataset previously expected it to pass. The eval is meant
    # to be a comparison tool, not a binary gate — so don't fail on
    # "hybrid beat token-recall" (that's the point). We exit nonzero
    # only if the baseline scorer (token_recall) got worse: total
    # recall@5 for token_recall must be >= the prior baseline.
    # For v0.8.1 ship the baseline is measured at eval run time and
    # a drop from "all three scorers agree" would be suspicious.
    # Conservative gate: only fail if token_recall has zero passes
    # across the whole dataset (catastrophic regression).
    return 0 if total_by_scorer["token_recall"]["r5"] > 0 else 1


def _print_matrix(
    grid: dict[str, dict[ScorerName, dict[str, int]]],
    total_by_scorer: dict[ScorerName, dict[str, int]],
    scorers: list[ScorerName],
) -> None:
    """Print the category × scorer recall matrix (recall@1 / recall@5)."""

    def cell(counts: dict[str, int]) -> str:
        return f"{counts['r1']}/{counts['total']} / {counts['r5']}/{counts['total']}"

    # Column widths
    cat_width = max(len(c) for c in grid) if grid else 10
    cat_width = max(cat_width, len("Category"))
    col_width = 18

    header = f"  {'Category'.ljust(cat_width)}  "
    for scorer in scorers:
        header += scorer.ljust(col_width)
    print(header)
    print(f"  {'─' * cat_width}  " + "─" * (col_width * len(scorers)))

    for category in sorted(grid.keys()):
        row = f"  {category.ljust(cat_width)}  "
        for scorer in scorers:
            row += cell(grid[category][scorer]).ljust(col_width)
        print(row)

    print(f"  {'─' * cat_width}  " + "─" * (col_width * len(scorers)))
    totals_row = f"  {'Total'.ljust(cat_width)}  "
    for scorer in scorers:
        totals_row += cell(total_by_scorer[scorer]).ljust(col_width)
    print(totals_row)
    print()
    print("  Legend: recall@1 / recall@5 — out of total cases in that category/scorer.")
    print()


def _print_case_details(
    per_case_detail: list[
        tuple[str, str, dict[ScorerName, tuple[bool, bool, list[str]]]]
    ],
    scorers: list[ScorerName],
) -> None:
    """Print per-case outcome rows when --verbose is set."""

    print("Per-case outcomes:")
    for case_id, category, scorer_results in per_case_detail:
        print(f"  [{category}] {case_id}")
        for scorer in scorers:
            hit1, hit5, keys = scorer_results[scorer]
            marker1 = "✓" if hit1 else "✗"
            marker5 = "✓" if hit5 else "✗"
            print(f"    {scorer:16s} r@1={marker1} r@5={marker5}  returned={keys}")
    print()


def _print_disagreements(
    per_case_detail: list[
        tuple[str, str, dict[ScorerName, tuple[bool, bool, list[str]]]]
    ],
    scorers: list[ScorerName],
) -> None:
    """Print cases where scorers disagreed on recall@5.

    This is the most useful diagnostic: which cases does each scorer
    win or lose, and where does hybrid RRF pull ahead of (or fall
    behind) pure embedding? Rendered as a table with ✓/✗ per scorer
    so you can eyeball which cases hybrid flipped.
    """

    if len(scorers) < 2:
        return

    disagreements: list[
        tuple[str, str, dict[ScorerName, tuple[bool, bool, list[str]]]]
    ] = []
    for case_id, category, scorer_results in per_case_detail:
        r5_values = {s: scorer_results[s][1] for s in scorers}
        if len(set(r5_values.values())) > 1:
            disagreements.append((case_id, category, scorer_results))

    if not disagreements:
        print("No disagreements on recall@5 — all scorers agree on every case.")
        print()
        return

    print(f"Disagreements on recall@5 ({len(disagreements)} case(s)):")
    print()
    header = "  " + "case_id".ljust(42) + "category".ljust(18)
    for scorer in scorers:
        header += scorer.ljust(16)
    print(header)
    print("  " + "─" * (42 + 18 + 16 * len(scorers)))
    for case_id, category, scorer_results in disagreements:
        row = "  " + case_id.ljust(42) + category.ljust(18)
        for scorer in scorers:
            _, hit5, _ = scorer_results[scorer]
            row += "✓ HIT".ljust(16) if hit5 else "✗ MISS".ljust(16)
        print(row)
    print()


def main() -> int:
    """Entry point for the retrieval eval runner."""

    args = _build_parser().parse_args()
    return asyncio.run(_run(args.mode, args.dataset, args.verbose))


if __name__ == "__main__":
    raise SystemExit(main())
