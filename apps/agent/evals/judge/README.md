# Reply-quality judge (HY-240)

The [rubric](rubric.md) is a product-owner review draft. Scores are advisory until
HY-242 checks agreement with human-confirmed labels; this is not clinical
validation. Seven binary hard failures are combined with fail-on-any. Five
anchored 1–5 ratings remain separate. One request returns all twelve decisions.

Run these commands from `apps/agent`. They are paid and opt-in. CI and the
runner's ordinary `--rescore` remain offline. Keys belong in the ignored
repo-root `.env`; never put them in committed files or shell arguments.

```bash
# Known negatives and positive control, twice each to record judge variance:
uv run --env-file ../../.env python -m evals.judge --sanity --reps 2

# Judge every reply in a previously saved coach run; do not regenerate replies:
uv run --env-file ../../.env python -m evals.judge --run evals/results/<coach-run>

# Cheap pilot: first two replies only; their preceding history is preserved:
uv run --env-file ../../.env python -m evals.judge \
  --run evals/results/<coach-run> --limit 2 --reps 2

# Compare two runs of the same suite/split in both blind candidate orders:
uv run --env-file ../../.env python -m evals.judge \
  --run evals/results/<first> --compare evals/results/<second>

# Explicit Gateway route, for testing or an Anthropic-credit outage:
uv run --env-file ../../.env python -m evals.judge --sanity --route gateway
```

The default route chain is direct Anthropic `claude-opus-5-5`
(`ANTHROPIC_API_KEY`), Vercel `anthropic/claude-opus-5.5`
(`AI_GATEWAY_API_KEY`, funded Gateway credits and Gateway-managed credentials),
then direct OpenAI Responses (`OPENAI_API_KEY`, `gpt-6.1-sol` by default;
override with `--openai-model`). `--route openai` selects the final route
explicitly. Changing to OpenAI changes the judge and loses the cross-vendor
separation from the coach; reports separate metrics by served model and route.

Fallback occurs only after a documented billing/credit-exhaustion error on the
preceding route. A missing key, authentication error, malformed request, ordinary
rate limit, timeout, refusal, truncation, or invalid verdict stops that request.
Missing optional fallback keys do not prevent the primary from working. A
missing Gateway key does not skip straight to OpenAI. Each route makes at most
one model request: no SDK retries or output-repair calls. All routes use high
effort, no sampling settings, a 4,096-output-token cap and a 120-second timeout.
Gateway retains the exact Opus profile and native structured output.

Each run creates a new directory under `evals/results/judge-<timestamp>/` (or
`--output-dir`, which must not exist):

- `results.jsonl`: source identity, full contextual judge input, verdict,
  judge repetition, and every route attempt.
- `pairs.jsonl`: both blind orderings, each with both contexts and the preferred
  source run. Neither prompt identifies a baseline.
- `errors.jsonl`: unscorable requests and their safe failure class/attempts;
  they never contribute quality scores. Completed rows survive interruption.
- `meta.json`, `rubric.md`: exact prompt/schema/input digests, model settings
  and the rubric used for the run.
- `summary.json`: hard-fail counts, each mean score, repeated-reply population
  variance, hard-fail disagreement, cost completeness, and per-judge grouping.
- `report.txt`: Pydantic Evals advisory metrics for full conversations when
  judging a saved run. Partially judged conversations report an evaluator
  failure rather than an invented score. The original recording is untouched.

`SavedQualityEvaluator` can project existing verdicts into `evaluate_saved_run`
without any LLM calls. It verifies the full current reply context, including
history, path, resource names, quality focus and seeded program records. Older
verdicts without record context cannot score cases with seeded records.
For repeated judgments, pass `judge_reps` from the judge run's `meta.json` to
`SavedQualityEvaluator` (default: 1). Every turn must have exactly that many
distinct repetitions; missing or duplicate judgments produce evaluator failures.
Changing the rubric and asking for fresh LLM verdicts still costs money;
deterministic aggregation is free.

Costs use returned usage and the SDK price catalog when resolvable. Missing
prices are null and counted in `attempts_without_cost`, never claimed as free.
Thinking tokens are included in provider output billing; do not add them again.
`--limit` limits source replies before judge reps; comparisons double requests
because both candidate orders are evaluated. `--sanity` expects all four controls,
so using `--limit` there can exit nonzero even if each returned verdict is valid.

## Initial live pilot, 2026-11-05

Direct Opus 5.5, high effort, no sampling, eight judgments: empty reply,
"I don't know", unrelated confident reply, and a positive control, each twice.
All three negatives scored 1 on attunement and conversational fit in both
repetitions. The positive control scored 4 on conversational fit twice.
No judge errors or hard failures; these negatives test poor quality, not harm.
Measured judge cost: **$0.21116** for 26,100 input and 5,338 output tokens.
All repeated score variances were zero except positive-control attunement
(population variance 0.25). This tiny sanity set is not calibration.

A separate Gateway request returned the same Opus version and a valid verdict;
the SDK could not resolve its cost. Its usage is retained in the ignored live
recording, and the summary explicitly marks the cost as unknown.
