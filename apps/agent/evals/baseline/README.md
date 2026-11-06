# Offline coach baseline (HY-241)

Run from `apps/agent`. These commands read saved synthetic recordings and make
no model calls. JSON snapshots omit transcripts, judge reasons and provider
error payloads. Original recordings remain the source evidence.

## Pre-merge comparison

After recording and judging a candidate on the same dev suite:

```bash
uv run python -m evals.baseline compare evals/results/CANDIDATE \
  --judge evals/results/CANDIDATE-JUDGE
```

The default baseline is `snapshots/coach-dev-2026-11-05.json`. The output names
new/resolved conversation checks and judge hard failures, then prints absolute
metrics and deltas. Existing failures remain visible in the baseline; exit 0
does not mean every scenario passes.

- **0:** compatible, no new deterministic failures, complete conversations/judging.
- **1:** new deterministic failure or incomplete conversation/judge coverage.
- **2:** invalid evidence or incompatible experiments.

Judge hard failures and scores are advisory, pending human calibration (HY-242).
Inspect them even when the command exits 0. Missing timings and costs are reported
explicitly; they do not change the exit code. A report without `--judge` measures
no judged quality and cannot compare against a judged baseline.

Compatibility requires the same suite digest, split, repetitions, evaluator,
rubric, judge schema/settings and observed judge models. Coach model and prompt
changes are allowed and shown. Judge inputs must match the supplied coach replies.
For historical YAML, pass `--suite PATH`; its digest must match the recording.
An evaluator change requires an explicitly reviewed baseline refresh, not a silent
comparison across different checks.

## Reproduce the snapshots

```bash
uv run python -m evals.baseline report evals/results/hy429-coach-dev-traced \
  --judge evals/results/hy429-replies-rubric-v2 \
  --output evals/baseline/snapshots/coach-dev-2026-11-05.json

uv run python -m evals.baseline report evals/results/hy241-coach-holdout-20261105 \
  --judge evals/results/hy241-judge-holdout-20261105 \
  --output evals/baseline/snapshots/coach-holdout-2026-11-05.json
```

Keep holdout confirmation separate. Select its snapshot explicitly with
`--baseline`; dev-versus-holdout comparisons are rejected. Snapshot generation
is deterministic given the artifacts, evaluator source, git history and prices.
Do not overwrite an accepted baseline just to remove a regression.

## First baseline

| Measurement | Dev | Holdout confirmation |
| --- | ---: | ---: |
| Scenarios × repetitions | 5 × 3 | 3 × 3 |
| Behaviour passes | 12/15 | 9/9 |
| Replies | 54 | 21 |
| Judgments | 108 | 42 |
| Judge hard-fail judgments | 0 | 0 |
| Recorded judge cost (USD) | 3.641700 | 1.384356 |

The three dev failures are the clearer-recovery scenario: the minute-12 denial
still screens at 0.51–0.61, delaying the offered check-in until minute 25.
Approved expectations and thresholds remain unchanged. See
[`RECOVERY.md`](../crisis/RECOVERY.md) for diagnosis and screening provenance.
These small coach suites do not replace the broader safety-detection evaluations.

## Interpretation and missing evidence

- **Behaviour:** passes divided by completed conversations. Infrastructure errors
  and missing conversations are separate, never model failures or silent passes.
  Empty/truncated replies fail deterministic checks and count in
  `quality.invalid_replies`, independently of the advisory judge verdict.
- **Uncertainty:** 2,000 deterministic whole-scenario bootstrap draws; repeated
  conversations, turns and judgments remain clustered. Dev's exploratory pass
  interval is [0.40, 1.00]. Null intervals mean fewer than two clusters or constant
  cluster means, not zero uncertainty. The planned 40-conversation noise estimate
  does not apply to five dev scenarios. One scenario is 20 percentage points of
  dev coverage (33.3 on holdout). Investigate any new deterministic failure;
  small score movements alone are not actionable evidence of improvement.
- **Latency:** `first_token` is time to first emitted text; `completion` is whole
  turn latency. Both include the measured wait, with p50/p90 and counts/shares
  above three seconds. `model_request` wraps the model request/stream and may
  include retries. Isolated final-successful-attempt timing was not recorded.
  Retry counts are SDK-reported; missing retry fields are counted separately.
- **Cost:** coach/Jev dollar costs were not saved. They remain null. Judge cost
  is the sum of recorded attempt costs, with missing-cost attempts counted.
  Coach token estimates can be supplied with `--prices rates.json`:

  ```json
  {
    "source": "verified provider pricing URL or version",
    "as_of": "YYYY-MM-DD",
    "models": {
      "exact-served-model": {"input": 2, "cached_input": 0.5, "output": 10}
    }
  }
  ```

  These example rates are illustrative, not current prices. Rates are USD per
  million tokens. Cached input is a subset of input; reasoning is already part
  of output. Estimates cover recorded successful calls only, not unrecorded
  failed attempts or Jev. Partial price coverage never becomes a full-run total.
  Different price snapshots suppress coach-cost deltas.
- **Provenance:** prompt hashes use source files at the recorded git revision.
  They do not capture runtime instructions, dirty-tree edits or environment
  overrides. Missing git history produces a null hash. Screening counts separate
  scripted, live and fake inputs, plus turns without traces.
