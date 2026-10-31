# Jev drift check (HY-345)

Run the human-labelled BHH validation split through the same Jev alias, questions
and transcript framing as the live gate. Compare it with the fixed 2026-10-28
recording behind `baselines/bhh-validation.json`. The reference in
`baselines/jev-drift.json` adds case IDs/levels, the case digest and scoring digest;
it contains no benchmark text. Monitoring never rewrites the reference or policy.

From `apps/agent`:

```sh
uv run python -m evals.crisis.bhh --split validation
uv run --env-file .env python -m evals.crisis.drift --report evals/results/jev-drift.json
# Re-score an existing complete run without a paid call:
uv run python -m evals.crisis.drift --run-dir evals/results/crisis/<run> --report evals/results/jev-drift.json
```

Exit codes: **0 pass**, **1 measured regression**, **2 measurement unavailable**.
Missing/duplicate/unexpected IDs, invalid answers, request errors, changed source
text/labels, a different dataset revision, questions or scoring logic all prevent a pass.
A setup or download failure also fails the workflow. API failures are never scored
as safe cases. Retry only failed cases with the existing `record --resume <run>`
command, then compare offline; do not discard hard cases to make a run pass.

## Alert rules

These operational tolerances were fixed before the 2026-10-31 recording. They
are absolute percentage-point changes, not confidence intervals or clinical
acceptance criteria. The small sample makes a single missed positive actionable.

| Metric | Reference | Fails when |
| --- | --- | --- |
| Suicide/self-harm recall at ELEVATED | 49/49 (100%) | Any decrease |
| Suicide/self-harm recall at WATCH | 49/49 (100%) | Any decrease |
| Violent-thoughts recall at WATCH | 5/5 (100%) | Any decrease |
| No-crisis false alarms at WATCH | 24/94 (25.53%) | Increase >5 points (at least 29/94) |
| No-crisis false alarms at ELEVATED | 7/94 (7.45%) | Increase >3 points (at least 10/94) |

The report includes both AUC values and every case whose policy level changed,
including changes that stay within a tolerance. Inspect those IDs locally in the
recording. BHH has type labels rather than severity labels; this does not measure
CRISIS step-down, abuse detection, medical detection or response quality.

## Scheduled operation

`.github/workflows/jev-drift.yml` runs Mondays at 03:17 UTC and supports manual
execution. It checks out `develop`, installs locked dependencies, downloads only
validation, and publishes a job summary plus a 30-day JSON artifact containing
metrics, provenance and case IDs. It never uploads raw BHH data or recordings.
A failed job is the alert; enable GitHub Actions failure notifications for the
account that enables the schedule, or watch the Actions page. No email/Slack
integration or automatic threshold tuning is added.

GitHub only schedules workflows from the default branch (`main` here). The
scheduler-only PR to `main` must be merged **after** the implementation PR to
`develop`. An Actions secret named `AI_GATEWAY_API_KEY` is required. A file only
on `develop` does not activate a schedule. Run `gh workflow run jev-drift.yml`
once after installation to verify credentials, downloading and report upload.

## Model provenance

Checked on 2026-10-31 using the [live model catalog](https://ai-gateway.vercel.sh/v1/models)
and [Vercel's TypeSafe API documentation](https://vercel.com/docs/ai-gateway/sdks-and-apis/typesafe).
The catalog lists `typesafe-ai/jev`, `typesafe-ai/jev-1.13.0` and
`typesafe-ai/jev-1.14.0`. Explicit version IDs can be requested, but this monitor
keeps the alias used by production so it measures the deployed dependency.
Changing production to a versioned ID requires a separate evaluation and review.

Each new drift recording records the requested ID, every served model ID observed
across the batch, SDK version, recording time, dataset revision, case digest, scoring digest and
Jev-input digest. Resumed recordings merge observed model IDs from each attempt
and reject a changed SDK version. A no-op resume leaves provenance unchanged.
The gateway currently reports `typesafe-ai/jev` for the alias;
that is not proof of an immutable underlying version. Older recordings lack
served-model metadata; their unknown version is not filled in retroactively.

## Responding to an alert

First distinguish an unavailable measurement from a regression. For a measured
regression, rerun once to check persistence, inspect changed cases, and re-score
candidate thresholds on the same recording. Protect recall first; validate any
candidate on a fresh held-out recording and conversation/step-down sets before
changing `policy.py`. If the change comes from Jev, raise the evidence with the
provider or evaluate a version pin. Neither response is automatic.

The scoring digest covers policy thresholds, signal decoding, label grouping and
metric computation (`policy.py`, `questions.py`, `detection.py`, `score.py`). Even
a non-behavioral edit to those files requires checking and regenerating the digest.
An intentional change to the input/scoring/dataset needs an explicit baseline
review. After that review, `snapshot(Path(<complete run>))` in `evals.crisis.drift`
can be serialized with `model_dump_json()` to replace the reference in a PR.
Document the reason and measurements; never refresh it just because a check failed.

## 2026-10-31 decision: retain the current thresholds

Accepted the current calibration for this iteration without retuning. The fresh
190-case validation recording preserved 49/49 suicide/self-harm detections at
ELEVATED and 5/5 violent-thought detections at WATCH. False alarms were 27/94 at
WATCH (28.72%, +3.19 points) and 6/94 at ELEVATED (6.38%, -1.06 points), compared
with the committed October 28 reference. AUC was 0.99262 versus 0.99229. Seven
case levels changed: six `no_crisis` cases and one suicidal-ideation case moving
from CRISIS to ELEVATED, still detected. The first attempt had four timeouts;
retrying only those cases produced a complete measurement with no remaining errors.
The initial 186 responses reported the alias. The four retries preceded the resume
provenance fix, so their served IDs were not captured; the evidence marks this
limitation rather than inferring their version.

The increased WATCH false-alarm rate remains a cost, not a claim of good
calibration. This run preserves the recall priority, has fewer ELEVATED false
alarms, and stays inside the bands declared before recording. It does not justify
loosening safety thresholds to optimize a small validation set. The earlier
October 19–23 shift is accepted provisionally under this monitoring policy,
not erased by a refreshed baseline. Keep the October 28 reference fixed and
investigate if subsequent runs cross a band. This is an engineering acceptance
of the current measured behavior, not clinical validation or a resolution of
HY-208's pending human label review.

The aggregate evidence is committed in `baselines/jev-drift-2026-10-31.json`.
