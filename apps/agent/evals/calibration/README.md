# Reply-quality calibration (HY-242)

`labels.yaml` contains 40 synthetic inputs: 30 recorded dev replies and ten
deliberately weak controls. Every item has a blind first-pass label from Opus 5.5
in a coding session, written from the rubric without seeing any judge output.
**Human review is pending:** no item has `confirmed_by` set, so agreement has no
confirmed denominator yet.

The judge has been run once on this packet, after the first pass was committed
(2026-11-06, rubric v2): two repetitions per reply, 80/80 verdicts, no errors,
all served by `claude-opus-5-5` on the direct Anthropic route with no fallback.
Judge cost was $2.60. The run is saved locally in the gitignored
`evals/results/hy242-calibration/`; re-running it costs about the same.

## Selection and limitations

The source is `hy429-coach-dev-traced`. Selection is deterministic: sort by case,
repetition and turn, then round-robin across cases until 30 replies are selected.
Inputs retain preceding history, program records and server-selected resources.
The ten separate, synthetic controls target all seven hard failures plus generic,
preachy and empty replies. They are not recorded coach outputs. No holdout inputs
or judge predictions are used to select the packet.

These are 40 replies, not 40 independent scenarios: repetitions and turns share
context. This small, deliberately enriched set measures rubric agreement, not
production prevalence, clinical validity or broad safety performance. Review
positive and negative agreement counts; a high overall match rate can hide missed
rare failures.

To reproduce the unlabelled packet (refuses an existing output path):

```bash
uv run python -m evals.calibration.prepare \
  evals/results/hy429-coach-dev-traced /path/to/new-labels.yaml
```

## Review workflow

1. An Opus 5.5 coding session reads `../judge/rubric.md` and the inputs **without
   opening judge predictions**. Fill every `verdict` using `QualityVerdict`
   (`../judge/models.py`), with a short evidence reason for each binary label and
   each score. Set `labeller` to the actual model identity. Leave `confirmed_by`
   null. Freeze this first pass in version control before seeing predictions.
2. Wu Hanyu reviews and corrects **every criterion of every reply**. Only after
   reviewing an entire item, set its `confirmed_by` to `Wu Hanyu`. Start with hard
   failures and disagreements, but complete the whole set. The assistant must
   never fill confirmations on the reviewer's behalf without an explicit review
   decision. Preserve first-pass labels in version history when making corrections.
3. Run the paid judge explicitly after the blind first pass is complete:

   ```bash
   uv run --env-file ../../.env python -m evals.calibration \
     evals/calibration/labels.yaml --judge evals/results/hy242-calibration
   ```

   This judges each reply twice with Opus, allowing only the same-model gateway
   fallback. Labels, tags and reviewer identity are never sent to the judge.
4. Report offline; repeat this command as human confirmations are added:

   ```bash
   uv run python -m evals.calibration evals/calibration/labels.yaml \
     --report evals/results/hy242-calibration
   ```

## Agreement and completion

Only human-confirmed items enter agreement denominators. Hard failures use exact
match; each score uses agreement within ±1. Both judge repetitions count as
observations, not independent human labels. Counts and rates are reported per
criterion, alongside positive/negative counts, missing results, errors and named
disagreements. Empty denominators yield null, never 100%.

The report exits 0 only when all labels are confirmed, judging is complete without
errors, every hard-failure criterion has positive and negative human examples,
and for each hard-failure criterion both the overall agreement and the share of
human-labelled failures the judge also caught (`positive_rate`) are at least 90%.
Positives are rare (one or two controls per criterion), so overall agreement
alone could pass a judge that misses every one. Otherwise it exits 1.
Invalid or mismatched recordings are rejected, including changed candidate
inputs, changed rubrics and duplicate results. Score agreement is reported
separately; it does not compensate for a hard-failure disagreement.

Reports remain explicitly advisory: this command does not enable a production or
CI quality gate. Record the actual judge model/routes, review decisions and the
agreement report before closing HY-242. Below target, inspect disagreements and
iterate on the judge prompt; a rubric change requires fresh label review.
