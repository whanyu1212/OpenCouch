---
title: Crisis Gate
---

# Crisis Gate

The crisis gate is the first non-negotiable safety boundary in OpenCouch.

Every inbound turn is assessed before normal support continues.

## Current flow

```text
deterministic override check
  -> obvious imminent risk or idiomatic safe case
  -> otherwise classifier path
  -> normalized crisis assessment
  -> route decision
```

### Deterministic fast paths

Three hard-coded overrides fire before the LLM classifier is ever invoked:

| Path | Condition | Result |
|---|---|---|
| **Imminent-risk override** | Matches imminent self-harm or suicide patterns | Level 3 — immediate crisis response |
| **Clear self-harm patterns** | Matches unambiguous self-harm language | Level 2 — safety check |
| **Idiomatic-safe override** | Common safe phrases that happen to contain trigger words (e.g. "dying to try it") | Level 0 — no risk |

### LLM fallback

When no deterministic path matches, the message is sent to a lightweight LLM classifier. The classifier returns a level with sharp boundaries — there is no fuzzy middle ground. Each level maps directly to a route decision.

## Why it exists

OpenCouch is not an emergency service, but users may disclose:
- suicidal ideation
- self-harm
- imminent intent
- ambiguous but concerning distress

If those are treated as ordinary support turns, the failure is serious.

## Response outcomes

- `crisis_response`
  - when clear risk requires immediate safety-oriented handling
- `safety_check`
  - when the message is concerning but still ambiguous
- `support`
  - when no crisis behavior is required

## Privacy asymmetry

The crisis log writes regardless of the user's memory mode. Even if the user has opted out of memory persistence or turned off proactive recall, crisis assessments are always recorded. This is an intentional asymmetry — safety telemetry is not subject to user memory preferences.

The `/memory purge-crisis [days]` command enforces a 90-day retention policy: crisis log entries older than the specified window are permanently deleted.

## Diagnostics

The crisis gate populates three keys in `state["diagnostics"]`:

| Key | Value |
|---|---|
| `crisis_gate_ms` | Wall-clock time for the full crisis assessment |
| `crisis_classifier_path` | Which path resolved the assessment: `deterministic` or `llm` |
| `crisis_level` | Final normalized level (0–3) |

## Important rule

The normal reply path should not proceed until the crisis gate finishes.

Safety sequencing matters more than shaving latency off the first response.
