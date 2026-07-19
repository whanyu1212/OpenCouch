---
title: Routing & Classifiers
sidebar_position: 4
---

# Routing & Classifiers

OpenCouch uses LLM-primary routing for ambiguous language and keeps
local deterministic code for schema validation, confirmation gates,
and state consistency. Natural-language fallback ladders are not part
of the intended design.

## Turn routing order

| Step | Primary mechanism | Deterministic role |
|---|---|---|
| Crisis gate | LLM crisis classifier | Truth-table normalization and route selection |
| Memory control gate | LLM classifier for memory commands | Slash commands plus explicit confirmation/cancel gates |
| Grounded lookup gate | LLM classifier for explicit factual lookup requests | Structured output validation and routing |
| Therapeutic dispatcher | LLM classifier for response style + therapeutic approach | Active-flow consistency and schema validation |
| Guided exercise selection | LLM classifier for selected vs ambiguous exercise request | Pending option state and catalog validation |
| Memory write policy | LLM candidate extraction followed by policy classification | Deterministic commit, hold, repetition, and drop decisions |

## Why LLM-primary

Mental-health support requests are high-variety natural language.
Regex-first routing becomes brittle as coverage grows: each new
phrase fixes one case but can regress another. LLM-primary routing
lets the classifier reason over intent, while local code protects the
places where ambiguity is unacceptable.

## Ambiguity behavior

Ambiguous user intent should not silently fall into a convenient
default. Current examples:

- Exercise selection offers two or three concrete options when the
  user asks for "an exercise" but does not specify the kind.
- Memory delete flows ask for confirmation before destructive action.
- Crisis level 1 asks one safety clarification instead of routing
  straight to a full crisis response.

## Worked example

The app-owned `TextTurnGraph` route planner resolves one route plan per turn in
a fixed precedence, short-circuiting at the first match:

1. **Eligibility** — empty/ineligible turns produce no plan.
2. **Crisis gate** — if the crisis classifier set a crisis runtime mode, route
   straight to the crisis plan. Nothing below runs.
3. **Grounded lookup** — if a prior step pre-routed the turn to
   `grounded_lookup`, emit that plan.
4. **Guided exercise** — load memory, check exercise lifecycle, and if the turn
   should run as an exercise, emit the `guided_exercise` plan.
5. **Memory control** — if the turn was routed to a memory command, emit
   `memory_control`.
6. **Therapeutic** — the default for everything else.

Consider *"My chest is tight and I keep thinking everyone would be better off
without me."* The crisis classifier fires at step 2, sets a crisis runtime
mode, and the turn resolves to a crisis plan before the therapeutic dispatcher
is ever consulted — the deterministic precedence guarantees safety routing wins
over response-style selection.

Now consider *"What's the national crisis text line number?"* during a calm
session. The crisis gate does not fire; the grounded-lookup classifier
recognizes an explicit factual request and the turn resolves to
`grounded_lookup`, which answers only from a verified tool result.

The plan that routing produces is what the [execution flows](/docs/agent/flows)
then run.

## Regression coverage

Routing changes should update backend tests and targeted live-provider checks.
