---
title: Routing & Classifiers
sidebar_position: 3
---

# Routing & Classifiers

OpenCouch uses LLM-primary routing for ambiguous language and keeps
deterministic rules for hard overrides, safety bounds, and no-LLM
fallbacks.

## Turn routing order

| Step | Primary mechanism | Deterministic role |
|---|---|---|
| Crisis gate | LLM crisis classifier | Imminent-risk overrides, idiomatic-safe overrides, and fallback ladder |
| Memory control gate | LLM classifier for memory commands | Slash commands, confirmation/cancel handling, and no-LLM fallback |
| Grounded lookup gate | LLM classifier for explicit factual lookup requests | Hard-yes patterns and no-LLM fallback |
| Therapeutic dispatcher | LLM classifier for response style + therapeutic approach | Clear exercise-exit overrides and no-LLM fallback |
| Guided exercise selection | LLM classifier for selected vs ambiguous exercise request | Pending option choices and no-LLM selector fallback |
| Memory write policy | LLM candidate extraction followed by policy classification | Deterministic commit, hold, repetition, and drop decisions |

## Why LLM-primary

Mental-health support requests are high-variety natural language.
Regex-first routing becomes brittle as coverage grows: each new
phrase fixes one case but can regress another. LLM-primary routing
lets the classifier reason over intent, while deterministic logic
still protects the places where ambiguity is unacceptable.

## Ambiguity behavior

Ambiguous user intent should not silently fall into a convenient
default. Current examples:

- Exercise selection offers two or three concrete options when the
  user asks for "an exercise" but does not specify the kind.
- Memory delete flows ask for confirmation before destructive action.
- Crisis level 1 asks one safety clarification instead of routing
  straight to a full crisis response.

## Regression coverage

The eval harness is being rebuilt. Until the replacement lands, routing changes
should update backend tests and targeted live-provider checks.
