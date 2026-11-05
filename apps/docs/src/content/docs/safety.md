---
title: Safety
description: How screening, conversation risk and crisis check-ins control each turn.
---

## Screen before generation

Every accepted user turn goes through the safety gate before an agent runs.
Jev answers a fixed set of questions; code maps those readings into risk evidence.
The conversation level, rather than one isolated message, selects the reply path.

If screening fails, risk never drops. At ELEVATED or above the crisis responder
replies; below ELEVATED the coach is told screening did not run. The missed turn
is retained for rechecking, with bounded retry and retention limits. Unresolved
missed evidence cannot justify a safe step-down.

## Conversation risk

Levels are NONE, WATCH, ELEVATED, and CRISIS. Acute evidence can trigger CRISIS;
an accumulated score can raise WATCH or ELEVATED. That score is a policy measure,
not a probability, and decays per **turn**, not per elapsed minute.

A per-user floor carries risk into new threads for 72 hours without carrying
conversation text. Starting another thread is not a way to reset active risk.

## Crisis response and check-ins

The crisis responder has no tools. Code selects resources from the vetted
Singapore catalog, with emergency resources first when danger may be immediate.
The UI renders the resource card; generated text is checked against the supplied
contacts. Exact vetted numbers are allowed, but incorrect/unapproved numbers
and contact URLs are not.

CRISIS can step down to ELEVATED only through the server-controlled check-in:

- The evidence score is below the exit threshold, and the calm-time requirement
  has elapsed (currently ten minutes since the last qualifying risk signal).
- A calm turn makes the server offer a check-in.
- The next turn contains a valid “I'm safe right now” answer to that offer,
  and the risk checks still permit step-down.

Typed reassurance alone is not a check-in. New risk can return the conversation
to CRISIS. Once a conversation has reached CRISIS, it never falls below ELEVATED.
Resources remain pinned temporarily after step-down. These are engineering
policy rules, not a clinical recovery timetable.

## Evidence and limitations

Risk changes and screening failures produce audit records without message text.
Scheduled Jev drift checks monitor changes in classifier behaviour. Neither
mechanism proves that a classifier reading is correct.

The [latest live eval](/OpenCouch/evaluations/2026-11-05/) exposes a known failure:
explicit recovery language delays the offered check-in by one turn in all three
repetitions. The expectations and thresholds were preserved. Broader independent
review of step-down labels and human calibration of the reply judge remain open.

Source: [safety implementation](https://github.com/whanyu1212/OpenCouch/tree/develop/apps/agent/src/opencouch_agent/safety),
[recovery audit](https://github.com/whanyu1212/OpenCouch/blob/develop/apps/agent/evals/crisis/RECOVERY.md),
and [drift monitoring](https://github.com/whanyu1212/OpenCouch/blob/develop/apps/agent/evals/crisis/DRIFT.md).
