---
title: OpenCouch, rebuilt
description: A guided self-help program for anxiety, with code-owned safety and an AI coach.
---

OpenCouch is being rebuilt as a **guided self-help program for anxiety**.
The aim is to help people practise skills rather than repeatedly seek reassurance.
It is not therapy, a diagnostic service, or an open-ended companion.

These pages describe the rewrite on `develop`, reviewed November 6, 2026.
The clinical content and program sequence are still open. Engineering evals
measure system behaviour; they do not establish improved anxiety outcomes.

## Start here

- [Quickstart](/OpenCouch/quickstart/): run the service and documentation locally.
- [Architecture](/OpenCouch/architecture/): follow a turn from request to saved reply.
- [Safety](/OpenCouch/safety/): screening, crisis routing, and recovery checks.
- [Evaluations](/OpenCouch/evaluations/methodology/): what we measure and what the numbers mean.
- [Latest results](/OpenCouch/evaluations/2026-11-05/): complete coverage of the current eight-scenario coach suite, including failures.

## Implemented today

The Python service and Next.js client provide authenticated, user-owned threads
and invite-only beta access. The server owns conversation history and state.
Safety screening runs before every agent reply, with conversation-level risk,
a cross-thread risk floor, a tool-less crisis responder, vetted Singapore
resources, check-in buttons, an audit log, and scheduled Jev drift monitoring.

Per-user stores hold program position, worries, check-ins, and formulation
entries. The coach receives instructions built from server-owned records and
risk state. A multi-turn runner and reply-quality judge evaluate saved conversations.

## Still being built

The program state machine and reviewed lesson content, a dedicated reassurance
classifier, questionnaires, reminders, and complete practice flows are future
work. Prompt guidance to redirect reassurance is already present; it is not the
planned classifier. The current coach has no registered tools.

Voice and inferred long-term memory belong to the previous companion direction
and are not features of this rewrite.

The [product vision](https://github.com/whanyu1212/OpenCouch/blob/develop/docs/vision.md)
explains the hypothesis and open clinical questions. The
[architecture plan (#356)](https://github.com/whanyu1212/OpenCouch/issues/356)
includes future work as well as implemented components.
