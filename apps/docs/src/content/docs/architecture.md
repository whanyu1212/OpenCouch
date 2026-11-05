---
title: Architecture
description: Code owns the workflow; the coach chooses the words.
---

**Code owns the workflow; the model chooses the words.** FastAPI and Pydantic AI
serve AG-UI events to a Next.js client using `@ag-ui/client`.

## A chat turn

1. Authenticate the caller and check beta access. Validate the newest user
   message, claim thread ownership, and acquire a single-turn lease.
2. Load server-owned history and state. Validate any check-in answer against
   the server's outstanding offer.
3. Screen the new message with Jev and recheck pending missed turns. Update
   conversation risk and choose the reply path in code.
4. Persist the user message and risk **before generation**, so a failed reply
   cannot erase an escalation. Record relevant audit events.
5. At CRISIS, run the tool-less crisis responder. Otherwise load program
   position, parked worries, and formulation entries for the coach's instructions.
6. Return AG-UI state and reply events. Ordinary replies stream; crisis replies
   are checked before being sent whole. Save completed or interrupted output
   and release the lease.

The adapter ignores client-provided history/state as sources of authority.
A caller cannot lower risk or replace stored history by sending an AG-UI state
snapshot. A concurrent turn on the same thread receives HTTP 409; a thread
owned by someone else returns 404.

## Where state lives

| Area        | Responsibility                                                               |
| ----------- | ---------------------------------------------------------------------------- |
| `sessions/` | Thread ownership, history, state, risk, and turn leases                      |
| `program/`  | Per-user records spanning threads: position, worries, check-ins, formulation |
| `coach/`    | Coach agent and per-turn instructions built from verified records            |
| `safety/`   | Screening, risk policy, crisis response, resources, follow-up and audit      |
| `beta/`     | Invites and early-access requests                                            |
| `apps/web`  | Sign-in, conversation UI, thread history, resource cards and check-ins       |

Stores use Postgres when configured and in-memory implementations for local
development and evals. The record store does not imply that all record-editing
screens or program flows are built. The current coach has no tools; future tools
will request changes that code validates. Model calls are bounded by `UsageLimits`.

## Scope of this implementation

The reassurance classifier and program state machine in the architecture plan
are not yet in the turn pipeline. Current instructions provide interim skill
and reassurance guidance. Reviewed program content will be versioned data,
not generated lessons written at runtime.

Source: [turn.py](https://github.com/whanyu1212/OpenCouch/blob/develop/apps/agent/src/opencouch_agent/turn.py),
[coach](https://github.com/whanyu1212/OpenCouch/tree/develop/apps/agent/src/opencouch_agent/coach),
and [program records](https://github.com/whanyu1212/OpenCouch/tree/develop/apps/agent/src/opencouch_agent/program).
