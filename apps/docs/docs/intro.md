---
sidebar_position: 1
title: Overview
---

import SystemAtAGlance from '@site/src/components/SystemAtAGlance';
import MemoryLayers from '@site/src/components/MemoryLayers';

# OpenCouch

A mental health support agent that **remembers you across sessions**,
**checks for safety before every response**, and **adapts its style
to how you prefer to be supported**.

:::note Active development
OpenCouch moves quickly. Docs stay close to the code, but some pages
may lag briefly after larger refactors or dogfood changes.
:::

---

## The system at a glance

Every turn enters through the same safety gate. If the gate stays
silent, an LLM-primary triage picks one of four routes — and only
that route runs. After the reply is sealed, an off-turn extract step
decides whether anything from the exchange is worth remembering.

<SystemAtAGlance />

Triage decisions live in [`agent/specialists/triage.py`](https://github.com/whanyu1212/OpenCouch/blob/main/apps/backend/agent/specialists/triage.py).
[See the full runtime →](/docs/agent/graph)

---

## Text and voice surfaces

One product runtime, two live transports. Both share memory, crisis
resources, guided exercises, and session persistence — but voice
trades the per-turn pipeline for low-latency Realtime function tools.

| Surface | Runtime |
|---|---|
| **Text** | CLI, web chat, and chat APIs. One OpenAI Agents SDK turn per message, with safety-first routing, memory loading, streaming status events, and post-response extraction. |
| **Voice** | Browser speech-to-speech over OpenAI Realtime WebRTC. The backend creates the session, injects compact memory context, executes app-owned function tools, and finalizes persistent sessions through the shared runtime. |

---

## How memory works

Three memory layers, retrieved per turn and loaded into prompts on
demand. Most LLM agents implement only semantic memory; OpenCouch
implements all three, which is what makes cross-session
personalization possible.

<MemoryLayers />

Inspired by [CoALA](https://arxiv.org/abs/2309.02427) (Cognitive
Architectures for Language Agents, Princeton 2023).
[Memory architecture →](/docs/memory/overview)

---

## Three pillars

<div className="doc-card-grid">
  <div className="doc-card">
    <div className="doc-card__header">
      <svg className="doc-card__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
      <strong>Safety first</strong>
    </div>
    <p>LLM-only crisis gate runs before anything else, with local truth-table normalization for levels 0–3. Region-aware hotline lookup overlays verified resources onto crisis replies. Always-on audit log, even in incognito.</p>
  </div>

  <div className="doc-card">
    <div className="doc-card__header">
      <svg className="doc-card__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
      <strong>Adaptive response</strong>
    </div>
    <p>An LLM-primary triage picks a response style and a therapeutic approach per turn — CBT, ACT, DBT skills, MI, IPT, grief, PFA. Guided exercises pin their starting approach in <code>exercise_state</code> so side-turns don't drift the protocol.</p>
  </div>

  <div className="doc-card">
    <div className="doc-card__header">
      <svg className="doc-card__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 19.5A2.5 2.5 0 016.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z"/></svg>
      <strong>Persistent memory</strong>
    </div>
    <p>Semantic facts, episodic session arcs, and procedural style rules. Extracted with structured LLM output, classified by LLM-primary write policy with local safety guards, retrieved via hybrid search with Reciprocal Rank Fusion.</p>
  </div>
</div>

---

## Under the hood

<div className="doc-card-grid doc-card-grid--two">
  <div className="doc-card">
    <div className="doc-card__header">
      <svg className="doc-card__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
      <strong>Observability</strong>
    </div>
    <p>Per-turn stage timings, classifier paths, retrieval mode, write-policy decisions, and side-effect counters merge through one structured diagnostic channel. Opik captures the full trace.</p>
  </div>

  <div className="doc-card">
    <div className="doc-card__header">
      <svg className="doc-card__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
      <strong>Session continuity</strong>
    </div>
    <p>A 20-minute inactivity sweeper auto-finalizes idle sessions through the same end-session path as <code>/end</code>. Held memory candidates persist across restarts so delayed promotion never silently disappears.</p>
  </div>
</div>

For Postgres-first persistence, privacy controls, and latency tuning, see [Backend Architecture](/docs/backend/overview) and [Privacy Controls](/docs/memory/privacy).

---

## Important to know

OpenCouch is a **research and development prototype** — not a
substitute for professional mental health care. It is not a licensed
therapist, a diagnostic tool, or an emergency service.
