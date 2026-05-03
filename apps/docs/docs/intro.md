---
sidebar_position: 1
title: Overview
---

import FlowVisualizer from '@site/src/components/FlowVisualizer';
import MemoryLayers from '@site/src/components/MemoryLayers';

# OpenCouch

A mental health support agent that **remembers you across sessions**,
**checks for safety before every response**, and **adapts its style
to how you prefer to be supported**.

:::note Active development
OpenCouch moves quickly. These docs are kept close to the code, but
some pages may lag briefly after larger refactors or dogfood changes.
:::

---

## How a turn flows

Every message passes through the same pipeline. Safety runs first.
Then a sequence of operational gates short-circuits the turn for
explicit memory commands and factual lookup requests; everything
else loads memory and routes through the therapeutic subgraph.
After the reply is sealed, two extractor lanes fan out in parallel
to evaluate what (if anything) is worth remembering.

<FlowVisualizer />

The therapeutic path can end in one of four ways:

- **Memory command** (`"forget that"`, `"recall off"`,
  `"remember that I prefer…"`) → `memory_control_node` runs
  deterministically and replies with a confirmation. No memory
  retrieval, no LLM.
- **Factual lookup** (`"verify…"`, `"look up the latest…"`) →
  `grounded_answer_node` runs a search-grounded LLM call and
  replies with sources. No therapeutic framing.
- **Therapeutic turn** (the default) → memory loads, the dispatcher
  picks one of seven response styles plus a therapeutic approach, and
  the matching response node generates the reply.
- **Crisis turn** (when `crisis_gate_node` raises level ≥ 2) →
  region-aware hotline lookup, crisis reply, audit log. Memory is
  never loaded on this branch.

[See the full graph →](/docs/agent/graph)

---

## How memory works

Three memory layers give the agent context that survives across
sessions — retrieved per turn, loaded into prompts on demand.

<MemoryLayers />

Inspired by [CoALA](https://arxiv.org/abs/2309.02427) (Cognitive
Architectures for Language Agents, Princeton 2023). Most LLM agents
only implement semantic memory; OpenCouch implements all three,
which is what makes cross-session personalization possible.
[Learn more →](/docs/memory/overview)

---

## Three pillars

<div className="doc-card-grid">
  <div className="doc-card">
    <div className="doc-card__header">
      <svg className="doc-card__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
      <strong>Safety first</strong>
    </div>
    <p>LLM-primary crisis gate runs before anything else, with deterministic overrides for imminent risk and idiomatic-safe phrases. Region-aware hotline lookup overlays verified resources onto crisis replies. Always-on audit log even in incognito.</p>
  </div>

  <div className="doc-card">
    <div className="doc-card__header">
      <svg className="doc-card__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
      <strong>Adaptive response</strong>
    </div>
    <p>Seven response styles — supportive, reflective, clarifying, psychoeducation, technique, guided exercise, closing — paired with a therapeutic approach (CBT, ACT, DBT skills, MI, IPT, grief, PFA) by an LLM-primary dispatcher per turn. A guided exercise pins its starting approach in <code>exercise_state.exercise_therapeutic_approach</code> so guidance resumes without approach drift after side-turns.</p>
  </div>

  <div className="doc-card">
    <div className="doc-card__header">
      <svg className="doc-card__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 19.5A2.5 2.5 0 016.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z"/></svg>
      <strong>Persistent memory</strong>
    </div>
    <p>Semantic facts, episodic session arcs, and procedural style rules — extracted with structured LLM output, gated by deterministic write policy, retrieved via hybrid search with Reciprocal Rank Fusion.</p>
  </div>
</div>

---

## Under the hood

<div className="doc-card-grid">
  <div className="doc-card">
    <div className="doc-card__header">
      <svg className="doc-card__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z"/></svg>
      <strong>Durable persistence</strong>
    </div>
    <p>Postgres-first durable storage for threads, memory, crisis log, and session feedback, with legacy SQLite fallback for compatibility. Audit stores live in their own package (<code>agent/audit/</code>) so they're never touched by user-facing memory recall toggles.</p>
  </div>

  <div className="doc-card">
    <div className="doc-card__header">
      <svg className="doc-card__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0110 0v4"/></svg>
      <strong>Privacy controls</strong>
    </div>
    <p>Memory commands are first-class graph traffic — natural-language <code>list</code>, <code>forget</code>, <code>recall on/off</code>, and explicit-preference saves all run through <code>memory_control_node</code>. Destructive deletes carry a pending action across turns for confirm/cancel.</p>
  </div>

  <div className="doc-card">
    <div className="doc-card__header">
      <svg className="doc-card__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
      <strong>Observability</strong>
    </div>
    <p>Per-turn stage timings, classifier paths, retrieval-path mode, write-policy decisions. Every node writes diagnostics through a <code>_merge_dicts</code> reducer; Opik captures the full trace.</p>
  </div>

  <div className="doc-card">
    <div className="doc-card__header">
      <svg className="doc-card__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
      <strong>Cost &amp; latency levers</strong>
    </div>
    <p>Pre-extractor small-talk gate skips most acknowledgment turns. Two extractor lanes fan out in parallel after the reply, so memory writes don't block the next user turn. Stream emits <code>response_ready</code> as soon as <code>finalize_turn_node</code> seals the reply — clients can render before the post-response tail finishes.</p>
  </div>

  <div className="doc-card">
    <div className="doc-card__header">
      <svg className="doc-card__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z"/><path d="M19 10v2a7 7 0 01-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
      <strong>Voice chat (LiveKit)</strong>
    </div>
    <p>LiveKit-native worker under <code>voice/livekit/</code>. Browser joins a LiveKit room over WebRTC; the worker dispatches into the room, runs <code>TherapeuticAgent</code> with handoffs to <code>CrisisAgent</code> and bounded <code>GroundingTask</code>s. Three-phase memory (startup load → mid-session retrieval → shutdown transcript replay).</p>
  </div>

  <div className="doc-card">
    <div className="doc-card__header">
      <svg className="doc-card__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
      <strong>Session continuity</strong>
    </div>
    <p>20-minute inactivity sweeper auto-finalizes idle sessions through the same end-session seam as <code>/end</code>. Held memory candidates persist across restarts via an active-session SQLite table — delayed promotion never silently disappears.</p>
  </div>
</div>

---

## Important to know

OpenCouch is a **research and development prototype** — not a
substitute for professional mental health care. It is not a licensed
therapist, a diagnostic tool, or an emergency service.
