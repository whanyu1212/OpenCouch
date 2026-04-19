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

---

## How a turn flows

Every message passes through the same pipeline. Safety runs first,
then memory, then one of six therapeutic response modes, then
extraction for future sessions.

<FlowVisualizer />

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
    <p>Hybrid regex + LLM crisis gate on every message. Three deterministic fast paths before the LLM fallback. 42-case eval dataset. Always-on audit log even in incognito.</p>
  </div>

  <div className="doc-card">
    <div className="doc-card__header">
      <svg className="doc-card__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
      <strong>Adaptive response</strong>
    </div>
    <p>Six therapeutic modes — supportive, reflective, clarifying, psychoeducation, guided exercise, closing — selected by a hybrid dispatcher per turn.</p>
  </div>

  <div className="doc-card">
    <div className="doc-card__header">
      <svg className="doc-card__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 19.5A2.5 2.5 0 016.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z"/></svg>
      <strong>Persistent memory</strong>
    </div>
    <p>Semantic facts, episodic session arcs, and procedural style rules — all extracted automatically, stored in SQLite, retrieved via hybrid search with Reciprocal Rank Fusion.</p>
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
    <p>Four SQLite files under <code>.store/</code> — threads, memory, crisis log, session feedback. Each owns its schema independently.</p>
  </div>

  <div className="doc-card">
    <div className="doc-card__header">
      <svg className="doc-card__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0110 0v4"/></svg>
      <strong>Privacy controls</strong>
    </div>
    <p>Per-record deletion, namespace-wide wipe, retention purge, recall toggle. Users can inspect and delete anything the agent remembers.</p>
  </div>

  <div className="doc-card">
    <div className="doc-card__header">
      <svg className="doc-card__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
      <strong>CLI observability</strong>
    </div>
    <p>Per-turn stage timings, raw state dump, transcript mode annotations. Every node writes diagnostics via merge reducers.</p>
  </div>

  <div className="doc-card">
    <div className="doc-card__header">
      <svg className="doc-card__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
      <strong>Cost optimization</strong>
    </div>
    <p>Pre-extractor small-talk gate, parallel extractor fan-out, deterministic-first dispatch. Most turns never need an LLM classification call.</p>
  </div>

  <div className="doc-card">
    <div className="doc-card__header">
      <svg className="doc-card__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z"/><path d="M19 10v2a7 7 0 01-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
      <strong>Voice chat</strong>
    </div>
    <p>Experimental speech preview via OpenAI Realtime. Current path is low-latency and memory-backed, but speech-only and non-agentic.</p>
  </div>
</div>

---

## Important to know

OpenCouch is a **research and development prototype** — not a
substitute for professional mental health care. It is not a licensed
therapist, a diagnostic tool, or an emergency service.
