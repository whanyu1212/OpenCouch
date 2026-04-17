import React, { useState, useCallback } from 'react';
import styles from './AgentGraph.module.css';

/* ── Data definitions ──────────────────────────────────────────────────────── */

interface Detail {
  what: string;
  how: string;
  emits?: string;
}

interface StepDef {
  id: string;
  label: string;
  sub: string;
  badges?: { label: string; llm?: boolean; crisis?: boolean; retry?: boolean; reducer?: boolean; parallel?: boolean }[];
  detail: Detail;
  branch?: { condition: string; targetA: string; targetB: string; crisis?: boolean };
}

interface ModeDef {
  id: string;
  label: string;
  detail: Detail;
}

/* ── Pipeline steps ───────────────────────────────────────────────────────── */

const STEPS: StepDef[] = [
  {
    id: 'input',
    label: 'User message',
    sub: 'build_initial_state emits only the current user turn',
    detail: {
      what: 'Entry point. Accepts a message, channel, and session IDs. Emits only the current user turn — the checkpointer + operator.add reducer accumulates prior turns automatically.',
      how: 'build_initial_state creates the AgentState dict. Persistent sessions pass prior_turn_count from the checkpoint instead of deserializing the full transcript. One-shot callers can opt into include_input_history=True for testing.',
      emits: 'AgentState with history=[user_turn], transcript=[user_turn]',
    },
  },
  {
    id: 'crisis_gate',
    label: 'crisis_gate_node',
    sub: 'Safety first — every message, no exceptions',
    badges: [
      { label: 'deterministic' },
      { label: 'optional LLM', llm: true },
      { label: 'retry', retry: true },
    ],
    branch: {
      condition: 'needs_crisis_response',
      targetA: 'crisis_response → crisis_log → finalize',
      targetB: 'load_memory → therapeutic_subgraph → finalize',
      crisis: true,
    },
    detail: {
      what: 'Hard safety boundary. Runs BEFORE memory retrieval — there is no path that loads context without first passing the safety check. Returns a Command(goto=...) that routes the turn.',
      how: 'Layer 1: deterministic override (imminent risk, idiomatic safe). Layer 2: regex ladder for clear patterns. Layer 3: optional LLM classifier for ambiguous cases. Layer 4: policy normalization. Output: CrisisAssessment with level 0–3 and routing decision.',
      emits: 'state.crisis + state.routing.route ("crisis" | "therapeutic")',
    },
  },
  {
    id: 'load_memory',
    label: 'load_memory_node',
    sub: 'Therapeutic branch only — hybrid retrieval across 3 namespaces',
    badges: [
      { label: 'hybrid RRF' },
      { label: 'retry', retry: true },
    ],
    detail: {
      what: 'Retrieves semantic facts, episodic session arcs, and procedural style rules. Returns structured WorkingMemoryEntry dicts — formatting happens on demand at prompt-build time.',
      how: 'Semantic + episodic retrieval run in parallel via asyncio.gather. Each uses the store\'s hybrid path: token-recall + optional cosine-similarity fused via Reciprocal Rank Fusion. Procedural profile loaded separately. First-turn episodic catch-up injects the most recent session arc.',
      emits: 'state.working_memory (structured entries) + state.memory.procedural_rules',
    },
  },
  {
    id: 'therapeutic',
    label: 'therapeutic_subgraph',
    sub: 'Dispatcher picks one of 6 modes, mode node generates response',
    badges: [
      { label: 'subgraph' },
      { label: 'all nodes retry', retry: true },
    ],
    detail: {
      what: 'Compiled StateGraph registered as a single parent node. Contains a dispatcher + 6 mode nodes (supportive, reflective, clarifying, psychoeducation, guided_exercise, closing). Uses a narrow output schema (TherapeuticSubgraphOutput) so only routing, response, and progress flow back to the parent — preventing reducer double-counting on history/transcript.',
      how: 'Dispatcher uses hybrid classification: regex fast paths for obvious cases, LLM classifier for the ambiguous middle, regex fallback when no LLM available. Active-exercise fast path bypasses classification when an exercise is in progress.',
      emits: 'state.routing.mode + state.response.text + state.progress (via merge reducer)',
    },
  },
  {
    id: 'finalize',
    label: 'finalize_turn_node',
    sub: 'Append assistant reply — single-element delta via operator.add reducer',
    badges: [
      { label: 'pure state' },
      { label: 'no retry' },
    ],
    detail: {
      what: 'Appends the assistant response to transcript and history as a 1-element list. The operator.add reducer handles merging with the accumulated state from the checkpoint. Empty/whitespace responses produce an empty delta to keep the transcript clean.',
      how: 'Reads state.response.text, stamps the routing mode onto the assistant turn dict. Returns {transcript: [turn], history: [turn]}. No I/O — pure state manipulation, so no RetryPolicy.',
      emits: 'state.transcript += [assistant_turn], state.history += [assistant_turn]',
    },
  },
  {
    id: 'extractors',
    label: 'extract_facts + extract_procedural',
    sub: 'Parallel fan-out — both run simultaneously after finalize',
    badges: [
      { label: 'parallel', parallel: true },
      { label: 'LLM structured output', llm: true },
      { label: 'retry', retry: true },
      { label: 'diagnostics reducer', reducer: true },
    ],
    detail: {
      what: 'Two independent LLM extraction nodes that fan out in parallel from finalize_turn_node. Semantic extractor writes persistent facts; procedural extractor writes style rules. Both are gated by a small-talk check. Both write to the diagnostics dict via _merge_dicts reducer — no manual spreading needed.',
      how: 'Gating: crisis path → skip, no LLM → skip, incognito → skip, small talk → skip. Otherwise: LLM structured-output call, batch-embed candidates, write to store. Diagnostics record timing + write counts + skip reason. Parallel execution saves 3–5s per turn when LLM extraction is active.',
      emits: 'state.diagnostics (extract_facts_ms, extract_procedural_ms, etc.)',
    },
  },
  {
    id: 'output',
    label: 'AgentOutput',
    sub: 'Normalized public response returned to the API layer',
    detail: {
      what: 'state_to_output extracts the public response shape from the final state. The checkpoint stores the full accumulated state for the next turn — including the reducer-merged transcript, diagnostics, and progress.',
      how: 'Extracts response_text, crisis assessment, mode, mode_type, mode_source, diagnostics (including turn_total_ms stamped by the runtime).',
      emits: 'AgentOutput',
    },
  },
];

/* ── Therapeutic modes ────────────────────────────────────────────────────── */

const THERAPEUTIC_MODES: ModeDef[] = [
  {
    id: 'supportive', label: 'supportive',
    detail: {
      what: 'Default — user sharing feelings, seeking support, or greeting. Three sub-strategies: hold_space (venting), strengths_based (progress), supportive_guidance (validate + next step).',
      how: 'Reachable from all routing layers. Most common mode.',
      emits: 'response.kind = THERAPEUTIC',
    },
  },
  {
    id: 'reflective', label: 'reflective',
    detail: {
      what: 'User describing a recurring pattern they\'ve already named. Reflects on themes, connections, cycles.',
      how: 'Regex fast path fires on "always", "every time", "pattern" keywords.',
      emits: 'response.kind = THERAPEUTIC',
    },
  },
  {
    id: 'clarifying', label: 'clarifying',
    detail: {
      what: 'Ambiguous or very short message — agent needs context before responding.',
      how: 'Regex fast path fires on messages < 3 words or confusion markers ("huh?", "ok").',
      emits: 'response.kind = THERAPEUTIC',
    },
  },
  {
    id: 'psychoeducation', label: 'psychoeducation',
    detail: {
      what: 'User describes a reaction AND seeks understanding. Brief normalizing explanation, permission before explaining, pivot back to experience.',
      how: 'LLM classifier picks this for "why do I feel..." patterns.',
      emits: 'response.kind = THERAPEUTIC',
    },
  },
  {
    id: 'guided_exercise', label: 'guided_exercise',
    detail: {
      what: 'Multi-turn structured technique. Exercise state (type + step) persists across turns via the progress merge reducer.',
      how: 'Active-exercise fast path bypasses classification when progress.exercise_type is set. 12 exercises across grounding, behavioral activation, thought work, and acceptance.',
      emits: 'response.kind = THERAPEUTIC + progress.exercise_type/step',
    },
  },
  {
    id: 'closing', label: 'closing',
    detail: {
      what: 'User signals wind-down ("I should go", "thanks, this helped"). Graceful session close.',
      how: 'Regex fast path fires on closing language.',
      emits: 'response.kind = THERAPEUTIC',
    },
  },
];

/* ── Detail panel ──────────────────────────────────────────────────────────── */

function DetailPanel({ detail }: { detail: Detail }) {
  return (
    <div className={styles.detail}>
      <div className={styles.detailRow}>
        <span className={styles.detailKey}>What</span>
        <span className={styles.detailVal}>{detail.what}</span>
      </div>
      <div className={styles.detailRow}>
        <span className={styles.detailKey}>How</span>
        <span className={styles.detailVal}>{detail.how}</span>
      </div>
      {detail.emits && (
        <div className={styles.detailRow}>
          <span className={styles.detailKey}>Emits</span>
          <code className={styles.detailEmits}>{detail.emits}</code>
        </div>
      )}
    </div>
  );
}

/* ── Main component ────────────────────────────────────────────────────────── */

export default function AgentGraph() {
  const [active, setActive] = useState<string | null>(null);

  const toggle = useCallback((id: string) => {
    setActive(p => p === id ? null : id);
  }, []);

  const activeModeId = active?.startsWith('mode:') ? active.replace('mode:', '') : null;
  const isTherapeuticExpanded = active === 'therapeutic' || activeModeId !== null;

  return (
    <div className={styles.root}>
      <p className={styles.hint}>Click any step to expand. The pipeline shows the v0.9+ safety-first topology with parallel extractors.</p>

      <div className={styles.pipeline}>
        {STEPS.map((step) => {
          const isActive = active === step.id;
          const isTherapeuticStep = step.id === 'therapeutic';
          const stepIsExpanded = isActive || (isTherapeuticStep && isTherapeuticExpanded);

          return (
            <div
              key={step.id}
              className={[
                styles.step,
                stepIsExpanded ? styles.stepActive : '',
                step.id === 'crisis_gate' ? styles.stepCrisis : '',
              ].join(' ')}
            >
              <div className={styles.stepDot} />

              <button className={styles.stepCard} onClick={() => toggle(step.id)}>
                <div className={styles.stepHeader}>
                  <span className={styles.stepLabel}>{step.label}</span>
                  <span className={styles.stepSub}>{step.sub}</span>
                </div>

                {step.badges && (
                  <div className={styles.badges}>
                    {step.badges.map((b, i) => (
                      <span
                        key={i}
                        className={[
                          styles.badge,
                          b.llm ? styles.badgeLlm : '',
                          b.crisis ? styles.badgeCrisis : '',
                          b.retry ? styles.badgeRetry : '',
                          b.reducer ? styles.badgeReducer : '',
                          b.parallel ? styles.badgeParallel : '',
                        ].join(' ')}
                      >
                        {b.label}
                      </span>
                    ))}
                  </div>
                )}

                {/* Branch callout */}
                {step.branch && (
                  <div className={styles.branchContainer}>
                    <div className={styles.branchRow}>
                      <span className={[styles.badge, styles.badgeCrisis].join(' ')}>
                        if {step.branch.condition}
                      </span>
                      <span className={styles.branchArrow}>{'\u2192'}</span>
                      <span className={styles.stepSub}>{step.branch.targetA}</span>
                    </div>
                    <div className={styles.branchRow}>
                      <span className={styles.badge}>else</span>
                      <span className={styles.branchArrow}>{'\u2192'}</span>
                      <span className={styles.stepSub}>{step.branch.targetB}</span>
                    </div>
                  </div>
                )}
              </button>

              {/* Step detail panel */}
              {isActive && !isTherapeuticStep && <DetailPanel detail={step.detail} key={`d-${step.id}`} />}

              {/* Therapeutic subgraph expansion */}
              {isTherapeuticStep && isTherapeuticExpanded && (
                <>
                  {isActive && <DetailPanel detail={step.detail} key={`d-${step.id}`} />}

                  <div className={styles.modeSection}>
                    <div className={styles.modeSectionHeader}>
                      <span className={styles.modeGridLabel}>Therapeutic modes — dispatcher picks exactly one per turn</span>
                    </div>
                    <div className={styles.modeGrid}>
                      {THERAPEUTIC_MODES.map(m => {
                        const mActive = activeModeId === m.id;
                        return (
                          <button
                            key={m.id}
                            className={[
                              styles.modeChip,
                              styles.modeChipTherapeutic,
                              mActive ? styles.modeChipActive : '',
                            ].join(' ')}
                            onClick={(e) => { e.stopPropagation(); toggle(`mode:${m.id}`); }}
                          >
                            <span className={styles.modeChipLabel}>{m.label}</span>
                          </button>
                        );
                      })}
                    </div>
                    {activeModeId && THERAPEUTIC_MODES.find(m => m.id === activeModeId) && (
                      <DetailPanel
                        detail={THERAPEUTIC_MODES.find(m => m.id === activeModeId)!.detail}
                        key={`d-${activeModeId}`}
                      />
                    )}
                  </div>
                </>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
