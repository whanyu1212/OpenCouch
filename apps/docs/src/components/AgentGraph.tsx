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

interface ResponseStyleDef {
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
      { label: 'LLM structured output', llm: true },
      { label: 'retry', retry: true },
    ],
    branch: {
      condition: 'needs_crisis_response',
      targetA: 'crisis_response → crisis_log → finalize',
      targetB: 'turn_dispatch → memory_control | grounded_answer | therapeutic',
      crisis: true,
    },
    detail: {
      what: 'Hard safety boundary. Runs BEFORE memory retrieval — there is no path that loads context without first passing the safety check. Returns a Command(goto=...) that routes the turn.',
      how: 'LLM-only classifier returns a structured CrisisAssessment with level 0–3. Local normalization enforces needs_crisis_response for levels 2–3 and needs_clarification for level 1; provider failures retry or surface instead of falling back to regex.',
      emits: 'state.crisis + state.routing.route ("crisis" | "therapeutic")',
    },
  },
  {
    id: 'turn_dispatch',
    label: 'turn_dispatch_node',
    sub: 'LLM route plan for safe turns',
    badges: [
      { label: 'LLM structured output', llm: true },
      { label: 'retry', retry: true },
    ],
    detail: {
      what: 'Safe-turn router. Routes explicit saved-memory commands to memory_control_node, factual lookup requests to grounded_answer_node, and ordinary support to load_memory_node.',
      how: 'LLM-primary structured decision with local validation for active-flow actions, memory-control payloads, grounded lookup queries, and explicit memory-reference mode.',
      emits: 'state.route + state.memory_control.action + state.grounded_lookup.query + Command(goto=...)',
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
      emits: 'state.working_memory (structured entries) + state.procedural_profile.procedural_rules',
    },
  },
  {
    id: 'therapeutic',
    label: 'therapeutic_subgraph',
    sub: 'Dispatcher routes to shared response or guided exercise node',
    badges: [
      { label: 'subgraph' },
      { label: 'all nodes retry', retry: true },
    ],
    detail: {
      what: 'Compiled StateGraph registered as a single parent node. Contains a dispatcher, one shared therapeutic response node, and one guided-exercise response node. Uses a narrow output schema so only response, approach, session action, diagnostics, and exercise state flow back to the parent.',
      how: 'Dispatcher is LLM-owned: the model picks response_style + therapeutic_approach, with local validation around active exercise state. Mid-exercise side-turns preserve the approach stored in exercise_therapeutic_approach.',
      emits: 'response_style + response_text + therapeutic_approach + exercise_state',
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
      how: 'Reads state.response_text, stamps routing metadata onto the assistant turn dict. Returns {transcript: [turn], history: [turn]}. No I/O — pure state manipulation, so no RetryPolicy.',
      emits: 'state.transcript += [assistant_turn], state.history += [assistant_turn]',
    },
  },
  {
    id: 'extractors',
    label: 'runtime extraction',
    sub: 'Off-graph side effect after response finalization',
    badges: [
      { label: 'parallel', parallel: true },
      { label: 'LLM structured output', llm: true },
      { label: 'retry', retry: true },
      { label: 'diagnostics reducer', reducer: true },
    ],
    detail: {
      what: 'After the graph reaches END, the runtime schedules semantic and procedural extraction. Each lane extracts candidates with structured LLM output, then runs LLM-primary write policy with hard local safety/storage guards.',
      how: 'Gating: crisis path -> skip, no LLM -> skip/error by path, incognito -> skip, small talk -> skip. Otherwise: structured-output extraction, policy decision, then immediate commit vs persisted active-session buffer vs drop. Session-end promotion later runs via runtime session finalization. Diagnostics record timing, write counts, policy drops, and policy errors.',
      emits: 'state.diagnostics (extract_facts_ms, extract_procedural_ms, etc.)',
    },
  },
  {
    id: 'output',
    label: 'AgentOutput',
    sub: 'Normalized public response returned to the API layer',
    detail: {
      what: 'state_to_output extracts the public response shape from the final state. The checkpoint stores the full accumulated state for the next turn — including the reducer-merged transcript, diagnostics, and progress.',
      how: 'Extracts response_text, crisis assessment, response_style, therapeutic_approach, session_action, and diagnostics. Public response_type is derived from crisis.level.',
      emits: 'AgentOutput',
    },
  },
];

/* ── Therapeutic response styles ──────────────────────────────────────────── */

const THERAPEUTIC_RESPONSE_STYLES: ResponseStyleDef[] = [
  {
    id: 'supportive', label: 'supportive',
    detail: {
      what: 'Default — user sharing feelings, seeking support, or greeting. Three sub-strategies: hold_space (venting), strengths_based (progress), supportive_guidance (validate + next step).',
      how: 'Reachable from all routing layers. Most common response style.',
      emits: 'AgentOutput.response_type = therapeutic',
    },
  },
  {
    id: 'reflective', label: 'reflective',
    detail: {
      what: 'User describing a recurring pattern they\'ve already named. Reflects on themes, connections, cycles.',
      how: 'Picked by the LLM dispatcher when the user is already naming a pattern.',
      emits: 'AgentOutput.response_type = therapeutic',
    },
  },
  {
    id: 'clarifying', label: 'clarifying',
    detail: {
      what: 'Ambiguous or very short message — agent needs context before responding.',
      how: 'Picked by the LLM dispatcher when the next useful move is one small clarifying question.',
      emits: 'AgentOutput.response_type = therapeutic',
    },
  },
  {
    id: 'psychoeducation', label: 'psychoeducation',
    detail: {
      what: 'User describes a reaction AND seeks understanding. Brief normalizing explanation, permission before explaining, pivot back to experience.',
      how: 'LLM classifier picks this for "why do I feel..." patterns.',
      emits: 'AgentOutput.response_type = therapeutic',
    },
  },
  {
    id: 'technique', label: 'technique',
    detail: {
      what: 'User wants structured therapeutic work without launching a named exercise — examining a thought, weighing evidence for a belief, working through a dilemma. The therapeutic_approach knowledge drives the response shape.',
      how: 'Picked by the LLM dispatcher when the user asks for structure but not a specific exercise. Requires an active therapeutic_approach.',
      emits: 'therapeutic_approach',
    },
  },
  {
    id: 'guided_exercise', label: 'guided_exercise',
    detail: {
      what: 'Multi-turn structured exercise. exercise_state (type + step + pinned approach stored in exercise_therapeutic_approach) persists across turns via the _merge_dicts reducer. Mid-exercise side-turns preserve the approach so it does not drift.',
      how: 'Active-exercise context is passed to the LLM dispatcher. 13 exercises across grounding, breathing, thought work, behavioral activation, acceptance, emotion regulation, and self-compassion.',
      emits: 'exercise_state.{exercise_type, exercise_step, exercise_therapeutic_approach}',
    },
  },
  {
    id: 'closing', label: 'closing',
    detail: {
      what: 'User signals wind-down ("I should go", "thanks, this helped"). Graceful session close. May set should_persist_memory=True.',
      how: 'LLM dispatcher distinguishes a real wind-down ("I have to head out") from mid-conversation thanks ("thanks, that helps").',
      emits: 'response_text',
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

  const activeResponseStyleId = active?.startsWith('style:') ? active.replace('style:', '') : null;
  const isTherapeuticExpanded = active === 'therapeutic' || activeResponseStyleId !== null;

  return (
    <div className={styles.root}>
      <p className={styles.hint}>Click any step to expand. The pipeline shows the safety-first topology with runtime-owned post-response memory evaluation.</p>

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

                  <div className={styles.responseStyleSection}>
                    <div className={styles.responseStyleSectionHeader}>
                      <span className={styles.responseStyleGridLabel}>Therapeutic responses — dispatcher picks exactly one per turn</span>
                    </div>
                    <div className={styles.responseStyleGrid}>
                      {THERAPEUTIC_RESPONSE_STYLES.map(m => {
                        const responseStyleActive = activeResponseStyleId === m.id;
                        return (
                          <button
                            key={m.id}
                            className={[
                              styles.responseStyleChip,
                              styles.responseStyleChipTherapeutic,
                              responseStyleActive ? styles.responseStyleChipActive : '',
                            ].join(' ')}
                            onClick={(e) => { e.stopPropagation(); toggle(`style:${m.id}`); }}
                          >
                            <span className={styles.responseStyleChipLabel}>{m.label}</span>
                          </button>
                        );
                      })}
                    </div>
                    {activeResponseStyleId && THERAPEUTIC_RESPONSE_STYLES.find(m => m.id === activeResponseStyleId) && (
                      <DetailPanel
                        detail={THERAPEUTIC_RESPONSE_STYLES.find(m => m.id === activeResponseStyleId)!.detail}
                        key={`d-${activeResponseStyleId}`}
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
