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
  badges?: { label: string; llm?: boolean; crisis?: boolean }[];
  detail: Detail;
  branch?: { condition: string; target: string; crisis?: boolean };
}

interface ModeDef {
  id: string;
  label: string;
  conditional?: boolean;
  conditionLabel?: string;
  detail: Detail;
}

/* ── Pipeline steps (the main spine — what every message goes through) ───── */

const STEPS: StepDef[] = [
  {
    id: 'input',
    label: 'User message',
    sub: 'prepare_turn_state re-derives context from the persisted transcript',
    detail: {
      what: 'Entry point. Accepts a message, channel, transcript, and session IDs. Re-derives all context from the persisted transcript each turn.',
      how: 'Trims history, extracts concerns/loops/goal, updates sticky session intent, builds rolling summary, resets turn-scoped fields.',
      emits: 'AgentState with route="therapeutic" mode="supportive_conversation"',
    },
  },
  {
    id: 'crisis_gate',
    label: 'Crisis gate',
    sub: 'Hard safety boundary — every message, no exceptions',
    badges: [
      { label: 'deterministic' },
      { label: 'optional LLM', llm: true },
    ],
    branch: {
      condition: 'crisis.level >= 2',
      target: 'Crisis response — acknowledge, surface resources, one safety question. Skips all normal routing.',
      crisis: true,
    },
    detail: {
      what: 'Assesses every inbound message for self-harm or suicide risk. Cannot be bypassed or weakened by mode, modality, or tone settings.',
      how: 'Layer 1: deterministic regex for obvious cases (imminent risk, idiomatic safe). Layer 2: optional LLM classifier for ambiguous cases. Layer 3: policy normalization. Output: CrisisAssessment with level 0–3 and two routing flags.',
      emits: 'state.crisis  state.route = "crisis" | "therapeutic"',
    },
  },
  {
    id: 'session_stage',
    label: 'Session stage',
    sub: 'opening → deepening → stabilizing → closing',
    detail: {
      what: 'Infers where the conversation is in its natural arc. Influences prompt tone and response shape without changing the mode.',
      how: 'Deterministic first: wrap-up language detection, turn count heuristics, mode history. Optional LLM refinement. Stage is sticky — does not flap without a clear cue.',
      emits: 'state.session_stage + session_stage_source + session_stage_reason',
    },
  },
  {
    id: 'mode_router',
    label: 'Mode router',
    sub: 'Three-layer selection — keyword patterns → session intent → LLM classifier',
    badges: [
      { label: 'keyword' },
      { label: 'intent' },
      { label: 'LLM fallback', llm: true },
    ],
    detail: {
      what: 'Selects one of 8 response modes. Safety-critical modes are exclusively deterministic — the LLM classifier can never override them.',
      how: 'Layer 1: keyword regex patterns match the message. Layer 2: sticky session intent steers ambiguous follow-ups. Layer 3: optional LLM classifier selects among 4 therapeutic modes when layers 1–2 miss. Layer 4: default to supportive_conversation.',
      emits: 'state.mode + state.mode_source (keyword | session_intent | llm | default)',
    },
  },
  {
    id: 'semantic_signals',
    label: 'Semantic signals',
    sub: 'Shared boolean flags derived from history + concerns + message',
    badges: [
      { label: 'deterministic' },
      { label: 'cached' },
    ],
    detail: {
      what: 'Computes 13 boolean semantic flags once per turn: theme detection (grief, anxiety, stress, relational), intent signals (wants_grounding, wants_cbt, wants_explanation, wants_pattern_reflection, wants_behavioral_activation), emotional state (is_venting, is_progress_update), and lifecycle (safety_sensitive, is_closing).',
      how: 'Combines recent history, active concerns, current goal, and the current message into a single lowercased text, then matches against curated term lists. Cached on state.semantic_signals for reuse by mode router, modality selector, and response shaping.',
      emits: 'state.semantic_signals',
    },
  },
  {
    id: 'modality_selection',
    label: 'Modality selection',
    sub: 'Picks therapeutic technique overlays based on semantic signals',
    badges: [
      { label: 'deterministic' },
    ],
    detail: {
      what: 'Selects 0–3 modality overlays (pfa, cbt, grief_support, act) based on the chosen mode and semantic signals. MI is applied as a baseline to certain modes automatically via MODE_BASELINE_FILES.',
      how: 'Per-mode rules in modality_selector.py read semantic signals and session stage to compose a priority-ordered modality list. Bounded to 3, deduplicated. Modes like realignment and safety_check get fixed modalities (empty or pfa only).',
      emits: 'state.active_modalities',
    },
  },
  {
    id: 'response_shaping',
    label: 'Response shaping',
    sub: 'Infers sub-strategy within the selected mode',
    badges: [
      { label: 'deterministic' },
    ],
    detail: {
      what: 'Deterministic inference of how the response should be shaped within the chosen mode. This is where support strategy, exercise focus, and psychoeducation topic are decided.',
      how: 'Support: hold_space / strengths_based / supportive_guidance. Exercise: grounding / behavioral_activation / thought_work / acceptance. Psychoeducation: anxiety_response / stress_response / grief_process / general. Pattern reflection: modality-aware guidance variants (grief-aware, strengths-aware, relational, ACT).',
      emits: 'state.response_guidance',
    },
  },
  {
    id: 'response',
    label: 'Response generation',
    sub: 'Registry-backed prompt assembly → LLM call (or deterministic fallback) → finalize',
    detail: {
      what: 'Uses the therapeutic_mode_registry to assemble the full prompt and generate the reply. Each mode has a registered config with prompt builders, system builders, temperature, and fallback templates.',
      how: 'The prompt is layered: soul.md and policy (always present), mode file + MI baseline if applicable (selected by router), modality overlays (from modality_selector), response guidance (from shaping), session context and history (from state). LLM generates text; per-mode deterministic fallback responses used if LLM unavailable.',
      emits: 'state.response_text + state.response_type + state.mode_type',
    },
  },
  {
    id: 'output',
    label: 'AgentOutput',
    sub: 'Normalized public response returned to the API layer',
    detail: {
      what: 'finalize_turn_state appends the user message and assistant response to the durable transcript, then state_to_output extracts the public response shape.',
      how: 'Extracts response_text, response_type, crisis assessment, mode, mode_type, mode_source, and should_persist_memory.',
      emits: 'AgentOutput',
    },
  },
];

/* ── Modes ─────────────────────────────────────────────────────────────────── */

const CONDITIONAL_MODES: ModeDef[] = [
  {
    id: 'safety_check', label: 'safety_check', conditional: true,
    conditionLabel: 'crisis gate sets needs_clarification',
    detail: {
      what: 'Asks one direct safety question before proceeding. Triggered when the message is concerning but ambiguous.',
      how: 'Keyword-only routing (deterministic). Uses PFA modality. Does not assume intent — clarifies first.',
      emits: 'mode_source = keyword',
    },
  },
  {
    id: 'orientation', label: 'orientation', conditional: true,
    conditionLabel: 'first message + intake language',
    detail: {
      what: 'Warm introduction to what OpenCouch can and cannot do. Only fires on the first turn with intake patterns.',
      how: 'Keyword-only routing (deterministic). Triggered by "how does this work", "I\'m new here", etc. Disabled if conversation has history.',
      emits: 'mode_source = keyword',
    },
  },
  {
    id: 'out_of_scope', label: 'out_of_scope', conditional: true,
    conditionLabel: 'diagnosis, medication, or legal request',
    detail: {
      what: 'For requests outside boundaries. States limits clearly without shame, redirects to appropriate help.',
      how: 'Keyword-only routing (deterministic). Matches "diagnose", "medication", "legal advice", etc.',
      emits: 'mode_source = keyword',
    },
  },
  {
    id: 'realignment', label: 'realignment', conditional: true,
    conditionLabel: 'user signals a mismatch',
    detail: {
      what: 'When the user says the previous reply missed — "that\'s not helpful", "you missed the point", "wrong direction".',
      how: 'Keyword-only routing (deterministic). Acknowledges the miss, re-attunes without defensiveness.',
      emits: 'mode_source = keyword',
    },
  },
];

const THERAPEUTIC_MODES: ModeDef[] = [
  {
    id: 'support', label: 'supportive_conversation',
    detail: {
      what: 'Default support with three sub-strategies: hold_space (venting, no advice), strengths_based (progress/wins), supportive_guidance (validate + one next step).',
      how: 'Reachable from all three routing layers. Most common mode. Sub-strategy inferred by response_shaping.',
      emits: 'response_type = THERAPEUTIC',
    },
  },
  {
    id: 'guided_exercise', label: 'guided_exercise',
    detail: {
      what: 'Structured self-help with four subtypes: grounding (sensory/breathing), behavioral_activation (tiny action), thought_work (CBT-style), acceptance (ACT defusion).',
      how: 'Reachable from all three routing layers. Subtype inferred from keywords, modality, and context.',
      emits: 'response_type = THERAPEUTIC',
    },
  },
  {
    id: 'reflection', label: 'pattern_reflection',
    detail: {
      what: 'Reflects on recurring themes, connections, and cycles. Names patterns tentatively, invites testing.',
      how: 'Reachable from all three routing layers. Guidance varies: grief-aware, strengths-aware, relational, ACT.',
      emits: 'response_type = THERAPEUTIC',
    },
  },
  {
    id: 'psychoeducation', label: 'psychoeducation',
    detail: {
      what: 'Brief normalizing explanations of anxiety, stress, grief, or emotional responses. Permission before explaining, pivot back to experience.',
      how: 'Reachable from all three routing layers. Topic inferred: anxiety_response, stress_response, grief_process, general.',
      emits: 'response_type = THERAPEUTIC',
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
  const isRouterExpanded = active === 'mode_router' || activeModeId !== null;

  return (
    <div className={styles.root}>
      <p className={styles.hint}>Click any step to expand. The main spine shows what every message goes through.</p>

      <div className={styles.pipeline}>
        {STEPS.map((step) => {
          const isActive = active === step.id;
          const isRouterStep = step.id === 'mode_router';
          const stepIsExpanded = isActive || (isRouterStep && isRouterExpanded);

          return (
            <div
              key={step.id}
              className={[
                styles.step,
                stepIsExpanded ? styles.stepActive : '',
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
                        ].join(' ')}
                      >
                        {b.label}
                      </span>
                    ))}
                  </div>
                )}

                {/* Conditional branch callout */}
                {step.branch && (
                  <div className={styles.branchRow}>
                    <span className={[styles.badge, step.branch.crisis ? styles.badgeCrisis : ''].join(' ')}>
                      if {step.branch.condition}
                    </span>
                    <span className={styles.branchArrow}>→</span>
                    <span className={styles.stepSub}>{step.branch.target}</span>
                  </div>
                )}
              </button>

              {/* Step detail panel */}
              {isActive && <DetailPanel detail={step.detail} key={`d-${step.id}`} />}

              {/* Mode router expansion */}
              {isRouterStep && isRouterExpanded && (
                <>
                  {/* Conditional modes section */}
                  <div className={styles.modeSection}>
                    <div className={styles.modeSectionHeader}>
                      <span className={styles.modeGridLabel}>Conditional modes — keyword-only, deterministic</span>
                    </div>
                    {CONDITIONAL_MODES.map(m => {
                      const mActive = activeModeId === m.id;
                      return (
                        <div key={m.id} className={styles.conditionalMode}>
                          <button
                            className={[styles.modeChip, mActive ? styles.modeChipActive : ''].join(' ')}
                            onClick={(e) => { e.stopPropagation(); toggle(`mode:${m.id}`); }}
                          >
                            <span className={styles.modeChipLabel}>{m.label}</span>
                            <span className={styles.modeCondition}>{m.conditionLabel}</span>
                          </button>
                          {mActive && <DetailPanel detail={m.detail} key={`d-${m.id}`} />}
                        </div>
                      );
                    })}
                  </div>

                  {/* Therapeutic modes section */}
                  <div className={styles.modeSection}>
                    <div className={styles.modeSectionHeader}>
                      <span className={styles.modeGridLabel}>Therapeutic modes — reachable from all 3 routing layers</span>
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
