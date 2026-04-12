import React, { useState } from 'react';
import styles from './StateFields.module.css';

/* ── Data ───────────────────────────────────────────────────────────────────── */

type Lifecycle = 'input' | 'persisted' | 'derived' | 'turn' | 'reserved';

interface FieldDef {
  name: string;
  type: string;
  setBy: string;
  lifecycle: Lifecycle;
  desc: string;
}

interface GroupDef {
  id: string;
  label: string;
  icon: string;
  fields: FieldDef[];
}

const GROUPS: GroupDef[] = [
  {
    id: 'input', label: 'Input', icon: '\u25B6',
    fields: [
      { name: 'message', type: 'str', setBy: 'caller', lifecycle: 'input', desc: 'Current user message being processed' },
      { name: 'channel', type: 'Channel', setBy: 'caller', lifecycle: 'input', desc: 'Normalized transport: TEST, WEB, SMS, WHATSAPP, TELEGRAM, VOICE' },
      { name: 'user_id', type: 'str | None', setBy: 'caller', lifecycle: 'input', desc: 'Reserved for future auth and ownership checks' },
      { name: 'session_id', type: 'str | None', setBy: 'caller', lifecycle: 'input', desc: 'Thread identifier used by the persistence layer' },
      { name: 'installed_skills', type: 'list[str]', setBy: 'caller', lifecycle: 'input', desc: 'Skill names resolved into prompt behavior by the graph' },
    ],
  },
  {
    id: 'transcript', label: 'Transcript & history', icon: '\u2630',
    fields: [
      { name: 'transcript', type: 'list[dict]', setBy: 'finalize_turn', lifecycle: 'persisted', desc: 'Full durable conversation record \u2014 source of truth for all context' },
      { name: 'history', type: 'list[dict]', setBy: 'prepare_turn', lifecycle: 'derived', desc: 'Last 8 turns sliced from transcript, injected into prompts' },
      { name: 'turn_count', type: 'int', setBy: 'prepare_turn', lifecycle: 'derived', desc: 'Count of user turns including current message' },
    ],
  },
  {
    id: 'context', label: 'Session context', icon: '\u2261',
    fields: [
      { name: 'active_concerns', type: 'list[str]', setBy: 'prepare_turn', lifecycle: 'derived', desc: 'Up to 3 labeled concerns: overwhelm, anxiety, grief, etc.' },
      { name: 'open_loops', type: 'list[str]', setBy: 'prepare_turn', lifecycle: 'derived', desc: 'Up to 3 unresolved threads the agent should track' },
      { name: 'current_goal', type: 'str | None', setBy: 'prepare_turn', lifecycle: 'derived', desc: 'Best-effort guess at what the user wants this session' },
      { name: 'session_summary', type: 'str', setBy: 'prepare_turn', lifecycle: 'derived', desc: 'Rolling summary of concerns, goal, and recent themes' },
    ],
  },
  {
    id: 'intent', label: 'Session intent & stage', icon: '\u2736',
    fields: [
      { name: 'session_intent', type: 'str | None', setBy: 'prepare_turn', lifecycle: 'persisted', desc: 'Sticky session direction: CBT work, grounding, pattern reflection, venting, supportive conversation' },
      { name: 'session_intent_source', type: 'str | None', setBy: 'prepare_turn', lifecycle: 'persisted', desc: 'How intent was set: "explicit" or "inferred"' },
      { name: 'session_stage', type: 'str', setBy: 'session_stage node', lifecycle: 'persisted', desc: 'Conversation arc: opening, deepening, stabilizing, closing' },
      { name: 'session_stage_source', type: 'str | None', setBy: 'session_stage node', lifecycle: 'persisted', desc: 'Deterministic logic or LLM refinement' },
      { name: 'session_stage_reason', type: 'str', setBy: 'session_stage node', lifecycle: 'persisted', desc: 'Short rationale for debugging and evals' },
    ],
  },
  {
    id: 'safety', label: 'Safety & routing', icon: '\u26A0',
    fields: [
      { name: 'crisis', type: 'CrisisAssessment', setBy: 'crisis_gate', lifecycle: 'turn', desc: 'Level 0\u20133, confidence, reason, needs_crisis_response, needs_clarification' },
      { name: 'route', type: 'str', setBy: 'crisis_gate', lifecycle: 'turn', desc: '"crisis" or "therapeutic" \u2014 decides which subgraph runs' },
    ],
  },
  {
    id: 'response', label: 'Response', icon: '\u2190',
    fields: [
      { name: 'mode', type: 'str', setBy: 'router / crisis subgraph', lifecycle: 'turn', desc: 'supportive_conversation, safety_check, orientation, guided_exercise, psychoeducation, pattern_reflection, out_of_scope, realignment, crisis_response' },
      { name: 'mode_source', type: 'str | None', setBy: 'therapeutic router', lifecycle: 'turn', desc: 'How the mode was selected: keyword, session_intent, llm, or default' },
      { name: 'mode_type', type: 'ModeType', setBy: 'therapeutic/crisis router', lifecycle: 'turn', desc: 'Higher-level mode category: therapeutic, operational, or crisis' },
      { name: 'active_modalities', type: 'list[str]', setBy: 'modality_selector', lifecycle: 'turn', desc: 'Active overlays: pfa (+ DBT skills), cbt, grief_support, act. MI applied as baseline via MODE_BASELINE_FILES.' },
      { name: 'semantic_signals', type: 'dict[str, bool]', setBy: 'prepare_turn / routing', lifecycle: 'turn', desc: 'Cached semantic interpretation (13 boolean fields) shared by routing, modality selection, and prompt shaping' },
      { name: 'response_guidance', type: 'str', setBy: 'therapeutic router', lifecycle: 'turn', desc: 'Turn-specific prompt shaping derived after mode selection' },
      { name: 'response_type', type: 'ResponseKind', setBy: 'response node', lifecycle: 'turn', desc: '"therapeutic" or "crisis"' },
      { name: 'response_text', type: 'str', setBy: 'response node', lifecycle: 'turn', desc: 'Generated reply from whichever node wins the route' },
    ],
  },
  {
    id: 'reserved', label: 'Reserved', icon: '\u2026',
    fields: [
      { name: 'working_memory', type: 'list[str]', setBy: '(future)', lifecycle: 'reserved', desc: 'Scratch space for retrieved facts \u2014 currently empty every turn' },
      { name: 'should_persist_memory', type: 'bool', setBy: '(future)', lifecycle: 'reserved', desc: 'Signal to persist after turn \u2014 currently always false' },
    ],
  },
];

const LC: Record<Lifecycle, { label: string; cls: string }> = {
  input:     { label: 'input',       cls: 'lcInput' },
  persisted: { label: 'persisted',   cls: 'lcPersisted' },
  derived:   { label: 're-derived',  cls: 'lcDerived' },
  turn:      { label: 'turn-scoped', cls: 'lcTurn' },
  reserved:  { label: 'reserved',    cls: 'lcReserved' },
};

/* ── Component ──────────────────────────────────────────────────────────────── */

export default function StateFields() {
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const [hoveredField, setHoveredField] = useState<string | null>(null);

  const toggle = (id: string) =>
    setCollapsed(prev => ({ ...prev, [id]: !prev[id] }));

  return (
    <div className={styles.root}>
      {/* Legend */}
      <div className={styles.legend}>
        {Object.entries(LC).map(([key, val]) => (
          <span key={key} className={styles.legendItem}>
            <span className={[styles.pill, styles[val.cls]].join(' ')}>{val.label}</span>
          </span>
        ))}
      </div>

      {/* Groups */}
      <div className={styles.groups}>
        {GROUPS.map(group => {
          const isCollapsed = collapsed[group.id] ?? false;
          return (
            <div key={group.id} className={styles.section}>
              <button
                className={styles.sectionHeader}
                onClick={() => toggle(group.id)}
                aria-expanded={!isCollapsed}
              >
                <span className={styles.sectionIcon}>{group.icon}</span>
                <span className={styles.sectionLabel}>{group.label}</span>
                <span className={styles.sectionCount}>{group.fields.length}</span>
                <span className={[styles.sectionChevron, isCollapsed ? styles.chevronCollapsed : ''].join(' ')}>
                  {'\u25BE'}
                </span>
              </button>

              {!isCollapsed && (
                <div className={styles.fieldTable}>
                  {group.fields.map(f => {
                    const lc = LC[f.lifecycle];
                    const isHovered = hoveredField === f.name;
                    return (
                      <div
                        key={f.name}
                        className={[styles.fieldRow, isHovered ? styles.fieldRowHover : ''].join(' ')}
                        onMouseEnter={() => setHoveredField(f.name)}
                        onMouseLeave={() => setHoveredField(null)}
                      >
                        <div className={styles.fieldLeft}>
                          <code className={styles.fieldName}>{f.name}</code>
                          <span className={styles.fieldType}>{f.type}</span>
                          <span className={[styles.pill, styles.pillSmall, styles[lc.cls]].join(' ')}>{lc.label}</span>
                        </div>
                        <div className={styles.fieldRight}>
                          <span className={styles.fieldDesc}>{f.desc}</span>
                          {isHovered && (
                            <span className={styles.fieldSetBy}>set by {f.setBy}</span>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
