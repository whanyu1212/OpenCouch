import React, { useState } from 'react';
import styles from './StateFields.module.css';

/* ── Data ───────────────────────────────────────────────────────────────────── */

type Lifecycle = 'input' | 'persisted' | 'reducer' | 'turn' | 'loaded';

interface FieldDef {
  name: string;
  type: string;
  setBy: string;
  lifecycle: Lifecycle;
  desc: string;
  reducer?: string;
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
      { name: 'user_id', type: 'str | None', setBy: 'caller', lifecycle: 'input', desc: 'Optional until auth/session plumbing exists, reserved for ownership checks' },
      { name: 'session_id', type: 'str | None', setBy: 'caller', lifecycle: 'input', desc: 'Thread identifier used by the persistence layer' },
      { name: 'installed_skills', type: 'list[str]', setBy: 'caller', lifecycle: 'input', desc: 'Skill names resolved into prompt behavior by the graph' },
    ],
  },
  {
    id: 'transcript', label: 'Transcript & history', icon: '\u2630',
    fields: [
      { name: 'history', type: 'Annotated[list[dict], operator.add]', setBy: 'build_initial_state + finalize_turn', lifecycle: 'reducer', desc: 'Accumulated via operator.add reducer. Each turn emits only new entries; the checkpointer merges them automatically across turns.', reducer: 'operator.add' },
      { name: 'transcript', type: 'Annotated[list[dict], operator.add]', setBy: 'build_initial_state + finalize_turn', lifecycle: 'reducer', desc: 'Full durable conversation record. Same reducer semantics as history — each turn appends, never reconstructs.', reducer: 'operator.add' },
    ],
  },
  {
    id: 'memory', label: 'Memory & working context', icon: '\u2261',
    fields: [
      { name: 'working_memory', type: 'list[WorkingMemoryEntry]', setBy: 'load_memory_node', lifecycle: 'loaded', desc: 'Structured dicts: SemanticWorkingMemoryEntry (evidence_quote) and EpisodicWorkingMemoryEntry (summary, themes, is_catch_up). Formatted on demand by prompt builders and CLI.' },
      { name: 'memory.summary', type: 'str', setBy: 'load_memory_node', lifecycle: 'loaded', desc: 'Retrieval summary for diagnostics: hit counts, store sizes, retrieval path' },
      { name: 'memory.procedural_rules', type: 'list[str]', setBy: 'load_memory_node', lifecycle: 'loaded', desc: 'Style rules from the user\'s procedural profile — directives that shape response style' },
      { name: 'memory.proactive_recall_enabled', type: 'bool', setBy: 'load_memory_node', lifecycle: 'loaded', desc: 'Whether to surface recalled context proactively. From the procedural profile\'s recall toggle.' },
    ],
  },
  {
    id: 'progress', label: 'Session progress', icon: '\u2736',
    fields: [
      { name: 'progress', type: 'Annotated[SessionProgressState, _merge_dicts]', setBy: 'build_initial_state + nodes', lifecycle: 'reducer', desc: 'Uses _merge_dicts reducer so per-turn fields (turn_count, stage) merge with cross-turn fields (exercise_type, exercise_step) from the checkpoint.', reducer: '_merge_dicts' },
      { name: 'progress.turn_count', type: 'int', setBy: 'build_initial_state', lifecycle: 'reducer', desc: 'Persistent callers derive from checkpoint; one-shot callers count from history' },
      { name: 'progress.stage', type: 'str', setBy: 'build_initial_state', lifecycle: 'reducer', desc: 'Conversation arc: opening, deepening, stabilizing, closing' },
      { name: 'progress.exercise_type', type: 'str | None', setBy: 'guided_exercise_node', lifecycle: 'reducer', desc: 'Multi-turn exercise identifier. Persists via merge reducer across turns without manual carry-forward.' },
      { name: 'progress.exercise_step', type: 'int | None', setBy: 'guided_exercise_node', lifecycle: 'reducer', desc: 'Current step index within the active exercise. Cleared when exercise completes.' },
    ],
  },
  {
    id: 'safety', label: 'Safety & routing', icon: '\u26A0',
    fields: [
      { name: 'crisis', type: 'CrisisAssessment', setBy: 'crisis_gate_node', lifecycle: 'turn', desc: 'Level 0\u20133, confidence, reason, needs_crisis_response, needs_clarification' },
      { name: 'routing.route', type: 'str', setBy: 'crisis_gate_node', lifecycle: 'turn', desc: '"crisis" or "therapeutic" \u2014 decides which branch runs' },
      { name: 'routing.mode', type: 'str', setBy: 'therapeutic_dispatch', lifecycle: 'turn', desc: 'supportive, reflective, clarifying, psychoeducation, guided_exercise, closing' },
      { name: 'routing.mode_source', type: 'str', setBy: 'therapeutic_dispatch', lifecycle: 'turn', desc: 'How the mode was selected: keyword, llm, default, or crisis_gate' },
      { name: 'routing.mode_type', type: 'ModeType', setBy: 'crisis_gate / dispatch', lifecycle: 'turn', desc: 'THERAPEUTIC, OPERATIONAL, or CRISIS' },
      { name: 'routing.modality', type: 'str | None', setBy: 'therapeutic_dispatch', lifecycle: 'turn', desc: 'Therapeutic modality: MI, CBT, ACT, DBT, grief, IPT, PFA, or None' },
    ],
  },
  {
    id: 'response', label: 'Response', icon: '\u2190',
    fields: [
      { name: 'response.text', type: 'str', setBy: 'mode node / crisis_response', lifecycle: 'turn', desc: 'Generated reply from whichever node wins the route' },
      { name: 'response.kind', type: 'ResponseKind', setBy: 'mode node', lifecycle: 'turn', desc: 'THERAPEUTIC or CRISIS' },
      { name: 'response.guidance', type: 'str', setBy: 'mode node', lifecycle: 'turn', desc: 'Turn-specific prompt shaping hint' },
    ],
  },
  {
    id: 'diagnostics', label: 'Diagnostics', icon: '\u2699',
    fields: [
      { name: 'diagnostics', type: 'Annotated[dict, _merge_dicts]', setBy: 'all I/O nodes', lifecycle: 'reducer', desc: 'Per-turn timing and write-count metadata. Uses _merge_dicts reducer so nodes write their own keys independently \u2014 no manual dict spreading needed. Parallel extractors write simultaneously without racing.', reducer: '_merge_dicts' },
      { name: 'diagnostics.load_memory_ms', type: 'float', setBy: 'load_memory_node', lifecycle: 'reducer', desc: 'Total time for memory retrieval (all 3 namespaces)' },
      { name: 'diagnostics.crisis_gate_ms', type: 'float', setBy: 'crisis_gate_node', lifecycle: 'reducer', desc: 'Time for crisis classification (regex + optional LLM)' },
      { name: 'diagnostics.extract_facts_ms', type: 'float', setBy: 'extract_facts_node', lifecycle: 'reducer', desc: 'Time for semantic fact extraction (LLM call)' },
      { name: 'diagnostics.extract_procedural_ms', type: 'float', setBy: 'extract_procedural_node', lifecycle: 'reducer', desc: 'Time for procedural rule extraction (LLM call)' },
    ],
  },
];

const LC: Record<Lifecycle, { label: string; cls: string }> = {
  input:  { label: 'input',        cls: 'lcInput' },
  persisted: { label: 'persisted', cls: 'lcPersisted' },
  reducer: { label: 'reducer',     cls: 'lcReducer' },
  turn:   { label: 'turn-scoped',  cls: 'lcTurn' },
  loaded: { label: 'loaded',       cls: 'lcLoaded' },
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
                          {f.reducer && (
                            <span className={[styles.pill, styles.pillSmall, styles.lcReducer].join(' ')}>{f.reducer}</span>
                          )}
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
