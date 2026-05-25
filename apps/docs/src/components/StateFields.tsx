import React, { useState } from 'react';
import styles from './StateFields.module.css';

/* ── Data ───────────────────────────────────────────────────────────────────── */

type Lifecycle = 'input' | 'reducer' | 'turn' | 'loaded';

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
  blurb: string;
  fields: FieldDef[];
}

const GROUPS: GroupDef[] = [
  {
    id: 'identity',
    label: 'Identity',
    blurb: 'Who and where the turn is from. Seeded once at turn start.',
    fields: [
      { name: 'message', type: 'str', setBy: 'caller', lifecycle: 'input', desc: 'Current user message being processed.' },
      { name: 'channel', type: 'Channel', setBy: 'caller', lifecycle: 'input', desc: 'Transport surface: TEST, WEB, VOICE.' },
      { name: 'user_id', type: 'str | None', setBy: 'caller', lifecycle: 'input', desc: 'Persistent owner. resolve_owner_id() namespaces memory by this; mandatory if session_id is absent.' },
      { name: 'session_id', type: 'str | None', setBy: 'caller', lifecycle: 'input', desc: 'Thread identifier used by persistence and as a fallback memory owner.' },
      { name: 'installed_skills', type: 'list[str]', setBy: 'caller', lifecycle: 'input', desc: 'Skill names resolved into prompt behavior by the runtime.' },
    ],
  },
  {
    id: 'conversation',
    label: 'Conversation',
    blurb: 'Append-only transcript channels — each turn emits only its own delta.',
    fields: [
      { name: 'history', type: 'Annotated[list[dict], operator.add]', setBy: 'build_initial_state + turn finalization', lifecycle: 'reducer', reducer: 'operator.add', desc: 'build_initial_state emits [user_turn]; turn finalization emits [assistant_turn]. The checkpointer accumulates prior turns automatically.' },
      { name: 'transcript', type: 'Annotated[list[dict], operator.add]', setBy: 'build_initial_state + finalize_turn', lifecycle: 'reducer', reducer: 'operator.add', desc: 'Full durable conversation record. Same reducer as history — never reconstructs from scratch.' },
    ],
  },
  {
    id: 'memory',
    label: 'Memory & working context',
    blurb: 'Loaded each turn by the runtime turn memory context.',
    fields: [
      { name: 'working_memory', type: 'list[WorkingMemoryEntry]', setBy: 'turn memory context', lifecycle: 'loaded', desc: 'SemanticWorkingMemoryEntry (category/subject/predicate/object + evidence_quote) and EpisodicWorkingMemoryEntry (summary, themes, is_catch_up, approach_used). Formatted on demand at prompt-build time.' },
      { name: 'session_memory', type: 'Annotated[SessionMemoryState, _merge_dicts]', setBy: 'turn memory context', lifecycle: 'reducer', reducer: '_merge_dicts', desc: 'Prompt-visible session continuity: summary, active_concerns, open_loops, current_goal.' },
      { name: 'procedural_profile.procedural_rules', type: 'list[str]', setBy: 'turn memory context', lifecycle: 'reducer', desc: 'Style directives that shape every reply. Always applied — the recall toggle is content-recall only.' },
      { name: 'procedural_profile.proactive_recall_enabled', type: 'bool', setBy: 'turn memory context', lifecycle: 'reducer', desc: 'Whether the agent may proactively reference recalled content. Procedural rules apply regardless.' },
    ],
  },
  {
    id: 'progress',
    label: 'Session progress',
    blurb: 'Per-turn counters that merge into the checkpoint.',
    fields: [
      { name: 'session_progress', type: 'Annotated[SessionProgressState, _merge_dicts]', setBy: 'build_initial_state', lifecycle: 'reducer', reducer: '_merge_dicts', desc: 'Turn-count continuity. The merge reducer preserves checkpointed counters and sibling flags.' },
      { name: 'session_progress.turn_count', type: 'int', setBy: 'build_initial_state', lifecycle: 'reducer', desc: 'Persistent callers derive from checkpoint; one-shot callers count from history.' },
    ],
  },
  {
    id: 'exercise',
    label: 'Exercise & memory-control continuity',
    blurb: 'Multi-turn state that survives mid-exercise side-turns.',
    fields: [
      { name: 'exercise_state', type: 'Annotated[ExerciseState, _merge_dicts]', setBy: 'guided exercise flow + triage', lifecycle: 'reducer', reducer: '_merge_dicts', desc: 'Active guided-exercise continuity. Cleared by runtime logic when the user exits or the exercise completes.' },
      { name: 'exercise_state.exercise_type', type: 'str | None', setBy: 'guided exercise flow', lifecycle: 'reducer', desc: 'Active exercise identifier (e.g., "grounding_5_4_3_2_1").' },
      { name: 'exercise_state.exercise_step', type: 'int | None', setBy: 'guided exercise flow', lifecycle: 'reducer', desc: 'Current step index. Cleared when the exercise completes or the user exits.' },
      { name: 'exercise_state.exercise_therapeutic_approach', type: 'str | None', setBy: 'guided exercise flow + triage', lifecycle: 'reducer', desc: 'Approach pinned at exercise start. Reused when guidance resumes and for narrow clarifying side-turns; psychoeducation side-turns keep the exercise active but may use a fresh top-level approach.' },
      { name: 'memory_control.pending_action', type: 'dict | None', setBy: 'memory-control service', lifecycle: 'reducer', reducer: '_merge_dicts', desc: 'Carries a destructive memory action across turns so the next reply can confirm or cancel without LLM inference.' },
    ],
  },
  {
    id: 'safety',
    label: 'Crisis & safety',
    blurb: 'Set by the crisis gate on every turn; consumed by crisis logging.',
    fields: [
      { name: 'crisis', type: 'CrisisAssessment', setBy: 'crisis gate', lifecycle: 'turn', desc: 'level (0–3), confidence, reason, needs_crisis_response, needs_clarification.' },
      { name: 'crisis_audit', type: 'CrisisAuditState', setBy: 'crisis gate', lifecycle: 'turn', desc: 'Classifier provenance: override_kind, classifier_path (llm_primary), llm_failure_occurred. Read by crisis logging.' },
      { name: 'route', type: 'str', setBy: 'crisis_gate + turn_dispatch', lifecycle: 'turn', desc: '"crisis" / "therapeutic" / "memory_control" / "grounded_lookup". Drives extractor skip logic.' },
    ],
  },
  {
    id: 'response',
    label: 'Routing & response',
    blurb: 'Whichever runtime branch wins the route writes these. Returned via AgentOutput.',
    fields: [
      { name: 'response_text', type: 'str', setBy: 'reply node (response style / crisis_response / memory_control / grounded_answer)', lifecycle: 'turn', desc: 'Generated reply for the turn.' },
      { name: 'response_style', type: 'str', setBy: 'reply node + gates', lifecycle: 'turn', desc: 'supportive · reflective · clarifying · psychoeducation · technique · guided_exercise · closing · safety_check · crisis_response · memory_control · grounded_lookup' },
      { name: 'therapeutic_approach', type: 'str | None', setBy: 'TherapeuticAgent', lifecycle: 'turn', desc: 'motivational_interviewing · cbt · act · dbt_skills · grief_support · interpersonal_therapy · pfa · none' },
      { name: 'should_persist_memory', type: 'bool', setBy: 'guided exercise flow', lifecycle: 'turn', desc: 'Set on exercise completion as a hint that the turn is worth summarizing.' },
    ],
  },
  {
    id: 'lookup',
    label: 'Lookup scratch fields',
    blurb: 'Turn-scoped IO between routing and worker nodes.',
    fields: [
      { name: 'memory_control.action', type: 'dict', setBy: 'turn triage', lifecycle: 'turn', desc: 'Detected memory command (kind, args). Consumed by the memory-control service.' },
      { name: 'grounded_lookup.query', type: 'str', setBy: 'turn triage', lifecycle: 'turn', desc: 'Factual query extracted by triage. Consumed by the grounded lookup tool.' },
      { name: 'grounded_lookup.status', type: 'str', setBy: 'grounded lookup tool', lifecycle: 'turn', desc: 'answered · no_verified_answer · not_attempted' },
      { name: 'inferred_location', type: 'str', setBy: 'crisis resource lookup', lifecycle: 'turn', desc: 'Region extracted from recent context for hotline lookup.' },
      { name: 'found_resources', type: 'list[dict]', setBy: 'crisis resource lookup', lifecycle: 'turn', desc: 'Verified hotlines (name / phone / website / region). Empty on failure or missing location.' },
      { name: 'resource_lookup_status', type: 'str', setBy: 'crisis resource lookup', lifecycle: 'turn', desc: 'found · no_location · location_refused · no_verified_results · not_attempted' },
    ],
  },
  {
    id: 'diagnostics',
    label: 'Diagnostics',
    blurb: 'Per-turn observability. _merge_dicts lets runtime stages and services write their own keys without racing.',
    fields: [
      { name: 'diagnostics', type: 'Annotated[dict, _merge_dicts]', setBy: 'runtime stages + services', lifecycle: 'reducer', reducer: '_merge_dicts', desc: 'Timing, routing, and retrieval metadata.' },
      { name: 'diagnostics.crisis_gate_ms · crisis_classifier_path', type: 'float · str', setBy: 'crisis gate', lifecycle: 'reducer', desc: 'Time spent classifying + which path decided it.' },
      { name: 'diagnostics.load_memory_ms · retrieval_path', type: 'float · str', setBy: 'turn memory context', lifecycle: 'reducer', desc: 'Retrieval time + which path ran (hybrid_rrf / token_recall / token_recall_after_embed_error).' },
      { name: 'diagnostics.turn_total_ms', type: 'float', setBy: 'runtime', lifecycle: 'reducer', desc: 'Total turn wall-clock, stamped outside the graph.' },
    ],
  },
];

const LC: Record<Lifecycle, { label: string; cls: string; hint: string }> = {
  input:    { label: 'input',       cls: 'lcInput',    hint: 'Set once by the caller at turn start.' },
  reducer:  { label: 'reducer',     cls: 'lcReducer',  hint: 'Annotated with a reducer — accumulates across turns via the checkpointer.' },
  turn:     { label: 'turn-scoped', cls: 'lcTurn',     hint: 'Fresh each turn. Last-write-wins.' },
  loaded:   { label: 'loaded',      cls: 'lcLoaded',   hint: 'Re-fetched each turn from the memory store.' },
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
          <span key={key} className={styles.legendItem} title={val.hint}>
            <span className={[styles.pill, styles[val.cls]].join(' ')}>{val.label}</span>
            <span className={styles.legendHint}>{val.hint}</span>
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
                <div className={styles.sectionHeaderText}>
                  <span className={styles.sectionLabel}>{group.label}</span>
                  <span className={styles.sectionBlurb}>{group.blurb}</span>
                </div>
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
                        </div>
                        <div className={styles.fieldRight}>
                          <p className={styles.fieldDesc}>{f.desc}</p>
                          <div className={styles.fieldMeta}>
                            <span className={[styles.pill, styles.pillSmall, styles[lc.cls]].join(' ')}>{lc.label}</span>
                            {f.reducer && (
                              <span className={[styles.pill, styles.pillSmall, styles.lcReducer].join(' ')}>{f.reducer}</span>
                            )}
                            <span className={styles.fieldSetBy}>set by <code>{f.setBy}</code></span>
                          </div>
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
