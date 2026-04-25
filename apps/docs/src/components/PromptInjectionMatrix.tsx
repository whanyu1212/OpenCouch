import React, { useState } from 'react';
import styles from './PromptInjectionMatrix.module.css';

interface ModeRow {
  mode: string;
  responseGuidance: boolean;
  stageGuidance: boolean;
  sessionContext: boolean;
  recentHistory: boolean;
  currentMessage: boolean;
  note?: string;
}

const MODES: ModeRow[] = [
  { mode: 'supportive',         responseGuidance: true,  stageGuidance: true,  sessionContext: true,  recentHistory: true,  currentMessage: true, note: 'Default mode — full context for warm validation' },
  { mode: 'reflective',         responseGuidance: true,  stageGuidance: true,  sessionContext: true,  recentHistory: true,  currentMessage: true, note: 'Pattern-naming needs cross-turn context' },
  { mode: 'clarifying',         responseGuidance: true,  stageGuidance: false, sessionContext: false, recentHistory: true,  currentMessage: true, note: 'One focused question — narrow context only' },
  { mode: 'psychoeducation',    responseGuidance: true,  stageGuidance: true,  sessionContext: true,  recentHistory: true,  currentMessage: true, note: 'Educational explanation needs the framing context' },
  { mode: 'technique',          responseGuidance: true,  stageGuidance: true,  sessionContext: true,  recentHistory: true,  currentMessage: true, note: 'Structured therapeutic work — full context plus modality' },
  { mode: 'guided_exercise',    responseGuidance: true,  stageGuidance: true,  sessionContext: true,  recentHistory: true,  currentMessage: true, note: 'Exercise state and pinned modality preserved across turns' },
  { mode: 'closing',            responseGuidance: true,  stageGuidance: true,  sessionContext: true,  recentHistory: true,  currentMessage: true, note: 'Wrap-up — context lets the farewell summarize the arc' },
  { mode: 'safety_check',       responseGuidance: true,  stageGuidance: false, sessionContext: false, recentHistory: true,  currentMessage: true, note: 'Crisis level 1 — narrow safety probe inside the therapeutic branch' },
  { mode: 'crisis_response',    responseGuidance: true,  stageGuidance: false, sessionContext: false, recentHistory: true,  currentMessage: true, note: 'Safety-focused — skip memory context, include any found_resources' },
  { mode: 'memory_control',     responseGuidance: false, stageGuidance: false, sessionContext: false, recentHistory: false, currentMessage: true, note: 'Deterministic — no LLM prompt assembled' },
  { mode: 'grounded_lookup',    responseGuidance: false, stageGuidance: false, sessionContext: false, recentHistory: true,  currentMessage: true, note: 'Search-grounded factual reply — minimal context' },
];

const LAYERS = ['responseGuidance', 'stageGuidance', 'sessionContext', 'recentHistory', 'currentMessage'] as const;
const LAYER_LABELS: Record<typeof LAYERS[number], string> = {
  responseGuidance: 'Response guidance',
  stageGuidance: 'Stage guidance',
  sessionContext: 'Session context',
  recentHistory: 'Recent history',
  currentMessage: 'Current message',
};

export default function PromptInjectionMatrix() {
  const [hoveredMode, setHoveredMode] = useState<string | null>(null);
  const [hoveredLayer, setHoveredLayer] = useState<string | null>(null);

  const activeRow = hoveredMode ? MODES.find(m => m.mode === hoveredMode) : null;

  return (
    <div className={styles.root}>
      <div className={styles.matrix}>
        {/* Header row */}
        <div className={styles.headerCorner} />
        {LAYERS.map(layer => (
          <div
            key={layer}
            className={[styles.headerCell, hoveredLayer === layer ? styles.headerActive : ''].join(' ')}
            onMouseEnter={() => setHoveredLayer(layer)}
            onMouseLeave={() => setHoveredLayer(null)}
          >
            {LAYER_LABELS[layer]}
          </div>
        ))}

        {/* Data rows */}
        {MODES.map(row => (
          <React.Fragment key={row.mode}>
            <div
              className={[styles.modeCell, hoveredMode === row.mode ? styles.modeCellActive : ''].join(' ')}
              onMouseEnter={() => setHoveredMode(row.mode)}
              onMouseLeave={() => setHoveredMode(null)}
            >
              {row.mode}
            </div>
            {LAYERS.map(layer => {
              const on = row[layer];
              const highlighted = hoveredMode === row.mode || hoveredLayer === layer;
              return (
                <div
                  key={layer}
                  className={[
                    styles.cell,
                    on ? styles.cellOn : styles.cellOff,
                    highlighted ? styles.cellHighlight : '',
                  ].join(' ')}
                  onMouseEnter={() => { setHoveredMode(row.mode); setHoveredLayer(layer); }}
                  onMouseLeave={() => { setHoveredMode(null); setHoveredLayer(null); }}
                >
                  {on ? '\u2713' : '\u2013'}
                </div>
              );
            })}
          </React.Fragment>
        ))}
      </div>

      {/* Note below on hover */}
      {activeRow?.note && (
        <p className={styles.note}>{activeRow.note}</p>
      )}
    </div>
  );
}
