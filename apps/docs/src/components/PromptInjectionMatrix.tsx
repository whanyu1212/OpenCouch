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
  { mode: 'supportive_conversation', responseGuidance: true,  stageGuidance: true,  sessionContext: true,  recentHistory: true,  currentMessage: true },
  { mode: 'pattern_reflection',      responseGuidance: true,  stageGuidance: true,  sessionContext: true,  recentHistory: true,  currentMessage: true },
  { mode: 'guided_exercise',         responseGuidance: true,  stageGuidance: true,  sessionContext: true,  recentHistory: true,  currentMessage: true },
  { mode: 'psychoeducation',         responseGuidance: true,  stageGuidance: true,  sessionContext: true,  recentHistory: true,  currentMessage: true },
  { mode: 'realignment',             responseGuidance: true,  stageGuidance: false, sessionContext: true,  recentHistory: true,  currentMessage: true, note: 'Gets session context and history to re-attune, but no stage guidance' },
  { mode: 'crisis_response',         responseGuidance: false, stageGuidance: false, sessionContext: false, recentHistory: true,  currentMessage: true, note: 'Safety-focused — skip context to stay narrow' },
  { mode: 'crisis_classifier',       responseGuidance: false, stageGuidance: false, sessionContext: false, recentHistory: true,  currentMessage: true, note: 'Classification only — minimal context' },
  { mode: 'safety_check',            responseGuidance: false, stageGuidance: false, sessionContext: false, recentHistory: false, currentMessage: false, note: 'Deterministic template — no LLM prompt assembled' },
  { mode: 'orientation',             responseGuidance: true,  stageGuidance: false, sessionContext: false, recentHistory: false, currentMessage: true, note: 'New user — response guidance + message only' },
  { mode: 'out_of_scope',            responseGuidance: true,  stageGuidance: false, sessionContext: false, recentHistory: false, currentMessage: true, note: 'Boundary response — response guidance + request only' },
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
