import React, { useState } from 'react';
import styles from './PromptInjectionMatrix.module.css';

interface ResponseStyleRow {
  responseStyle: string;
  responseGuidance: boolean;
  stageGuidance: boolean;
  sessionContext: boolean;
  recentHistory: boolean;
  currentMessage: boolean;
  note?: string;
}

const RESPONSE_STYLES: ResponseStyleRow[] = [
  { responseStyle: 'supportive',         responseGuidance: true,  stageGuidance: true,  sessionContext: true,  recentHistory: true,  currentMessage: true, note: 'Default response style — full context for warm validation' },
  { responseStyle: 'reflective',         responseGuidance: true,  stageGuidance: true,  sessionContext: true,  recentHistory: true,  currentMessage: true, note: 'Pattern-naming needs cross-turn context' },
  { responseStyle: 'clarifying',         responseGuidance: true,  stageGuidance: false, sessionContext: false, recentHistory: true,  currentMessage: true, note: 'One focused question — narrow context only' },
  { responseStyle: 'psychoeducation',    responseGuidance: true,  stageGuidance: true,  sessionContext: true,  recentHistory: true,  currentMessage: true, note: 'Educational explanation needs the framing context' },
  { responseStyle: 'technique',          responseGuidance: true,  stageGuidance: true,  sessionContext: true,  recentHistory: true,  currentMessage: true, note: 'Structured therapeutic work — full context plus therapeutic approach' },
  { responseStyle: 'guided_exercise',    responseGuidance: true,  stageGuidance: true,  sessionContext: true,  recentHistory: true,  currentMessage: true, note: 'Exercise state and pinned approach preserved across turns' },
  { responseStyle: 'closing',            responseGuidance: true,  stageGuidance: true,  sessionContext: true,  recentHistory: true,  currentMessage: true, note: 'Wrap-up — context lets the farewell summarize the arc' },
  { responseStyle: 'safety_check',       responseGuidance: true,  stageGuidance: false, sessionContext: false, recentHistory: true,  currentMessage: true, note: 'Crisis level 1 — narrow safety probe inside the therapeutic branch' },
  { responseStyle: 'crisis_response',    responseGuidance: true,  stageGuidance: false, sessionContext: false, recentHistory: true,  currentMessage: true, note: 'Safety-focused — skip memory context, include any found_resources' },
  { responseStyle: 'memory_control',     responseGuidance: false, stageGuidance: false, sessionContext: false, recentHistory: false, currentMessage: true, note: 'Deterministic — no LLM prompt assembled' },
  { responseStyle: 'grounded_lookup',    responseGuidance: false, stageGuidance: false, sessionContext: false, recentHistory: true,  currentMessage: true, note: 'Search-grounded factual reply — minimal context' },
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
  const [hoveredResponseStyle, setHoveredResponseStyle] = useState<string | null>(null);
  const [hoveredLayer, setHoveredLayer] = useState<string | null>(null);

  const activeRow = hoveredResponseStyle ? RESPONSE_STYLES.find(m => m.responseStyle === hoveredResponseStyle) : null;

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
        {RESPONSE_STYLES.map(row => (
          <React.Fragment key={row.responseStyle}>
            <div
              className={[styles.responseStyleCell, hoveredResponseStyle === row.responseStyle ? styles.responseStyleCellActive : ''].join(' ')}
              onMouseEnter={() => setHoveredResponseStyle(row.responseStyle)}
              onMouseLeave={() => setHoveredResponseStyle(null)}
            >
              {row.responseStyle}
            </div>
            {LAYERS.map(layer => {
              const on = row[layer];
              const highlighted = hoveredResponseStyle === row.responseStyle || hoveredLayer === layer;
              return (
                <div
                  key={layer}
                  className={[
                    styles.cell,
                    on ? styles.cellOn : styles.cellOff,
                    highlighted ? styles.cellHighlight : '',
                  ].join(' ')}
                  onMouseEnter={() => { setHoveredResponseStyle(row.responseStyle); setHoveredLayer(layer); }}
                  onMouseLeave={() => { setHoveredResponseStyle(null); setHoveredLayer(null); }}
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
