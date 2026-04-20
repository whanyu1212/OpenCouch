import React, { useState } from 'react';
import styles from './RuntimeLayers.module.css';

/* ── Data ───────────────────────────────────────────────────────────────────── */

interface LayerDef {
  id: string;
  label: string;
  entry: string;
  desc: string;
  usedBy: string[];
  persists: boolean;
}

const LAYERS: LayerDef[] = [
  {
    id: 'stateless',
    label: 'Stateless',
    entry: 'run_agent(input)',
    desc: 'Takes AgentInput, returns AgentOutput. No persistence, no thread awareness. Caller manages history — single-turn contract.',
    usedBy: ['Tests', 'Evals', 'One-shot scripts'],
    persists: false,
  },
  {
    id: 'persistent',
    label: 'Persistent',
    entry: 'PersistentAgentRuntime.run_turn()',
    desc: 'Thread-aware runtime with SQLite checkpoint persistence. Conversations resume across sessions. Also exposes run_turn_stream() for real-time token streaming.',
    usedBy: ['CLI', 'API (WebSocket)', 'Multi-turn sessions'],
    persists: true,
  },
];

/* ── Component ──────────────────────────────────────────────────────────────── */

export default function RuntimeLayers() {
  const [activeLayer, setActiveLayer] = useState<string | null>(null);
  const layer = activeLayer ? LAYERS.find(l => l.id === activeLayer) ?? null : null;

  return (
    <div className={styles.root}>
      <div className={styles.diagram}>
        {/* ── Left: two entry points ─────────────── */}
        <div className={styles.entries}>
          {LAYERS.map((l) => (
            <button
              key={l.id}
              className={[styles.layer, styles[`layer_${l.id}`], activeLayer === l.id ? styles.layerActive : ''].join(' ')}
              onClick={() => setActiveLayer(p => p === l.id ? null : l.id)}
            >
              <span className={styles.layerAccent} />
              <div className={styles.layerContent}>
                <span className={styles.layerLabel}>{l.label}</span>
                <code className={styles.layerEntry}>{l.entry}</code>
                <div className={styles.layerTags}>
                  {l.usedBy.map(u => <span key={u} className={styles.tag}>{u}</span>)}
                  {l.persists && <span className={[styles.tag, styles.tagSqlite].join(' ')}>SQLite</span>}
                </div>
              </div>
            </button>
          ))}
        </div>

        {/* ── Center: shared core ────────────────── */}
        <div className={styles.core}>
          <div className={styles.coreInner}>
            <span className={styles.coreTitle}>Same graph</span>
            <span className={styles.coreSub}>Same nodes · same order · same safety</span>
          </div>
          <svg className={styles.coreArrow} viewBox="0 0 40 24" fill="none">
            <path d="M 2 12 H 32 M 28 6 L 36 12 L 28 18" stroke="var(--oc-accent)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>

        {/* ── Right: output ──────────────────────── */}
        <div className={styles.output}>
          <code className={styles.outputLabel}>AgentOutput</code>
          <span className={styles.outputSub}>response + crisis + diagnostics</span>
        </div>
      </div>

      {/* ── Detail panel ─────────────────────────── */}
      {layer && (
        <div className={styles.detail} key={layer.id}>
          <div className={styles.detailHeader}>
            <code className={styles.detailTitle}>{layer.entry}</code>
          </div>
          <p className={styles.detailText}>{layer.desc}</p>
          <div className={styles.detailMeta}>
            <span className={styles.metaKey}>Used by</span>
            <span className={styles.metaVal}>{layer.usedBy.join(', ')}</span>
          </div>
          <div className={styles.detailMeta}>
            <span className={styles.metaKey}>Persists</span>
            <span className={styles.metaVal}>{layer.persists ? 'Yes — SQLite via LangGraph checkpointer' : 'No — caller manages history'}</span>
          </div>
        </div>
      )}
    </div>
  );
}
