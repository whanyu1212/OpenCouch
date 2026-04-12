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
    entry: 'run_agent()',
    desc: 'Takes AgentInput, returns AgentOutput. No persistence, no thread awareness. The caller is responsible for storing and replaying history.',
    usedBy: ['API routes', 'unit tests'],
    persists: false,
  },
  {
    id: 'persistent',
    label: 'Persistent',
    entry: 'PersistentAgentRuntime',
    desc: 'Wraps the LangGraph workflow with SQLite checkpoint persistence. Thread-aware — conversations resume across sessions via run_turn(thread_id, message).',
    usedBy: ['CLI', 'evals', 'multi-turn tests'],
    persists: true,
  },
];

interface NodeDef {
  id: string;
  label: string;
  kind: 'node' | 'fork' | 'compact';
  detail: string;
}

const PIPELINE: NodeDef[] = [
  { id: 'prepare', label: 'prepare_turn', kind: 'node', detail: 'Re-derives all context fields (concerns, loops, goal, intent, summary) from the persisted transcript. Resets turn-scoped fields (crisis, route, mode, response).' },
  { id: 'crisis', label: 'crisis_gate', kind: 'node', detail: 'Hybrid crisis detection: deterministic regex first, optional LLM classifier for ambiguous cases. Sets route to "crisis" or "therapeutic".' },
  { id: 'stage', label: 'session_stage', kind: 'node', detail: 'Infers conversation progression (opening → deepening → stabilizing → closing). Deterministic first, optional LLM refinement. Skipped on crisis turns.' },
  { id: 'fork', label: 'crisis | therapeutic', kind: 'fork', detail: 'Conditional branch based on route. Crisis path runs crisis_response + PFA. Therapeutic path runs the mode router then the selected response node.' },
  { id: 'finalize', label: 'finalize_turn', kind: 'node', detail: 'Appends the user message and assistant response to the durable transcript. Updates the trimmed history window.' },
  { id: 'compact', label: 'compact', kind: 'compact', detail: 'Strips derived history before checkpoint persistence. The transcript is the source of truth — history is re-derived next turn.' },
];

/* ── Component ──────────────────────────────────────────────────────────────── */

export default function RuntimeLayers() {
  const [activeLayer, setActiveLayer] = useState<string | null>(null);
  const [activeNode, setActiveNode] = useState<string | null>(null);

  const layer = activeLayer ? LAYERS.find(l => l.id === activeLayer) ?? null : null;
  const node = activeNode ? PIPELINE.find(n => n.id === activeNode) ?? null : null;
  const activeDetail = node || layer;

  return (
    <div className={styles.root}>
      <p className={styles.hint}>Click a layer or pipeline node to see details.</p>

      <div className={styles.diagram}>
        {/* Stateless layer */}
        <button
          className={[styles.layer, styles.layerStateless, activeLayer === 'stateless' ? styles.layerActive : ''].join(' ')}
          onClick={() => { setActiveLayer(p => p === 'stateless' ? null : 'stateless'); setActiveNode(null); }}
        >
          <span className={styles.layerAccent} />
          <div className={styles.layerContent}>
            <div className={styles.layerHeader}>
              <span className={styles.layerLabel}>Stateless</span>
              <code className={styles.layerEntry}>run_agent()</code>
            </div>
            <div className={styles.layerTags}>
              <span className={styles.tag}>API routes</span>
              <span className={styles.tag}>unit tests</span>
            </div>
          </div>
        </button>

        {/* Shared pipeline */}
        <div className={styles.pipeline}>
          <span className={styles.pipelineLabel}>shared pipeline</span>
          <div className={styles.pipelineTrack}>
            {PIPELINE.map((n, i) => (
              <React.Fragment key={n.id}>
                <button
                  className={[
                    styles.pipelineNode,
                    n.kind === 'fork' ? styles.nodeFork : '',
                    n.kind === 'compact' ? styles.nodeCompact : '',
                    activeNode === n.id ? styles.nodeActive : '',
                  ].join(' ')}
                  onClick={() => { setActiveNode(p => p === n.id ? null : n.id); setActiveLayer(null); }}
                  title={n.label}
                >
                  {n.label}
                </button>
                {i < PIPELINE.length - 1 && (
                  <span className={styles.pipelineArrow}>{'\u2192'}</span>
                )}
              </React.Fragment>
            ))}
          </div>
        </div>

        {/* Persistent layer */}
        <button
          className={[styles.layer, styles.layerPersistent, activeLayer === 'persistent' ? styles.layerActive : ''].join(' ')}
          onClick={() => { setActiveLayer(p => p === 'persistent' ? null : 'persistent'); setActiveNode(null); }}
        >
          <span className={styles.layerAccent} />
          <div className={styles.layerContent}>
            <div className={styles.layerHeader}>
              <span className={styles.layerLabel}>Persistent</span>
              <code className={styles.layerEntry}>PersistentAgentRuntime</code>
            </div>
            <div className={styles.layerTags}>
              <span className={styles.tag}>CLI</span>
              <span className={styles.tag}>evals</span>
              <span className={styles.tag}>thread-aware</span>
              <span className={[styles.tag, styles.tagSqlite].join(' ')}>SQLite</span>
            </div>
          </div>
        </button>
      </div>

      {/* Detail panel */}
      {activeDetail && (
        <div className={styles.detail} key={'id' in activeDetail ? activeDetail.id : ''}>
          <div className={styles.detailHeader}>
            <span className={styles.detailTitle}>
              {'entry' in activeDetail ? activeDetail.entry : activeDetail.label}
            </span>
            <button className={styles.detailClose} onClick={() => { setActiveLayer(null); setActiveNode(null); }}>&#10005;</button>
          </div>
          <div className={styles.detailBody}>
            {'desc' in activeDetail ? (
              <>
                <p className={styles.detailText}>{activeDetail.desc}</p>
                <div className={styles.detailMeta}>
                  <span className={styles.metaKey}>Used by</span>
                  <span className={styles.metaVal}>{activeDetail.usedBy.join(', ')}</span>
                </div>
                <div className={styles.detailMeta}>
                  <span className={styles.metaKey}>Persists</span>
                  <span className={styles.metaVal}>{activeDetail.persists ? 'Yes — SQLite via LangGraph checkpointer' : 'No — caller manages history'}</span>
                </div>
              </>
            ) : (
              <p className={styles.detailText}>{activeDetail.detail}</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
