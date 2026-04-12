import React, { useState } from 'react';
import styles from './ThreadLifecycle.module.css';

/* ── Data ───────────────────────────────────────────────────────────────────── */

interface FieldDef {
  name: string;
  persisted: boolean;
}

const FIELDS: FieldDef[] = [
  { name: 'transcript', persisted: true },
  { name: 'session_intent', persisted: true },
  { name: 'session_stage', persisted: true },
  { name: 'active_concerns', persisted: false },
  { name: 'open_loops', persisted: false },
  { name: 'current_goal', persisted: false },
  { name: 'session_summary', persisted: false },
];

interface OpDef {
  id: string;
  label: string;
  desc: string;
  returns: string;
}

const OPS: OpDef[] = [
  { id: 'get_state', label: 'get_state()', desc: 'Load the latest persisted state snapshot for a thread.', returns: 'AgentState | None' },
  { id: 'get_history', label: 'get_history()', desc: 'Materialize the stored transcript as validated Message objects.', returns: 'list[Message]' },
  { id: 'reset_thread', label: 'reset_thread()', desc: 'Delete all checkpoints for a thread. The next turn starts fresh.', returns: 'None' },
];

const TURNS = [
  { label: 'Turn 1', message: '"I feel overwhelmed at work"' },
  { label: 'Turn 2', message: '"Can we do a grounding exercise?"' },
  { label: 'Turn 3', message: '"That helped, can you summarize?"' },
];

/* ── Component ──────────────────────────────────────────────────────────────── */

export default function ThreadLifecycle() {
  const [hoveredTurn, setHoveredTurn] = useState<number | null>(null);
  const [activeOp, setActiveOp] = useState<string | null>(null);
  const op = activeOp ? OPS.find(o => o.id === activeOp) ?? null : null;

  return (
    <div className={styles.root}>
      {/* Timeline */}
      <div className={styles.timeline}>
        {TURNS.map((turn, i) => (
          <React.Fragment key={i}>
            {/* Turn column */}
            <div
              className={[styles.turnCol, hoveredTurn === i ? styles.turnColActive : ''].join(' ')}
              onMouseEnter={() => setHoveredTurn(i)}
              onMouseLeave={() => setHoveredTurn(null)}
            >
              <div className={styles.turnHeader}>
                <span className={styles.turnDot} />
                <span className={styles.turnLabel}>{turn.label}</span>
              </div>
              <p className={styles.turnMessage}>{turn.message}</p>

              {/* Field chips */}
              <div className={styles.fieldGroup}>
                <span className={styles.fieldGroupLabel}>Persisted</span>
                <div className={styles.fieldChips}>
                  {FIELDS.filter(f => f.persisted).map(f => (
                    <span key={f.name} className={[styles.chip, styles.chipPersisted].join(' ')}>{f.name}</span>
                  ))}
                </div>
              </div>
              <div className={styles.fieldGroup}>
                <span className={styles.fieldGroupLabel}>Re-derived</span>
                <div className={styles.fieldChips}>
                  {FIELDS.filter(f => !f.persisted).map(f => (
                    <span key={f.name} className={[styles.chip, styles.chipDerived].join(' ')}>{f.name}</span>
                  ))}
                </div>
              </div>
            </div>

            {/* Checkpoint marker between turns */}
            {i < TURNS.length - 1 && (
              <div className={styles.checkpoint}>
                <div className={styles.checkpointLine} />
                <div className={styles.checkpointIcon} title="SQLite checkpoint">
                  <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                    <rect x="2" y="1" width="10" height="12" rx="2" stroke="currentColor" strokeWidth="1.2" />
                    <line x1="2" y1="4.5" x2="12" y2="4.5" stroke="currentColor" strokeWidth="1" />
                  </svg>
                </div>
                <div className={styles.checkpointLine} />
              </div>
            )}
          </React.Fragment>
        ))}
      </div>

      {/* Operations row */}
      <div className={styles.opsRow}>
        <span className={styles.opsLabel}>Thread operations</span>
        <div className={styles.opsChips}>
          {OPS.map(o => (
            <button
              key={o.id}
              className={[styles.opChip, activeOp === o.id ? styles.opChipActive : ''].join(' ')}
              onClick={() => setActiveOp(p => p === o.id ? null : o.id)}
            >
              {o.label}
            </button>
          ))}
        </div>
      </div>

      {/* Op detail */}
      {op && (
        <div className={styles.detail} key={op.id}>
          <div className={styles.detailHeader}>
            <span className={styles.detailTitle}>{op.label}</span>
            <button className={styles.detailClose} onClick={() => setActiveOp(null)}>&#10005;</button>
          </div>
          <div className={styles.detailBody}>
            <p className={styles.detailText}>{op.desc}</p>
            <div className={styles.detailMeta}>
              <span className={styles.metaKey}>Returns</span>
              <code className={styles.metaCode}>{op.returns}</code>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
