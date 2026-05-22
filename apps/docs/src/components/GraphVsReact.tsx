import React from 'react';
import styles from './GraphVsReact.module.css';

/* ── Data ─────────────────────────────────────────────────────────────────── */

interface Trait {
  label: string;
  runtimeValue: string;
  reactValue: string;
}

const TRAITS: Trait[] = [
  { label: 'Execution order', runtimeValue: 'Topology enforces order', reactValue: 'LLM chooses order' },
  { label: 'Safety gate', runtimeValue: 'Cannot be skipped', reactValue: 'Can drift with prompt changes' },
  { label: 'Memory extraction', runtimeValue: 'Scheduled after response', reactValue: 'Optional — LLM may skip' },
  { label: 'Tool invocation', runtimeValue: 'Fixed runtime branches', reactValue: 'Emergent from LLM output' },
  { label: 'Routing decisions', runtimeValue: 'State delta + branch selection', reactValue: 'Parsed from LLM text' },
  { label: 'Enforcement', runtimeValue: 'Application invariant', reactValue: 'Runtime guideline' },
];

/* ── Component ────────────────────────────────────────────────────────────── */

export default function GraphVsReact() {
  return (
    <div className={styles.root}>
      <div className={styles.columns}>
        {/* ── Runtime column ─────────────────────────── */}
        <div className={styles.column}>
          <div className={styles.columnHeader}>
            <span className={styles.columnDot + ' ' + styles.dotGraph} />
            <span className={styles.columnTitle}>Runtime topology</span>
            <span className={styles.columnTag + ' ' + styles.tagGraph}>OpenCouch</span>
          </div>
          <div className={styles.spine + ' ' + styles.spineGraph}>
            <div className={styles.spinePulse + ' ' + styles.pulseGraph} />
            {TRAITS.map((t) => (
              <div key={t.label} className={styles.traitCard + ' ' + styles.traitGraph}>
                <span className={styles.traitDot + ' ' + styles.dotGraph} />
                <div className={styles.traitContent}>
                  <span className={styles.traitLabel}>{t.label}</span>
                  <span className={styles.traitValue}>{t.runtimeValue}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* ── Divider ────────────────────────────────── */}
        <div className={styles.divider}>
          <span className={styles.dividerLabel}>vs</span>
        </div>

        {/* ── ReAct column ───────────────────────────── */}
        <div className={styles.column}>
          <div className={styles.columnHeader}>
            <span className={styles.columnDot + ' ' + styles.dotReact} />
            <span className={styles.columnTitle}>ReAct loop</span>
            <span className={styles.columnTag + ' ' + styles.tagReact}>alternative</span>
          </div>
          <div className={styles.spine + ' ' + styles.spineReact}>
            <div className={styles.spinePulse + ' ' + styles.pulseReact} />
            {TRAITS.map((t) => (
              <div key={t.label} className={styles.traitCard + ' ' + styles.traitReact}>
                <span className={styles.traitDot + ' ' + styles.dotReact} />
                <div className={styles.traitContent}>
                  <span className={styles.traitLabel}>{t.label}</span>
                  <span className={styles.traitValue}>{t.reactValue}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className={styles.verdict}>
        The runtime topology doesn&apos;t make OpenCouch <em>less capable</em> &mdash; it makes it
        <strong> incapable of the specific failures that matter most</strong> in a mental health context.
      </div>
    </div>
  );
}
