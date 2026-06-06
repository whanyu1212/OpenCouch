import React, { useState } from 'react';
import styles from './AxisCombination.module.css';

/* ── Two-axis composition ──────────────────────────────────────────────────────
   Visualizes how each therapeutic turn = one response style (the shape) overlaid
   with one therapeutic approach (the framework). The two axes are orthogonal and
   chosen at different points in the pipeline; this panel shows them composing. */

interface Pairing {
  id: string;
  style: string;
  approach: string;
  result: string;
}

const PAIRINGS: Pairing[] = [
  {
    id: 'supportive-mi',
    style: 'supportive',
    approach: 'motivational_interviewing',
    result: 'Validate the feeling, then evoke the user’s own reasons and a small next step.',
  },
  {
    id: 'technique-cbt',
    style: 'technique',
    approach: 'cbt',
    result: 'Walk a thought through evidence step by step — the CBT overlay carries the shape.',
  },
  {
    id: 'psychoeducation-grief',
    style: 'psychoeducation',
    approach: 'grief_support',
    result: 'A short, normalizing explanation of grief, framed for this user’s situation.',
  },
];

export default function AxisCombination() {
  const [active, setActive] = useState<string>(PAIRINGS[0].id);
  const pairing = PAIRINGS.find(p => p.id === active) ?? PAIRINGS[0];

  return (
    <div className={styles.root}>
      <div className={styles.formula} role="group" aria-label="How a turn is composed">
        <div className={`${styles.axis} ${styles.axisStyle}`}>
          <span className={styles.axisLabel}>Response style</span>
          <span className={styles.axisValue}>{pairing.style}</span>
          <span className={styles.axisCaption}>the shape</span>
        </div>

        <span className={styles.operator} aria-hidden="true">+</span>

        <div className={`${styles.axis} ${styles.axisApproach}`}>
          <span className={styles.axisLabel}>Therapeutic approach</span>
          <span className={styles.axisValue}>{pairing.approach}</span>
          <span className={styles.axisCaption}>the framework</span>
        </div>

        <span className={styles.operator} aria-hidden="true">=</span>

        <div className={`${styles.axis} ${styles.axisResult}`}>
          <span className={styles.axisLabel}>This turn</span>
          <span className={styles.axisResultText}>{pairing.result}</span>
        </div>
      </div>

      <div className={styles.switcher} role="tablist" aria-label="Example pairings">
        {PAIRINGS.map(p => (
          <button
            key={p.id}
            role="tab"
            aria-selected={active === p.id}
            className={`${styles.chip} ${active === p.id ? styles.chipActive : ''}`}
            onClick={() => setActive(p.id)}
          >
            {p.style} × {p.approach}
          </button>
        ))}
      </div>
    </div>
  );
}
