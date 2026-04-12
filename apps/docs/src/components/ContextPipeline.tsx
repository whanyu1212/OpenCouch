import React, { useState } from 'react';
import styles from './ContextPipeline.module.css';

interface Step {
  id: string;
  label: string;
  icon: string;
  inputs: string;
  outputs: string;
  detail: string;
}

const STEPS: Step[] = [
  {
    id: 'trim',
    label: 'Trim history',
    icon: '01',
    inputs: 'full transcript',
    outputs: 'last 8 turns',
    detail: 'Simple tail slice of the persisted transcript. Older turns fall off so the prompt window stays bounded regardless of session length.',
  },
  {
    id: 'concerns',
    label: 'Extract concerns',
    icon: '02',
    inputs: 'all user turns + current message',
    outputs: 'up to 3 concern labels',
    detail: 'Regex patterns match themes like overwhelm, anxiety, grief, self-worth, relationships, work pressure, sleep. Falls back to raw utterances if nothing matches.',
  },
  {
    id: 'loops',
    label: 'Extract open loops',
    icon: '03',
    inputs: 'last 6 user turns + current message',
    outputs: 'up to 3 unresolved threads',
    detail: 'Identifies questions and requests that haven\'t been resolved — "why do I keep", "help me understand", anything with a question mark.',
  },
  {
    id: 'goal',
    label: 'Infer goal',
    icon: '04',
    inputs: 'current message (fallback: last user turn)',
    outputs: 'goal string or null',
    detail: 'Pattern-matches against known goal templates: understand patterns, feel calmer, work through exercise, learn how OpenCouch works.',
  },
  {
    id: 'intent',
    label: 'Update session intent',
    icon: '05',
    inputs: 'current message + existing intent',
    outputs: 'sticky intent label + source',
    detail: 'Classifies the user\'s overall session intent (CBT work, grounding, reflection, psychoeducation, venting, support). Sticky — explicit requests override inferred ones and persist across turns.',
  },
  {
    id: 'summary',
    label: 'Build summary',
    icon: '06',
    inputs: 'concerns + goal + recent user snippets',
    outputs: 'rolling session summary',
    detail: 'Compiles everything into one deterministic string that recomputes every turn. Same inputs always produce the same summary.',
  },
];

export default function ContextPipeline() {
  const [active, setActive] = useState<string | null>(null);
  const activeStep = active ? STEPS.find(s => s.id === active) ?? null : null;

  return (
    <div className={styles.root}>
      <p className={styles.hint}>Click a step to see what it does.</p>

      {/* Pipeline steps */}
      <div className={styles.pipeline}>
        {STEPS.map((step, i) => (
          <React.Fragment key={step.id}>
            <button
              className={[styles.step, active === step.id ? styles.stepActive : ''].join(' ')}
              onClick={() => setActive(p => p === step.id ? null : step.id)}
            >
              <span className={styles.stepIcon}>{step.icon}</span>
              <span className={styles.stepLabel}>{step.label}</span>
              <span className={styles.stepIO}>
                <span className={styles.ioIn}>{step.inputs}</span>
                <span className={styles.ioArrow}>&rarr;</span>
                <span className={styles.ioOut}>{step.outputs}</span>
              </span>
            </button>
            {i < STEPS.length - 1 && (
              <div className={styles.connector}>
                <div className={styles.connectorLine} />
                <div className={styles.connectorHead} />
              </div>
            )}
          </React.Fragment>
        ))}
      </div>

      {/* Detail panel */}
      {activeStep && (
        <div className={styles.detail} key={activeStep.id}>
          <div className={styles.detailHeader}>
            <span className={styles.detailIcon}>{activeStep.icon}</span>
            <span className={styles.detailTitle}>{activeStep.label}</span>
            <button className={styles.detailClose} onClick={() => setActive(null)}>&#10005;</button>
          </div>
          <div className={styles.detailBody}>
            <p className={styles.detailText}>{activeStep.detail}</p>
            <div className={styles.detailMeta}>
              <div className={styles.metaRow}>
                <span className={styles.metaKey}>In</span>
                <span className={styles.metaVal}>{activeStep.inputs}</span>
              </div>
              <div className={styles.metaRow}>
                <span className={styles.metaKey}>Out</span>
                <span className={styles.metaVal}>{activeStep.outputs}</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
