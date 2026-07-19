import React from 'react';
import styles from './AppVsAgent.module.css';

interface Responsibility {
  label: string;
  detail: string;
}

const APP_OWNED: Responsibility[] = [
  { label: 'Safety classification', detail: 'Runs before any specialist is built' },
  { label: 'Specialist selection', detail: 'Therapeutic / Crisis / Guided Exercise' },
  { label: 'Memory mutation gating', detail: 'When writes are allowed, what is held' },
  { label: 'Exercise state', detail: 'Consent, current step, exit semantics' },
  { label: 'Persistence & audit', detail: 'Thread locks, crisis log, finalization' },
];

const AGENT_OWNED: Responsibility[] = [
  { label: 'Which tool to call', detail: 'Memory, grounded lookup, response-style, exercise tools' },
  { label: 'Response style choice', detail: 'supportive · reflective · clarifying · psychoeducation · technique · closing' },
  { label: 'Response wording', detail: 'Drafting the actual prose for the user' },
  { label: 'In-slot reasoning', detail: 'When to look up a hotline, when to record exercise progress' },
];

function ResponsibilityList({ items, dotClass }: { items: Responsibility[]; dotClass: string }): JSX.Element {
  return (
    <ul className={styles.list}>
      {items.map((item) => (
        <li key={item.label} className={styles.listItem}>
          <span className={`${styles.dot} ${dotClass}`} aria-hidden="true" />
          <div className={styles.listItemContent}>
            <span className={styles.itemLabel}>{item.label}</span>
            <span className={styles.itemDetail}>{item.detail}</span>
          </div>
        </li>
      ))}
    </ul>
  );
}

export default function AppVsAgent(): JSX.Element {
  return (
    <div className={styles.root}>
      <section className={styles.outer} aria-label="App-owned lifecycle">
        <header className={styles.outerHeader}>
          <span className={styles.eyebrow}>OUTER LAYER</span>
          <h4 className={styles.title}>App-owned</h4>
          <p className={styles.subtitle}>
            Lifecycle and ordering. Deterministic. Cannot be skipped or reordered by the model.
          </p>
        </header>

        <ResponsibilityList items={APP_OWNED} dotClass={styles.dotApp} />

        <section className={styles.inner} aria-label="Agent-owned reasoning">
          <header className={styles.innerHeader}>
            <span className={styles.eyebrow}>INNER LAYER</span>
            <h4 className={styles.title}>Agent-owned</h4>
            <p className={styles.subtitle}>
              Judgment within the assigned specialist slot. ReAct loop over attached tools.
            </p>
          </header>

          <ResponsibilityList items={AGENT_OWNED} dotClass={styles.dotAgent} />
        </section>
      </section>

      <p className={styles.caption}>
        The agent runs <em>inside</em> a slot the app defined. The outer ring decides who runs and when;
        the inner ring decides what to do once selected.
      </p>
    </div>
  );
}
