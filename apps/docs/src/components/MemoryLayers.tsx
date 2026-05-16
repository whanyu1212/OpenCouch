import React from 'react';
import styles from './MemoryLayers.module.css';

const semanticFacts = ['KNOWS Sarah', 'USES fluoxetine', 'WORRIES_ABOUT work'];
const episodicSessions = [
  'Session 1: panic attacks, did grounding',
  'Session 2: work stress and sleep',
];
const proceduralRules = ["Don't suggest meditation", 'Prefer shorter responses'];

function MemoryColumn({
  title,
  variant,
  items,
}: {
  title: string;
  variant: 'semantic' | 'episodic' | 'procedural';
  items: string[];
}): JSX.Element {
  const variantClass = {
    semantic: styles.semantic,
    episodic: styles.episodic,
    procedural: styles.procedural,
  }[variant];

  return (
    <article className={`${styles.column} ${variantClass}`}>
      <h4>{title}</h4>
      <ul>
        {items.map((item) => (
          <li key={item}>
            <code>{item}</code>
          </li>
        ))}
      </ul>
    </article>
  );
}

export default function MemoryLayers(): JSX.Element {
  return (
    <section className={styles.container} aria-label="Three memory layers">
      <div className={styles.topGrid}>
        <MemoryColumn title="Semantic Memory" variant="semantic" items={semanticFacts} />
        <MemoryColumn title="Episodic Memory" variant="episodic" items={episodicSessions} />
        <MemoryColumn title="Procedural Memory" variant="procedural" items={proceduralRules} />
      </div>

      <div className={styles.connectors} aria-hidden="true">
        <span />
        <span />
        <span />
      </div>

      <div className={styles.bottomGrid}>
        <div className={styles.workingMemory}>
          <strong>Working Memory</strong>
          <p>Retrieved per turn via hybrid search and session catch-up.</p>
        </div>
        <div className={styles.promptSuffix}>
          <strong>System Prompt Suffix</strong>
          <p>Procedural rules loaded as style directives.</p>
        </div>
      </div>

      <div className={styles.finalConnector} aria-hidden="true" />

      <div className={styles.responseNode}>
        <strong>Response Generation</strong>
      </div>
    </section>
  );
}
