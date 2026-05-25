import React, { useState } from 'react';
import styles from './MemoryLayers.module.css';

type Variant = 'semantic' | 'episodic' | 'procedural';
type Destination = 'working' | 'prompt';

const DESTINATION_OF: Record<Variant, Destination> = {
  semantic: 'working',
  episodic: 'working',
  procedural: 'prompt',
};

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
  onHover,
  isDimmed,
  isHovered,
}: {
  title: string;
  variant: Variant;
  items: string[];
  onHover: (v: Variant | null) => void;
  isDimmed: boolean;
  isHovered: boolean;
}): JSX.Element {
  const variantClass = {
    semantic: styles.semantic,
    episodic: styles.episodic,
    procedural: styles.procedural,
  }[variant];

  return (
    <article
      className={`${styles.column} ${variantClass}`}
      data-dimmed={isDimmed}
      data-hovered={isHovered}
      onMouseEnter={() => onHover(variant)}
      onMouseLeave={() => onHover(null)}
      onFocus={() => onHover(variant)}
      onBlur={() => onHover(null)}
      tabIndex={0}
      aria-label={`${title}, feeds ${DESTINATION_OF[variant] === 'working' ? 'Working Memory' : 'System Prompt Suffix'}`}
    >
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
  const [hovered, setHovered] = useState<Variant | null>(null);
  const activeDestination = hovered ? DESTINATION_OF[hovered] : null;

  return (
    <section
      className={styles.container}
      aria-label="Three memory layers"
      data-has-hover={hovered !== null}
    >
      <div className={styles.topGrid}>
        <MemoryColumn
          title="Semantic Memory"
          variant="semantic"
          items={semanticFacts}
          onHover={setHovered}
          isHovered={hovered === 'semantic'}
          isDimmed={hovered !== null && hovered !== 'semantic'}
        />
        <MemoryColumn
          title="Episodic Memory"
          variant="episodic"
          items={episodicSessions}
          onHover={setHovered}
          isHovered={hovered === 'episodic'}
          isDimmed={hovered !== null && hovered !== 'episodic'}
        />
        <MemoryColumn
          title="Procedural Memory"
          variant="procedural"
          items={proceduralRules}
          onHover={setHovered}
          isHovered={hovered === 'procedural'}
          isDimmed={hovered !== null && hovered !== 'procedural'}
        />
      </div>

      <div className={styles.connectors} aria-hidden="true">
        <span data-active={hovered === 'semantic'} data-dimmed={hovered !== null && hovered !== 'semantic'} />
        <span data-active={hovered === 'episodic'} data-dimmed={hovered !== null && hovered !== 'episodic'} />
        <span data-active={hovered === 'procedural'} data-dimmed={hovered !== null && hovered !== 'procedural'} />
      </div>

      <div className={styles.bottomGrid}>
        <div
          className={styles.workingMemory}
          data-active={activeDestination === 'working'}
          data-dimmed={activeDestination === 'prompt'}
        >
          <strong>Working Memory</strong>
          <p>Retrieved per turn via hybrid search and session catch-up.</p>
        </div>
        <div
          className={styles.promptSuffix}
          data-active={activeDestination === 'prompt'}
          data-dimmed={activeDestination === 'working'}
        >
          <strong>System Prompt Suffix</strong>
          <p>Procedural rules loaded as style directives.</p>
        </div>
      </div>

      <div className={styles.finalConnector} aria-hidden="true" />

      <div className={styles.responseNode}>
        <strong>Response Generation</strong>
      </div>

      <p className={styles.hint} aria-live="polite">
        {hovered
          ? `${hovered.charAt(0).toUpperCase() + hovered.slice(1)} memory → ${activeDestination === 'working' ? 'Working Memory' : 'System Prompt Suffix'}`
          : 'Hover a layer to see where it lands in the prompt.'}
      </p>
    </section>
  );
}
