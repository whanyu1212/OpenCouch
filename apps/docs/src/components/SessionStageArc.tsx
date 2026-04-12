import React, { useState } from 'react';
import styles from './SessionStageArc.module.css';

interface Stage {
  id: string;
  label: string;
  icon: string;
  guidance: string;
  signals: string[];
}

const STAGES: Stage[] = [
  {
    id: 'opening',
    label: 'Opening',
    icon: '\u25CB',  // ○
    guidance: 'Orient, validate, and avoid jumping too fast into heavy structure.',
    signals: ['Early turns (1–2)', 'Orientation or supportive_conversation mode active', 'No strong deepening cues'],
  },
  {
    id: 'deepening',
    label: 'Deepening',
    icon: '\u25CF',  // ●
    guidance: 'Stay with emotional material or pattern exploration. Keep the reply contained.',
    signals: ['Turn count \u2265 3 with structured work intent', 'Guided exercise or reflection in progress', 'Active exploratory/reflective pattern'],
  },
  {
    id: 'stabilizing',
    label: 'Stabilizing',
    icon: '\u25D1',  // ◑
    guidance: 'Help consolidate. Favor grounding, integration, or next-step planning over deeper excavation.',
    signals: ['"that helped" / "that makes sense"', '"I feel calmer" / "I feel a bit better"', '"what should I do next"', 'Turn count \u2265 6 without deepening cues'],
  },
  {
    id: 'closing',
    label: 'Closing',
    icon: '\u25A0',  // ■
    guidance: 'Briefly summarize the key takeaway. At most one next step. End with a gentle landing.',
    signals: ['"before we wrap up"', '"I need to go"', '"that\'s enough for today"', '"can you summarize this"'],
  },
];

export default function SessionStageArc() {
  const [active, setActive] = useState<string | null>(null);
  const activeStage = active ? STAGES.find(s => s.id === active) ?? null : null;

  return (
    <div className={styles.root}>
      {/* Arc bar */}
      <div className={styles.arc}>
        {STAGES.map((stage, i) => (
          <React.Fragment key={stage.id}>
            <button
              className={[styles.stage, active === stage.id ? styles.stageActive : ''].join(' ')}
              onClick={() => setActive(p => p === stage.id ? null : stage.id)}
            >
              <span className={styles.stageIcon}>{stage.icon}</span>
              <span className={styles.stageLabel}>{stage.label}</span>
            </button>
            {i < STAGES.length - 1 && (
              <div className={styles.arcConnector}>
                <div className={styles.arcLine} />
                <div className={styles.arcArrow}>{'\u203A'}</div>
              </div>
            )}
          </React.Fragment>
        ))}
      </div>

      {/* Detail */}
      {activeStage && (
        <div className={styles.detail} key={activeStage.id}>
          <div className={styles.detailHeader}>
            <span className={styles.detailIcon}>{activeStage.icon}</span>
            <span className={styles.detailTitle}>{activeStage.label}</span>
            <button className={styles.detailClose} onClick={() => setActive(null)}>&#10005;</button>
          </div>
          <div className={styles.detailBody}>
            <div className={styles.detailSection}>
              <span className={styles.sectionKey}>Guidance</span>
              <p className={styles.sectionVal}>{activeStage.guidance}</p>
            </div>
            <div className={styles.detailSection}>
              <span className={styles.sectionKey}>Detected by</span>
              <ul className={styles.signalList}>
                {activeStage.signals.map((s, i) => (
                  <li key={i} className={styles.signalItem}>{s}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
