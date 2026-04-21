import React, { useState } from 'react';
import styles from './PromptLayers.module.css';

interface Layer {
  id: string;
  depth: number;
  label: string;
  source: string;
  desc: string;
}

const LAYERS: Layer[] = [
  {
    id: 'posture',
    depth: 0,
    label: '6. Turn posture',
    source: 'build_therapeutic_response_prompt()',
    desc: 'Working memory block, recent history, current message, and step directives for multi-turn exercises.',
  },
  {
    id: 'state',
    depth: 1,
    label: '5. Dynamic state',
    source: 'procedural rules + recall toggle',
    desc: 'Active procedural style rules from memory. Recall toggle constraint when proactive content recall is disabled.',
  },
  {
    id: 'instructions',
    depth: 2,
    label: '4. Instructions',
    source: 'Inline in prompts.py per mode',
    desc: 'Mode-specific behavioral instructions. Each mode has a unique instruction block shaping response style and constraints.',
  },
  {
    id: 'modality',
    depth: 3,
    label: '3. Approach overlay',
    source: 'agent/prompts/sources/modalities/*.md',
    desc: 'Therapeutic framework overlay — MI, PFA, CBT, grief, IPT, ACT, or DBT. Selected per turn by the dispatcher. Optional.',
  },
  {
    id: 'mode',
    depth: 4,
    label: '2. Mode knowledge',
    source: 'agent/prompts/sources/response_modes/*.md',
    desc: 'Mode-specific knowledge file. Supportive, reflective, psychoeducation, closing, and guided exercise each have a dedicated file.',
  },
  {
    id: 'core',
    depth: 5,
    label: '1. Core identity',
    source: 'soul.md + identity.md + policy/',
    desc: 'Agent identity, boundaries, and privacy policy. Loaded for every mode. The immutable foundation of the prompt.',
  },
];

export default function PromptLayersVisual() {
  const [active, setActive] = useState<string | null>(null);
  const activeLayer = active ? LAYERS.find(l => l.id === active) : null;

  return (
    <div className={styles.root}>
      <p className={styles.intro}>
        Six layers composed per turn, outermost first. Click a layer to see its source.
      </p>

      <div className={styles.stack}>
        {LAYERS.map((layer) => {
          const isActive = active === layer.id;
          return (
            <button
              key={layer.id}
              className={[styles.layer, isActive ? styles.layerActive : ''].join(' ')}
              style={{
                '--layer-depth': layer.depth,
                '--layer-hue': `${140 + layer.depth * 30}`,
              } as React.CSSProperties}
              onClick={() => setActive(isActive ? null : layer.id)}
            >
              <span className={styles.layerLabel}>{layer.label}</span>
              <span className={styles.layerSource}>{layer.source}</span>
            </button>
          );
        })}
      </div>

      {activeLayer && (
        <div className={styles.detail} key={activeLayer.id}>
          <span className={styles.detailTitle}>{activeLayer.label}</span>
          <p className={styles.detailDesc}>{activeLayer.desc}</p>
          <code className={styles.detailSource}>{activeLayer.source}</code>
        </div>
      )}
    </div>
  );
}
