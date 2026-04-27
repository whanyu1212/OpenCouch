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
    source: 'build_*_system_prompt() + format_recent_history()',
    desc: 'Working memory block, recent history, current message, and step directives for multi-turn exercises (including the pinned exercise_modality).',
  },
  {
    id: 'state',
    depth: 1,
    label: '5. Procedural rules',
    source: 'procedural_profile.procedural_rules',
    desc: 'Active style directives from the user\'s procedural profile, injected as a system prompt suffix. Always applied — the recall toggle controls content recall, not directive application.',
  },
  {
    id: 'instructions',
    depth: 2,
    label: '4. Response instructions',
    source: 'agent/therapeutic/prompting/instructions.py',
    desc: 'Per-response-style behavioral instructions assembled by build_*_system_prompt(). Supportive, reflective, clarifying, psychoeducation, technique, guided_exercise, and closing each have their own instruction block.',
  },
  {
    id: 'approach',
    depth: 3,
    label: '3. Approach overlay',
    source: 'agent/prompts/sources/modalities/*.md',
    desc: 'Therapeutic framework overlay — MI, CBT (with cbt_arc), ACT, DBT skills, grief support, IPT, or PFA. Selected per turn as therapeutic_approach by the LLM dispatcher.',
  },
  {
    id: 'response',
    depth: 4,
    label: '2. Response knowledge',
    source: 'agent/prompts/sources/response_modes/*.md',
    desc: 'Response-style knowledge file. Seven files: support, reflection, psychoeducation, guided_exercise, closing, crisis_response, safety_check. Technique responses use the core sources plus the active therapeutic approach.',
  },
  {
    id: 'core',
    depth: 5,
    label: '1. Core identity',
    source: 'CORE_SOURCES — soul.md + identity.md + policy/boundaries.md + policy/privacy.md',
    desc: 'Agent identity, boundaries, and privacy policy. Loaded via compose_sources() for every response. The immutable foundation of every system prompt.',
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
