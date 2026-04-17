import React, { useState } from 'react';
import s from './MemoryArchitecture.module.css';

/* ================================================================
   MemoryArchitecture — Write/read flow + layer detail cards

   Replaces two mermaid diagrams with a single interactive component
   showing how memory flows through the system. Click a layer to
   expand its detail card.
   ================================================================ */

interface LayerDef {
  id: string;
  label: string;
  icon: string;
  color: 'semantic' | 'episodic' | 'procedural';
  writeHow: string;
  writeWhen: string;
  readHow: string;
  readInto: string;
  outputType: string;
  storage: string;
  examples: string[];
}

const LAYERS: LayerDef[] = [
  {
    id: 'semantic',
    label: 'Semantic',
    icon: '\u25C6',
    color: 'semantic',
    writeHow: 'LLM structured output via extract_semantic_facts_node',
    writeWhen: 'After every response (parallel with procedural). Small-talk gate skips greetings.',
    readHow: 'Hybrid RRF — embedding cosine + token-recall fused per turn',
    readInto: 'SemanticWorkingMemoryEntry in working_memory',
    outputType: '{ type: "semantic", evidence_quote: "..." }',
    storage: 'One row per fact, namespaced (owner_id, "semantic"). Embedding stored as BLOB.',
    examples: ['KNOWS Sarah — "my sister Sarah visited"', 'USES fluoxetine — "I take fluoxetine daily"'],
  },
  {
    id: 'episodic',
    label: 'Episodic',
    icon: '\u25CB',
    color: 'episodic',
    writeHow: 'Single LLM call via run_summarize_session',
    writeWhen: 'Once per session on /end or /exit. Produces a StoredSessionArc.',
    readHow: 'Hybrid RRF + first-turn catch-up (most recent arc injected automatically)',
    readInto: 'EpisodicWorkingMemoryEntry in working_memory',
    outputType: '{ type: "episodic", summary: "...", primary_themes: [...], is_catch_up: true }',
    storage: 'One row per arc, namespaced (owner_id, "episodic").',
    examples: ['Session 1: panic attacks, did grounding exercise', 'Session 2: work stress and sleep issues'],
  },
  {
    id: 'procedural',
    label: 'Procedural',
    icon: '\u25A0',
    color: 'procedural',
    writeHow: 'LLM structured output via extract_procedural_rules_node',
    writeWhen: 'After every response (parallel with semantic). "Did the user ask me to change how I respond?"',
    readHow: 'Full rule set loaded every turn — not query-based',
    readInto: 'System prompt suffix (always applied, regardless of recall toggle)',
    outputType: 'ProceduralProfile with rules[] and proactive_recall_enabled',
    storage: 'Single profile document per user, namespaced (owner_id, "procedural").',
    examples: ['"Don\'t suggest meditation"', '"Prefer shorter responses"'],
  },
];

function LayerCard({ layer, expanded, onToggle }: {
  layer: LayerDef;
  expanded: boolean;
  onToggle: () => void;
}) {
  return (
    <div className={`${s.layerCard} ${s[layer.color]} ${expanded ? s.layerCardExpanded : ''}`}>
      <button className={s.layerHeader} onClick={onToggle} aria-expanded={expanded}>
        <span className={s.layerIcon}>{layer.icon}</span>
        <span className={s.layerLabel}>{layer.label}</span>
        <span className={`${s.chevron} ${expanded ? s.chevronOpen : ''}`}>{'\u25BE'}</span>
      </button>

      {expanded && (
        <div className={s.layerBody}>
          {/* Examples */}
          <div className={s.examples}>
            {layer.examples.map((ex) => (
              <code key={ex} className={s.example}>{ex}</code>
            ))}
          </div>

          {/* Write/Read grid */}
          <div className={s.flowGrid}>
            <div className={s.flowCol}>
              <span className={s.flowLabel}>{'\u25B8'} Write</span>
              <span className={s.flowValue}>{layer.writeHow}</span>
              <span className={s.flowSub}>{layer.writeWhen}</span>
            </div>
            <div className={s.flowCol}>
              <span className={s.flowLabel}>{'\u25B8'} Read</span>
              <span className={s.flowValue}>{layer.readHow}</span>
              <span className={s.flowSub}>{'\u2192'} {layer.readInto}</span>
            </div>
          </div>

          {/* Output shape */}
          <div className={s.outputRow}>
            <span className={s.outputLabel}>Output shape</span>
            <code className={s.outputCode}>{layer.outputType}</code>
          </div>

          {/* Storage */}
          <div className={s.storageRow}>
            <span className={s.storageLabel}>Storage</span>
            <span className={s.storageValue}>{layer.storage}</span>
          </div>
        </div>
      )}
    </div>
  );
}

export default function MemoryArchitecture(): React.JSX.Element {
  const [expanded, setExpanded] = useState<string | null>('semantic');

  return (
    <div className={s.root}>
      {/* Flow header */}
      <div className={s.flowHeader}>
        <div className={s.flowDirection}>
          <span className={s.dirLabel}>Write</span>
          <span className={s.dirArrow}>{'\u2193'}</span>
          <span className={s.dirSub}>after response (parallel)</span>
        </div>
        <div className={s.flowDirection}>
          <span className={s.dirLabel}>Read</span>
          <span className={s.dirArrow}>{'\u2191'}</span>
          <span className={s.dirSub}>before response (per turn)</span>
        </div>
      </div>

      {/* Layer cards */}
      <div className={s.layers}>
        {LAYERS.map((layer) => (
          <LayerCard
            key={layer.id}
            layer={layer}
            expanded={expanded === layer.id}
            onToggle={() => setExpanded(p => p === layer.id ? null : layer.id)}
          />
        ))}
      </div>

      {/* Convergence */}
      <div className={s.convergence}>
        <div className={s.convConn} />
        <div className={s.convNode}>
          <span className={s.convIcon}>{'\u2192'}</span>
          <span>Response Generation</span>
        </div>
      </div>
    </div>
  );
}
