import React, { useState } from 'react';
import styles from './SystemAtAGlance.module.css';

type RouteId = 'therapeutic' | 'memory_control' | 'grounded_lookup' | 'guided_exercise' | 'crisis';

interface RouteDef {
  id: RouteId;
  name: string;
  tag: string;
  color: string;
  desc: string;
  readsMemory: boolean;
  writesMemory: boolean;
  bypassesTriage: boolean;
  codeRef: string;
  agentName: string;
  isAgent: boolean;
  details: { label: string; value: string }[];
}

const ROUTES: RouteDef[] = [
  {
    id: 'therapeutic',
    name: 'Therapeutic',
    tag: 'DEFAULT',
    color: '#14b8a6', // teal-500
    desc: 'The primary conversational engine. Evaluates clinical approaches and generates empathetic, style-matched responses.',
    readsMemory: true,
    writesMemory: true,
    bypassesTriage: false,
    codeRef: 'agent/specialists/therapeutic.py',
    agentName: 'TherapeuticAgent',
    isAgent: true,
    details: [
      { label: 'Styles', value: 'Supportive, Reflective, Clarifying, Psychoeducation, Technique, Closing' },
      { label: 'Approaches', value: 'CBT, ACT, DBT skills, MI, IPT, Grief, PFA' }
    ]
  },
  {
    id: 'memory_control',
    name: 'Memory Control',
    tag: 'EXPLICIT',
    color: '#8b5cf6', // violet-500
    desc: 'Direct manipulation of the semantic layer. Bypasses therapeutic generation for direct state changes.',
    readsMemory: true,
    writesMemory: true,
    bypassesTriage: false,
    codeRef: 'agent/runtime/memory.py',
    agentName: 'App Runtime Tool',
    isAgent: false,
    details: [
      { label: 'Triggers', value: '"forget that", "recall off", "remember..."' },
      { label: 'Action', value: 'Executes operation & replies neutrally' }
    ]
  },
  {
    id: 'grounded_lookup',
    name: 'Grounded Lookup',
    tag: 'TOOL',
    color: '#0ea5e9', // sky-500
    desc: 'Search-grounded LLM retrieval for fact-checking or looking up resources. No therapeutic framing.',
    readsMemory: false,
    writesMemory: false,
    bypassesTriage: false,
    codeRef: 'agent/runtime/lookup.py',
    agentName: 'App Runtime Tool',
    isAgent: false,
    details: [
      { label: 'Triggers', value: '"verify...", "look up the latest..."' },
      { label: 'Format', value: 'Returns cited sources' }
    ]
  },
  {
    id: 'guided_exercise',
    name: 'Guided Exercise',
    tag: 'PINNED',
    color: '#f59e0b', // amber-500
    desc: 'Stateful therapeutic workflow. Pins the approach in exercise_state to prevent drift during side-turns.',
    readsMemory: true,
    writesMemory: true,
    bypassesTriage: false,
    codeRef: 'agent/specialists/guided_exercise.py',
    agentName: 'GuidedExerciseAgent',
    isAgent: true,
    details: [
      { label: 'Triggers', value: 'Explicit request to start/continue exercise' },
      { label: 'State', value: 'Maintains step-by-step progress' }
    ]
  },
  {
    id: 'crisis',
    name: 'Crisis Gate',
    tag: 'OVERRIDE',
    color: '#ef4444', // red-500
    desc: 'Pre-triage audit layer. Detects imminent harm and overrides standard routing.',
    readsMemory: false,
    writesMemory: false,
    bypassesTriage: true,
    codeRef: 'agent/specialists/crisis.py',
    agentName: 'CrisisAgent',
    isAgent: true,
    details: [
      { label: 'Action', value: 'Hotline lookup & immediate off-ramp' },
      { label: 'Security', value: 'Always-on, skips standard memory' }
    ]
  }
];

export default function SystemAtAGlance() {
  const [activeId, setActiveId] = useState<RouteId>('therapeutic');
  const activeRoute = ROUTES.find(r => r.id === activeId)!;
  const isCrisis = activeId === 'crisis';
  const defaultColor = 'var(--oc-text-muted)';

  return (
    <div className={styles.wrapper} style={{ '--theme-color': activeRoute.color } as any}>
      <div className={styles.header}>
        <div className={styles.eyebrow}>Architecture Overview</div>
        <h3 className={styles.title}>The OpenCouch Pipeline</h3>
        <p className={styles.subtitle}>
          Select a route to trace the chronological execution of a single turn.
        </p>
      </div>

      <div className={styles.layout}>
        {/* Sidebar: Route Selection */}
        <div className={styles.sidebar}>
          <div className={styles.sidebarTitle}>Execution Routes</div>
          <div className={styles.routeList}>
            {ROUTES.map(r => (
              <button
                key={r.id}
                className={`${styles.routeBtn} ${activeId === r.id ? styles.activeRouteBtn : ''}`}
                onClick={() => setActiveId(r.id)}
                style={{ '--route-color': r.color } as any}
              >
                <span className={styles.routeIndicator} />
                <span className={styles.routeName}>{r.name}</span>
                {r.id === 'crisis' && <span className={styles.alertIcon}>🚨</span>}
              </button>
            ))}
          </div>
        </div>

        {/* Main Area: Graph & Details */}
        <div className={styles.main}>

          {/* Horizontal Linear Graph */}
          <div className={styles.graphContainer}>
            <GraphNode icon="💬" title="Message" active={true} />
            <GraphLine active={true} color={isCrisis ? activeRoute.color : defaultColor} />

            <GraphNode icon="🛡️" title="Crisis Gate" active={true} color={isCrisis ? activeRoute.color : undefined} />
            <GraphLine active={!isCrisis} color={!isCrisis ? defaultColor : undefined} />

            <GraphNode icon="🔀" title="Turn Triage" active={!isCrisis} />
            <GraphLine active={!isCrisis} color={activeRoute.color} />

            <GraphNode
              icon={activeRoute.isAgent ? '🤖' : '⚙️'}
              title={activeRoute.name}
              active={true}
              color={activeRoute.color}
            />
            <GraphLine active={true} color={activeRoute.color} />

            <GraphNode icon="📦" title="Finalize" active={true} color={activeRoute.color} />
            <GraphLine active={activeRoute.writesMemory || activeRoute.readsMemory} color={activeRoute.color} />

            <GraphNode
              icon="💾"
              title="Memory"
              active={activeRoute.writesMemory || activeRoute.readsMemory}
              color={activeRoute.color}
            />
          </div>

          {/* Details Card */}
          <div className={styles.activeCard}>
            <div className={styles.cardHeader}>
              <div className={styles.cardTitleGroup}>
                <h4 className={styles.cardTitle}>{activeRoute.name}</h4>
                <span className={styles.cardAgent}>
                  {activeRoute.isAgent ? 'Autonomous Agent' : 'Deterministic Tool'} · {activeRoute.agentName}
                </span>
              </div>
              <span className={styles.cardTag}>{activeRoute.tag}</span>
            </div>

            <p className={styles.cardDesc}>{activeRoute.desc}</p>

            <div className={styles.cardDetailsGrid}>
              {activeRoute.details.map((d, i) => (
                <div key={i} className={styles.detailItem}>
                  <span className={styles.detailLabel}>{d.label}</span>
                  <span className={styles.detailValue}>{d.value}</span>
                </div>
              ))}
              <div className={styles.detailItem} style={{ gridColumn: '1 / -1' }}>
                <span className={styles.detailLabel}>Source Code</span>
                <code className={styles.detailCode}>{activeRoute.codeRef}</code>
              </div>
            </div>

            {/* Memory Impact */}
            <div className={styles.memorySection}>
              <div className={styles.memoryLabel}>Memory Impact</div>
              <div className={styles.memoryGrid}>
                <MemoryCell label="Semantic" desc="Facts & Entities" reads={activeRoute.readsMemory} writes={activeRoute.writesMemory} />
                <MemoryCell label="Episodic" desc="Session Arcs" reads={activeRoute.readsMemory} writes={activeRoute.writesMemory} />
                <MemoryCell label="Procedural" desc="Style Rules" reads={activeRoute.readsMemory} writes={activeRoute.writesMemory} />
              </div>
            </div>

          </div>
        </div>
      </div>
    </div>
  );
}

function GraphNode({ icon, title, active, color }: { icon: string, title: string, active: boolean, color?: string }) {
  return (
    <div
      className={styles.graphNode}
      data-active={active}
      style={{ '--node-color': color || 'var(--oc-text-muted)' } as any}
    >
      <div className={styles.graphIcon}>{icon}</div>
      <div className={styles.graphTitle}>{title}</div>
    </div>
  );
}

function GraphLine({ active, color }: { active: boolean, color?: string }) {
  return (
    <div
      className={styles.graphLine}
      data-active={active}
      style={{ '--line-color': color || 'var(--oc-border)' } as any}
    >
      <div className={styles.graphLineFill} />
    </div>
  );
}

function MemoryCell({ label, desc, reads, writes }: { label: string, desc: string, reads: boolean, writes: boolean }) {
  const active = reads || writes;
  return (
    <div className={styles.memoryCell} data-active={active}>
      <div className={styles.memoryCellHeader}>
        <span className={styles.memoryCellLabel}>{label}</span>
        <div className={styles.ioBadges}>
          <span className={styles.ioBadge} data-on={reads}>READ</span>
          <span className={styles.ioBadge} data-on={writes}>WRITE</span>
        </div>
      </div>
      <span className={styles.memoryDesc}>{desc}</span>
    </div>
  );
}
