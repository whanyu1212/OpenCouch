import React from 'react';
import styles from './TurnPipeline.module.css';

/**
 * Turn pipeline as a structured outline — no arrows, no connectors.
 * Branching is communicated through indentation and color-coded borders.
 */

export default function TurnPipeline() {
  return (
    <div className={styles.outline}>
      {/* ── Entry ──────────────────────────────────────── */}
      <div className={styles.section}>
        <Node
          num="1"
          label="crisis_gate"
          desc="Every message — no exceptions. LLM-only classifier with local truth-table normalization."
          accent="var(--gate-color)"
        />
        <span className={styles.routeTag}>Command(goto=...) routes to one branch:</span>
      </div>

      {/* ── Branches ───────────────────────────────────── */}
      <div className={styles.branchGrid}>
        <div className={styles.branchCard + ' ' + styles.crisisCard}>
          <div className={styles.branchTitle}>
            <span className={styles.branchDot} style={{ background: 'var(--crisis-color)' }} />
            crisis path
          </div>
          <Node num="2a" label="crisis_resource_lookup" desc="Region-aware hotline lookup via web search grounding" accent="var(--crisis-color)" compact />
          <Node num="3a" label="CrisisAgent" desc="Crisis reply with optional resource overlay" accent="var(--crisis-color)" compact />
          <Node num="4a" label="crisis_log" desc="Always-on audit trail — writes regardless of memory mode" accent="var(--crisis-color)" compact />
        </div>

        <div className={styles.branchCard + ' ' + styles.safeCard}>
          <div className={styles.branchTitle}>
            <span className={styles.branchDot} style={{ background: 'var(--safe-color)' }} />
            safe-turn path
          </div>
          <Node num="2b" label="turn_dispatch" desc="LLM routes safe turns to memory control, grounded lookup, or therapeutic flow" accent="var(--safe-color)" compact />
          <Node num="3b" label="memory_control" desc="Operational memory replies for list/status/forget/recall/preference turns" accent="var(--safe-color)" compact />
          <Node num="3b" label="grounded_lookup" desc="Search-grounded answer for explicit factual lookup turns" accent="var(--safe-color)" compact />
          <Node num="4b" label="turn_memory_context" desc="Runtime-owned retrieval across 3 namespaces for ordinary support" accent="var(--safe-color)" compact />
          <Node num="5b" label="TherapeuticAgent" desc="Response style selection or guided-exercise handoff" accent="var(--safe-color)" compact />
        </div>
      </div>

      {/* ── Shared terminal ────────────────────────────── */}
      <div className={styles.section}>
        <div className={styles.sectionHeader}>both paths converge</div>
        <Node num="6" label="turn_finalization" desc="Append response to transcript via operator.add reducer. No I/O — no retry. Stream emits response_ready here." accent="var(--oc-accent)" />
        <div className={styles.parallelGroup}>
          <div className={styles.parallelTag}>runtime side effects after response</div>
          <div className={styles.parallelNodes}>
            <Node num="7" label="semantic extraction" desc="Candidate extraction → LLM-primary write policy → commit-now / hold / require-repetition / drop" accent="var(--oc-accent)" compact />
            <Node num="7" label="procedural extraction" desc="Style rules → immediate commit or session-end hold. Safety-conflict requests dropped." accent="var(--oc-accent)" compact />
          </div>
        </div>
      </div>

      {/* ── Session-end ────────────────────────────────── */}
      <div className={styles.section + ' ' + styles.sessionSection}>
        <div className={styles.sectionHeader}>
          <span className={styles.sessionBadge}>session end</span>
          /end · 20-min inactivity sweeper · shutdown · API end-session · voice disconnect
        </div>
        <Node num="8" label="summarize_session" desc="Episodic arc for cross-session continuity" accent="var(--session-color)" />
        <Node num="9" label="commit_session_memory" desc="Promote held semantic + procedural candidates from the active-session buffer" accent="var(--session-color)" />
      </div>

      <p className={styles.footnote}>
        Every I/O stage has <code>RetryPolicy(max_attempts=2)</code> as defense-in-depth.
      </p>
    </div>
  );
}

function Node({ num, label, desc, accent, compact }: {
  num: string;
  label: string;
  desc: string;
  accent: string;
  compact?: boolean;
}) {
  return (
    <div className={`${styles.node} ${compact ? styles.nodeCompact : ''}`} style={{ '--accent': accent } as React.CSSProperties}>
      <span className={styles.nodeNum}>{num}</span>
      <div className={styles.nodeBody}>
        <span className={styles.nodeLabel}>{label}</span>
        <span className={styles.nodeDesc}>{desc}</span>
      </div>
    </div>
  );
}
