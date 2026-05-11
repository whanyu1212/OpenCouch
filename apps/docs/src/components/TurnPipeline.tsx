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
          label="crisis_gate_node"
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
          <Node num="2a" label="crisis_resource_lookup_node" desc="Region-aware hotline lookup via web search grounding" accent="var(--crisis-color)" compact />
          <Node num="3a" label="crisis_response_node" desc="Crisis reply with optional resource overlay" accent="var(--crisis-color)" compact />
          <Node num="4a" label="crisis_log_node" desc="Always-on audit trail — writes regardless of memory mode" accent="var(--crisis-color)" compact />
        </div>

        <div className={styles.branchCard + ' ' + styles.safeCard}>
          <div className={styles.branchTitle}>
            <span className={styles.branchDot} style={{ background: 'var(--safe-color)' }} />
            therapeutic path
          </div>
          <Node num="2b" label="turn_dispatch_node" desc="LLM routes safe turns to memory control, grounded lookup, or therapeutic flow" accent="var(--safe-color)" compact />
          <Node num="3b" label="grounded_answer_node" desc="Search-grounded answer for explicit factual lookup turns" accent="var(--safe-color)" compact />
          <Node num="4b" label="load_memory_node" desc="Hybrid RRF retrieval across 3 namespaces" accent="var(--safe-color)" compact />
          <Node num="5b" label="therapeutic_subgraph" desc="LLM dispatcher → 1 of 7 response styles × 7 approaches" accent="var(--safe-color)" compact />
        </div>
      </div>

      {/* ── Shared terminal ────────────────────────────── */}
      <div className={styles.section}>
        <div className={styles.sectionHeader}>both paths converge</div>
        <Node num="6" label="finalize_turn_node" desc="Append response to transcript via operator.add reducer. No I/O — no retry. Stream emits response_ready here." accent="var(--oc-accent)" />
        <div className={styles.parallelGroup}>
          <div className={styles.parallelTag}>parallel fan-out</div>
          <div className={styles.parallelNodes}>
            <Node num="7" label="extract_semantic_facts_node" desc="Candidate extraction → deterministic write policy → commit-now / hold / require-repetition / drop" accent="var(--oc-accent)" compact />
            <Node num="7" label="extract_procedural_rules_node" desc="Style rules → immediate commit (explicit) or session-end hold (implicit). Safety-conflict requests dropped." accent="var(--oc-accent)" compact />
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
        Every I/O node has <code>RetryPolicy(max_attempts=2)</code> as defense-in-depth.
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
