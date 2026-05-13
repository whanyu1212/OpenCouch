import React from 'react';
import styles from './FlowVisualizer.module.css';

type FlowNode = {
  id: string;
  label: string;
  variant?: 'default' | 'warm' | 'safe' | 'danger';
};

const preGate: FlowNode[] = [
  {id: 'user', label: 'User message', variant: 'default'},
  {id: 'gate', label: 'Crisis Gate', variant: 'danger'},
];

// Safe branch — one turn dispatcher runs before memory loads.
// Most turns end up at the therapeutic subgraph; memory commands
// ("forget #3", "recall off") and factual lookups ("look up the
// crisis line for X") short-circuit to dedicated reply nodes.
const safePath: FlowNode[] = [
  {id: 'turn_dispatch', label: 'Turn Dispatch', variant: 'default'},
  {id: 'memory_control', label: 'Memory Control / Grounded Lookup', variant: 'default'},
  {id: 'load', label: 'Load Memory', variant: 'warm'},
  {id: 'therapy', label: 'Therapeutic Subgraph', variant: 'safe'},
];

// Crisis branch — region-aware hotline lookup runs before the reply
// so any verified resources land in the same crisis turn.
const crisisPath: FlowNode[] = [
  {id: 'resources', label: 'Crisis Resource Lookup', variant: 'danger'},
  {id: 'crisis', label: 'Crisis Response', variant: 'danger'},
  {id: 'log', label: 'Crisis Log', variant: 'danger'},
];

const postMerge: FlowNode[] = [
  {id: 'finalize', label: 'Finalize Turn', variant: 'default'},
  {id: 'extract', label: 'Runtime Extract Facts & Rules', variant: 'warm'},
];

function Node({label, variant = 'default'}: {label: string; variant?: FlowNode['variant']}): JSX.Element {
  const variantClass = {
    default: styles.nodeDefault,
    warm: styles.nodeWarm,
    safe: styles.nodeSafe,
    danger: styles.nodeDanger,
  }[variant];

  return (
    <div className={`${styles.node} ${variantClass}`}>
      <span>{label}</span>
    </div>
  );
}

function Connector(): JSX.Element {
  return <div className={styles.connector} aria-hidden="true" />;
}

export default function FlowVisualizer(): JSX.Element {
  return (
    <section className={styles.container} aria-label="How a turn flows">
      <div className={styles.preGate}>
        {preGate.map((node, index) => (
          <React.Fragment key={node.id}>
            <Node label={node.label} variant={node.variant} />
            {index < preGate.length - 1 ? <Connector /> : null}
          </React.Fragment>
        ))}
      </div>

      <div className={styles.branchSplit} aria-hidden="true">
        <div className={styles.branchLine} />
        <div className={styles.branchLine} />
      </div>

      <div className={styles.branches}>
        <div className={styles.branchColumn}>
          <div className={styles.branchLabelSafe}>therapeutic path</div>
          {safePath.map((node, index) => (
            <React.Fragment key={node.id}>
              <Node label={node.label} variant={node.variant} />
              {index < safePath.length - 1 ? <Connector /> : null}
            </React.Fragment>
          ))}
        </div>

        <div className={styles.branchColumn}>
          <div className={styles.branchLabelDanger}>crisis path</div>
          {crisisPath.map((node, index) => (
            <React.Fragment key={node.id}>
              <Node label={node.label} variant={node.variant} />
              {index < crisisPath.length - 1 ? <Connector /> : null}
            </React.Fragment>
          ))}
        </div>
      </div>

      <div className={styles.mergeConnector} aria-hidden="true" />

      <div className={styles.postMerge}>
        {postMerge.map((node, index) => (
          <React.Fragment key={node.id}>
            <Node label={node.label} variant={node.variant} />
            {index < postMerge.length - 1 ? <Connector /> : null}
          </React.Fragment>
        ))}
      </div>
    </section>
  );
}
