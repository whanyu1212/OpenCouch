import React, { useState, useEffect, useRef } from 'react';
import s from './GraphTopology.module.css';

/* ================================================================
   GraphTopology — Animated agent graph with branching + parallel lanes

   Auto-plays the message flow through the safety-first topology.
   Shows the crisis/therapeutic branch and the parallel extractor
   fan-out as distinct execution lanes.
   ================================================================ */

interface NodeState {
  status: 'idle' | 'active' | 'done';
}

/*
  Animation steps (therapeutic path):
  0: idle
  1: crisis_gate active
  2: crisis_gate done, load_memory active
  3: load_memory done, therapeutic active
  4: therapeutic done, finalize active
  5: finalize done, both extractors active (parallel)
  6: extractors done, output ready
*/
const TOTAL_STEPS = 7;
const STEP_DELAY = 1000;
const RESTART_DELAY = 3500;

type NodeId =
  | 'start'
  | 'crisis_gate'
  | 'crisis_response'
  | 'crisis_log'
  | 'load_memory'
  | 'therapeutic'
  | 'finalize'
  | 'extract_facts'
  | 'extract_procedural'
  | 'end';

function getNodeStates(step: number): Record<NodeId, NodeState> {
  const idle: NodeState = { status: 'idle' };
  const active: NodeState = { status: 'active' };
  const done: NodeState = { status: 'done' };

  const base: Record<NodeId, NodeState> = {
    start: idle,
    crisis_gate: idle,
    crisis_response: idle,
    crisis_log: idle,
    load_memory: idle,
    therapeutic: idle,
    finalize: idle,
    extract_facts: idle,
    extract_procedural: idle,
    end: idle,
  };

  if (step >= 1) { base.start = done; base.crisis_gate = step === 1 ? active : done; }
  if (step >= 2) { base.load_memory = step === 2 ? active : done; }
  if (step >= 3) { base.therapeutic = step === 3 ? active : done; }
  if (step >= 4) { base.finalize = step === 4 ? active : done; }
  if (step >= 5) {
    base.extract_facts = step === 5 ? active : done;
    base.extract_procedural = step === 5 ? active : done;
  }
  if (step >= 6) { base.end = done; }

  return base;
}

function NodeBox({
  label,
  sub,
  state,
  variant = 'default',
}: {
  label: string;
  sub: string;
  state: NodeState;
  variant?: 'default' | 'safety' | 'memory' | 'therapy' | 'finalize' | 'extract' | 'terminal';
}) {
  const statusIcon =
    state.status === 'active' ? (
      <span className={s.spinner} />
    ) : state.status === 'done' ? (
      <span className={s.check}>{'\u2713'}</span>
    ) : null;

  return (
    <div
      className={[
        s.node,
        s[`node_${variant}`],
        state.status === 'active' ? s.nodeActive : '',
        state.status === 'done' ? s.nodeDone : '',
        state.status === 'idle' ? s.nodeIdle : '',
      ].join(' ')}
    >
      <div className={s.nodeHeader}>
        {statusIcon}
        <span className={s.nodeLabel}>{label}</span>
      </div>
      <span className={s.nodeSub}>{sub}</span>
    </div>
  );
}

function Connector({
  active,
  variant = 'default',
}: {
  active: boolean;
  variant?: 'default' | 'branch' | 'parallel';
}) {
  return (
    <div
      className={[
        s.conn,
        active ? s.connActive : '',
        variant === 'branch' ? s.connBranch : '',
        variant === 'parallel' ? s.connParallel : '',
      ].join(' ')}
    />
  );
}

export default function GraphTopology(): React.JSX.Element {
  const [step, setStep] = useState(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let current = 0;

    const tick = () => {
      current++;
      if (current >= TOTAL_STEPS) {
        timerRef.current = setTimeout(() => {
          current = 0;
          setStep(0);
          timerRef.current = setTimeout(tick, STEP_DELAY);
        }, RESTART_DELAY);
        return;
      }
      setStep(current);
      timerRef.current = setTimeout(tick, STEP_DELAY);
    };

    timerRef.current = setTimeout(tick, 800);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  const ns = getNodeStates(step);
  const isRunning = step > 0 && step < TOTAL_STEPS - 1;
  const progress = Math.round((step / (TOTAL_STEPS - 1)) * 100);

  return (
    <div className={s.container}>
      {/* Header */}
      <div className={s.header}>
        <span className={s.headerTitle}>build_agent_workflow()</span>
        <span className={s.headerBadges}>
          <span className={s.headerBadge}>RetryPolicy</span>
          <span className={s.headerBadge}>operator.add</span>
          <span className={s.headerBadge}>_merge_dicts</span>
        </span>
        <span className={s.headerStatus}>
          <span className={isRunning ? s.runDot : s.pausedDot} />
          {step === 0 ? 'idle' : step >= TOTAL_STEPS - 1 ? 'complete' : 'executing'}
        </span>
      </div>

      {/* Graph body */}
      <div className={s.body}>
        {/* ── Entry ── */}
        <div className={s.spine}>
          <NodeBox
            label="START"
            sub="build_initial_state"
            state={ns.start}
            variant="terminal"
          />
          <Connector active={step >= 1} />

          {/* ── Crisis gate ── */}
          <NodeBox
            label="crisis_gate_node"
            sub="safety first"
            state={ns.crisis_gate}
            variant="safety"
          />
        </div>

        {/* ── Branch split ── */}
        <div className={s.branchZone}>
          <div className={s.branchHeader}>
            <span className={s.branchLabel}>Command(goto=...)</span>
          </div>
          <div className={s.branchLanes}>
            {/* Crisis branch (dimmed — we animate the therapeutic path) */}
            <div className={s.branchLane}>
              <div className={s.branchLaneHeader}>
                <span className={s.branchDot + ' ' + s.dotCrisis} />
                <span className={s.branchLaneLabel}>Crisis branch</span>
              </div>
              <div className={s.branchNodeStack}>
                <NodeBox
                  label="crisis_response"
                  sub="PFA overlay"
                  state={ns.crisis_response}
                  variant="safety"
                />
                <Connector active={false} variant="branch" />
                <NodeBox
                  label="crisis_log"
                  sub="audit record"
                  state={ns.crisis_log}
                  variant="safety"
                />
              </div>
            </div>

            {/* Therapeutic branch (active) */}
            <div className={s.branchLane}>
              <div className={s.branchLaneHeader}>
                <span className={s.branchDot + ' ' + s.dotTherapeutic} />
                <span className={s.branchLaneLabel}>Therapeutic branch</span>
              </div>
              <div className={s.branchNodeStack}>
                <NodeBox
                  label="load_memory"
                  sub="hybrid RRF retrieval"
                  state={ns.load_memory}
                  variant="memory"
                />
                <Connector active={step >= 3} variant="branch" />
                <NodeBox
                  label="therapeutic_subgraph"
                  sub="dispatcher + 7 response styles"
                  state={ns.therapeutic}
                  variant="therapy"
                />
              </div>
            </div>
          </div>
        </div>

        {/* ── Converge ── */}
        <div className={s.spine}>
          <Connector active={step >= 4} />
          <NodeBox
            label="finalize_turn_node"
            sub="operator.add reducer"
            state={ns.finalize}
            variant="finalize"
          />
        </div>

        {/* ── Parallel extractor fan-out ── */}
        <div className={s.parallelZone}>
          <div className={s.parallelHeader}>
            <span className={s.parallelLabel}>parallel fan-out</span>
          </div>
          <div className={s.parallelLanes}>
            <NodeBox
              label="extract_facts"
              sub="semantic LLM"
              state={ns.extract_facts}
              variant="extract"
            />
            <NodeBox
              label="extract_procedural"
              sub="style rules LLM"
              state={ns.extract_procedural}
              variant="extract"
            />
          </div>
          <div className={s.parallelReducerTag}>
            diagnostics via _merge_dicts reducer
          </div>
        </div>

        {/* ── End ── */}
        <div className={s.spine}>
          <Connector active={step >= 6} />
          <NodeBox
            label="END"
            sub="AgentOutput"
            state={ns.end}
            variant="terminal"
          />
        </div>
      </div>

      {/* Footer */}
      <div className={s.footer}>
        <span className={s.footerLabel}>
          {step >= TOTAL_STEPS - 1
            ? 'turn complete'
            : step === 0
              ? 'waiting for message'
              : `step ${step}/${TOTAL_STEPS - 1}`}
        </span>
        <div className={s.progressTrack}>
          <div className={s.progressFill} style={{ width: `${progress}%` }} />
        </div>
      </div>
    </div>
  );
}
