import React, { useState, useEffect, useRef } from 'react';
import s from './GraphTopology.module.css';

interface NodeState {
  status: 'idle' | 'active' | 'done';
}

/*
  Animation steps (ordinary therapeutic path):
  0: idle
  1: crisis_gate active
  2: turn_dispatch active
  3: load_memory active
  4: therapeutic_subgraph active
  5: finalize active
  6: graph END
*/
const TOTAL_STEPS = 7;
const STEP_DELAY = 1000;
const RESTART_DELAY = 3500;

type NodeId =
  | 'start'
  | 'crisis_gate'
  | 'crisis_resource_lookup'
  | 'crisis_response'
  | 'crisis_log'
  | 'turn_dispatch'
  | 'memory_control'
  | 'grounded_answer'
  | 'load_memory'
  | 'therapeutic'
  | 'finalize'
  | 'end';

function getNodeStates(step: number): Record<NodeId, NodeState> {
  const idle: NodeState = { status: 'idle' };
  const active: NodeState = { status: 'active' };
  const done: NodeState = { status: 'done' };

  const base: Record<NodeId, NodeState> = {
    start: idle,
    crisis_gate: idle,
    crisis_resource_lookup: idle,
    crisis_response: idle,
    crisis_log: idle,
    turn_dispatch: idle,
    memory_control: idle,
    grounded_answer: idle,
    load_memory: idle,
    therapeutic: idle,
    finalize: idle,
    end: idle,
  };

  if (step >= 1) { base.start = done; base.crisis_gate = step === 1 ? active : done; }
  if (step >= 2) { base.turn_dispatch = step === 2 ? active : done; }
  if (step >= 3) { base.load_memory = step === 3 ? active : done; }
  if (step >= 4) { base.therapeutic = step === 4 ? active : done; }
  if (step >= 5) { base.finalize = step === 5 ? active : done; }
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
      <div className={s.header}>
        <span className={s.headerTitle}>OpenAITextRuntime</span>
        <span className={s.headerBadges}>
          <span className={s.headerBadge}>runtime routing</span>
          <span className={s.headerBadge}>SDK Runner</span>
          <span className={s.headerBadge}>SDK session memory</span>
        </span>
        <span className={s.headerStatus}>
          <span className={isRunning ? s.runDot : s.pausedDot} />
          {step === 0 ? 'idle' : step >= TOTAL_STEPS - 1 ? 'complete' : 'executing'}
        </span>
      </div>

      <div className={s.body}>
        <div className={s.spine}>
          <NodeBox
            label="START"
            sub="build_initial_state"
            state={ns.start}
            variant="terminal"
          />
          <Connector active={step >= 1} />
          <NodeBox
            label="crisis_gate"
            sub="LLM-only safety classifier"
            state={ns.crisis_gate}
            variant="safety"
          />
        </div>

        <div className={s.branchZone}>
          <div className={s.branchHeader}>
            <span className={s.branchLabel}>crisis gate route</span>
          </div>
          <div className={s.branchLanes}>
            <div className={s.branchLane}>
              <div className={s.branchLaneHeader}>
                <span className={s.branchDot + ' ' + s.dotCrisis} />
                <span className={s.branchLaneLabel}>Crisis branch</span>
              </div>
              <div className={s.branchNodeStack}>
                <NodeBox
                  label="crisis_resource_lookup"
                  sub="location-aware resources"
                  state={ns.crisis_resource_lookup}
                  variant="safety"
                />
                <Connector active={false} variant="branch" />
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

            <div className={s.branchLane}>
              <div className={s.branchLaneHeader}>
                <span className={s.branchDot + ' ' + s.dotTherapeutic} />
                <span className={s.branchLaneLabel}>Safe-turn branch</span>
              </div>
              <div className={s.branchNodeStack}>
                <NodeBox
                  label="turn_dispatch"
                  sub="memory · lookup · support"
                  state={ns.turn_dispatch}
                  variant="default"
                />
                <div className={s.routeOptions}>
                  <NodeBox
                    label="memory_control"
                    sub="saved-memory command"
                    state={ns.memory_control}
                    variant="default"
                  />
                  <NodeBox
                    label="grounded_lookup"
                    sub="factual lookup"
                    state={ns.grounded_answer}
                    variant="default"
                  />
                </div>
                <Connector active={step >= 3} variant="branch" />
                <NodeBox
                  label="load_memory"
                  sub="hybrid RRF retrieval"
                  state={ns.load_memory}
                  variant="memory"
                />
                <Connector active={step >= 4} variant="branch" />
                <NodeBox
                  label="TherapeuticAgent"
                  sub="response | guided exercise"
                  state={ns.therapeutic}
                  variant="therapy"
                />
              </div>
            </div>
          </div>
        </div>

        <div className={s.spine}>
          <Connector active={step >= 5} />
          <NodeBox
            label="finalization"
            sub="assistant turn"
            state={ns.finalize}
            variant="finalize"
          />
          <Connector active={step >= 6} />
          <NodeBox
            label="END"
            sub="runtime response returned"
            state={ns.end}
            variant="terminal"
          />
        </div>

      </div>

      <div className={s.footer}>
        <span className={s.footerLabel}>
          {step >= TOTAL_STEPS - 1
            ? 'runtime side effects'
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
