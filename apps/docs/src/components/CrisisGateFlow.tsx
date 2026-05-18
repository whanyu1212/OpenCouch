import React, { useEffect, useMemo, useRef, useState } from 'react';
import styles from './CrisisGateFlow.module.css';

/* ================================================================
   CrisisGateFlow — Interactive walkthrough of the crisis gate.

   Each example input triggers a precomputed trace through the
   two-step classifier:

     1. LLM classifier (structured output)
     2. Truth-table normalization

   The right panel shows the resulting state delta (level, route,
   classifier_path, override_kind, response_style).

   This is documentation, not a live classifier — every trace is
   hand-authored to mirror what `agent/runtime/guardrails/assessment.py` and
   `agent/runtime/guardrails/service.py` would actually emit.
   ================================================================ */

type ClassifierPath = 'llm_primary';

type OverrideOutcome = 'none';

type Stage =
  | { id: 'llm'; status: 'firing' | 'success' | 'failed' | 'skipped'; rawLevel?: 0 | 1 | 2 | 3; note: string }
  | { id: 'normalize'; status: 'firing' | 'done'; level: 0 | 1 | 2 | 3; note: string };

interface FinalState {
  level: 0 | 1 | 2 | 3;
  route: 'crisis' | 'therapeutic';
  classifierPath: ClassifierPath;
  overrideKind: OverrideOutcome;
  llmFailureOccurred: boolean;
  responseStyle: string;
  pipeline: string;
}

interface Trace {
  id: string;
  label: string;
  blurb: string;
  message: string;
  /** Lower-tier stages that are skipped fall to "skipped" once the
   *  active stage produces a verdict. The component animates through
   *  these in order, dwelling on each. */
  stages: Stage[];
  final: FinalState;
}

const TRACES: Trace[] = [
  {
    id: 'imminent',
    label: 'Imminent risk',
    blurb: 'Plan + means + timing. The LLM classifier returns level 3.',
    message:
      "I have the pills set aside. I'm planning to take them tonight.",
    stages: [
      {
        id: 'llm',
        status: 'success',
        rawLevel: 3,
        note:
          'Structured output: level=3, confidence=high, reason="plan, means, and timing are present".',
      },
      {
        id: 'normalize',
        status: 'done',
        level: 3,
        note:
          'Truth table: level=3 → needs_crisis_response=true, needs_clarification=false.',
      },
    ],
    final: {
      level: 3,
      route: 'crisis',
      classifierPath: 'llm_primary',
      overrideKind: 'none',
      llmFailureOccurred: false,
      responseStyle: 'safety_check',
      pipeline:
        'crisis_resource_lookup → crisis_response → crisis_log → finalization',
    },
  },
  {
    id: 'self_harm',
    label: 'Clear self-harm',
    blurb: 'Unambiguous suicidal ideation. The LLM classifier returns level 2.',
    message: 'I keep thinking about ending things. I just want it to stop.',
    stages: [
      {
        id: 'llm',
        status: 'success',
        rawLevel: 2,
        note:
          'Structured output: level=2, confidence=high, reason="user describes wanting suicide without explicit plan".',
      },
      {
        id: 'normalize',
        status: 'done',
        level: 2,
        note:
          'Truth table: level=2 → needs_crisis_response=true, needs_clarification=false.',
      },
    ],
    final: {
      level: 2,
      route: 'crisis',
      classifierPath: 'llm_primary',
      overrideKind: 'none',
      llmFailureOccurred: false,
      responseStyle: 'safety_check',
      pipeline:
        'crisis_resource_lookup → crisis_response → crisis_log → finalization',
    },
  },
  {
    id: 'ambiguous',
    label: 'Ambiguous (level 1)',
    blurb: 'Concerning but not explicit. The LLM picks level 1; the turn stays therapeutic.',
    message:
      "Sometimes I wonder if anyone would even notice if I wasn't here.",
    stages: [
      {
        id: 'llm',
        status: 'success',
        rawLevel: 1,
        note:
          'Structured output: level=1, confidence=medium, reason="ambiguous absence-as-relief framing".',
      },
      {
        id: 'normalize',
        status: 'done',
        level: 1,
        note:
          'Truth table: level=1 → needs_crisis_response=false, needs_clarification=true. Stays in therapeutic branch.',
      },
    ],
    final: {
      level: 1,
      route: 'therapeutic',
      classifierPath: 'llm_primary',
      overrideKind: 'none',
      llmFailureOccurred: false,
      responseStyle: '(picked downstream by therapeutic dispatcher)',
      pipeline: 'turn_dispatch → load_memory → TherapeuticAgent',
    },
  },
  {
    id: 'idiomatic_safe',
    label: 'Idiomatic safe',
    blurb: '"Killing me" as a colloquialism. The LLM classifies it as level 0.',
    message: "Work has been killing me lately, I'm dead tired.",
    stages: [
      {
        id: 'llm',
        status: 'success',
        rawLevel: 0,
        note:
          'Structured output: level=0, confidence=high, reason="idiomatic work stress phrase without self-harm intent".',
      },
      {
        id: 'normalize',
        status: 'done',
        level: 0,
        note:
          'Truth table: level=0 → needs_crisis_response=false, needs_clarification=false.',
      },
    ],
    final: {
      level: 0,
      route: 'therapeutic',
      classifierPath: 'llm_primary',
      overrideKind: 'none',
      llmFailureOccurred: false,
      responseStyle: '(picked downstream by therapeutic dispatcher)',
      pipeline: 'turn_dispatch → load_memory → TherapeuticAgent',
    },
  },
  {
    id: 'safety_denial',
    label: 'Safety denial',
    blurb: 'After a safety check, the user de-escalates. Recent history helps the LLM classify level 0.',
    message: "No, I'm safe. I was just venting.",
    stages: [
      {
        id: 'llm',
        status: 'success',
        rawLevel: 0,
        note:
          'Structured output: level=0, confidence=medium, reason="user denies immediate safety risk after a safety check".',
      },
      {
        id: 'normalize',
        status: 'done',
        level: 0,
        note:
          'Truth table: level=0 → therapeutic branch resumes normally.',
      },
    ],
    final: {
      level: 0,
      route: 'therapeutic',
      classifierPath: 'llm_primary',
      overrideKind: 'none',
      llmFailureOccurred: false,
      responseStyle: '(picked downstream by therapeutic dispatcher)',
      pipeline: 'turn_dispatch → load_memory → TherapeuticAgent',
    },
  },
];

/* ── Animation timing ──────────────────────────────────────────── */

const STAGE_DWELL_MS: Record<Stage['id'], number> = {
  llm: 1300,
  normalize: 1300,
};

/* ── Component ─────────────────────────────────────────────────── */

type Phase = 'idle' | 'playing' | 'done';

export default function CrisisGateFlow(): React.JSX.Element {
  const [traceId, setTraceId] = useState<string>(TRACES[0].id);
  const [stageIdx, setStageIdx] = useState(0);
  const [phase, setPhase] = useState<Phase>('idle');
  const [autoplay, setAutoplay] = useState(true);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const trace = useMemo(
    () => TRACES.find((t) => t.id === traceId) ?? TRACES[0],
    [traceId],
  );

  /* Reset and start whenever the trace changes (or autoplay flips on). */
  useEffect(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    setStageIdx(0);
    setPhase('idle');

    if (!autoplay) return;

    // Small lead-in pause before the first stage fires.
    timerRef.current = setTimeout(() => {
      setPhase('playing');
      setStageIdx(0);
    }, 400);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [traceId, autoplay]);

  /* Drive stage transitions. */
  useEffect(() => {
    if (phase !== 'playing') return;
    const stage = trace.stages[stageIdx];
    if (!stage) {
      setPhase('done');
      return;
    }

    const delay = STAGE_DWELL_MS[stage.id];
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      if (stageIdx + 1 >= trace.stages.length) {
        setPhase('done');
      } else {
        setStageIdx(stageIdx + 1);
      }
    }, delay);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [phase, stageIdx, trace.stages]);

  function replay() {
    if (timerRef.current) clearTimeout(timerRef.current);
    setStageIdx(0);
    setPhase('playing');
  }

  function selectTrace(id: string) {
    setTraceId(id);
  }

  // Shape the stages for rendering — each has a current "playState"
  // relative to the active stageIdx.
  const renderedStages = trace.stages.map((stage, i) => {
    const playState: 'pending' | 'firing' | 'settled' =
      phase === 'idle'
        ? 'pending'
        : i < stageIdx
          ? 'settled'
          : i === stageIdx
            ? phase === 'done'
              ? 'settled'
              : 'firing'
            : 'pending';
    return { stage, playState };
  });

  return (
    <div className={styles.root}>
      {/* Trace picker */}
      <div className={styles.traceBar}>
        <div className={styles.traceBarLabel}>Sample input:</div>
        <div className={styles.traceChips}>
          {TRACES.map((t) => (
            <button
              key={t.id}
              className={[styles.traceChip, traceId === t.id ? styles.traceChipActive : ''].join(' ')}
              onClick={() => selectTrace(t.id)}
              aria-pressed={traceId === t.id}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      <div className={styles.shell}>
        {/* Left column: input + classifier tiers */}
        <div className={styles.left}>
          {/* User message bubble */}
          <div className={styles.messageCard}>
            <div className={styles.messageHeader}>
              <span className={styles.messageRole}>user</span>
              <span className={styles.messageHint}>{trace.blurb}</span>
            </div>
            <p className={styles.messageBody}>{trace.message}</p>
          </div>

          {/* Tier 1: LLM classifier */}
          <Tier
            number="01"
            title="LLM classifier"
            subtitle="generate_structured(CrisisAssessmentSchema)"
            data={renderedStages[0]}
            kind="llm"
          />

          <BranchArrow stage={renderedStages[0]} kind="llm" />

          {/* Tier 2: Normalize */}
          <Tier
            number="02"
            title="Truth-table normalization"
            subtitle="enforce_crisis_truth_table(assessment)"
            data={renderedStages[1]}
            kind="normalize"
          />
        </div>

        {/* Right column: state delta */}
        <div className={styles.right}>
          <StatePanel trace={trace} stageIdx={stageIdx} phase={phase} />
        </div>
      </div>

      {/* Footer controls */}
      <div className={styles.footerBar}>
        <label className={styles.autoplayToggle}>
          <input
            type="checkbox"
            checked={autoplay}
            onChange={(e) => setAutoplay(e.target.checked)}
          />
          <span>Autoplay on input change</span>
        </label>
        <div className={styles.footerStatus}>
          {phase === 'playing' && (
            <span className={styles.statusRunning}>
              <span className={styles.statusDot} />
              tier {stageIdx + 1} / {trace.stages.length}
            </span>
          )}
          {phase === 'done' && (
            <span className={styles.statusDone}>
              <span className={styles.statusCheck}>{'\u2713'}</span> trace complete
            </span>
          )}
          {phase === 'idle' && (
            <span className={styles.statusIdle}>idle</span>
          )}
          <button
            className={styles.replayBtn}
            onClick={replay}
            disabled={phase === 'playing'}
          >
            {'\u21BB'} Replay trace
          </button>
        </div>
      </div>
    </div>
  );
}

/* ── Tier card ─────────────────────────────────────────────────── */

interface TierProps {
  number: string;
  title: string;
  subtitle: string;
  data: { stage: Stage; playState: 'pending' | 'firing' | 'settled' };
  kind: Stage['id'];
}

function Tier({ number, title, subtitle, data, kind }: TierProps) {
  const { stage, playState } = data;

  const tone = stageTone(stage);
  const verdict = stageVerdict(stage);

  return (
    <div
      className={[
        styles.tier,
        styles[`tier_${tone}`],
        playState === 'firing' ? styles.tierFiring : '',
        playState === 'settled' ? styles.tierSettled : '',
        playState === 'pending' ? styles.tierPending : '',
      ].join(' ')}
    >
      <div className={styles.tierHeader}>
        <span className={styles.tierNumber}>{number}</span>
        <div className={styles.tierTitles}>
          <span className={styles.tierTitle}>{title}</span>
          <span className={styles.tierSubtitle}>{subtitle}</span>
        </div>
        <span className={[styles.tierBadge, styles[`tierBadge_${tone}`]].join(' ')}>
          {verdict}
        </span>
      </div>
      <p className={styles.tierNote}>{stage.note}</p>
    </div>
  );
}

/* ── Branch arrow between tiers ────────────────────────────────── */

function BranchArrow({
  stage: { stage, playState },
  kind,
}: {
  stage: { stage: Stage; playState: 'pending' | 'firing' | 'settled' };
  kind: Stage['id'];
}) {
  const tone = stageTone(stage);
  const branchLabel = branchLabelFor(stage);

  return (
    <div
      className={[
        styles.branch,
        styles[`branch_${tone}`],
        playState === 'pending' ? styles.branchPending : '',
        playState === 'settled' ? styles.branchSettled : '',
        playState === 'firing' ? styles.branchFiring : '',
      ].join(' ')}
    >
      <div className={styles.branchLine} />
      <span className={styles.branchTag}>{branchLabel}</span>
    </div>
  );
}

/* ── State panel ───────────────────────────────────────────────── */

function StatePanel({ trace, stageIdx, phase }: { trace: Trace; stageIdx: number; phase: Phase }) {
  const llmSettled = phase !== 'idle' && (stageIdx > 0 || phase === 'done');
  const normalized = phase === 'done';

  const { final } = trace;

  // Build progressively-revealed state values. Classifier metadata is
  // known after the LLM step, while route-level fields land after
  // normalization.
  return (
    <div className={styles.statePanel}>
      <div className={styles.statePanelHeader}>
        <div className={styles.statePanelDots}>
          <span className={styles.dotR} />
          <span className={styles.dotY} />
          <span className={styles.dotG} />
        </div>
        <span className={styles.statePanelTitle}>state delta</span>
      </div>
      <div className={styles.stateBody}>
        <StateLine
          label="crisis.level"
          value={normalized ? String(final.level) : null}
          comment="0 safe · 1 ambiguous · 2 clear self-harm · 3 imminent"
        />
        <StateLine
          label="route"
          value={normalized ? final.route : null}
          comment="needs_crisis_response decides this"
        />
        <StateLine
          label="response_style"
          value={normalized ? final.responseStyle : null}
          comment={
            final.route === 'crisis'
              ? 'safety_check stamped on entry to the crisis branch'
              : 'left to the therapeutic dispatcher'
          }
        />
        <div className={styles.stateDivider} />
        <StateLine
          label="crisis_audit.crisis_classifier_path"
          value={
            llmSettled || phase === 'done'
              ? final.classifierPath
              : null
          }
          comment="written for every completed crisis-gate turn"
        />
        <StateLine
          label="crisis_audit.crisis_override_kind"
          value={llmSettled ? final.overrideKind : null}
          comment="always none in the LLM-only gate"
        />
        <StateLine
          label="crisis_audit.crisis_llm_failure_occurred"
          value={llmSettled ? String(final.llmFailureOccurred) : null}
          comment="failed LLM calls retry or surface instead of writing fallback state"
        />
        <div className={styles.stateDivider} />
        <div className={styles.pipelineRow}>
          <span className={styles.pipelineLabel}>next pipeline:</span>
          <span className={[
            styles.pipelineValue,
            normalized ? styles.pipelineValueOn : '',
          ].join(' ')}>
            {normalized ? final.pipeline : '…'}
          </span>
        </div>
      </div>
    </div>
  );
}

function StateLine({ label, value, comment }: {
  label: string;
  value: string | null;
  comment: string;
}) {
  const filled = value !== null;
  return (
    <div className={[styles.stateLine, filled ? styles.stateLineFilled : ''].join(' ')}>
      <code className={styles.stateKey}>{label}</code>
      <code className={styles.stateValue}>
        {filled ? value : '…'}
      </code>
      <span className={styles.stateComment}>{comment}</span>
    </div>
  );
}

/* ── Helpers ───────────────────────────────────────────────────── */

function stageTone(stage: Stage): 'crisis' | 'safe' | 'pending' | 'fail' | 'neutral' {
  switch (stage.id) {
    case 'llm':
      if (stage.status === 'success') {
        if (stage.rawLevel != null && stage.rawLevel >= 2) return 'crisis';
        return stage.rawLevel === 0 ? 'safe' : 'neutral';
      }
      if (stage.status === 'failed') return 'fail';
      return 'pending';
    case 'normalize':
      return stage.level >= 2 ? 'crisis' : stage.level === 0 ? 'safe' : 'neutral';
  }
}

function stageVerdict(stage: Stage): string {
  switch (stage.id) {
    case 'llm':
      if (stage.status === 'success') return `level=${stage.rawLevel}`;
      if (stage.status === 'failed') return 'failed';
      return 'skipped';
    case 'normalize':
      return `level=${stage.level} (final)`;
  }
}

function branchLabelFor(stage: Stage): string {
  switch (stage.id) {
    case 'llm':
      if (stage.status === 'success') return 'LLM verdict → normalize';
      if (stage.status === 'failed') return 'LLM failed → retry/error';
      return 'skipped';
    case 'normalize':
      return 'enforce truth table';
  }
}
