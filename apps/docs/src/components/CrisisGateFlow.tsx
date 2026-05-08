import React, { useEffect, useMemo, useRef, useState } from 'react';
import styles from './CrisisGateFlow.module.css';

/* ================================================================
   CrisisGateFlow — Interactive walkthrough of the crisis gate.

   Each example input triggers a precomputed trace through the
   three-tier classifier:

     1. Deterministic overrides (imminent_risk / idiomatic_safe / safety_denial)
     2. LLM classifier (structured output, primary path)
     3. Deterministic regex ladder (fallback when LLM unavailable)

   …followed by the truth-table normalization step. The right panel
   shows the resulting state delta (level, route, classifier_path,
   override_kind, response_style).

   This is documentation, not a live classifier — every trace is
   hand-authored to mirror what `agent/nodes/crisis_gate.py` and
   `agent/safety/crisis_rules.py` would actually emit.
   ================================================================ */

type ClassifierPath =
  | 'override'
  | 'llm_primary'
  | 'llm_fallback'
  | 'deterministic';

type OverrideOutcome = 'imminent_risk' | 'idiomatic_safe' | 'safety_denial' | 'none';

type Stage =
  | { id: 'override'; status: 'firing' | 'pass' | 'matched'; matchKind?: OverrideOutcome; note: string }
  | { id: 'llm'; status: 'firing' | 'success' | 'failed' | 'skipped'; rawLevel?: 0 | 1 | 2 | 3; note: string }
  | { id: 'regex'; status: 'firing' | 'matched' | 'skipped'; rawLevel?: 0 | 1 | 2 | 3; note: string }
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
    blurb: 'Plan + means + timing. The deterministic override fires before any LLM call.',
    message:
      "I have the pills set aside. I'm planning to take them tonight.",
    stages: [
      {
        id: 'override',
        status: 'matched',
        matchKind: 'imminent_risk',
        note:
          'IMMINENT_PATTERNS matched: plan ("set aside") + means ("pills") + timing ("tonight"). Deterministic L3, no network call.',
      },
      {
        id: 'llm',
        status: 'skipped',
        note: 'Skipped — override already produced a verdict.',
      },
      {
        id: 'regex',
        status: 'skipped',
        note: 'Skipped — only used when no LLM is configured or the LLM call fails.',
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
      classifierPath: 'override',
      overrideKind: 'imminent_risk',
      llmFailureOccurred: false,
      responseStyle: 'safety_check',
      pipeline:
        'crisis_resource_lookup_node → crisis_response_node → crisis_log_node → finalize_turn_node',
    },
  },
  {
    id: 'self_harm',
    label: 'Clear self-harm',
    blurb: 'Unambiguous suicidal ideation. The LLM classifier returns level 2.',
    message: 'I keep thinking about ending things. I just want it to stop.',
    stages: [
      {
        id: 'override',
        status: 'pass',
        matchKind: 'none',
        note:
          'No imminent-risk plan markers, no idiomatic-safe phrasing, no safety-denial framing. Pass through to the LLM.',
      },
      {
        id: 'llm',
        status: 'success',
        rawLevel: 2,
        note:
          'Structured output: level=2, confidence=high, reason="user describes wanting suicide without explicit plan".',
      },
      {
        id: 'regex',
        status: 'skipped',
        note: 'Skipped — LLM call succeeded.',
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
        'crisis_resource_lookup_node → crisis_response_node → crisis_log_node → finalize_turn_node',
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
        id: 'override',
        status: 'pass',
        matchKind: 'none',
        note: 'No deterministic patterns match.',
      },
      {
        id: 'llm',
        status: 'success',
        rawLevel: 1,
        note:
          'Structured output: level=1, confidence=medium, reason="ambiguous absence-as-relief framing".',
      },
      {
        id: 'regex',
        status: 'skipped',
        note: 'Skipped — LLM call succeeded.',
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
      pipeline: 'turn_dispatch_node → load_memory_node → therapeutic_subgraph',
    },
  },
  {
    id: 'idiomatic_safe',
    label: 'Idiomatic safe',
    blurb: '"Killing me" as a colloquialism. The override pre-empts the LLM.',
    message: "Work has been killing me lately, I'm dead tired.",
    stages: [
      {
        id: 'override',
        status: 'matched',
        matchKind: 'idiomatic_safe',
        note:
          'IDIOMATIC_SAFE_PATTERNS matched: "killing me" (work context) + "dead tired" (idiom). Forced level=0.',
      },
      {
        id: 'llm',
        status: 'skipped',
        note:
          'Skipped — even capable LLMs occasionally false-positive on these idioms; the override prevents that.',
      },
      {
        id: 'regex',
        status: 'skipped',
        note: 'Skipped.',
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
      classifierPath: 'override',
      overrideKind: 'idiomatic_safe',
      llmFailureOccurred: false,
      responseStyle: '(picked downstream by therapeutic dispatcher)',
      pipeline: 'turn_dispatch_node → load_memory_node → therapeutic_subgraph',
    },
  },
  {
    id: 'llm_fail',
    label: 'LLM failure → fallback',
    blurb: 'Provider outage. The deterministic regex ladder catches the safety-relevant pattern.',
    message: 'I cannot do this anymore.',
    stages: [
      {
        id: 'override',
        status: 'pass',
        matchKind: 'none',
        note: 'No override patterns match.',
      },
      {
        id: 'llm',
        status: 'failed',
        note:
          'Structured-output call raised: provider timeout. crisis_llm_failure_occurred set in audit; no level produced.',
      },
      {
        id: 'regex',
        status: 'matched',
        rawLevel: 1,
        note:
          'AMBIGUOUS_PATTERNS matched "i can\'t do this anymore" → level=1. Degraded coverage but the safety floor holds.',
      },
      {
        id: 'normalize',
        status: 'done',
        level: 1,
        note:
          'Truth table: level=1 → needs_clarification=true. classifier_path=llm_fallback so dashboards can track failure rate.',
      },
    ],
    final: {
      level: 1,
      route: 'therapeutic',
      classifierPath: 'llm_fallback',
      overrideKind: 'none',
      llmFailureOccurred: true,
      responseStyle: '(picked downstream by therapeutic dispatcher)',
      pipeline: 'turn_dispatch_node → load_memory_node → therapeutic_subgraph',
    },
  },
  {
    id: 'safety_denial',
    label: 'Safety denial',
    blurb: 'After a safety check, the user de-escalates. The override pulls the level back down.',
    message: "No, I'm safe. I was just venting.",
    stages: [
      {
        id: 'override',
        status: 'matched',
        matchKind: 'safety_denial',
        note:
          'Previous assistant turn was a safety_check, current turn matches SAFETY_DENIAL_PATTERNS ("I\'m safe", "just venting"). Forced level=0.',
      },
      {
        id: 'llm',
        status: 'skipped',
        note: 'Skipped — denial after a safety check is the override\'s purpose.',
      },
      {
        id: 'regex',
        status: 'skipped',
        note: 'Skipped.',
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
      classifierPath: 'override',
      overrideKind: 'safety_denial',
      llmFailureOccurred: false,
      responseStyle: '(picked downstream by therapeutic dispatcher)',
      pipeline: 'turn_dispatch_node → load_memory_node → therapeutic_subgraph',
    },
  },
];

/* ── Animation timing ──────────────────────────────────────────── */

const STAGE_DWELL_MS: Record<Stage['id'], number> = {
  override: 1100,
  llm: 1300,
  regex: 1100,
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

          {/* Tier 1: Override */}
          <Tier
            number="01"
            title="Deterministic overrides"
            subtitle="agent/safety/crisis_rules.py · detect_crisis_override()"
            data={renderedStages[0]}
            kind="override"
          />

          {/* Branch indicator */}
          <BranchArrow stage={renderedStages[0]} kind="override" />

          {/* Tier 2: LLM classifier */}
          <Tier
            number="02"
            title="LLM classifier (primary)"
            subtitle="generate_structured(CrisisAssessmentSchema)"
            data={renderedStages[1]}
            kind="llm"
          />

          <BranchArrow stage={renderedStages[1]} kind="llm" />

          {/* Tier 3: Regex ladder */}
          <Tier
            number="03"
            title="Regex ladder (fallback)"
            subtitle="agent/safety/crisis_rules.py · pattern tuples"
            data={renderedStages[2]}
            kind="regex"
          />

          <BranchArrow stage={renderedStages[2]} kind="regex" />

          {/* Tier 4: Normalize */}
          <Tier
            number="04"
            title="Truth-table normalization"
            subtitle="enforce_crisis_truth_table(assessment)"
            data={renderedStages[3]}
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
  // Until the override stage settles we don't know any verdict.
  const overrideSettled = phase !== 'idle' && (stageIdx > 0 || phase === 'done');
  const llmSettled = phase !== 'idle' && (stageIdx > 1 || phase === 'done');
  const regexSettled = phase !== 'idle' && (stageIdx > 2 || phase === 'done');
  const normalized = phase === 'done';

  const { final } = trace;

  // Build progressively-revealed state values. Some keys land
  // earlier than others — `crisis_audit.crisis_override_kind` is
  // known after stage 0, `level` only after normalize.
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
            overrideSettled || phase === 'done'
              ? final.classifierPath
              : null
          }
          comment="written for every turn — drives fallback-rate dashboards"
        />
        <StateLine
          label="crisis_audit.crisis_override_kind"
          value={overrideSettled ? final.overrideKind : null}
          comment="imminent_risk · idiomatic_safe · safety_denial · none"
        />
        <StateLine
          label="crisis_audit.crisis_llm_failure_occurred"
          value={
            llmSettled || regexSettled
              ? String(final.llmFailureOccurred)
              : null
          }
          comment="true when the LLM call raised; surfaces in the audit log"
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
    case 'override':
      if (stage.status === 'matched') {
        return stage.matchKind === 'imminent_risk' ? 'crisis' : 'safe';
      }
      return 'neutral';
    case 'llm':
      if (stage.status === 'success') {
        if (stage.rawLevel != null && stage.rawLevel >= 2) return 'crisis';
        return stage.rawLevel === 0 ? 'safe' : 'neutral';
      }
      if (stage.status === 'failed') return 'fail';
      return 'pending';
    case 'regex':
      if (stage.status === 'matched') {
        if (stage.rawLevel != null && stage.rawLevel >= 2) return 'crisis';
        return 'neutral';
      }
      return 'pending';
    case 'normalize':
      return stage.level >= 2 ? 'crisis' : stage.level === 0 ? 'safe' : 'neutral';
  }
}

function stageVerdict(stage: Stage): string {
  switch (stage.id) {
    case 'override':
      if (stage.status === 'matched') return `matched · ${stage.matchKind}`;
      return 'no match';
    case 'llm':
      if (stage.status === 'success') return `level=${stage.rawLevel}`;
      if (stage.status === 'failed') return 'failed → fallback';
      return 'skipped';
    case 'regex':
      if (stage.status === 'matched') return `level=${stage.rawLevel}`;
      return 'skipped';
    case 'normalize':
      return `level=${stage.level} (final)`;
  }
}

function branchLabelFor(stage: Stage): string {
  switch (stage.id) {
    case 'override':
      return stage.status === 'matched' ? 'short-circuit to normalize' : 'no override → try LLM';
    case 'llm':
      if (stage.status === 'success') return 'LLM verdict → normalize';
      if (stage.status === 'failed') return 'LLM failed → regex fallback';
      return 'skipped';
    case 'regex':
      if (stage.status === 'matched') return 'fallback verdict → normalize';
      return 'skipped';
    case 'normalize':
      return 'enforce truth table';
  }
}
