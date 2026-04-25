import React, { useState } from 'react';
import styles from './ExerciseCatalog.module.css';

interface Exercise {
  id: string;
  name: string;
  subtype: string;
  subtypeLabel: string;
  steps: number;
  completionMode: string;
  triggers: string[];
  description: string;
  stepSummary: string[];
}

const EXERCISES: Exercise[] = [
  {
    id: 'grounding_5_4_3_2_1',
    name: '5-4-3-2-1 Grounding',
    subtype: 'grounding',
    subtypeLabel: 'Grounding',
    steps: 5,
    completionMode: 'Item count',
    triggers: ['"ground me"', '"grounding"', 'default fallback'],
    description: 'Sensory anchoring exercise. The user identifies items across five senses to return to the present moment. Steps are independent — order matters less than engagement.',
    stepSummary: ['Name 5 things you can see', 'Name 4 things you can hear', 'Name 3 things you can feel', 'Name 2 things you can smell', 'Name 1 thing you can taste'],
  },
  {
    id: 'grounding_box_breathing',
    name: 'Box Breathing',
    subtype: 'grounding',
    subtypeLabel: 'Grounding',
    steps: 4,
    completionMode: 'User confirmation',
    triggers: ['"breathing"', '"breathe"', '"box breathing"'],
    description: 'Structured 4-phase breathing cycle targeting somatic regulation. Each step is a single breathing action confirmed by the user. Complements 5-4-3-2-1 (sensory) with a respiratory channel.',
    stepSummary: ['Inhale for 4 counts', 'Hold for 4 counts', 'Exhale for 4 counts', 'Hold (empty) for 4 counts'],
  },
  {
    id: 'grounding_stop_technique',
    name: 'STOP Technique',
    subtype: 'grounding',
    subtypeLabel: 'Grounding / DBT',
    steps: 4,
    completionMode: 'Mixed',
    triggers: ['"stop technique"', '"STOP"', '"pause"', '"slow down"'],
    description: 'DBT-informed distress tolerance skill. Each letter is a discrete step. S-T steps use confirmation mode (somatic); O-P steps use item count (cognitive).',
    stepSummary: ['Stop — pause everything', 'Take a breath', 'Observe — name what you notice', 'Proceed — pick one next action'],
  },
  {
    id: 'grounding_muscle_relaxation',
    name: 'Progressive Muscle Relaxation',
    subtype: 'grounding',
    subtypeLabel: 'Grounding (Somatic)',
    steps: 5,
    completionMode: 'User confirmation',
    triggers: ['"muscle"', '"tension"', '"relax my body"', '"PMR"'],
    description: 'Body-focused relaxation: tense and release 5 muscle groups. Each step is a simple "tense, hold, release, notice." Complements breathing (respiratory) and 5-4-3-2-1 (sensory) with a muscular channel.',
    stepSummary: ['Hands — clench fists, release', 'Shoulders — shrug up, drop', 'Face — scrunch, go slack', 'Stomach — brace, let go', 'Legs — press feet down, release'],
  },
  {
    id: 'thought_work_simple_record',
    name: 'Simple Thought Record',
    subtype: 'thought_work',
    subtypeLabel: 'Thought Work (CBT)',
    steps: 4,
    completionMode: 'Item count',
    triggers: ['"thought record"', '"examine this thought"', '"belief"'],
    description: 'Simplified 4-step CBT thought record. Sequential — each step depends on the prior. Exit is the only valid off-ramp. The user discovers the reframe rather than being told what the "correct" thought is.',
    stepSummary: ['Describe the situation', 'State the specific thought', 'Find evidence against it', 'Form a balanced alternative'],
  },
  {
    id: 'thought_work_behavioral_experiment',
    name: 'Behavioral Experiment',
    subtype: 'thought_work',
    subtypeLabel: 'Thought Work (CBT)',
    steps: 4,
    completionMode: 'Item count',
    triggers: ['"test this belief"', '"behavioral experiment"', '"check if"'],
    description: 'Tests beliefs in the real world. The thought record examines evidence in the mind; this tests it in reality. Step 3→4 may span hours (the user does something IRL between turns).',
    stepSummary: ['State the belief', 'Plan a small test', 'Predict what will happen', 'Reflect on what actually happened'],
  },
  {
    id: 'thought_work_continuum',
    name: 'Continuum (All-or-Nothing)',
    subtype: 'thought_work',
    subtypeLabel: 'Thought Work (CBT)',
    steps: 5,
    completionMode: 'Item count',
    triggers: ['"continuum"', '"all-or-nothing"', '"black and white"', '"I\'m a terrible…"', '"I always fail"'],
    description: 'Targets rigid all-or-nothing self-labels by converting an absolute ("I\'m a terrible parent") into a 0-100 dimension. The user defines both endpoints, then places themselves honestly. Most discover they\'re mid-range, not at zero — which is already a shift from the absolute framing.',
    stepSummary: ['State the absolute belief', 'Define what 0 (worst-case) looks like', 'Define what 100 (impossibly perfect) looks like', 'Place yourself honestly on the scale', 'Identify a small move +5 points up'],
  },
  {
    id: 'behavioral_activation_tiny_action',
    name: 'Tiny Action Experiment',
    subtype: 'behavioral_activation',
    subtypeLabel: 'Behavioral Activation',
    steps: 4,
    completionMode: 'Item count',
    triggers: ['"stuck"', '"can\'t start"', '"depleted"', '"motivation"'],
    description: 'One small action framed as an experiment, not a commitment. Targets avoidance and low activation. The ask is intentionally tiny — "just to see what happens."',
    stepSummary: ['Name one small action', 'Specify when/where', 'Anticipate obstacles', 'Check feasibility (1–10)'],
  },
  {
    id: 'defusion_leaves_on_stream',
    name: 'Leaves on a Stream',
    subtype: 'defusion',
    subtypeLabel: 'Acceptance / ACT',
    steps: 5,
    completionMode: 'Mixed',
    triggers: ['"let go"', '"leaves"', '"defusion"', '"stop fighting"'],
    description: 'ACT defusion exercise. The user names a sticky thought, places it on an imagined leaf, watches it float, notices what remains, then identifies a values-aligned step. The exercise is about acting alongside the thought, not removing it.',
    stepSummary: ['Name the sticky thought', 'Place it on a leaf (visualize)', 'Watch the leaf drift', 'Notice what\'s different', 'Identify a values step'],
  },
  {
    id: 'defusion_values_compass',
    name: 'Values Compass',
    subtype: 'defusion',
    subtypeLabel: 'Acceptance / ACT',
    steps: 4,
    completionMode: 'Item count',
    triggers: ['"values"', '"what matters"', '"purpose"', '"direction"'],
    description: 'ACT values clarification. Counterpart to defusion — Leaves on a Stream helps with letting go; Values Compass helps with moving toward. Together they cover the full ACT arc.',
    stepSummary: ['Pick a life domain', 'Describe why it matters', 'Rate alignment (1–10)', 'Identify one small step toward it'],
  },
  {
    id: 'self_compassion_break',
    name: 'Self-Compassion Break',
    subtype: 'self_compassion',
    subtypeLabel: 'Self-Compassion',
    steps: 3,
    completionMode: 'Mixed',
    triggers: ['"self-compassion"', '"hard on myself"', '"self-critical"'],
    description: 'Kristin Neff\'s 3-component model. Very short by design — self-compassion exercises lose power when long. The user chooses their own kind words rather than following a script.',
    stepSummary: ['Acknowledge suffering', 'Common humanity', 'Kind wish to self'],
  },
  {
    id: 'emotion_regulation_improve',
    name: 'IMPROVE the Moment',
    subtype: 'emotion_regulation',
    subtypeLabel: 'Emotion Regulation (DBT)',
    steps: 4,
    completionMode: 'Mixed',
    triggers: ['"improve"', '"overwhelmed"', '"cope"', '"get through this"'],
    description: 'Core DBT distress tolerance skill. Uses 4 of the 7 IMPROVE letters: Imagery, Meaning, One thing, Encouragement. For users in acute overwhelm who need help tolerating the current moment.',
    stepSummary: ['Imagery — visualize a safe place', 'Meaning — name one reason to endure', 'One thing — pick a single focus', 'Encouragement — say something kind'],
  },
  {
    id: 'emotion_regulation_gratitude',
    name: 'Gratitude Inventory',
    subtype: 'emotion_regulation',
    subtypeLabel: 'Emotion Regulation',
    steps: 3,
    completionMode: 'Item count',
    triggers: ['"grateful"', '"gratitude"', '"positive"', '"thankful"'],
    description: 'Short positive-psychology exercise for building positive affect. Best for neutral-to-low-mood states or session closers. Not appropriate during acute distress.',
    stepSummary: ['Name 3 things you\'re grateful for', 'Pick one — describe why it matters', 'Notice your body\'s response'],
  },
];

const SUBTYPES = [
  { id: 'all', label: 'All' },
  { id: 'grounding', label: 'Grounding' },
  { id: 'thought_work', label: 'Thought Work' },
  { id: 'behavioral_activation', label: 'Activation' },
  { id: 'defusion', label: 'ACT / Defusion' },
  { id: 'self_compassion', label: 'Self-Compassion' },
  { id: 'emotion_regulation', label: 'Emotion Reg.' },
];

export default function ExerciseCatalog() {
  const [filter, setFilter] = useState('all');
  const [expanded, setExpanded] = useState<string | null>(null);

  const filtered = filter === 'all'
    ? EXERCISES
    : EXERCISES.filter(e => e.subtype === filter);

  return (
    <div className={styles.root}>
      {/* Filter tabs */}
      <div className={styles.filterRow}>
        {SUBTYPES.map(s => (
          <button
            key={s.id}
            className={[styles.filterTab, filter === s.id ? styles.filterActive : ''].join(' ')}
            onClick={() => setFilter(s.id)}
          >
            {s.label}
            {s.id !== 'all' && (
              <span className={styles.filterCount}>
                {EXERCISES.filter(e => e.subtype === s.id).length}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Exercise cards */}
      <div className={styles.grid}>
        {filtered.map(ex => (
          <div
            key={ex.id}
            className={[styles.card, expanded === ex.id ? styles.cardExpanded : ''].join(' ')}
            onClick={() => setExpanded(p => p === ex.id ? null : ex.id)}
          >
            <div className={styles.cardHeader}>
              <div className={styles.cardMeta}>
                <span className={styles.subtypeTag}>{ex.subtypeLabel}</span>
                <span className={styles.stepCount}>{ex.steps} steps</span>
                <span className={styles.completionTag}>{ex.completionMode}</span>
              </div>
              <h4 className={styles.cardTitle}>{ex.name}</h4>
              <p className={styles.cardDesc}>{ex.description}</p>
            </div>

            {expanded === ex.id && (
              <div className={styles.cardDetail}>
                {/* Steps */}
                <div className={styles.stepList}>
                  <span className={styles.detailLabel}>Steps</span>
                  <ol className={styles.steps}>
                    {ex.stepSummary.map((step, i) => (
                      <li key={i} className={styles.stepItem}>{step}</li>
                    ))}
                  </ol>
                </div>

                {/* Triggers */}
                <div className={styles.triggerList}>
                  <span className={styles.detailLabel}>Trigger keywords</span>
                  <div className={styles.triggers}>
                    {ex.triggers.map(t => (
                      <code key={t} className={styles.triggerCode}>{t}</code>
                    ))}
                  </div>
                </div>

                {/* Code reference */}
                <div className={styles.codeRef}>
                  <code className={styles.codeRefText}>{ex.id}</code>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
