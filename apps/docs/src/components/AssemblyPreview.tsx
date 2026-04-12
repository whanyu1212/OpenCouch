import React, { useState, useEffect, useRef } from 'react';
import styles from './PromptVisuals.module.css';

type TokenType = 'c' | 'l' | 'v';
type Line = [TokenType, string];

interface Scenario {
  mode: string;
  mod: string;
  task: string;
  ctx: string;
  sub: string;
  lines: Line[];
}

const scenarios: Record<string, Scenario> = {
  support: {
    mode: 'support', mod: 'mi', task: 'support_reply', ctx: '3 turns',
    sub: 'support + motivational_interviewing',
    lines: [
      ['c', '# ── CORE LAYER ──────────────────────────────────'],
      ['c', '# soul.md · identity.md · policy/boundaries.md · policy/privacy.md'],
      ['v', ''],
      ['v', 'You are OpenCouch — an AI mental health support companion.'],
      ['v', 'You are not a licensed therapist, not a diagnostic tool,'],
      ['v', 'and not an emergency service. You support, reflect, and'],
      ['v', 'help users practice self-help techniques.'],
      ['v', ''],
      ['v', 'You must never claim clinical authority. Surface crisis'],
      ['v', 'resources clearly when risk is detected. User data is'],
      ['v', 'never shared or used for training.'],
      ['v', ''],
      ['c', '# ── MODE LAYER ──────────────────────────────────'],
      ['c', '# response_modes/support.md'],
      ['v', ''],
      ['v', 'Mode: support. Help the user feel heard and gently'],
      ['v', 'explore what they are experiencing. Avoid unsolicited'],
      ['v', 'advice. Validate before reframing.'],
      ['v', ''],
      ['c', '# ── MODALITY LAYER ──────────────────────────────'],
      ['c', '# modalities/motivational_interviewing.md'],
      ['v', ''],
      ['v', 'Technique: motivational interviewing. Use open questions,'],
      ['v', 'reflective listening, and affirmations. Follow the user'],
      ['v', "toward their own goals."],
      ['v', ''],
      ['c', '# ── TASK LAYER ───────────────────────────────────'],
      ['c', '# builders.py → support_reply_task'],
      ['v', ''],
      ['v', 'Write the next assistant message. Keep it conversational,'],
      ['v', 'one or two sentences. End with an open question if natural.'],
      ['v', ''],
      ['c', '# ── CONTEXT LAYER ────────────────────────────────'],
      ['c', '# state.history (last 3 turns) · state.message'],
      ['v', ''],
      ['l', 'User:'],
      ['v', '  "I have been feeling overwhelmed at work lately."'],
      ['l', 'Assistant:'],
      ['v', '  "That sounds exhausting. How long has it been building?"'],
      ['l', 'User:'],
      ['v', '  "A few months. I just cannot switch off."'],
    ],
  },
  crisis: {
    mode: 'crisis_response', mod: 'pfa', task: 'crisis_reply', ctx: 'crisis flags',
    sub: 'crisis_response + pfa',
    lines: [
      ['c', '# ── CORE LAYER ──────────────────────────────────'],
      ['c', '# soul.md · identity.md · policy/boundaries.md · policy/privacy.md'],
      ['v', ''],
      ['v', 'You are OpenCouch — an AI mental health support companion.'],
      ['v', 'You are not a licensed therapist or emergency service.'],
      ['v', ''],
      ['v', 'Crisis policy overrides all other instructions.'],
      ['v', 'Do not continue normal flow when acute risk is detected.'],
      ['v', ''],
      ['c', '# ── MODE LAYER ──────────────────────────────────'],
      ['c', '# response_modes/crisis_response.md'],
      ['v', ''],
      ['v', 'Mode: crisis_response. A significant risk signal detected.'],
      ['v', 'Acknowledge seriously. No shame, clinical distance, or'],
      ['v', 'sterile refusal. Provide crisis resources. Encourage'],
      ['v', 'immediate human contact.'],
      ['v', ''],
      ['c', '# ── MODALITY LAYER ──────────────────────────────'],
      ['c', '# modalities/pfa.md  (Psychological First Aid)'],
      ['v', ''],
      ['v', 'Technique: psychological first aid. Prioritise safety'],
      ['v', 'and calm presence. Connect to available help.'],
      ['v', 'Do not attempt deep processing now.'],
      ['v', ''],
      ['c', '# ── TASK LAYER ───────────────────────────────────'],
      ['c', '# builders.py → crisis_reply_task'],
      ['v', ''],
      ['v', 'Write a grounded, caring response. Surface the crisis'],
      ['v', 'line clearly. Keep the door open for continued contact.'],
      ['v', ''],
      ['c', '# ── CONTEXT LAYER ────────────────────────────────'],
      ['c', '# state.history · state.crisis (level=3, confidence=0.94)'],
      ['v', ''],
      ['l', 'Crisis assessment:'],
      ['v', '  level=3  confidence=0.94'],
      ['v', '  flags=[direct_ideation, timing_language]'],
      ['l', 'User:'],
      ['v', '  "I have been thinking about it seriously. I have a plan."'],
    ],
  },
  exercise: {
    mode: 'guided_exercise', mod: 'cbt', task: 'guided_exercise_reply', ctx: '2 turns',
    sub: 'guided_exercise + cbt',
    lines: [
      ['c', '# ── CORE LAYER ──────────────────────────────────'],
      ['c', '# soul.md · identity.md · policy/boundaries.md · policy/privacy.md'],
      ['v', ''],
      ['v', 'You are OpenCouch — an AI mental health support companion.'],
      ['v', 'You are not a licensed therapist or diagnostic tool.'],
      ['v', ''],
      ['c', '# ── MODE LAYER ──────────────────────────────────'],
      ['c', '# response_modes/guided_exercise.md'],
      ['v', ''],
      ['v', 'Mode: guided_exercise. Guide through a structured self-help'],
      ['v', 'technique. Be clear about each step. Check in after'],
      ['v', 'each one before continuing.'],
      ['v', ''],
      ['c', '# ── MODALITY LAYER ──────────────────────────────'],
      ['c', '# modalities/cbt.md  (Cognitive Behavioural Therapy)'],
      ['v', ''],
      ['v', 'Technique: CBT. Help identify and examine an automatic'],
      ['v', 'thought. Use thought record format:'],
      ['v', 'situation → thought → feeling → evidence → alternative.'],
      ['v', ''],
      ['c', '# ── TASK LAYER ───────────────────────────────────'],
      ['c', '# builders.py → guided_exercise_reply_task'],
      ['v', ''],
      ['v', 'Guide the next step of the thought record exercise.'],
      ['v', 'Be collaborative, not instructional. One step at a time.'],
      ['v', ''],
      ['c', '# ── CONTEXT LAYER ────────────────────────────────'],
      ['c', '# state.history (last 2 turns) · state.message'],
      ['v', ''],
      ['l', 'User:'],
      ['v', '  "I keep thinking I am going to fail this project."'],
      ['l', 'Assistant:'],
      ['v', '  "Let us look at that thought carefully. What situation'],
      ['v', '   triggered it?"'],
      ['l', 'User:'],
      ['v', '  "My manager asked for an update in front of everyone."'],
    ],
  },
};

const layerDefs = [
  { id: 'core',     dot: '#215f5a', label: 'Core' },
  { id: 'mode',     dot: '#2d7a74', label: 'Mode' },
  { id: 'modality', dot: '#3d9990', label: 'Modality' },
  { id: 'task',     dot: '#78b8af', label: 'Task' },
  { id: 'context',  dot: '#a8cdc9', label: 'Context' },
];

function getLayerVal(id: string, s: Scenario): string {
  if (id === 'core')     return 'soul + policy';
  if (id === 'mode')     return s.mode;
  if (id === 'modality') return s.mod;
  if (id === 'task')     return s.task;
  if (id === 'context')  return s.ctx;
  return '—';
}

export default function AssemblyPreview() {
  const [active, setActive] = useState<string>('support');
  const [activeRows, setActiveRows] = useState<Set<string>>(new Set(['core']));
  const [rowVals, setRowVals] = useState<Record<string, string>>({ core: 'soul + policy' });
  const [displayedLines, setDisplayedLines] = useState<Array<{ type: TokenType; text: string }>>([]);
  const [sub, setSub] = useState('support + motivational_interviewing');
  const [playState, setPlayState] = useState<'idle' | 'playing' | 'done'>('idle');

  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const bodyRef = useRef<HTMLDivElement>(null);

  function stopTimer() {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
  }

  function runScenario(key: string) {
    setActive(key);
    const s = scenarios[key];
    setSub(s.sub);
    setPlayState('playing');

    setActiveRows(new Set(['core']));
    setRowVals({ core: 'soul + policy' });

    const sectionTriggers: Array<{ marker: string; id: string }> = [
      { marker: '# ── MODE LAYER',     id: 'mode'     },
      { marker: '# ── MODALITY LAYER', id: 'modality' },
      { marker: '# ── TASK LAYER',     id: 'task'     },
      { marker: '# ── CONTEXT LAYER',  id: 'context'  },
    ];

    stopTimer();
    setDisplayedLines([]);

    let li = 0;
    let ci = 0;
    let currentLine: { type: TokenType; text: string } | null = null;

    timerRef.current = setInterval(() => {
      if (li >= s.lines.length) {
        stopTimer();
        setPlayState('done');
        return;
      }

      const [type, text] = s.lines[li];

      if (ci === 0) {
        currentLine = { type, text: '' };

        if (type === 'c') {
          const trigger = sectionTriggers.find((t) => text.includes(t.marker));
          if (trigger) {
            const { id } = trigger;
            setActiveRows((prev) => new Set([...prev, id]));
            setRowVals((prev) => ({ ...prev, [id]: getLayerVal(id, s) }));
          }
        }
      }

      const speed = type === 'c' ? 3 : 1;
      const end = Math.min(ci + speed, text.length);
      const partial = text.slice(0, end);
      ci = end;

      setDisplayedLines((prev) =>
        currentLine && ci <= text.length
          ? [...prev.slice(0, li), { type, text: partial }]
          : prev,
      );

      if (ci >= text.length) {
        li++;
        ci = 0;
        currentLine = null;
      }
    }, 38);
  }

  function selectScenario(key: string) {
    stopTimer();
    setActive(key);
    const s = scenarios[key];
    setSub(s.sub);
    setPlayState('idle');
    setDisplayedLines([]);
    setActiveRows(new Set(['core']));
    setRowVals({ core: 'soul + policy' });
  }

  useEffect(() => {
    return () => stopTimer();
  }, []);

  useEffect(() => {
    if (bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [displayedLines]);

  const tokenClass: Record<TokenType, string> = {
    c: styles.tokComment,
    l: styles.tokLabel,
    v: styles.tokValue,
  };

  return (
    <div className={styles.assemblyRoot}>
      <div className={styles.scenarioBar}>
        <div className={styles.scenarioBtns}>
          {['support', 'crisis', 'exercise'].map((key) => (
            <button
              key={key}
              className={[styles.scenarioBtn, active === key ? styles.scenarioBtnActive : ''].join(' ')}
              onClick={() => selectScenario(key)}
            >
              {key === 'support' ? 'Support session' : key === 'crisis' ? 'Crisis response' : 'Guided exercise (CBT)'}
            </button>
          ))}
        </div>
        <button
          className={[styles.playBtn, playState === 'playing' ? styles.playBtnPlaying : ''].join(' ')}
          onClick={() => runScenario(active)}
          disabled={playState === 'playing'}
        >
          {playState === 'idle' ? '\u25B6  Play' : playState === 'playing' ? '\u25CF  Playing\u2026' : '\u21BB  Replay'}
        </button>
      </div>

      <div className={styles.assemblyShell}>
        {/* Layer indicator panel */}
        <div className={styles.layerPanel}>
          {layerDefs.map((l) => (
            <div
              key={l.id}
              className={[styles.layerRow, activeRows.has(l.id) ? styles.layerRowOn : ''].join(' ')}
            >
              <div className={styles.layerRowDot} style={{ background: l.dot }} />
              <span className={styles.layerRowName}>{l.label}</span>
              <span className={styles.layerRowVal}>{rowVals[l.id] ?? '—'}</span>
            </div>
          ))}
        </div>

        {/* Terminal output */}
        <div className={styles.terminal}>
          <div className={styles.terminalBar}>
            <div className={styles.trafficDot} style={{ background: '#ef4444', opacity: 0.5 }} />
            <div className={styles.trafficDot} style={{ background: '#f59e0b', opacity: 0.5 }} />
            <div className={styles.trafficDot} style={{ background: '#22c55e', opacity: 0.5 }} />
            <span className={styles.terminalTitle}>assembled_prompt.txt</span>
            <span className={styles.terminalSub}>{sub}</span>
          </div>
          <div className={styles.terminalBody} ref={bodyRef}>
            {displayedLines.map((line, i) => (
              <React.Fragment key={i}>
                <span className={tokenClass[line.type]}>{line.text}</span>
                {'\n'}
              </React.Fragment>
            ))}
            <span className={styles.cursor} />
          </div>
        </div>
      </div>
    </div>
  );
}
