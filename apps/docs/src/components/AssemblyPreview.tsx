import React, { useEffect, useMemo, useRef, useState } from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './PromptVisuals.module.css';

/* ── Layer detection ──────────────────────────────────────────────────────────
 *
 * The dumps under apps/docs/static/prompt-dumps/*.txt are trimmed but real —
 * they keep the actual section headers from the live prompts. We detect which
 * layer a given line belongs to by matching prefix patterns. The first matched
 * layer for any line wins; everything before any match is treated as the
 * "preamble" (variant header / source comments / "===== SYSTEM PROMPT =====").
 *
 * Layer ids match the badges shown in the side panel.
 * ──────────────────────────────────────────────────────────────────────────── */

type LayerId =
  | 'preamble'
  | 'core'
  | 'response'
  | 'approach'
  | 'instructions'
  | 'rules'
  | 'recall'
  | 'task';

interface LayerDef {
  id: LayerId;
  label: string;
  dot: string;
  source: string;
  match: (line: string) => boolean;
}

// Match each line against the most specific layer first. The first hit wins.
// Order matters: e.g. "# Privacy Policy Notes" must match the core layer
// before any later response-style header could.
const LAYER_DEFS: LayerDef[] = [
  {
    id: 'task',
    label: 'Task & history',
    dot: '#a8cdc9',
    source: 'state.history + state.message',
    match: (l) => l.startsWith('===== USER/TASK PROMPT'),
  },
  {
    id: 'recall',
    label: 'Memory recall hint',
    dot: '#78b8af',
    source: 'load_memory_node → recall toggle',
    match: (l) => l.startsWith('═══ Memory reference guidance'),
  },
  {
    id: 'rules',
    label: 'Procedural rules',
    dot: '#5fa39a',
    source: 'procedural_profile.procedural_rules',
    match: (l) => l.startsWith('═══ Style rules'),
  },
  {
    id: 'instructions',
    label: 'Response instructions',
    dot: '#3d9990',
    source: 'agent/therapeutic/prompting/instructions.py',
    match: (l) => /^You are in [A-Z_]+ mode\./.test(l),
  },
  {
    id: 'approach',
    label: 'Approach',
    dot: '#2d7a74',
    source: 'agent/prompts/sources/modalities/*.md',
    match: (l) =>
      /^# (Motivational Interviewing|Cognitive Behavioural Therapy|Acceptance and Commitment Therapy|DBT Skills|Grief Support|Interpersonal Therapy|Psychological First Aid)/.test(
        l,
      ),
  },
  {
    id: 'response',
    label: 'Response knowledge',
    dot: '#3d9990',
    source: 'agent/prompts/sources/response_modes/*.md',
    match: (l) =>
      /^# (Supportive Conversation Mode|Pattern Reflection Mode|Psychoeducation Mode|Guided Exercise Mode|Closing Mode|Crisis (Response )?(Mode|Policy)|Safety Check Mode)/.test(
        l,
      ),
  },
  {
    id: 'core',
    label: 'Core identity',
    dot: '#215f5a',
    source: 'soul.md · identity.md · policy/*.md',
    match: (l) =>
      /^# (OpenCouch (Soul|Identity)|Boundaries Policy|Privacy Policy Notes)/.test(
        l,
      ),
  },
];

const LAYER_ORDER: LayerId[] = [
  'core',
  'response',
  'approach',
  'instructions',
  'rules',
  'recall',
  'task',
];

interface ScenarioMeta {
  id: string;
  label: string;
  file: string;
  sub: string;
  blurb: string;
  /** Layers we expect to see in this scenario; used to dim the inactive ones up front. */
  expects: LayerId[];
}

const SCENARIOS: ScenarioMeta[] = [
  {
    id: 'supportive_mi',
    label: 'Supportive · MI',
    file: 'prompt-dumps/supportive_mi.txt',
    sub: 'response_style=supportive · approach=motivational_interviewing',
    blurb:
      'The default therapeutic turn — soul + identity + boundaries + privacy + supportive response style + MI overlay + recall hint.',
    expects: ['core', 'response', 'approach', 'instructions', 'recall', 'task'],
  },
  {
    id: 'reflective_cbt_rules',
    label: 'Reflective · CBT + style rules',
    file: 'prompt-dumps/reflective_cbt_rules.txt',
    sub: 'response_style=reflective · approach=cbt · procedural rules loaded',
    blurb:
      'Same skeleton, plus a "Style rules" block injected near the end of the system prompt when the user has saved procedural preferences.',
    expects: ['core', 'response', 'approach', 'instructions', 'rules', 'recall', 'task'],
  },
  {
    id: 'guided_exercise_act_drift',
    label: 'Guided exercise · ACT (approach pinned)',
    file: 'prompt-dumps/guided_exercise_act_drift.txt',
    sub: 'response_style=guided_exercise · approach=act · mid-exercise side-turn',
    blurb:
      'Mid-exercise side-turn: the dispatcher could re-pick an approach, but exercise_state.exercise_modality is pinned at "act" so ACT framing stays loaded.',
    expects: ['core', 'response', 'approach', 'instructions', 'recall', 'task'],
  },
  {
    id: 'crisis_response',
    label: 'Crisis response',
    file: 'prompt-dumps/crisis_response.txt',
    sub: 'response_style=crisis_response · approach=pfa+dbt_skills',
    blurb:
      'Crisis path uses a different layer set: core + crisis policy + crisis_response style + PFA + DBT-skills. No procedural rules, no recall hint — by design.',
    expects: ['core', 'response', 'approach', 'instructions', 'task'],
  },
];

/* ── Component ──────────────────────────────────────────────────────────────── */

interface ParsedLine {
  text: string;
  layer: LayerId;
}

function classifyLines(raw: string): ParsedLine[] {
  const lines = raw.split('\n');
  const out: ParsedLine[] = [];
  let current: LayerId = 'preamble';
  for (const line of lines) {
    let next: LayerId | null = null;
    for (const def of LAYER_DEFS) {
      if (def.match(line)) {
        next = def.id;
        break;
      }
    }
    if (next) current = next;
    out.push({ text: line, layer: current });
  }
  return out;
}

// One body line per tick at the chosen speed — readers can see each
// section actually compose. Layer-header lines pause for an extra
// ``HEADER_DWELL_TICKS`` ticks so the side-panel update is legible.
const BASE_TYPE_INTERVAL_MS = 55;
const HEADER_DWELL_TICKS = 6;

type Speed = 0.5 | 1 | 2;

const SPEED_OPTIONS: { value: Speed; label: string }[] = [
  { value: 0.5, label: '0.5×' },
  { value: 1, label: '1×' },
  { value: 2, label: '2×' },
];

export default function AssemblyPreview() {
  const [scenarioId, setScenarioId] = useState<string>(SCENARIOS[0].id);
  const [parsed, setParsed] = useState<ParsedLine[]>([]);
  const [shown, setShown] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeLayer, setActiveLayer] = useState<LayerId | null>(null);
  const [seenLayers, setSeenLayers] = useState<Set<LayerId>>(new Set());
  const [speed, setSpeed] = useState<Speed>(1);

  const bodyRef = useRef<HTMLDivElement>(null);
  const tickRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const scenario = useMemo(
    () => SCENARIOS.find((s) => s.id === scenarioId) ?? SCENARIOS[0],
    [scenarioId],
  );
  const fileUrl = useBaseUrl(scenario.file);

  /* Load + parse the dump when the scenario changes. */
  useEffect(() => {
    let cancelled = false;
    setShown(0);
    setActiveLayer(null);
    setSeenLayers(new Set());
    setError(null);
    setParsed([]);
    setPlaying(false);

    fetch(fileUrl)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.text();
      })
      .then((text) => {
        if (cancelled) return;
        setParsed(classifyLines(text));
      })
      .catch((e) => {
        if (cancelled) return;
        setError(`Could not load ${scenario.file}: ${e.message}`);
      });

    return () => {
      cancelled = true;
      if (tickRef.current) clearTimeout(tickRef.current);
      tickRef.current = null;
    };
  }, [fileUrl, scenario.file]);

  /* Auto-scroll the terminal as new lines reveal. */
  useEffect(() => {
    if (bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [shown]);

  /* Drive the typing animation. One line per tick at the chosen
   * speed; layer-header lines linger for HEADER_DWELL_TICKS so the
   * side-panel update is visible. setTimeout chain instead of
   * setInterval so we can vary the delay per line. */
  useEffect(() => {
    if (!playing || parsed.length === 0) return;
    if (tickRef.current) clearTimeout(tickRef.current);

    const tickMs = Math.max(8, Math.round(BASE_TYPE_INTERVAL_MS / speed));

    function advance() {
      setShown((prev) => {
        if (prev >= parsed.length) {
          setPlaying(false);
          return prev;
        }

        // Reveal exactly one line per tick.
        const next = prev + 1;
        const justRevealed = parsed[prev];

        // Update side-panel indicator when the revealed line belongs
        // to a real layer (not the file preamble).
        if (justRevealed && justRevealed.layer !== 'preamble') {
          setActiveLayer(justRevealed.layer);
          setSeenLayers((s) => {
            if (s.has(justRevealed.layer)) return s;
            const ns = new Set(s);
            ns.add(justRevealed.layer);
            return ns;
          });
        }

        // Schedule the next reveal. Layer headers and dividers
        // dwell longer so readers can register the side-panel
        // change before more lines stream in.
        const isLayerSignal =
          justRevealed?.text.startsWith('# ') ||
          justRevealed?.text.startsWith('═══') ||
          justRevealed?.text.startsWith('=====');
        const delay = isLayerSignal ? tickMs * HEADER_DWELL_TICKS : tickMs;

        if (next < parsed.length) {
          tickRef.current = setTimeout(advance, delay);
        } else {
          // Final line just revealed — let the cursor blink one more
          // beat before flipping to "done".
          tickRef.current = setTimeout(() => setPlaying(false), tickMs * 2);
        }
        return next;
      });
    }

    tickRef.current = setTimeout(advance, tickMs);

    return () => {
      if (tickRef.current) clearTimeout(tickRef.current);
      tickRef.current = null;
    };
  }, [playing, parsed, speed]);

  /* Handlers */
  function play() {
    if (parsed.length === 0) return;
    setShown(0);
    setActiveLayer(null);
    setSeenLayers(new Set());
    setPlaying(true);
  }
  function showAll() {
    if (tickRef.current) {
      clearTimeout(tickRef.current);
      tickRef.current = null;
    }
    setPlaying(false);
    setShown(parsed.length);
    // Mark every layer present in the dump as "seen" so the indicator panel reflects it.
    const layersInDump = new Set<LayerId>();
    for (const p of parsed) {
      if (p.layer !== 'preamble') layersInDump.add(p.layer);
    }
    setSeenLayers(layersInDump);
    if (parsed.length > 0) {
      setActiveLayer(parsed[parsed.length - 1].layer);
    }
  }
  function selectScenario(id: string) {
    if (tickRef.current) {
      clearTimeout(tickRef.current);
      tickRef.current = null;
    }
    setScenarioId(id);
  }

  /* Render helpers */
  const visibleLines = parsed.slice(0, shown);
  const isDone = !playing && shown > 0 && shown >= parsed.length;
  const playLabel = playing
    ? '\u25CF  Playing\u2026'
    : isDone
      ? '\u21BB  Replay'
      : shown > 0
        ? '\u25B6  Continue'
        : '\u25B6  Play';

  return (
    <div className={styles.assemblyRoot}>
      {/* Scenario picker */}
      <div className={styles.scenarioBar}>
        <div className={styles.scenarioBtns}>
          {SCENARIOS.map((s) => (
            <button
              key={s.id}
              className={[
                styles.scenarioBtn,
                scenarioId === s.id ? styles.scenarioBtnActive : '',
              ].join(' ')}
              onClick={() => selectScenario(s.id)}
            >
              {s.label}
            </button>
          ))}
        </div>
        <div className={styles.scenarioActions}>
          <div className={styles.speedGroup} role="radiogroup" aria-label="Animation speed">
            {SPEED_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                className={[
                  styles.speedBtn,
                  speed === opt.value ? styles.speedBtnActive : '',
                ].join(' ')}
                onClick={() => setSpeed(opt.value)}
                role="radio"
                aria-checked={speed === opt.value}
              >
                {opt.label}
              </button>
            ))}
          </div>
          <button
            className={styles.skipBtn}
            onClick={showAll}
            disabled={parsed.length === 0}
          >
            Show all
          </button>
          <button
            className={[styles.playBtn, playing ? styles.playBtnPlaying : ''].join(' ')}
            onClick={play}
            disabled={playing || parsed.length === 0}
          >
            {playLabel}
          </button>
        </div>
      </div>

      <p className={styles.scenarioBlurb}>{scenario.blurb}</p>

      {/* Layer panel + terminal */}
      <div className={styles.assemblyShell}>
        {/* Layer indicator panel */}
        <div className={styles.layerPanel}>
          {LAYER_ORDER.map((id) => {
            const def = LAYER_DEFS.find((d) => d.id === id)!;
            const expected = scenario.expects.includes(id);
            const seen = seenLayers.has(id);
            const isActive = activeLayer === id;
            return (
              <div
                key={id}
                className={[
                  styles.layerRow,
                  seen ? styles.layerRowOn : '',
                  isActive ? styles.layerRowActive : '',
                  !expected ? styles.layerRowSkipped : '',
                ].join(' ')}
                title={
                  expected
                    ? `${def.label} — ${def.source}`
                    : `${def.label} — not used in this scenario`
                }
              >
                <div className={styles.layerRowDot} style={{ background: def.dot }} />
                <span className={styles.layerRowName}>{def.label}</span>
                {!expected && <span className={styles.layerRowSkippedTag}>skipped</span>}
              </div>
            );
          })}
          <p className={styles.layerPanelHint}>
            Layers light up as the typing animation walks past each section
            header in the prompt.
          </p>
        </div>

        {/* Terminal */}
        <div className={styles.terminal}>
          <div className={styles.terminalBar}>
            <div className={styles.trafficDot} style={{ background: '#ef4444', opacity: 0.5 }} />
            <div className={styles.trafficDot} style={{ background: '#f59e0b', opacity: 0.5 }} />
            <div className={styles.trafficDot} style={{ background: '#22c55e', opacity: 0.5 }} />
            <span className={styles.terminalTitle}>{scenario.file.split('/').pop()}</span>
            <span className={styles.terminalSub}>{scenario.sub}</span>
          </div>
          <div className={styles.terminalBody} ref={bodyRef}>
            {error && <span className={styles.terminalError}>{error}</span>}
            {!error && parsed.length === 0 && (
              <span className={styles.tokComment}>loading prompt dump…</span>
            )}
            {visibleLines.map((line, i) => (
              <React.Fragment key={i}>
                <PromptLine line={line} />
                {'\n'}
              </React.Fragment>
            ))}
            {playing && <span className={styles.cursor} />}
          </div>
        </div>
      </div>
    </div>
  );
}

/* Tokenize a line for syntax-ish coloring — comments, layer headers, "user:"
 * speaker labels, etc. Shows real prompt structure rather than fake DSL. */
function PromptLine({ line }: { line: ParsedLine }) {
  const t = line.text;

  // Layer/section headers — they look like markdown-ish or boxed dividers.
  if (t.startsWith('# ')) {
    return <span className={styles.tokLayerHeader}>{t}</span>;
  }
  if (t.startsWith('## ') || t.startsWith('### ')) {
    return <span className={styles.tokSubHeader}>{t}</span>;
  }
  if (t.startsWith('═══ ') || t.startsWith('=====')) {
    return <span className={styles.tokDivider}>{t}</span>;
  }
  // Bracketed "trimmed" notes we inserted.
  if (t.startsWith('[…') || t.startsWith('VARIANT:') || t.startsWith('SOURCE:')) {
    return <span className={styles.tokComment}>{t}</span>;
  }
  // Speaker labels in the task block.
  const speaker = t.match(/^(user|assistant): /);
  if (speaker) {
    return (
      <>
        <span className={styles.tokLabel}>{speaker[0]}</span>
        <span className={styles.tokValue}>{t.slice(speaker[0].length)}</span>
      </>
    );
  }
  return <span className={styles.tokValue}>{t}</span>;
}
