import React, { useEffect, useRef } from 'react';
import styles from './PromptVisuals.module.css';

interface FileInfo {
  name: string;
  kind: 'markdown' | 'code' | 'state';
  summary: string;
  excerpt: string;
}

const fileIndex: Record<string, FileInfo> = {
  'soul.md': {
    name: 'soul.md',
    kind: 'markdown',
    summary: 'Core character and values',
    excerpt:
      'Defines who OpenCouch is at its most fundamental level — grounded, warm, direct, and honest about its limits. Sets the emotional register and the non-negotiable stance that no mode or modality can override.',
  },
  'identity.md': {
    name: 'identity.md',
    kind: 'markdown',
    summary: 'Product identity and scope',
    excerpt:
      'How OpenCouch presents itself to users: an AI companion for talking through difficult moments, reflecting on patterns, and practising self-help techniques. Not a therapist. Not diagnostic. Not an emergency service.',
  },
  'policy/boundaries.md': {
    name: 'policy/boundaries.md',
    kind: 'markdown',
    summary: 'Hard behavioural constraints',
    excerpt:
      'Must not diagnose, claim therapeutic authority, give medication or legal advice, express false certainty about a user\'s internal state, or encourage dependency. Limits are stated clearly and alternatives surfaced.',
  },
  'policy/privacy.md': {
    name: 'policy/privacy.md',
    kind: 'markdown',
    summary: 'Data minimisation rules',
    excerpt:
      'Users share sensitive mental health information. Only necessary context is injected into prompts. Memory is minimal and reviewable. Principle of least necessary context applies at every step.',
  },
  'response_modes/support.md': {
    name: 'response_modes/support.md',
    kind: 'markdown',
    summary: 'Default supportive_conversation mode',
    excerpt:
      'Active when no crisis or safety check is needed. Validates before suggesting, reflects the user\'s emotional state, offers one helpful next step. Replies stay concise and grounded.',
  },
  'crisis_response.md': {
    name: 'response_modes/crisis_response.md',
    kind: 'markdown',
    summary: 'Crisis response mode',
    excerpt:
      'Activates when a risk signal is confirmed. Prioritises immediate safety, avoids clinical distance, encourages offline human support and emergency services where appropriate.',
  },
  'modes.py': {
    name: 'prompts/modes.py',
    kind: 'code',
    summary: 'Mode system prompt builder',
    excerpt:
      'Python code that reads the active mode\'s markdown, applies any approach overlay, and assembles the combined system prompt section. Enforces that only catalog-approved combinations are built.',
  },
  'pfa.md': {
    name: 'modalities/pfa.md + dbt_skills.md',
    kind: 'markdown',
    summary: 'Psychological First Aid (+ DBT skills)',
    excerpt:
      'Baseline for acute distress: calm presence, practical immediate next steps, emotional stabilisation. DBT skills bundled in. Avoids turning support into diagnosis or intensive therapy when someone is in crisis.',
  },
  'cbt.md': {
    name: 'modalities/cbt.md',
    kind: 'markdown',
    summary: 'CBT self-help overlay',
    excerpt:
      'Structured self-help for thought-checking, pattern identification, behaviour activation, and problem-solving. Avoids clinical phrasing and does not force reframes before the user is ready.',
  },
  'grief_support.md': {
    name: 'modalities/grief_support.md',
    kind: 'markdown',
    summary: 'Grief support overlay',
    excerpt:
      'Makes room for grief without rushing it. Validates mixed emotions, avoids silver linings, treats grief as non-pathological. Does not push the user toward resolution on any timeline.',
  },
  'act.md': {
    name: 'modalities/act.md',
    kind: 'markdown',
    summary: 'ACT defusion and values overlay',
    excerpt:
      'Acceptance and Commitment Therapy techniques: cognitive defusion, values clarification, willingness over control. Helps users relate differently to difficult thoughts rather than trying to eliminate them.',
  },
  'builders.py': {
    name: 'prompts/builders.py',
    kind: 'code',
    summary: 'Node task prompt builder',
    excerpt:
      'Generates node-specific task instructions in code — the final prompt section that tells the model exactly what to produce for this graph node (reply, classification, orientation question, etc.).',
  },
  'support_task': {
    name: 'support_reply_task (builders.py)',
    kind: 'code',
    summary: 'Task instruction for support replies',
    excerpt:
      'Tells the model to write the next conversational message: keep it short, stay warm, end with an open question if it feels natural. No unsolicited advice.',
  },
  'crisis_task': {
    name: 'crisis_reply_task (builders.py)',
    kind: 'code',
    summary: 'Task instruction for crisis replies',
    excerpt:
      'Tells the model to write a grounded, caring response that surfaces crisis resources clearly and keeps the door open for the user to continue.',
  },
  'state.history': {
    name: 'state.history',
    kind: 'state',
    summary: 'Recent conversation turns',
    excerpt:
      'The last N turns of the conversation, injected as a message history block. Length is bounded to stay within context limits.',
  },
  'state.message': {
    name: 'state.message',
    kind: 'state',
    summary: 'Current user message',
    excerpt:
      'The user\'s most recent message — appended as the final human turn so the model responds to exactly what was just said.',
  },
  'state.crisis': {
    name: 'state.crisis',
    kind: 'state',
    summary: 'Active crisis assessment',
    excerpt:
      'When a risk signal is present: the crisis level (0–3), confidence score, reason string, and pattern flags. Injected into context so the model can reference why crisis mode is active.',
  },
};

const kindLabel: Record<FileInfo['kind'], string> = {
  markdown: 'markdown',
  code: 'python',
  state: 'runtime state',
};

const layers = [
  {
    n: '1',
    label: 'Core',
    tag: 'always present',
    desc: "The permanent foundation: who OpenCouch is, what it will and won't do, and the hard safety and privacy boundaries that no mode can override.",
    files: ['soul.md', 'identity.md', 'policy/boundaries.md', 'policy/privacy.md'],
    dot: '#215f5a',
  },
  {
    n: '2',
    label: 'Mode',
    tag: 'graph-selected',
    desc: 'The response mode selected by the graph. Shapes the goal and tone of the entire response — supportive_conversation, crisis_response, orientation, guided_exercise, pattern_reflection, psychoeducation, and more.',
    files: ['response_modes/support.md', 'crisis_response.md', 'modes.py'],
    dot: '#2d7a74',
  },
  {
    n: '3',
    label: 'Approach',
    tag: 'optional overlay',
    desc: 'A therapeutic technique lens selected by the modality_selector based on semantic signals. MI is applied as a baseline to certain modes automatically, not as a selectable overlay.',
    files: ['pfa.md', 'cbt.md', 'grief_support.md', 'act.md'],
    dot: '#3d9990',
  },
  {
    n: '4',
    label: 'Task',
    tag: 'node-specific',
    desc: 'The exact instruction for this graph node — what to write, what format to use, what constraints apply. Includes response_guidance for turn-specific shaping.',
    files: ['builders.py', 'support_task', 'crisis_task'],
    dot: '#78b8af',
  },
  {
    n: '5',
    label: 'Context',
    tag: 'runtime',
    desc: 'Recent conversation history, current user message, and any active crisis signals injected from graph state.',
    files: ['state.history', 'state.message', 'state.crisis'],
    dot: '#a8cdc9',
  },
];

const legend = [
  { color: '#215f5a', label: 'Core — immutable' },
  { color: '#2d7a74', label: 'Mode — graph-selected' },
  { color: '#3d9990', label: 'Modality — optional' },
  { color: '#78b8af', label: 'Task — node-built' },
  { color: '#a8cdc9', label: 'Context — runtime state' },
];

function FileChip({ fileKey }: { fileKey: string }) {
  const info = fileIndex[fileKey];
  if (!info) {
    return <span className={styles.layerFile}>{fileKey}</span>;
  }
  return (
    <span className={styles.fileChipWrap}>
      <span className={[styles.layerFile, styles.layerFileClickable].join(' ')}>
        {fileKey}
        <span className={styles.layerFileIcon} aria-hidden>↗</span>
      </span>
      <span className={styles.hoverCard}>
        <span className={styles.hoverCardHeader}>
          <span className={styles.hoverCardName}>{info.name}</span>
          <span className={styles.hoverCardKind}>{kindLabel[info.kind]}</span>
        </span>
        <span className={styles.hoverCardSummary}>{info.summary}</span>
        <span className={styles.hoverCardExcerpt}>{info.excerpt}</span>
      </span>
    </span>
  );
}

export default function PromptLayerStack() {
  const stackRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const stack = stackRef.current;
    if (!stack) return;
    const items = stack.querySelectorAll<HTMLElement>('[data-layer-item]');
    const obs = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          items.forEach((el) => el.classList.add(styles.layerVisible));
          obs.disconnect();
        }
      },
      { threshold: 0.05 },
    );
    obs.observe(stack);
    return () => obs.disconnect();
  }, []);

  return (
    <div className={styles.layerStackRoot}>
      <p className={styles.alwaysLabel}>Always present in every prompt</p>

      <div className={styles.stack} ref={stackRef}>
        {[...layers].reverse().map((layer, i) => (
          <div
            key={layer.n}
            data-layer-item
            className={styles.layer}
            style={{ transitionDelay: `${i * 70}ms` }}
          >
            <span className={styles.layerNum}>{layer.n}</span>
            <div className={styles.layerBody}>
              <div className={styles.layerLabel}>
                {layer.label}
                <span className={styles.layerTag}>{layer.tag}</span>
              </div>
              <p className={styles.layerDesc}>{layer.desc}</p>
              <div className={styles.layerFiles}>
                {layer.files.map((f) => (
                  <FileChip key={f} fileKey={f} />
                ))}
              </div>
            </div>
            <div className={styles.layerDot} style={{ background: layer.dot }} />
          </div>
        ))}
      </div>

      <div className={styles.legend}>
        {legend.map((l) => (
          <div key={l.label} className={styles.legendItem}>
            <div className={styles.legendDot} style={{ background: l.color }} />
            {l.label}
          </div>
        ))}
      </div>
    </div>
  );
}
