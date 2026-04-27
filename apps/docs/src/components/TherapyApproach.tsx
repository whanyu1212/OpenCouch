import React, { useState } from 'react';
import Link from '@docusaurus/Link';
import styles from './TherapyApproach.module.css';

/* ── Therapeutic approaches ────────────────────────────────────────────────── */

interface Approach {
  id: string;
  label: string;
  full: string;
  role: string;
  desc: string;
  goodFor: string[];
  avoid: string[];
}

const APPROACHES: Approach[] = [
  {
    id: 'motivational_interviewing',
    label: 'MI ✓',
    full: 'Motivational Interviewing',
    role: 'Active — default for supportive style',
    desc: 'Partnership over authority. Reflective listening, respect for autonomy, evoking the user\'s own reasons and next steps. OARS: open questions, affirmations, reflections, summaries.',
    goodFor: ['Ambivalence', 'Stuck feelings', 'Exploring change', 'Building self-awareness'],
    avoid: ['Lecturing', 'Rushing to fix', 'Forcing change talk', 'Coach-script tone'],
  },
  {
    id: 'pfa',
    label: 'PFA ✓',
    full: 'Psychological First Aid',
    role: 'Active — LLM-routed for acute distress',
    desc: 'Humane, supportive, practical help. Calm presence, emotional stabilization, immediate needs. No forced disclosure or intensive processing.',
    goodFor: ['Acute distress', 'Emotional stabilization', 'Practical next steps', 'Crisis-adjacent support'],
    avoid: ['Deep interpretation', 'Probing for trauma details', 'Premature reframing'],
  },
  {
    id: 'cbt',
    label: 'CBT ✓',
    full: 'Cognitive Behavioral Therapy',
    role: 'Active — LLM-routed for thought work and psychoeducation',
    desc: 'Collaborative, concrete, one technique at a time. Identify a specific thought or behavior target, examine evidence, generate a balanced alternative. Small, doable practice steps.',
    goodFor: ['Thought records', 'Cognitive distortions', 'Behavioral activation', 'Problem-solving structure'],
    avoid: ['Forcing reframes before validation', 'Worksheet dumps', 'Clinical jargon', 'Debate-club tone'],
  },
  {
    id: 'grief_support',
    label: 'Grief ✓',
    full: 'Grief Support',
    role: 'Active — LLM-routed for loss and bereavement',
    desc: 'Make room for grief without rushing resolution. Validate mixed emotions. Respect that grief is not a problem to fix quickly.',
    goodFor: ['Bereavement', 'Loss processing', 'Mixed or contradictory emotions', 'Sitting with pain'],
    avoid: ['Cliché comfort', 'Silver linings', 'Treating grief as pathology', 'Tidy lessons'],
  },
  {
    id: 'interpersonal_therapy',
    label: 'IPT ✓',
    full: 'Interpersonal Therapy',
    role: 'Active — LLM-routed for relationship and role issues',
    desc: 'Focuses on how relationships and social roles affect mood. Helps the user understand interpersonal patterns and communication dynamics.',
    goodFor: ['Relationship strain', 'Role transitions', 'Social isolation', 'Communication patterns'],
    avoid: ['Blaming others', 'Overanalyzing every relationship', 'Ignoring internal experience'],
  },
  {
    id: 'act',
    label: 'ACT ✓',
    full: 'Acceptance & Commitment Therapy',
    role: 'Active — LLM-routed for avoidance and values work',
    desc: 'Psychological flexibility through acceptance, defusion, and values-driven action. Name the thought, hold it lightly, choose what matters.',
    goodFor: ['Avoidance patterns', 'Values clarification', 'Acceptance of difficult emotions', 'Committed action'],
    avoid: ['Forcing positivity', 'Dismissing pain', 'Overcomplicating metaphors', 'Abstract philosophizing'],
  },
  {
    id: 'dbt_skills',
    label: 'DBT ✓',
    full: 'DBT Skills Training',
    role: 'Active — LLM-routed for emotional overwhelm',
    desc: 'Concrete skills for emotional regulation, distress tolerance, and interpersonal effectiveness. TIPP, STOP, radical acceptance.',
    goodFor: ['Emotional overwhelm', 'Distress tolerance', 'Impulse management', 'Interpersonal effectiveness'],
    avoid: ['Full DBT protocol', 'Diagnostic framing', 'Assuming personality pathology', 'Skills without validation first'],
  },
];

/* ── Response styles ───────────────────────────────────────────────────────── */

interface ResponseStyle {
  id: string;
  label: string;
  when: string;
  goal: string;
}

const RESPONSE_STYLES: ResponseStyle[] = [
  // ── Therapeutic response styles (dispatched per turn by the subgraph) ──
  { id: 'supportive', label: 'Supportive', when: 'Default — user seeking emotional support, sharing feelings, or greeting', goal: 'Validate before suggesting. Reflect emotional state. One helpful next step. Concise.' },
  { id: 'reflective', label: 'Reflective', when: 'User is describing a recurring pattern they\'ve already named', goal: 'Name 1–2 patterns carefully. Tentative, testable. Preserve user\'s framing.' },
  { id: 'clarifying', label: 'Clarifying', when: 'Ambiguous message — agent doesn\'t know what "it" refers to', goal: 'One context-gathering question. About context, not content. No assumptions.' },
  { id: 'psychoeducation', label: 'Psychoeducation', when: 'User describes a reaction AND seeks understanding ("why am I crying?")', goal: 'One short normalizing explanation. Pivot back to user\'s experience. No clinical jargon.' },
  { id: 'technique', label: 'Technique', when: 'User wants structured therapeutic work without launching a named exercise (examine a thought, weigh evidence, work through a dilemma)', goal: 'Collaborative step-by-step shaped by the active therapeutic_approach. No free-form lecturing.' },
  { id: 'guided_exercise', label: 'Guided exercise', when: 'User explicitly requests a structured exercise (13 exercises available)', goal: 'One exercise at a time. Multi-turn step tracking with a pinned approach. Check pace between steps.' },
  { id: 'closing', label: 'Closing', when: 'User signals wind-down ("I should go", "thanks, this helped")', goal: 'Warm wrap-up. Don\'t ask a new question or pivot to a new topic.' },
];

/* ── Principles ─────────────────────────────────────────────────────────────── */

interface Principle {
  label: string;
  desc: string;
}

const PRINCIPLES: Principle[] = [
  { label: 'Supportive, not clinical', desc: 'Validates, reflects, offers structure — but does not diagnose, interpret deeply, or claim clinical authority.' },
  { label: 'Honest about limits', desc: 'Not a therapist, not a diagnostic tool, not an emergency service. Says so clearly in onboarding and prompts.' },
  { label: 'Safety overrides everything', desc: 'Crisis detection runs before every response. Approach overlays, tone, and skills can never weaken crisis policy.' },
  { label: 'Respect user agency', desc: 'Supports decision-making rather than directing it. Evokes the user\'s own reasons, not the agent\'s prescription.' },
  { label: 'Bridge, not replacement', desc: 'Fills gaps in access to therapy with immediate support and structured practice. Defers clinical work to professionals.' },
  { label: 'Useful without pretending', desc: 'Warm but not sugary. Direct but not blunt. Emotionally accurate without being performative or poetic.' },
];

/* ── Component ──────────────────────────────────────────────────────────────── */

export default function TherapyApproach() {
  const [activeApproach, setActiveApproach] = useState<string | null>(null);
  const mod = activeApproach ? APPROACHES.find(m => m.id === activeApproach) ?? null : null;

  return (
    <div className={styles.root}>
      {/* ── Principles ─────────────────────────────────────────── */}
      <section className={styles.principlesSection}>
        <h3 className={styles.sectionTitle}>Core beliefs</h3>
        <div className={styles.principleGrid}>
          {PRINCIPLES.map(p => (
            <div key={p.label} className={styles.principleCard}>
              <span className={styles.principleLabel}>{p.label}</span>
              <span className={styles.principleDesc}>{p.desc}</span>
            </div>
          ))}
        </div>
      </section>

      {/* ── Therapeutic approaches ─────────────────────────────── */}
      <section className={styles.approachSection}>
        <h3 className={styles.sectionTitle}>Therapeutic approaches</h3>
        <p className={styles.sectionSub}>
          Designed as overlays and stances — not full treatments. All seven approaches are wired and selected per turn by the LLM dispatcher based on the user&apos;s message context. Click to see what each is good for and what to avoid.
        </p>
        <div className={styles.approachRow}>
          {APPROACHES.map(m => (
            <button
              key={m.id}
              className={[styles.approachCard, activeApproach === m.id ? styles.approachActive : ''].join(' ')}
              onClick={() => setActiveApproach(p => p === m.id ? null : m.id)}
            >
              <span className={styles.approachAbbr}>{m.label}</span>
              <span className={styles.approachFull}>{m.full}</span>
              <span className={styles.approachRole}>{m.role}</span>
            </button>
          ))}
        </div>

        {mod && (
          <div className={styles.approachDetail} key={mod.id}>
            <p className={styles.approachDesc}>{mod.desc}</p>
            <div className={styles.approachColumns}>
              <div className={styles.approachCol}>
                <span className={styles.colLabel}>Good for</span>
                <ul className={styles.tagList}>
                  {mod.goodFor.map(g => <li key={g} className={styles.tagGood}>{g}</li>)}
                </ul>
              </div>
              <div className={styles.approachCol}>
                <span className={styles.colLabel}>Avoid</span>
                <ul className={styles.tagList}>
                  {mod.avoid.map(a => <li key={a} className={styles.tagAvoid}>{a}</li>)}
                </ul>
              </div>
            </div>
          </div>
        )}
      </section>

      {/* ── Response styles ────────────────────────────────────── */}
      <section className={styles.responseStylesSection}>
        <h3 className={styles.sectionTitle}>Therapeutic response styles</h3>
        <p className={styles.sectionSub}>
          Seven response styles are dispatched per turn by the therapeutic subgraph. The LLM classifier is the primary path; deterministic regex fires only for explicit exercise opt-out and as a fallback when no LLM is configured. Crisis responses bypass this subgraph entirely and are handled by the crisis gate (see <Link to="/docs/philosophy/crisis-gate">Crisis Gate</Link>).
        </p>
        <div className={styles.responseStyleTable}>
          {RESPONSE_STYLES.map(m => (
            <div key={m.id} className={styles.responseStyleRow}>
              <span className={styles.responseStyleName}>{m.label}</span>
              <span className={styles.responseStyleWhen}>{m.when}</span>
              <span className={styles.responseStyleGoal}>{m.goal}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
