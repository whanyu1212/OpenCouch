import React from 'react';
import s from './VoiceArchitecture.module.css';

/* ================================================================
   VoiceArchitecture
   -----------------
   Static reference diagram of the OpenAI Realtime voice runtime.

   Three labeled sections stacked vertically:
     1. Realtime transport — Browser → OpenAI Realtime → Backend tools
     2. Session core       — Realtime model + app-owned policy/tools
     3. Shared state       — PersistentAgentRuntime + memory backends

   No animation. Site-native white card aesthetic.
   ================================================================ */

export default function VoiceArchitecture(): React.JSX.Element {
  return (
    <div className={s.root}>
      {/* ── Section 1: transport ──────────────────────────────── */}
      <Section
        kicker="Realtime transport"
        title="Browser ⇢ OpenAI Realtime ⇢ Backend tools"
      >
        <div className={s.transportRow}>
          <Box title="Browser" sub="mic + speaker" />
          <Arrow label="client secret" />
          <Box title="OpenAI Realtime" sub="speech-to-speech · tools" emphasis />
          <Arrow label="function calls" />
          <Box title="FastAPI" sub="/api/voice/realtime/*" />
        </div>
      </Section>

      <Connector label="WebRTC audio · data-channel events call app-owned tools" />

      {/* ── Section 2: session core ───────────────────────────── */}
      <Section
        kicker="Session core"
        title="Realtime session · app-owned policy/tools"
      >
        <div className={s.sessionRow}>
          {/* TherapeuticAgent */}
          <article className={s.agentCard}>
            <header className={s.agentHead}>
              <span className={s.agentKicker}>default agent</span>
              <h4 className={s.agentTitle}>TherapeuticAgent</h4>
            </header>
            <p className={s.agentSub}>
              Holds the thread. <code>on_user_turn_completed</code> runs the
              LLM crisis gate, turn policy, and selective recall after each turn.
            </p>
            <ul className={s.agentList}>
              <li><code>start_grounding_exercise</code></li>
              <li><code>MemoryControlToolset</code></li>
              <li><code>GroundedLookupToolset</code></li>
            </ul>
          </article>

          {/* Handoff column */}
          <div className={s.handoffCol}>
            <span className={[s.handoff, s.handoffOut].join(' ')}>
              <span className={s.handoffLabel}>LLM crisis gate</span>
              <span className={s.handoffArrow} aria-hidden>{'\u27F6'}</span>
            </span>
            <span className={[s.handoff, s.handoffIn].join(' ')}>
              <span className={s.handoffArrow} aria-hidden>{'\u27F5'}</span>
              <span className={s.handoffLabel}>de_escalate</span>
            </span>
          </div>

          {/* CrisisAgent */}
          <article className={[s.agentCard, s.agentCardCrisis].join(' ')}>
            <header className={s.agentHead}>
              <span className={s.agentKicker}>handoff target</span>
              <h4 className={s.agentTitle}>CrisisAgent</h4>
            </header>
            <p className={s.agentSub}>
              Acknowledge · resources · stay present. Restores
              the therapeutic agent on de-escalate.
            </p>
            <ul className={s.agentList}>
              <li><code>CrisisResourceToolset</code></li>
              <li>verified hotlines · location-aware</li>
            </ul>
          </article>

          {/* Bounded task — sits beneath the therapist column */}
          <div className={s.taskWrap}>
            <span className={s.taskTetherLabel}>start_grounding_exercise</span>
            <article className={s.taskCard}>
              <header className={s.agentHead}>
                <span className={s.agentKicker}>bounded task</span>
                <h4 className={s.agentTitle}>VoiceExerciseTask</h4>
              </header>
              <p className={s.agentSub}>
                10 voice-allowlisted exercises. Owns the loop until the user
                completes or exits.
              </p>
            </article>
          </div>
        </div>
      </Section>

      <Connector label="reads user identity + memory · writes durable insights at session end" />

      {/* ── Section 3: shared state ───────────────────────────── */}
      <Section
        kicker="Shared state"
        title="Per-session userdata + worker-singleton runtime"
      >
        <div className={s.sharedRow}>
          <article className={s.sharedCard}>
            <header className={s.agentHead}>
              <span className={s.agentKicker}>typed userdata</span>
              <h4 className={s.agentTitle}>SessionData</h4>
            </header>
            <p className={s.agentSub}>
              Per-session state shared across agents, tools, and tasks.
            </p>
            <div className={s.fieldGrid}>
              {[
                'user_id',
                'thread_id',
                'memory_mode',
                'crisis_level',
                'max_crisis_level',
                'turn_index',
                'exercise_consent_turn_index',
                'recommended_exercise_type',
                'proactive_recall_enabled',
                'pending_memory_delete',
                'therapeutic_instructions',
                'injected_semantic_memory_keys',
                'recent_exercise_types',
              ].map((f) => (
                <code key={f} className={s.field}>{f}</code>
              ))}
            </div>
          </article>

          <article className={s.sharedCard}>
            <header className={s.agentHead}>
              <span className={s.agentKicker}>worker singleton</span>
              <h4 className={s.agentTitle}>PersistentAgentRuntime</h4>
            </header>
            <p className={s.agentSub}>
              One per worker process. Postgres-first with SQLite fallback.
              Survives across rooms.
            </p>
            <div className={s.fieldGrid}>
              {[
                'memory store',
                'crisis log',
                'session feedback',
                'embedding provider',
                'control LLM',
              ].map((f) => (
                <code key={f} className={s.field}>{f}</code>
              ))}
            </div>
          </article>
        </div>
      </Section>
    </div>
  );
}

/* ── Section wrapper ───────────────────────────────────────── */

function Section({
  kicker,
  title,
  children,
}: {
  kicker: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className={s.section}>
      <header className={s.sectionHead}>
        <span className={s.sectionKicker}>{kicker}</span>
        <h3 className={s.sectionTitle}>{title}</h3>
      </header>
      {children}
    </section>
  );
}

/* ── Box used in the transport row ─────────────────────────── */

function Box({
  title,
  sub,
  emphasis,
}: {
  title: string;
  sub: string;
  emphasis?: boolean;
}) {
  return (
    <div className={[s.box, emphasis ? s.boxEmphasis : ''].join(' ')}>
      <span className={s.boxTitle}>{title}</span>
      <span className={s.boxSub}>{sub}</span>
    </div>
  );
}

/* ── Horizontal arrow between transport boxes ──────────────── */

function Arrow({ label }: { label: string }) {
  return (
    <div className={s.arrow} role="presentation">
      <span className={s.arrowLabel}>{label}</span>
      <div className={s.arrowLine}>
        <span className={s.arrowHead} aria-hidden>{'\u25B6'}</span>
      </div>
    </div>
  );
}

/* ── Vertical connector between sections ───────────────────── */

function Connector({ label }: { label: string }) {
  return (
    <div className={s.connector} role="presentation" aria-hidden>
      <span className={s.connectorLine} />
      <span className={s.connectorLabel}>{label}</span>
      <span className={s.connectorLine} />
    </div>
  );
}
