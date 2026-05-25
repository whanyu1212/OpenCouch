import React from 'react';
import s from './VoiceArchitecture.module.css';

/* Static reference diagram of the OpenAI Realtime voice runtime. */

export default function VoiceArchitecture(): React.JSX.Element {
  return (
    <div className={s.root}>
      <Section
        kicker="Transport"
        title="Browser WebRTC -> OpenAI Realtime -> FastAPI tools"
      >
        <div className={s.transportRow}>
          <Box title="Browser" sub="mic · speaker · data channel" />
          <Arrow label="ephemeral secret + SDP" />
          <Box title="OpenAI Realtime" sub="speech loop · VAD · tools" emphasis />
          <Arrow label="function calls" />
          <Box title="FastAPI" sub="/api/voice/realtime/*" />
        </div>
      </Section>

      <Connector label="Realtime generates speech; OpenCouch executes product state and tools" />

      <Section
        kicker="Session core"
        title="Realtime model with app-owned policy"
      >
        <div className={s.sessionRow}>
          <article className={s.agentCard}>
            <header className={s.agentHead}>
              <span className={s.agentKicker}>live model</span>
              <h4 className={s.agentTitle}>Realtime session</h4>
            </header>
            <p className={s.agentSub}>
              Owns the live spoken response loop. Receives compact
              instructions, private memory context, Realtime tool schemas, and
              per-turn policy hints.
            </p>
            <ul className={s.agentList}>
              <li>server VAD with interrupt support</li>
              <li>input transcription for turn recording</li>
              <li><code>response.create</code> policy injection</li>
            </ul>
          </article>

          <div className={s.handoffCol}>
            <span className={[s.handoff, s.handoffOut].join(' ')}>
              <span className={s.handoffLabel}>final transcript</span>
              <span className={s.handoffArrow} aria-hidden>{'\u2192'}</span>
            </span>
            <span className={[s.handoff, s.handoffIn].join(' ')}>
              <span className={s.handoffArrow} aria-hidden>{'\u2190'}</span>
              <span className={s.handoffLabel}>tool output</span>
            </span>
          </div>

          <article className={[s.agentCard, s.agentCardCrisis].join(' ')}>
            <header className={s.agentHead}>
              <span className={s.agentKicker}>backend policy</span>
              <h4 className={s.agentTitle}>Voice endpoints</h4>
            </header>
            <p className={s.agentSub}>
              Create the session, execute tools, prepare observe-only turn
              policy, record finalized turns, and close persistent sessions.
            </p>
            <ul className={s.agentList}>
              <li><code>/session</code> builds config and client secret</li>
              <li><code>/tools</code> executes app-owned function calls</li>
              <li><code>/turn</code> records finalized transcripts</li>
              <li><code>/end</code> runs shared session finalization</li>
            </ul>
          </article>

          <div className={s.taskWrap}>
            <span className={s.taskTetherLabel}>shared tool services</span>
            <article className={s.taskCard}>
              <header className={s.agentHead}>
                <span className={s.agentKicker}>reused by text and voice</span>
                <h4 className={s.agentTitle}>Memory · lookup · exercises</h4>
              </header>
              <p className={s.agentSub}>
                Realtime schemas call the same service functions used by text
                SDK specialists.
              </p>
              <ul className={s.agentList}>
                <li>memory control and recall status</li>
                <li>grounded factual lookup and crisis resources</li>
                <li>therapeutic response skills and guided exercises</li>
              </ul>
            </article>
          </div>
        </div>
      </Section>

      <Connector label="Finalized voice turns are written into the shared runtime state" />

      <Section
        kicker="Persistence"
        title="Transcript recording plus shared session finalization"
      >
        <div className={s.sharedRow}>
          <article className={s.sharedCard}>
            <header className={s.agentHead}>
              <span className={s.agentKicker}>per-session identity</span>
              <h4 className={s.agentTitle}>Web setup state</h4>
            </header>
            <p className={s.agentSub}>
              Voice reuses the active web thread, memory mode, optional user id,
              and selected assistant voice.
            </p>
            <div className={s.fieldGrid}>
              {[
                'thread_id',
                'user_id',
                'memory_mode',
                'assistant_voice',
                'transcript',
                'tool_activity',
                'finalization_status',
              ].map((f) => (
                <code key={f} className={s.field}>{f}</code>
              ))}
            </div>
          </article>

          <article className={s.sharedCard}>
            <header className={s.agentHead}>
              <span className={s.agentKicker}>shared runtime</span>
              <h4 className={s.agentTitle}>PersistentAgentRuntime</h4>
            </header>
            <p className={s.agentSub}>
              Voice does not run a text turn, but it writes state through the
              same runtime stores and ends persistent sessions through the same
              finalizer.
            </p>
            <div className={s.fieldGrid}>
              {[
                'voice_session_memory_context',
                'build_voice_tool_context',
                'prepare_voice_turn_policy',
                'record_voice_turn',
                'end_session',
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

function Connector({ label }: { label: string }) {
  return (
    <div className={s.connector} role="presentation" aria-hidden>
      <span className={s.connectorLine} />
      <span className={s.connectorLabel}>{label}</span>
      <span className={s.connectorLine} />
    </div>
  );
}
