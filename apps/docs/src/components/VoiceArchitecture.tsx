import React from 'react';
import s from './VoiceArchitecture.module.css';

/* ================================================================
   VoiceArchitecture — Voice pipeline diagram

   Static visual showing the current experimental voice path:
   Browser ↔ FastAPI ↔ Realtime, with lightweight client/server
   interruption controls and bounded prompt preload.
   ================================================================ */

export default function VoiceArchitecture(): React.JSX.Element {
  return (
    <div className={s.root}>
      {/* Main pipeline — horizontal */}
      <div className={s.pipeline}>
        <div className={`${s.actor} ${s.actorBrowser}`}>
          <span className={s.actorIcon}>{'\uD83C\uDF10'}</span>
          <span className={s.actorLabel}>Browser</span>
          <span className={s.actorSub}>mic + speaker</span>
        </div>

        <div className={s.link}>
          <div className={s.linkLine} />
          <span className={s.linkLabel}>WebSocket</span>
          <span className={s.linkSub}>PCM16 audio</span>
        </div>

        <div className={`${s.actor} ${s.actorServer}`}>
          <span className={s.actorIcon}>{'\u2699'}</span>
          <span className={s.actorLabel}>FastAPI</span>
          <span className={s.actorSub}>voice bridge</span>
        </div>

        <div className={s.link}>
          <div className={s.linkLine} />
          <span className={s.linkLabel}>WebSocket</span>
          <span className={s.linkSub}>Realtime protocol</span>
        </div>

        <div className={`${s.actor} ${s.actorRealtime}`}>
          <span className={s.actorIcon}>{'\uD83C\uDF99'}</span>
          <span className={s.actorLabel}>OpenAI Realtime</span>
          <span className={s.actorSub}>STT + LLM + TTS</span>
        </div>
      </div>

      {/* Side-effects (below the server) */}
      <div className={s.sideEffects}>
        <div className={s.seConn} />

        <div className={s.seRow}>
          <div className={`${s.seBox} ${s.seCrisis}`}>
            <span className={s.seIcon}>{'\u25C6'}</span>
            <div className={s.seContent}>
              <span className={s.seLabel}>Prompt Preload</span>
              <span className={s.seSub}>bounded memory context on connect</span>
            </div>
          </div>

          <div className={`${s.seBox} ${s.seExtract}`}>
            <span className={s.seIcon}>{'\u25B6'}</span>
            <div className={s.seContent}>
              <span className={s.seLabel}>VAD + Truncate</span>
              <span className={s.seSub}>server interruption with client sync</span>
            </div>
          </div>

          <div className={`${s.seBox} ${s.seMemory}`}>
            <span className={s.seIcon}>{'\u25C6'}</span>
            <div className={s.seContent}>
              <span className={s.seLabel}>Local Ducking</span>
              <span className={s.seSub}>browser-side preemptive playback mute</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
