"use client";

import { useCallback, useEffect, useRef } from "react";
import { createVoiceSession } from "@/lib/api";
import {
  useSessionStore,
  getVoiceRefs,
  setVoiceNextPlayTime,
} from "@/lib/session";

const SAMPLE_RATE = 24000;

export default function VoicePage() {
  const {
    userId,
    threadId,
    voiceConnected,
    voiceAgentSpeaking,
    voiceTranscripts,
    voiceError,
    setVoiceConnected,
    setVoiceAgentSpeaking,
    addVoiceTranscript,
    setVoiceError,
    voiceSetRefs,
    voiceNewGeneration,
    clearVoiceTranscripts,
    voiceCleanup,
    voiceDisconnect,
  } = useSessionStore();

  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [voiceTranscripts]);

  // ── Audio playback (reads refs from module-level vars) ────────────
  const playAudioChunk = useCallback(
    (pcm16Bytes: Uint8Array) => {
      const { audioCtx: ctx, generation: gen } = getVoiceRefs();
      if (!ctx) return;

      const int16 = new Int16Array(
        pcm16Bytes.buffer,
        pcm16Bytes.byteOffset,
        pcm16Bytes.byteLength / 2
      );
      const float32 = new Float32Array(int16.length);
      for (let i = 0; i < int16.length; i++) {
        float32[i] = int16[i] / 32768.0;
      }

      const buffer = ctx.createBuffer(1, float32.length, SAMPLE_RATE);
      buffer.copyToChannel(float32, 0);

      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);

      const now = ctx.currentTime;
      const { nextPlayTime } = getVoiceRefs();
      const startTime = Math.max(now, nextPlayTime);
      source.start(startTime);
      setVoiceNextPlayTime(startTime + buffer.duration);

      source.onended = () => {
        // Ignore if this callback is from a stale session
        if (getVoiceRefs().generation !== gen) return;
        const { nextPlayTime: npt } = getVoiceRefs();
        if (ctx && ctx.currentTime >= npt - 0.05) {
          setVoiceAgentSpeaking(false);
        }
      };
    },
    [setVoiceAgentSpeaking]
  );

  // ── Connect ───────────────────────────────────────────────────────
  const connect = useCallback(async () => {
    // Guard against double-connect
    if (useSessionStore.getState().voiceConnected) return;

    setVoiceError(null);
    clearVoiceTranscripts();

    // Bump generation — any previous onclose becomes a no-op
    const gen = voiceNewGeneration();

    const ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
    voiceSetRefs({ audioCtx: ctx });
    setVoiceNextPlayTime(0);

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: SAMPLE_RATE,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      voiceSetRefs({ mediaStream: stream });
    } catch {
      // Clean up the AudioContext we just created
      ctx.close();
      voiceSetRefs({ audioCtx: null });
      setVoiceError("Microphone access denied");
      return;
    }

    const ws = createVoiceSession(userId, threadId, {
      onAudio: (bytes) => {
        setVoiceAgentSpeaking(true);
        playAudioChunk(bytes);
      },
      onTranscript: (role, text) => {
        addVoiceTranscript({ role, text });
      },
      onError: (msg) => {
        setVoiceError(msg);
      },
    });

    ws.onclose = () => {
      // Only act if this is still the active session
      if (getVoiceRefs().generation !== gen) return;
      setVoiceConnected(false);
      setVoiceAgentSpeaking(false);
      voiceCleanup(gen);
    };

    voiceSetRefs({ ws });

    const source = ctx.createMediaStreamSource(stream);
    const processor = ctx.createScriptProcessor(4096, 1, 1);
    voiceSetRefs({ processor });

    processor.onaudioprocess = (e) => {
      const { ws: activeWs } = getVoiceRefs();
      if (!activeWs || activeWs.readyState !== WebSocket.OPEN) return;
      if (useSessionStore.getState().voiceAgentSpeaking) return;

      const input = e.inputBuffer.getChannelData(0);
      const pcm16 = new Int16Array(input.length);
      for (let i = 0; i < input.length; i++) {
        pcm16[i] = Math.max(-32768, Math.min(32767, input[i] * 32768));
      }
      const bytes = new Uint8Array(pcm16.buffer);
      const base64 = btoa(String.fromCharCode(...bytes));
      activeWs.send(JSON.stringify({ type: "audio", data: base64 }));
    };

    source.connect(processor);
    processor.connect(ctx.destination);
    setVoiceConnected(true);
  }, [
    userId,
    threadId,
    playAudioChunk,
    setVoiceConnected,
    setVoiceAgentSpeaking,
    addVoiceTranscript,
    setVoiceError,
    voiceSetRefs,
    voiceNewGeneration,
    clearVoiceTranscripts,
    voiceCleanup,
  ]);

  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <header className="px-6 py-3.5 border-b border-oc-border flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          <h1 className="font-display text-lg text-oc-teal-900">Voice</h1>
          {voiceConnected && (
            <span className="text-[12px] font-mono text-oc-green">connected</span>
          )}
        </div>
        <div className="flex items-center gap-3">
          {voiceAgentSpeaking && (
            <div className="flex items-center gap-2 text-[13px] font-mono text-oc-cta">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-oc-cta opacity-60" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-oc-cta" />
              </span>
              speaking
            </div>
          )}
          {!voiceConnected ? (
            <button
              onClick={connect}
              className="bg-oc-teal-700 text-white px-5 py-2.5 rounded-xl text-[15px] font-medium hover:bg-oc-teal-600 transition-all shadow-sm"
            >
              Connect
            </button>
          ) : (
            <button
              onClick={voiceDisconnect}
              className="bg-oc-red-subtle text-oc-red border border-oc-red/20 px-5 py-2.5 rounded-xl text-[15px] font-medium hover:bg-red-100 transition-all"
            >
              Disconnect
            </button>
          )}
        </div>
      </header>

      {/* Voice area */}
      <div className="flex-1 flex flex-col items-center justify-center px-6">
        {!voiceConnected ? (
          <div className="text-center animate-fadeIn">
            <div className="w-24 h-24 rounded-2xl bg-oc-accent-glow flex items-center justify-center mx-auto mb-6">
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                className="w-12 h-12 text-oc-accent"
              >
                <path d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z" />
                <path d="M19 10v2a7 7 0 01-14 0v-2" />
                <line x1="12" y1="19" x2="12" y2="23" />
                <line x1="8" y1="23" x2="16" y2="23" />
              </svg>
            </div>
            <p className="font-display text-xl text-oc-text-secondary mb-2">Start a voice session</p>
            <p className="text-oc-text-muted text-sm font-mono">
              your browser will ask for microphone permission
            </p>
          </div>
        ) : (
          <div className="text-center animate-fadeIn">
            {/* Waveform visualizer */}
            <div className="flex items-center justify-center gap-1.5 h-28 mb-6">
              {Array.from({ length: 7 }).map((_, i) => (
                <div
                  key={i}
                  className={`w-1.5 rounded-full transition-all duration-300 ${
                    voiceAgentSpeaking
                      ? "bg-oc-teal-400"
                      : "bg-oc-warm-300"
                  }`}
                  style={{
                    height: voiceAgentSpeaking ? `${20 + Math.sin((i + 1) * 0.7) * 50}%` : "12%",
                    ...(voiceAgentSpeaking
                      ? {
                          animation: `waveBar 0.8s ease-in-out infinite`,
                          animationDelay: `${i * 0.08}s`,
                        }
                      : {}),
                  }}
                />
              ))}
            </div>
            <p className="text-oc-text-muted text-[15px] font-mono">
              {voiceAgentSpeaking ? "agent speaking..." : "listening \u2014 speak when ready"}
            </p>
          </div>
        )}

        {voiceError && (
          <div className="mt-5 px-5 py-3 bg-oc-red-subtle border border-oc-red/20 rounded-xl text-oc-red text-[15px]">
            {voiceError}
          </div>
        )}
      </div>

      {/* Transcript panel */}
      {voiceTranscripts.length > 0 && (
        <div className="border-t border-oc-border shrink-0 bg-oc-bg-card/50">
          <div className="px-6 py-2.5 border-b border-oc-border-subtle">
            <span className="text-[11px] font-mono font-medium uppercase tracking-widest text-oc-text-dim">
              Transcript
            </span>
          </div>
          <div
            ref={scrollRef}
            className="max-h-52 overflow-y-auto px-6 py-3 space-y-2.5"
          >
            {voiceTranscripts.map((t, i) => (
              <div key={i} className="flex items-start gap-3 text-[15px] animate-fadeIn">
                <span
                  className={`text-[12px] font-mono font-medium w-14 shrink-0 pt-0.5 ${
                    t.role === "user" ? "text-oc-cta" : "text-oc-accent"
                  }`}
                >
                  {t.role}
                </span>
                <span className="text-oc-text-secondary leading-relaxed">{t.text}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
