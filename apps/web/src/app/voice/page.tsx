"use client";

import { useCallback, useEffect, useRef } from "react";
import {
  REALTIME_VOICE_OPTIONS,
  TRANSCRIPTION_LANGUAGE_OPTIONS,
  createVoiceSession,
  sendVoiceTruncate,
} from "@/lib/api";
import type {
  RealtimeVoiceOption,
  TranscriptionLanguageOption,
} from "@/lib/api";
import {
  useSessionStore,
  getVoiceRefs,
  setVoiceNextPlayTime,
  setVoiceCurrentItem,
  setVoiceItemScheduledEnd,
  addVoiceActiveSource,
  removeVoiceActiveSource,
  flushVoicePlayback,
  computePlayedMs,
  setVoiceGain,
  activateVoiceLocalDuck,
  releaseVoiceLocalDuckIfReady,
  clearVoiceLocalDuck,
} from "@/lib/session";

const SAMPLE_RATE = 24000;
const LOCAL_DUCK_THRESHOLD = 0.02;
const LOCAL_DUCK_GAIN = 0.08;
const LOCAL_DUCK_HOLD_MS = 220;
const SERVER_VAD_SILENCE_MS = 300;
const MIC_CHUNK_SAMPLES = 512;
const MIC_CHUNK_MS = Math.round((MIC_CHUNK_SAMPLES / SAMPLE_RATE) * 1000);

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode(...chunk);
  }
  return btoa(binary);
}

export default function VoicePage() {
  const {
    userId,
    threadId,
    voiceConnected,
    voiceAgentSpeaking,
    voiceSelected,
    transcriptionLanguageSelected,
    voiceTranscripts,
    voiceError,
    setVoiceConnected,
    setVoiceAgentSpeaking,
    setVoiceSelected,
    setTranscriptionLanguageSelected,
    setVoiceError,
    addVoiceTranscript,
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

  // ── Audio playback — routes through GainNode, tracks sources ──────
  const playAudioChunk = useCallback(
    (pcm16Bytes: Uint8Array, itemId: string, contentIndex: number) => {
      const {
        audioCtx: ctx,
        gainNode: gain,
        generation: gen,
        playbackEpoch: epoch,
        currentItemId,
      } = getVoiceRefs();
      if (!ctx || !gain) return;

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
      source.connect(gain); // Route through GainNode, not destination

      const now = ctx.currentTime;
      const { nextPlayTime } = getVoiceRefs();
      const startTime = Math.max(now, nextPlayTime);
      source.start(startTime);
      const endTime = startTime + buffer.duration;
      setVoiceNextPlayTime(endTime);

      // Track per-item playback for truncation reporting
      if (!currentItemId || currentItemId !== itemId) {
        setVoiceCurrentItem(itemId, contentIndex, startTime);
      }
      setVoiceItemScheduledEnd(endTime);

      // Register for flush-on-interrupt
      addVoiceActiveSource(source);

      source.onended = () => {
        removeVoiceActiveSource(source);
        const refs = getVoiceRefs();
        // Ignore if session or playback epoch changed (interrupted/disconnected)
        if (refs.generation !== gen) return;
        if (refs.playbackEpoch !== epoch) return;
        const { nextPlayTime: npt } = refs;
        if (ctx.currentTime >= npt - 0.05) {
          clearVoiceLocalDuck();
          setVoiceGain(1, 10);
          setVoiceAgentSpeaking(false);
        }
      };
    },
    [setVoiceAgentSpeaking]
  );

  // ── Handle interruption from server ───────────────────────────────
  const handleInterrupted = useCallback(() => {
    const { ws, currentItemId, currentContentIndex } = getVoiceRefs();

    // Compute how much audio was actually played before flushing
    const playedMs = computePlayedMs();

    // Flush all queued audio immediately
    flushVoicePlayback();
    setVoiceAgentSpeaking(false);

    // Tell the server where we truncated so conversation history stays in sync
    if (ws && currentItemId) {
      sendVoiceTruncate(ws, currentItemId, currentContentIndex, playedMs);
    }
  }, [setVoiceAgentSpeaking]);

  // ── Connect ───────────────────────────────────────────────────────
  const connect = useCallback(async () => {
    // Guard against double-connect
    if (useSessionStore.getState().voiceConnected) return;
    const existingWs = getVoiceRefs().ws;
    if (existingWs && existingWs.readyState !== WebSocket.CLOSED) return;

    setVoiceError(null);
    clearVoiceTranscripts();

    // Bump generation — any previous onclose becomes a no-op
    const gen = voiceNewGeneration();

    const ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
    voiceSetRefs({ audioCtx: ctx });
    setVoiceNextPlayTime(0);

    // Create GainNode for clean interruption (mute → stop → restore)
    const gain = ctx.createGain();
    gain.connect(ctx.destination);
    voiceSetRefs({ gainNode: gain });

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
      voiceSetRefs({ audioCtx: null, gainNode: null });
      setVoiceError("Microphone access denied");
      return;
    }

    const ws = createVoiceSession(
      userId,
      threadId,
      voiceSelected,
      transcriptionLanguageSelected,
      {
        onReady: () => {
          setVoiceConnected(true);
        },
        onAudio: (bytes, itemId, contentIndex) => {
          setVoiceAgentSpeaking(true);
          playAudioChunk(bytes, itemId, contentIndex);
        },
        onTranscript: (role, text, itemId) => {
          addVoiceTranscript({ role, text, itemId });
        },
        onInterrupted: () => {
          handleInterrupted();
        },
        onError: (msg) => {
          setVoiceError(msg);
        },
      }
    );

    ws.onclose = () => {
      // Only act if this is still the active session
      if (getVoiceRefs().generation !== gen) return;
      setVoiceConnected(false);
      setVoiceAgentSpeaking(false);
      voiceCleanup(gen);
    };

    voiceSetRefs({ ws });

    const source = ctx.createMediaStreamSource(stream);
    const processor = ctx.createScriptProcessor(512, 1, 1);
    voiceSetRefs({ processor });

    processor.onaudioprocess = (e) => {
      const { ws: activeWs } = getVoiceRefs();
      if (!activeWs || activeWs.readyState !== WebSocket.OPEN) return;
      // Don't gate on voiceAgentSpeaking — server_vad needs continuous
      // audio to detect barge-in / interruption.

      const input = e.inputBuffer.getChannelData(0);
      const pcm16 = new Int16Array(input.length);
      let sumSquares = 0;
      for (let i = 0; i < input.length; i++) {
        sumSquares += input[i] * input[i];
        pcm16[i] = Math.max(-32768, Math.min(32767, input[i] * 32768));
      }

      const rms = Math.sqrt(sumSquares / input.length);
      const nowMs = performance.now();
      if (useSessionStore.getState().voiceAgentSpeaking && rms >= LOCAL_DUCK_THRESHOLD) {
        activateVoiceLocalDuck(LOCAL_DUCK_GAIN, LOCAL_DUCK_HOLD_MS);
      } else {
        releaseVoiceLocalDuckIfReady(nowMs);
      }

      const bytes = new Uint8Array(pcm16.buffer);
      const base64 = bytesToBase64(bytes);
      activeWs.send(JSON.stringify({ type: "audio", data: base64 }));
    };

    source.connect(processor);
    processor.connect(ctx.destination);
  }, [
    userId,
    threadId,
    voiceSelected,
    transcriptionLanguageSelected,
    playAudioChunk,
    handleInterrupted,
    setVoiceConnected,
    setVoiceAgentSpeaking,
    setVoiceError,
    addVoiceTranscript,
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
          <span className="text-[11px] font-mono uppercase tracking-widest text-oc-warm-700 bg-oc-warm-100 border border-oc-warm-200 rounded-full px-2 py-1">
            experimental
          </span>
          {voiceConnected && (
            <span className="text-[12px] font-mono text-oc-green">connected</span>
          )}
        </div>
        <div className="flex items-center gap-3 flex-wrap justify-end">
          <label className="flex items-center gap-2 text-[12px] font-mono text-oc-text-dim">
            <span>voice</span>
            <select
              value={voiceSelected}
              disabled={voiceConnected}
              onChange={(event) =>
                setVoiceSelected(event.target.value as RealtimeVoiceOption)
              }
              className="rounded-lg border border-oc-border bg-oc-bg-card px-2.5 py-1.5 text-[12px] font-mono text-oc-text-secondary disabled:opacity-60"
            >
              {REALTIME_VOICE_OPTIONS.map((voice) => (
                <option key={voice} value={voice}>
                  {voice}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2 text-[12px] font-mono text-oc-text-dim">
            <span>lang</span>
            <select
              value={transcriptionLanguageSelected}
              disabled={voiceConnected}
              onChange={(event) =>
                setTranscriptionLanguageSelected(
                  event.target.value as TranscriptionLanguageOption
                )
              }
              className="rounded-lg border border-oc-border bg-oc-bg-card px-2.5 py-1.5 text-[12px] font-mono text-oc-text-secondary disabled:opacity-60"
            >
              {TRANSCRIPTION_LANGUAGE_OPTIONS.map((language) => (
                <option key={language.value || "auto"} value={language.value}>
                  {language.label}
                </option>
              ))}
            </select>
          </label>
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
      <div className="flex-1 px-6">
        <div className="flex h-full w-full flex-col items-center justify-center">
        {!voiceConnected ? (
          <div className="w-full max-w-2xl text-center animate-fadeIn">
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
            <p className="mx-auto max-w-xl text-oc-text-muted text-sm font-mono">
              experimental speech preview only — the voice path does not have agentic tools or autonomous capability yet. your browser will ask for microphone permission
            </p>
            <p className="mx-auto mt-3 max-w-xl text-oc-text-dim text-[12px] font-mono">
              choose a voice and spoken language before connecting. cedar and marin are the best-quality voice options right now, and setting the language can help transcript accuracy.
            </p>
          </div>
        ) : (
          <div className="w-full max-w-2xl text-center animate-fadeIn">
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
              {voiceAgentSpeaking ? "agent speaking\u2026" : "listening \u2014 speak when ready"}
            </p>
            <p className="mt-3 text-oc-text-dim text-[12px] font-mono">
              experimental mode · speech-only conversation · no agentic actions
            </p>
          </div>
        )}

        <div className="mt-8 w-full max-w-2xl rounded-2xl border border-oc-border bg-oc-bg-card/70 px-5 py-4 text-left shadow-sm">
          <p className="text-[11px] font-mono uppercase tracking-widest text-oc-text-dim mb-2">
            Turn-Taking Note
          </p>
          <p className="text-sm text-oc-text-secondary leading-relaxed">
            This voice mode is responsive, but not instant. Turn-end detection waits for about{" "}
            <span className="font-mono text-oc-teal-700">{SERVER_VAD_SILENCE_MS} ms</span> of
            silence, and mic audio is streamed in roughly{" "}
            <span className="font-mono text-oc-teal-700">{MIC_CHUNK_MS} ms</span> chunks.
            Interruptions are supported with local ducking and server truncation, but they still
            take a beat to settle. If you want to cut in, a clear first word works better than an
            abrupt whisper or half-breathed aside. Transcript history is approximate, even with a
            language hint and transcription prompt, and can still miss names, numbers, accents, or
            overlapping speech.
          </p>
        </div>

        {voiceError && (
          <div className="mt-5 w-full max-w-2xl px-5 py-3 bg-oc-red-subtle border border-oc-red/20 rounded-xl text-oc-red text-[15px]">
            {voiceError}
          </div>
        )}
        </div>
      </div>

      {/* Transcript panel */}
      {voiceTranscripts.length > 0 && (
        <div className="border-t border-oc-border shrink-0 bg-oc-bg-card/50">
          <div className="px-6 py-2.5 border-b border-oc-border-subtle flex items-center justify-between gap-3">
            <span className="text-[11px] font-mono font-medium uppercase tracking-widest text-oc-text-dim">
              Transcript History
            </span>
            <span className="text-[10px] font-mono uppercase tracking-widest text-oc-text-dim/80">
              approximate
            </span>
          </div>
          <div
            ref={scrollRef}
            className="max-h-52 overflow-y-auto px-6 py-3 space-y-2.5"
          >
            {voiceTranscripts.map((t, i) => (
              <div
                key={t.itemId ? `${t.role}:${t.itemId}` : `${t.role}:${i}`}
                className="flex items-start gap-3 text-[15px] animate-fadeIn"
              >
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
