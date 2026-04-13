"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { createVoiceSession } from "@/lib/api";
import { useSessionStore } from "@/lib/session";

const SAMPLE_RATE = 24000;

interface Transcript {
  role: "user" | "assistant";
  text: string;
}

export default function VoicePage() {
  const { userId, threadId } = useSessionStore();
  const [isConnected, setIsConnected] = useState(false);
  const [isAgentSpeaking, setIsAgentSpeaking] = useState(false);
  const [transcripts, setTranscripts] = useState<Transcript[]>([]);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const nextPlayTimeRef = useRef(0);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [transcripts]);

  const playAudioChunk = useCallback((pcm16Bytes: Uint8Array) => {
    const ctx = audioContextRef.current;
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
    const startTime = Math.max(now, nextPlayTimeRef.current);
    source.start(startTime);
    nextPlayTimeRef.current = startTime + buffer.duration;

    source.onended = () => {
      if (ctx && ctx.currentTime >= nextPlayTimeRef.current - 0.05) {
        setIsAgentSpeaking(false);
      }
    };
  }, []);

  const connect = useCallback(async () => {
    setError(null);
    setTranscripts([]);

    const ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
    audioContextRef.current = ctx;
    nextPlayTimeRef.current = 0;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: SAMPLE_RATE,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      mediaStreamRef.current = stream;
    } catch {
      setError("Microphone access denied");
      return;
    }

    const ws = createVoiceSession(userId, threadId, {
      onAudio: (bytes) => {
        setIsAgentSpeaking(true);
        playAudioChunk(bytes);
      },
      onTranscript: (role, text) => {
        setTranscripts((prev) => [...prev, { role, text }]);
        if (role === "assistant") {
          setIsAgentSpeaking(false);
        }
      },
      onError: (msg) => {
        setError(msg);
      },
    });

    ws.onclose = () => {
      setIsConnected(false);
      cleanup();
    };

    wsRef.current = ws;

    const source = ctx.createMediaStreamSource(mediaStreamRef.current!);
    const processor = ctx.createScriptProcessor(4096, 1, 1);
    processorRef.current = processor;

    processor.onaudioprocess = (e) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
      if (isAgentSpeaking) return;

      const input = e.inputBuffer.getChannelData(0);
      const pcm16 = new Int16Array(input.length);
      for (let i = 0; i < input.length; i++) {
        pcm16[i] = Math.max(-32768, Math.min(32767, input[i] * 32768));
      }
      const bytes = new Uint8Array(pcm16.buffer);
      const base64 = btoa(String.fromCharCode(...bytes));
      wsRef.current.send(JSON.stringify({ type: "audio", data: base64 }));
    };

    source.connect(processor);
    processor.connect(ctx.destination);
    setIsConnected(true);
  }, [playAudioChunk, isAgentSpeaking]);

  const disconnect = useCallback(() => {
    wsRef.current?.close();
    cleanup();
    setIsConnected(false);
  }, []);

  function cleanup() {
    processorRef.current?.disconnect();
    processorRef.current = null;
    mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
    mediaStreamRef.current = null;
    audioContextRef.current?.close();
    audioContextRef.current = null;
    nextPlayTimeRef.current = 0;
  }

  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <header className="px-6 py-3.5 border-b border-oc-border flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          <h1 className="font-display text-lg text-oc-teal-900">Voice</h1>
          {isConnected && (
            <span className="text-[12px] font-mono text-oc-green">connected</span>
          )}
        </div>
        <div className="flex items-center gap-3">
          {isAgentSpeaking && (
            <div className="flex items-center gap-2 text-[13px] font-mono text-oc-cta">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-oc-cta opacity-60" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-oc-cta" />
              </span>
              speaking
            </div>
          )}
          {!isConnected ? (
            <button
              onClick={connect}
              className="bg-oc-teal-700 text-white px-5 py-2.5 rounded-xl text-[15px] font-medium hover:bg-oc-teal-600 transition-all shadow-sm"
            >
              Connect
            </button>
          ) : (
            <button
              onClick={disconnect}
              className="bg-oc-red-subtle text-oc-red border border-oc-red/20 px-5 py-2.5 rounded-xl text-[15px] font-medium hover:bg-red-100 transition-all"
            >
              Disconnect
            </button>
          )}
        </div>
      </header>

      {/* Voice area */}
      <div className="flex-1 flex flex-col items-center justify-center px-6">
        {!isConnected ? (
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
                    isAgentSpeaking
                      ? "bg-oc-teal-400"
                      : "bg-oc-warm-300"
                  }`}
                  style={{
                    height: isAgentSpeaking ? `${20 + Math.sin((i + 1) * 0.7) * 50}%` : "12%",
                    ...(isAgentSpeaking
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
              {isAgentSpeaking ? "agent speaking…" : "listening — speak when ready"}
            </p>
          </div>
        )}

        {error && (
          <div className="mt-5 px-5 py-3 bg-oc-red-subtle border border-oc-red/20 rounded-xl text-oc-red text-[15px]">
            {error}
          </div>
        )}
      </div>

      {/* Transcript panel */}
      {transcripts.length > 0 && (
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
            {transcripts.map((t, i) => (
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
