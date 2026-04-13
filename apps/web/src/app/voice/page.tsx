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
        <h1 className="text-sm font-semibold text-oc-teal-800">Voice Chat</h1>
        <div className="flex items-center gap-3">
          {isAgentSpeaking && (
            <div className="flex items-center gap-2 text-[12px] text-oc-cta">
              <div className="w-1.5 h-1.5 rounded-full bg-oc-cta animate-pulse" />
              Speaking
            </div>
          )}
          {!isConnected ? (
            <button
              onClick={connect}
              className="bg-gradient-to-r from-oc-teal-500 to-oc-teal-400 text-white px-4 py-2 rounded-lg text-[13px] font-medium hover:from-oc-teal-400 hover:to-oc-teal-300 transition-all shadow-sm"
            >
              Connect
            </button>
          ) : (
            <button
              onClick={disconnect}
              className="bg-oc-red/10 text-oc-red border border-oc-red/20 px-4 py-2 rounded-lg text-[13px] font-medium hover:bg-oc-red/20 transition-all"
            >
              Disconnect
            </button>
          )}
        </div>
      </header>

      {/* Voice area */}
      <div className="flex-1 flex flex-col items-center justify-center px-6">
        {!isConnected ? (
          <div className="text-center">
            <div className="w-20 h-20 rounded-3xl bg-oc-accent-glow flex items-center justify-center mx-auto mb-5">
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                className="w-10 h-10 text-oc-accent"
              >
                <path d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z" />
                <path d="M19 10v2a7 7 0 01-14 0v-2" />
                <line x1="12" y1="19" x2="12" y2="23" />
                <line x1="8" y1="23" x2="16" y2="23" />
              </svg>
            </div>
            <p className="text-oc-text-secondary text-sm mb-1">Click Connect to start</p>
            <p className="text-oc-text-dim text-xs">
              Your browser will ask for microphone permission.
            </p>
          </div>
        ) : (
          <div className="text-center">
            {/* Animated voice indicator */}
            <div className="relative w-28 h-28 mx-auto mb-6">
              <div
                className={`absolute inset-0 rounded-full transition-all duration-500 ease-out ${
                  isAgentSpeaking
                    ? "bg-oc-teal-400/20 scale-125"
                    : "bg-oc-teal-400/5 scale-100"
                }`}
              />
              <div
                className={`absolute inset-3 rounded-full transition-all duration-400 ease-out ${
                  isAgentSpeaking
                    ? "bg-oc-teal-400/30 scale-110"
                    : "bg-oc-teal-400/10 scale-100"
                }`}
              />
              <div className="absolute inset-6 rounded-full bg-gradient-to-br from-oc-teal-500/40 to-oc-teal-400/20 flex items-center justify-center backdrop-blur-sm">
                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  className="w-8 h-8 text-oc-accent-hover"
                >
                  <path d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z" />
                  <path d="M19 10v2a7 7 0 01-14 0v-2" />
                </svg>
              </div>
            </div>
            <p className="text-oc-text-muted text-[13px]">
              {isAgentSpeaking ? "Agent is speaking..." : "Listening — speak when ready"}
            </p>
          </div>
        )}

        {error && (
          <div className="mt-4 px-4 py-2.5 bg-oc-red/10 border border-oc-red/20 rounded-lg text-oc-red text-[13px]">
            {error}
          </div>
        )}
      </div>

      {/* Transcript panel */}
      {transcripts.length > 0 && (
        <div
          ref={scrollRef}
          className="border-t border-oc-border max-h-52 overflow-y-auto px-6 py-3 space-y-2 shrink-0 bg-oc-bg-card/50"
        >
          {transcripts.map((t, i) => (
            <div key={i} className="text-[13px]">
              <span
                className={`font-medium mr-2 ${
                  t.role === "user" ? "text-oc-cta" : "text-oc-accent"
                }`}
              >
                {t.role}:
              </span>
              <span className="text-oc-text-muted">{t.text}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
