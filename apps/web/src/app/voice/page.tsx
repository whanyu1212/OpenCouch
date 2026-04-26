"use client";

import { useCallback, useEffect, useRef } from "react";
import {
  StartAudio,
  useAgent,
  useSessionContext,
} from "@livekit/components-react";
import { ConnectionState } from "livekit-client";
import {
  TRANSCRIPTION_LANGUAGE_OPTIONS,
  type TranscriptionLanguageOption,
} from "@/lib/api";
import { useSessionStore } from "@/lib/session";

const VOICE_SESSION_START_OPTIONS = {
  tracks: {
    microphone: {
      enabled: true,
      publishOptions: {
        preConnectBuffer: false,
      },
    },
  },
} as const;

const ENABLE_VOICE_DEBUG = process.env.NEXT_PUBLIC_ENABLE_VOICE_DEBUG === "true";

function formatVoiceTimestamp(value: string | null | undefined): string {
  if (!value) {
    return "not yet";
  }

  const timestamp = Date.parse(value);
  if (Number.isNaN(timestamp)) {
    return "unknown";
  }

  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  }).format(timestamp);
}

function clipVoiceText(text: string, limit = 180): string {
  const normalized = text.replace(/\s+/g, " ").trim();
  if (normalized.length <= limit) {
    return normalized;
  }
  return `${normalized.slice(0, limit - 1).trim()}…`;
}

function safeServerHost(value: string | null | undefined): string {
  if (!value) {
    return "none";
  }

  try {
    return new URL(value).host || "configured";
  } catch {
    return "configured";
  }
}

function VoiceDebugRow({
  label,
  value,
}: {
  label: string;
  value: boolean | number | string | null | undefined;
}) {
  const displayValue =
    value === null || value === undefined || value === "" ? "none" : String(value);

  return (
    <div className="rounded-xl border border-oc-border-subtle bg-white/60 px-3 py-2">
      <p className="text-[10px] font-mono uppercase tracking-widest text-oc-text-dim">
        {label}
      </p>
      <p className="mt-1 break-words text-[12px] font-mono text-oc-text-secondary">
        {displayValue}
      </p>
    </div>
  );
}

export default function VoicePage() {
  const {
    userId,
    threadId,
    sessionMode,
    voiceConnected,
    voiceAgentSpeaking,
    voiceReadyToSpeak,
    transcriptionLanguageSelected,
    voiceTranscripts,
    voiceActivities,
    voiceFinalization,
    voiceSessionInfo,
    voiceError,
    setVoiceError,
    setTranscriptionLanguageSelected,
    clearVoiceTranscripts,
    clearVoiceActivities,
    voiceDisconnect,
  } = useSessionStore();
  const session = useSessionContext();
  const agent = useAgent();
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [voiceTranscripts]);

  const connect = useCallback(async () => {
    if (session.connectionState === ConnectionState.Connecting || session.isConnected) {
      return;
    }

    if (sessionMode !== "incognito" && !userId.trim()) {
      setVoiceError(
        "LiveKit voice currently requires a persistent session with a user id."
      );
      return;
    }

    setVoiceError(null);
    clearVoiceTranscripts();
    clearVoiceActivities();

    try {
      await session.start(VOICE_SESSION_START_OPTIONS);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unable to start LiveKit voice session.";
      setVoiceError(message);
    }
  }, [
    clearVoiceActivities,
    clearVoiceTranscripts,
    session,
    sessionMode,
    setVoiceError,
    userId,
  ]);

  const disconnect = useCallback(() => {
    voiceDisconnect();
    void session.end();
  }, [session, voiceDisconnect]);

  const isSavingDisconnectedSession =
    !voiceConnected &&
    voiceFinalization.threadId === threadId &&
    voiceFinalization.status === "in_progress";
  const finalizationFailedForCurrentThread =
    !voiceConnected &&
    voiceFinalization.threadId === threadId &&
    voiceFinalization.status === "failed";
  const finalizationCompletedForCurrentThread =
    !voiceConnected &&
    voiceFinalization.threadId === threadId &&
    voiceFinalization.status === "completed";
  const connectDisabled =
    session.connectionState === ConnectionState.Connecting ||
    isSavingDisconnectedSession;
  const latestTranscript = voiceTranscripts[voiceTranscripts.length - 1] ?? null;
  const latestActivity = voiceActivities[voiceActivities.length - 1] ?? null;
  const effectiveVoiceMemoryMode =
    voiceSessionInfo?.memoryMode || sessionMode;
  const memoryStatusText =
    effectiveVoiceMemoryMode === "incognito"
      ? "memory off for this call"
      : "memory can load and save";
  const finalizationStatusText = (() => {
    if (voiceConnected) {
      return "active call";
    }
    if (voiceFinalization.threadId !== threadId) {
      return "no voice save pending";
    }
    if (voiceFinalization.status === "in_progress") {
      return voiceFinalization.detail || "saving session memory";
    }
    if (voiceFinalization.status === "completed") {
      return voiceFinalization.detail || "session memory saved";
    }
    if (voiceFinalization.status === "failed") {
      return voiceFinalization.detail || "session memory save failed";
    }
    return "no voice save pending";
  })();
  const finalizationToneClass =
    finalizationFailedForCurrentThread
      ? "border-oc-red/20 bg-oc-red-subtle text-oc-red"
      : isSavingDisconnectedSession
        ? "border-oc-teal-200 bg-oc-teal-50 text-oc-teal-800"
        : finalizationCompletedForCurrentThread
          ? "border-oc-green/20 bg-green-50 text-oc-green"
          : "border-oc-border bg-white text-oc-text-secondary";
  const showMemoryConsolidationNote =
    effectiveVoiceMemoryMode !== "incognito" &&
    (voiceConnected ||
      isSavingDisconnectedSession ||
      finalizationCompletedForCurrentThread ||
      finalizationFailedForCurrentThread);
  const memoryConsolidationToneClass =
    finalizationFailedForCurrentThread
      ? "border-oc-red/20 bg-oc-red-subtle text-oc-red"
      : finalizationCompletedForCurrentThread
        ? "border-oc-green/20 bg-green-50 text-oc-green"
        : "border-oc-teal-200 bg-oc-teal-50 text-oc-teal-800";
  const memoryConsolidationMessage = (() => {
    if (isSavingDisconnectedSession) {
      return "Please wait here while the transcript is consolidated into memory. This can take a little while; reconnecting before it finishes may mean the latest session memory is not ready yet.";
    }
    if (finalizationCompletedForCurrentThread) {
      return "Memory consolidation finished. The saved summary and extracted memories are now available for future persistent sessions.";
    }
    if (finalizationFailedForCurrentThread) {
      return "Memory consolidation did not finish cleanly. You can reconnect, but the latest voice session may not be saved yet.";
    }
    return "After you click Disconnect, keep this page open for a moment while the transcript is consolidated and saved into memory.";
  })();

  const statusText = (() => {
    if (session.connectionState === ConnectionState.Connecting) {
      return "connecting to the livekit session…";
    }
    if (!voiceConnected) {
      return "connect when you’re ready";
    }
    if (!voiceReadyToSpeak || agent.isPending) {
      return "warming up audio path… one moment";
    }
    if (voiceAgentSpeaking) {
      return "agent speaking…";
    }
    if (agent.state === "thinking") {
      return "agent thinking…";
    }
    return "ready — speak when you want";
  })();

  return (
    <div className="flex flex-col h-screen">
      <header className="px-6 py-3.5 border-b border-oc-border flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          <h1 className="font-display text-lg text-oc-teal-900">Voice</h1>
          <span className="text-[11px] font-mono uppercase tracking-widest text-oc-teal-800 bg-oc-teal-50 border border-oc-teal-200 rounded-full px-2 py-1">
            livekit
          </span>
          {voiceConnected && (
            <span className="text-[12px] font-mono text-oc-green">connected</span>
          )}
          <span className="text-[12px] font-mono text-oc-text-dim">
            {effectiveVoiceMemoryMode}
          </span>
          {voiceFinalization.threadId === threadId &&
            voiceFinalization.status !== "idle" && (
              <span
                className={`rounded-full border px-2 py-1 text-[11px] font-mono ${finalizationToneClass}`}
              >
                {voiceFinalization.status.replace("_", " ")}
              </span>
            )}
        </div>
        <div className="flex items-center gap-3 flex-wrap justify-end">
          <label className="flex items-center gap-2 text-[11px] font-mono uppercase tracking-widest text-oc-text-dim">
            <span>transcript</span>
            <select
              value={transcriptionLanguageSelected}
              onChange={(event) =>
                setTranscriptionLanguageSelected(
                  event.target.value as TranscriptionLanguageOption
                )
              }
              disabled={voiceConnected}
              className="rounded-lg border border-oc-border bg-white px-2.5 py-2 text-[12px] font-mono normal-case tracking-normal text-oc-text-secondary disabled:opacity-60"
            >
              {TRANSCRIPTION_LANGUAGE_OPTIONS.map((option) => (
                <option key={option.label} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          {!voiceConnected ? (
            <button
              onClick={connect}
              disabled={connectDisabled}
              className="bg-oc-teal-700 text-white px-5 py-2.5 rounded-xl text-[15px] font-medium hover:bg-oc-teal-600 transition-all shadow-sm disabled:opacity-60"
            >
              {isSavingDisconnectedSession
                ? "Saving memory…"
                : session.connectionState === ConnectionState.Connecting
                ? "Connecting…"
                : "Connect"}
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
              <p className="font-display text-xl text-oc-text-secondary mb-2">
                Start a LiveKit voice session
              </p>
              <p className="mx-auto max-w-xl text-oc-text-muted text-sm font-mono">
                This voice page now runs on the LiveKit Session API instead of manual room
                event wiring, so room connection, agent lifecycle, and transcript updates stay
                aligned with LiveKit’s frontend model.
              </p>
              <p className="mx-auto mt-3 max-w-xl text-oc-text-dim text-[12px] font-mono">
                The agent now stays quiet on connect. Once the status switches to ready,
                you speak first.
              </p>
              {isSavingDisconnectedSession && (
                <p className="mx-auto mt-3 max-w-xl text-oc-teal-700 text-[12px] font-mono">
                  Saving session memory from the previous call. Reconnect will unlock
                  automatically when that finishes.
                </p>
              )}
              {finalizationCompletedForCurrentThread && (
                <p className="mx-auto mt-3 max-w-xl text-oc-green text-[12px] font-mono">
                  {voiceFinalization.detail || "Previous voice session memory saved."}
                </p>
              )}
              {finalizationFailedForCurrentThread && (
                <p className="mx-auto mt-3 max-w-xl text-oc-red text-[12px] font-mono">
                  {voiceFinalization.detail ||
                    "The previous voice session may still be finishing its memory save. You can reconnect now, but the latest memory may not be ready yet."}
                </p>
              )}
              <p className="mx-auto mt-3 max-w-xl text-oc-text-dim text-[12px] font-mono">
                Current mode:{" "}
                <span className="text-oc-teal-700">{sessionMode}</span> · user{" "}
                <span className="text-oc-teal-700">{userId || "none"}</span> · thread{" "}
                <span className="text-oc-teal-700">{threadId}</span>
              </p>
              <p className="mx-auto mt-2 max-w-xl text-oc-text-dim text-[12px] font-mono">
                Transcription language:{" "}
                <span className="text-oc-teal-700">
                  {transcriptionLanguageSelected || "auto detect"}
                </span>
              </p>
              {sessionMode === "incognito" && (
                <p className="mx-auto mt-3 max-w-xl text-oc-text-dim text-[12px] font-mono">
                  Incognito voice will not load or save memory for this call.
                </p>
              )}
            </div>
          ) : (
            <div className="w-full max-w-2xl text-center animate-fadeIn">
              <div className="flex items-center justify-center gap-1.5 h-28 mb-6">
                {Array.from({ length: 7 }).map((_, i) => (
                  <div
                    key={i}
                    className={`w-1.5 rounded-full transition-all duration-300 ${
                      voiceAgentSpeaking ? "bg-oc-teal-400" : "bg-oc-warm-300"
                    }`}
                    style={{
                      height: voiceAgentSpeaking
                        ? `${20 + Math.sin((i + 1) * 0.7) * 50}%`
                        : "12%",
                      ...(voiceAgentSpeaking
                        ? {
                            animation: "waveBar 0.8s ease-in-out infinite",
                            animationDelay: `${i * 0.08}s`,
                          }
                        : {}),
                    }}
                  />
                ))}
              </div>
              <p className="text-oc-text-muted text-[15px] font-mono">{statusText}</p>
              <p className="mt-3 text-oc-text-dim text-[12px] font-mono">
                {voiceReadyToSpeak
                  ? "livekit session api · shared room lifecycle · browser mic published directly"
                  : "livekit session api · warming the browser audio path before you speak"}
              </p>
              <div className="mt-5 flex items-center justify-center">
                <StartAudio
                  label="Allow audio"
                  room={session.room}
                  className="rounded-xl border border-oc-border bg-white px-4 py-2 text-[13px] font-mono text-oc-text-secondary shadow-sm"
                />
              </div>
            </div>
          )}

          <div className="mt-8 grid w-full max-w-3xl gap-3 text-left sm:grid-cols-2">
            <div className="rounded-2xl border border-oc-border bg-oc-bg-card/70 px-5 py-4 shadow-sm">
              <p className="text-[11px] font-mono uppercase tracking-widest text-oc-text-dim">
                Connection
              </p>
              <p className="mt-2 text-sm font-medium text-oc-text-secondary">
                {voiceConnected ? "LiveKit room connected" : "Not connected"}
              </p>
              <p className="mt-1 text-[12px] font-mono text-oc-text-dim">
                state {session.connectionState}
                {voiceSessionInfo?.roomName ? ` · room ${voiceSessionInfo.roomName}` : ""}
              </p>
            </div>

            <div className="rounded-2xl border border-oc-border bg-oc-bg-card/70 px-5 py-4 shadow-sm">
              <p className="text-[11px] font-mono uppercase tracking-widest text-oc-text-dim">
                Memory
              </p>
              <p className="mt-2 text-sm font-medium text-oc-text-secondary">
                {effectiveVoiceMemoryMode}
              </p>
              <p className="mt-1 text-[12px] font-mono text-oc-text-dim">
                {memoryStatusText}
                {voiceSessionInfo?.identity ? ` · participant ${voiceSessionInfo.identity}` : ""}
              </p>
            </div>

            <div className={`rounded-2xl border px-5 py-4 shadow-sm ${finalizationToneClass}`}>
              <p className="text-[11px] font-mono uppercase tracking-widest opacity-70">
                Finalization
              </p>
              <p className="mt-2 text-sm font-medium">{finalizationStatusText}</p>
              <p className="mt-1 text-[12px] font-mono opacity-70">
                updated {formatVoiceTimestamp(voiceFinalization.updatedAt)}
              </p>
            </div>

            <div className="rounded-2xl border border-oc-border bg-oc-bg-card/70 px-5 py-4 shadow-sm">
              <p className="text-[11px] font-mono uppercase tracking-widest text-oc-text-dim">
                Agent
              </p>
              <p className="mt-2 text-sm font-medium text-oc-text-secondary">
                {statusText}
              </p>
              <p className="mt-1 text-[12px] font-mono text-oc-text-dim">
                agent {agent.state}
                {voiceAgentSpeaking ? " · speaking" : ""}
              </p>
            </div>
          </div>

          {showMemoryConsolidationNote && (
            <div
              className={`mt-4 w-full max-w-3xl rounded-2xl border px-5 py-4 text-left shadow-sm ${memoryConsolidationToneClass}`}
            >
              <p className="text-[11px] font-mono uppercase tracking-widest opacity-70">
                Memory Save Timing
              </p>
              <p className="mt-2 text-sm font-medium leading-relaxed">
                {memoryConsolidationMessage}
              </p>
            </div>
          )}

          {ENABLE_VOICE_DEBUG && (
            <details className="mt-4 w-full max-w-3xl rounded-2xl border border-dashed border-oc-border bg-white/70 px-5 py-4 text-left shadow-sm">
              <summary className="cursor-pointer text-[11px] font-mono uppercase tracking-widest text-oc-text-dim">
                Voice Debug
              </summary>
              <div className="mt-4 grid gap-2 sm:grid-cols-2">
                <VoiceDebugRow label="connection" value={session.connectionState} />
                <VoiceDebugRow label="session connected" value={session.isConnected} />
                <VoiceDebugRow label="store connected" value={voiceConnected} />
                <VoiceDebugRow label="agent state" value={agent.state} />
                <VoiceDebugRow label="agent pending" value={agent.isPending} />
                <VoiceDebugRow label="ready to speak" value={voiceReadyToSpeak} />
                <VoiceDebugRow label="agent speaking" value={voiceAgentSpeaking} />
                <VoiceDebugRow label="memory mode" value={effectiveVoiceMemoryMode} />
                <VoiceDebugRow label="room" value={voiceSessionInfo?.roomName} />
                <VoiceDebugRow label="participant" value={voiceSessionInfo?.identity} />
                <VoiceDebugRow
                  label="server host"
                  value={safeServerHost(voiceSessionInfo?.serverUrl)}
                />
                <VoiceDebugRow
                  label="connected at"
                  value={formatVoiceTimestamp(voiceSessionInfo?.connectedAt)}
                />
                <VoiceDebugRow label="user id" value={userId || "none"} />
                <VoiceDebugRow label="thread id" value={threadId} />
                <VoiceDebugRow
                  label="transcript language"
                  value={transcriptionLanguageSelected || "auto detect"}
                />
                <VoiceDebugRow
                  label="finalization"
                  value={voiceFinalization.status}
                />
                <VoiceDebugRow
                  label="finalization detail"
                  value={voiceFinalization.detail}
                />
                <VoiceDebugRow
                  label="finalization updated"
                  value={formatVoiceTimestamp(voiceFinalization.updatedAt)}
                />
                <VoiceDebugRow
                  label="transcript count"
                  value={voiceTranscripts.length}
                />
                <VoiceDebugRow
                  label="activity count"
                  value={voiceActivities.length}
                />
                <VoiceDebugRow
                  label="latest activity"
                  value={
                    latestActivity
                      ? `${latestActivity.label}: ${latestActivity.status}`
                      : null
                  }
                />
                <VoiceDebugRow
                  label="latest transcript"
                  value={
                    latestTranscript
                      ? `${latestTranscript.role}: ${clipVoiceText(latestTranscript.text, 120)}`
                      : null
                  }
                />
              </div>
              <p className="mt-3 text-[11px] font-mono text-oc-text-dim">
                Developer-only readout. Enable with{" "}
                <span className="text-oc-teal-700">
                  NEXT_PUBLIC_ENABLE_VOICE_DEBUG=true
                </span>
                . Tokens and raw credentials are intentionally not shown.
              </p>
            </details>
          )}

          {voiceError && (
            <div className="mt-5 w-full max-w-2xl px-5 py-3 bg-oc-red-subtle border border-oc-red/20 rounded-xl text-oc-red text-[15px]">
              {voiceError}
            </div>
          )}
        </div>
      </div>

      {voiceTranscripts.length > 0 && (
        <div className="border-t border-oc-border shrink-0 bg-oc-bg-card/50">
          <div className="px-6 py-2.5 border-b border-oc-border-subtle flex items-center justify-between gap-3">
            <span className="text-[11px] font-mono font-medium uppercase tracking-widest text-oc-text-dim">
              Transcript
            </span>
            <span className="text-[10px] font-mono uppercase tracking-widest text-oc-text-dim/80">
              4o-transcribe
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
                  className={`text-[12px] font-mono font-medium w-16 shrink-0 pt-0.5 ${
                    t.role === "user"
                      ? "text-oc-cta"
                      : t.role === "assistant"
                        ? "text-oc-accent"
                        : "text-oc-text-dim"
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
