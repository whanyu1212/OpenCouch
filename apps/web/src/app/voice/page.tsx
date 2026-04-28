"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  useAgent,
  useAudioPlayback,
  useSessionContext,
} from "@livekit/components-react";
import { ConnectionState } from "livekit-client";
import {
  TRANSCRIPTION_LANGUAGE_OPTIONS,
  type TranscriptionLanguageOption,
} from "@/lib/api";
import { useSessionStore } from "@/lib/session";
import { CouchLogo } from "@/components/logo";
import {
  ConversationShell,
  MobileTabBar,
  SessionPill,
} from "@/components/conversation-shell";

// `preConnectBuffer: true` lets the mic capture speech locally while the
// LiveKit worker is still cold-starting the agent. The buffer is forwarded
// to the agent over the `lk.agent.pre-connect-audio-buffer` byte stream
// once it joins, removing the "click connect, then dead air" gap on the
// first turn.
const VOICE_SESSION_START_OPTIONS = {
  tracks: {
    microphone: {
      enabled: true,
      publishOptions: {
        preConnectBuffer: true,
      },
    },
  },
} as const;

const ENABLE_VOICE_DEBUG = process.env.NEXT_PUBLIC_ENABLE_VOICE_DEBUG === "true";
const VOICE_OUTPUT_WARMUP_TOPIC = "opencouch.voice_output_warmup";
const VOICE_OUTPUT_WARMUP_TIMEOUT_MS = 6_000;

type VoiceOutputWarmupStatus = "idle" | "requested" | "speaking" | "done";

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

function languageLabel(value: string): string {
  const match = TRANSCRIPTION_LANGUAGE_OPTIONS.find(
    (option) => option.value === value
  );
  return match?.label ?? "auto detect";
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

const IconMic = ({ size = 16 }: { size?: number }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.6"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <rect x="9" y="3" width="6" height="11" rx="3" />
    <path d="M5 11a7 7 0 0 0 14 0" />
    <path d="M12 18v3" />
  </svg>
);

function VoiceOrb({
  size = 280,
  active = false,
}: {
  size?: number;
  active?: boolean;
}) {
  return (
    <div
      className={`oc-orb-wrap${active ? " oc-orb-active" : ""}`}
      style={{ width: size, height: size }}
      aria-hidden="true"
    >
      <div className="oc-orb-ring" />
      <div className="oc-orb-ring oc-orb-ring-2" />
      <div className="oc-orb-core">
        <IconMic size={size * 0.18} />
      </div>
    </div>
  );
}

function formatElapsed(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

export default function VoicePage() {
  const {
    userId,
    threadId,
    sessionMode,
    chatLoading,
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
  const { canPlayAudio, startAudio } = useAudioPlayback(session.room);
  const [voiceOutputWarmupStatus, setVoiceOutputWarmupStatus] =
    useState<VoiceOutputWarmupStatus>("idle");
  const voiceOutputWarmupRequestedRef = useRef(false);
  const voiceOutputWarmupTimeoutRef = useRef<number | null>(null);
  const transcriptScrollRef = useRef<HTMLDivElement>(null);

  // Live call timer — derived from `connectedAt` so we don't need an effect
  // to reset state when the call ends. The interval ticks `now` purely to
  // re-render the formatted timer; React skips the update when the deps are
  // stable, so disconnect drops back to 00:00 with no cascade.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!voiceConnected) return;
    const interval = window.setInterval(() => {
      setNow(Date.now());
    }, 1000);
    return () => window.clearInterval(interval);
  }, [voiceConnected]);
  const connectedAtMs = voiceSessionInfo?.connectedAt
    ? Date.parse(voiceSessionInfo.connectedAt)
    : NaN;
  const callElapsedMs =
    voiceConnected && Number.isFinite(connectedAtMs)
      ? Math.max(0, now - connectedAtMs)
      : 0;

  const clearVoiceOutputWarmupTimeout = useCallback(() => {
    if (voiceOutputWarmupTimeoutRef.current !== null) {
      window.clearTimeout(voiceOutputWarmupTimeoutRef.current);
      voiceOutputWarmupTimeoutRef.current = null;
    }
  }, []);

  const resetVoiceOutputWarmup = useCallback(() => {
    voiceOutputWarmupRequestedRef.current = false;
    clearVoiceOutputWarmupTimeout();
    setVoiceOutputWarmupStatus("idle");
  }, [clearVoiceOutputWarmupTimeout]);

  useEffect(() => {
    return () => {
      clearVoiceOutputWarmupTimeout();
    };
  }, [clearVoiceOutputWarmupTimeout]);

  useEffect(() => {
    transcriptScrollRef.current?.scrollTo({
      top: transcriptScrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [voiceTranscripts]);

  const connect = useCallback(async () => {
    if (
      session.connectionState === ConnectionState.Connecting ||
      session.isConnected
    ) {
      return;
    }

    if (chatLoading) {
      setVoiceError("Wait for the current text reply to finish before starting voice.");
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
    resetVoiceOutputWarmup();

    try {
      await session.start(VOICE_SESSION_START_OPTIONS);
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "Unable to start LiveKit voice session.";
      setVoiceError(message);
    }
  }, [
    clearVoiceActivities,
    clearVoiceTranscripts,
    chatLoading,
    resetVoiceOutputWarmup,
    session,
    sessionMode,
    setVoiceError,
    userId,
  ]);

  const disconnect = useCallback(() => {
    voiceDisconnect();
    resetVoiceOutputWarmup();
    void session.end();
  }, [resetVoiceOutputWarmup, session, voiceDisconnect]);

  const allowAudioPlayback = useCallback(async () => {
    try {
      await startAudio();
      setVoiceError(null);
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "Unable to start audio playback.";
      setVoiceError(message);
    }
  }, [setVoiceError, startAudio]);

  useEffect(() => {
    if (
      !voiceConnected ||
      !canPlayAudio ||
      !voiceReadyToSpeak ||
      agent.isPending ||
      voiceOutputWarmupRequestedRef.current
    ) {
      return;
    }

    let cancelled = false;
    const requestTimeout = window.setTimeout(() => {
      if (cancelled) {
        return;
      }

      voiceOutputWarmupRequestedRef.current = true;
      setVoiceOutputWarmupStatus("requested");
      clearVoiceOutputWarmupTimeout();
      voiceOutputWarmupTimeoutRef.current = window.setTimeout(() => {
        if (cancelled) {
          return;
        }
        setVoiceOutputWarmupStatus("done");
        voiceOutputWarmupTimeoutRef.current = null;
      }, VOICE_OUTPUT_WARMUP_TIMEOUT_MS);

      void session.room.localParticipant
        .sendText("warmup", { topic: VOICE_OUTPUT_WARMUP_TOPIC })
        .catch((error) => {
          if (cancelled) {
            return;
          }
          const message =
            error instanceof Error
              ? error.message
              : "Unable to warm up voice output.";
          setVoiceError(message);
          clearVoiceOutputWarmupTimeout();
          setVoiceOutputWarmupStatus("done");
        });
    }, 0);

    return () => {
      cancelled = true;
      window.clearTimeout(requestTimeout);
    };
  }, [
    agent.isPending,
    canPlayAudio,
    clearVoiceOutputWarmupTimeout,
    session.room,
    setVoiceError,
    voiceConnected,
    voiceReadyToSpeak,
  ]);

  useEffect(() => {
    if (voiceOutputWarmupStatus === "requested" && voiceAgentSpeaking) {
      const transition = window.setTimeout(() => {
        setVoiceOutputWarmupStatus("speaking");
      }, 0);
      return () => {
        window.clearTimeout(transition);
      };
    }

    if (voiceOutputWarmupStatus === "speaking" && !voiceAgentSpeaking) {
      const transition = window.setTimeout(() => {
        clearVoiceOutputWarmupTimeout();
        setVoiceOutputWarmupStatus("done");
      }, 0);
      return () => {
        window.clearTimeout(transition);
      };
    }
  }, [
    clearVoiceOutputWarmupTimeout,
    voiceAgentSpeaking,
    voiceOutputWarmupStatus,
  ]);

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
    chatLoading ||
    session.connectionState === ConnectionState.Connecting ||
    isSavingDisconnectedSession;
  const latestTranscript = voiceTranscripts[voiceTranscripts.length - 1] ?? null;
  const latestActivity = voiceActivities[voiceActivities.length - 1] ?? null;
  const effectiveVoiceMemoryMode =
    voiceSessionInfo?.memoryMode || sessionMode;
  const isPersistent = effectiveVoiceMemoryMode === "persistent";

  const eyebrowState: "idle" | "ready" | "listening" | "speaking" = (() => {
    if (!voiceConnected) return "idle";
    if (voiceAgentSpeaking) return "speaking";
    if (voiceReadyToSpeak && canPlayAudio && voiceOutputWarmupStatus === "done") {
      return "listening";
    }
    return "ready";
  })();

  const eyebrowLabel = (() => {
    if (eyebrowState === "idle") return "agent · idle";
    if (eyebrowState === "speaking") return "agent · speaking";
    if (eyebrowState === "listening") return "agent · listening";
    return "agent · warming up";
  })();

  const statusText = (() => {
    if (session.connectionState === ConnectionState.Connecting) {
      return "connecting to the livekit session…";
    }
    if (!voiceConnected) {
      return "connect when you’re ready";
    }
    if (!canPlayAudio) {
      return "click Allow audio before speaking";
    }
    if (!voiceReadyToSpeak || agent.isPending) {
      return "warming up audio path… one moment";
    }
    if (voiceOutputWarmupStatus !== "done") {
      return "warming up voice output… one moment";
    }
    if (voiceAgentSpeaking) {
      return "agent speaking…";
    }
    if (agent.state === "thinking") {
      return "agent thinking…";
    }
    return "ready — speak when you want";
  })();

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
    return null;
  })();

  const showMemoryConsolidationNote =
    isPersistent && memoryConsolidationMessage !== null;

  const consolidationToneClass = finalizationFailedForCurrentThread
    ? "border-oc-red/20 bg-oc-red-subtle text-oc-red"
    : finalizationCompletedForCurrentThread
      ? "border-oc-green/20 bg-green-50 text-oc-green"
      : "border-oc-teal-200 bg-oc-teal-50 text-oc-teal-800";

  const callTimerText = formatElapsed(callElapsedMs);

  return (
    <ConversationShell withWash>
      {/* Desktop top bar — wrapper controls breakpoint visibility */}
      <div className="oc-app-top-wrap">
      <header className="oc-app-top">
        <div className="flex items-center gap-3">
          <span className="oc-mobile-mark">
            <CouchLogo className="w-4 h-4" />
          </span>
          <h2 className="oc-app-top-title">Voice</h2>
          {voiceConnected && (
            <span
              className="text-[11px] font-mono"
              style={{
                color: "var(--color-oc-primary)",
                letterSpacing: "0.06em",
              }}
            >
              ● live · {callTimerText}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2.5">
          <span className="text-[10px] font-mono uppercase tracking-[0.08em] text-oc-text-dim">
            livekit
          </span>
          <SessionPill />
        </div>
      </header>
      </div>

      {/* Mobile top bar — wrapper controls breakpoint visibility */}
      <div className="oc-mobile-top-wrap">
      <header className="oc-mobile-top">
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span className="oc-mobile-mark">
            <CouchLogo className="w-4 h-4" />
          </span>
          <h2
            className="oc-mobile-top-title"
            style={voiceConnected ? { fontStyle: "italic", fontWeight: 500 } : undefined}
          >
            {voiceConnected ? "listening" : "Voice"}
          </h2>
        </div>
        {voiceConnected ? (
          <span
            className="text-[10px] font-mono"
            style={{
              color: "var(--color-oc-primary)",
              letterSpacing: "0.06em",
            }}
          >
            ● live · {callTimerText}
          </span>
        ) : (
          <span className="text-[10px] font-mono uppercase tracking-[0.08em] text-oc-text-dim">
            livekit
          </span>
        )}
      </header>
      </div>

      {/* Voice stage — fills remaining height; pages use this whether
          connected or not, swapping content + CTA. */}
      <div className="oc-voice-stage">
        <div
          className={`oc-voice-eyebrow ${
            eyebrowState !== "idle" ? "is-ready" : ""
          }`}
        >
          <span className="dot" />
          {eyebrowLabel}
        </div>

        <VoiceOrb size={280} active={voiceAgentSpeaking} />

        {!voiceConnected ? (
          <>
            <h1 className="oc-voice-title">
              Take a breath. <em>Begin when ready.</em>
            </h1>
            <p className="oc-voice-sub">
              Voice runs with persistent memory.{" "}
              {sessionMode === "incognito"
                ? "Incognito · nothing is saved this call."
                : "We won’t speak until you do."}
            </p>
            {finalizationCompletedForCurrentThread && (
              <p className="text-[12px] font-mono text-oc-green mb-3 -mt-3">
                {voiceFinalization.detail || "Previous voice session memory saved."}
              </p>
            )}
            {finalizationFailedForCurrentThread && (
              <p className="text-[12px] font-mono text-oc-red mb-3 -mt-3">
                {voiceFinalization.detail ||
                  "The previous voice session may still be finishing its memory save. You can reconnect, but the latest memory may not be ready yet."}
              </p>
            )}
            {isSavingDisconnectedSession && (
              <p className="text-[12px] font-mono text-oc-teal-700 mb-3 -mt-3">
                Saving session memory from the previous call. Reconnect will
                unlock automatically when that finishes.
              </p>
            )}
            {chatLoading && (
              <p className="text-[12px] font-mono text-oc-teal-700 mb-3 -mt-3">
                Text reply in progress. Voice unlocks when it finishes.
              </p>
            )}
            <button
              type="button"
              className="oc-voice-cta"
              onClick={() => void connect()}
              disabled={connectDisabled}
            >
              <IconMic size={16} />
              {isSavingDisconnectedSession
                ? "Saving memory…"
                : chatLoading
                  ? "Replying…"
                  : session.connectionState === ConnectionState.Connecting
                  ? "Connecting…"
                  : "Connect & speak"}
            </button>

            {/* Language picker — quiet inline control, not a card */}
            <label
              className="mt-5 flex items-center gap-2"
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                color: "var(--color-oc-text-dim)",
                letterSpacing: "0.04em",
                textTransform: "uppercase",
              }}
            >
              <span>transcript</span>
              <select
                value={transcriptionLanguageSelected}
                onChange={(event) =>
                  setTranscriptionLanguageSelected(
                    event.target.value as TranscriptionLanguageOption
                  )
                }
                disabled={voiceConnected}
                className="rounded-lg border border-oc-border bg-white px-2.5 py-1.5 text-[12px] font-mono normal-case tracking-normal text-oc-text-secondary disabled:opacity-60"
              >
                {TRANSCRIPTION_LANGUAGE_OPTIONS.map((option) => (
                  <option key={option.label} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>

            {/* Single meta line — replaces the four debug cards. */}
            <div className="oc-voice-meta">
              <span>
                language <b>{languageLabel(transcriptionLanguageSelected)}</b>
              </span>
              <span className="sep">·</span>
              <span>
                memory <b>{isPersistent ? "persistent" : "off"}</b>
              </span>
              {voiceSessionInfo?.roomName && (
                <>
                  <span className="sep">·</span>
                  <span>
                    room <b>{voiceSessionInfo.roomName}</b>
                  </span>
                </>
              )}
            </div>
          </>
        ) : (
          <>
            {latestTranscript ? (
              <p
                style={{
                  fontFamily: "var(--font-display)",
                  fontStyle: "italic",
                  fontSize: 22,
                  color: "var(--color-oc-ink-2)",
                  lineHeight: 1.4,
                  maxWidth: 360,
                  margin: "0 auto",
                  textAlign: "center",
                }}
              >
                &ldquo;
                <span
                  style={{
                    color:
                      latestTranscript.role === "user"
                        ? "var(--color-oc-primary)"
                        : "var(--color-oc-ink-2)",
                  }}
                >
                  {clipVoiceText(latestTranscript.text, 140)}
                </span>
                &rdquo;
              </p>
            ) : (
              <p
                style={{
                  fontFamily: "var(--font-display)",
                  fontStyle: "italic",
                  fontSize: 22,
                  color: "var(--color-oc-muted)",
                  lineHeight: 1.4,
                  maxWidth: 320,
                  margin: "0 auto",
                  textAlign: "center",
                }}
              >
                {statusText}
              </p>
            )}
            <div className="mt-6 flex items-center gap-3">
              {!canPlayAudio && (
                <button
                  type="button"
                  className="oc-voice-cta oc-voice-cta--secondary"
                  onClick={() => {
                    void allowAudioPlayback();
                  }}
                >
                  Allow audio
                </button>
              )}
              <button
                type="button"
                className="oc-voice-cta oc-voice-cta--danger"
                onClick={disconnect}
              >
                End session
              </button>
            </div>

            {latestActivity && (
              <div className="oc-voice-meta" style={{ marginTop: 18 }}>
                <span>
                  {latestActivity.label} <b>{latestActivity.status}</b>
                </span>
              </div>
            )}
          </>
        )}
      </div>

      {/* Memory consolidation note — surfaced inline, not as a card. */}
      {showMemoryConsolidationNote && (
        <div className="px-6 pb-4 flex justify-center">
          <div
            className={`max-w-2xl rounded-2xl border px-4 py-3 text-[13px] leading-relaxed ${consolidationToneClass}`}
          >
            {memoryConsolidationMessage}
          </div>
        </div>
      )}

      {voiceError && (
        <div className="px-6 pb-4 flex justify-center">
          <div className="max-w-2xl rounded-2xl border border-oc-red/20 bg-oc-red-subtle px-4 py-3 text-[13px] text-oc-red">
            {voiceError}
          </div>
        </div>
      )}

      {/* Transcript scroller — shown only when there's content; also kept on
          desktop when connected so the user can review what was said. */}
      {voiceTranscripts.length > 0 && (
        <div className="border-t border-oc-line-2 bg-white/40 backdrop-blur-sm">
          <div className="flex items-center justify-between gap-3 px-4 py-2 border-b border-oc-line-2 md:px-6">
            <span className="text-[11px] font-mono font-medium uppercase tracking-widest text-oc-text-dim">
              Transcript
            </span>
            <span className="text-[10px] font-mono uppercase tracking-widest text-oc-text-dim/80">
              {languageLabel(transcriptionLanguageSelected)}
            </span>
          </div>
          <div
            ref={transcriptScrollRef}
            className="max-h-44 overflow-y-auto px-4 py-2.5 space-y-2 md:px-6"
          >
            {voiceTranscripts.map((t, i) => (
              <div
                key={t.itemId ? `${t.role}:${t.itemId}` : `${t.role}:${i}`}
                className="flex items-start gap-3 text-[14.5px] animate-fadeIn"
              >
                <span
                  className={`text-[11px] font-mono font-medium w-14 shrink-0 pt-1 ${
                    t.role === "user"
                      ? "text-oc-primary"
                      : t.role === "assistant"
                        ? "text-oc-cta"
                        : "text-oc-text-dim"
                  }`}
                >
                  {t.role}
                </span>
                <span className="text-oc-ink-2 leading-relaxed">{t.text}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Verbose debug overlay — replaces the four always-visible status cards. */}
      {ENABLE_VOICE_DEBUG && (
        <details className="mx-4 mb-4 rounded-2xl border border-dashed border-oc-border bg-white/70 px-5 py-4 text-left shadow-sm md:mx-6">
          <summary className="cursor-pointer text-[11px] font-mono uppercase tracking-widest text-oc-text-dim">
            Voice debug
          </summary>
          <div className="mt-4 grid gap-2 sm:grid-cols-2">
            <VoiceDebugRow label="connection" value={session.connectionState} />
            <VoiceDebugRow label="session connected" value={session.isConnected} />
            <VoiceDebugRow label="store connected" value={voiceConnected} />
            <VoiceDebugRow label="agent state" value={agent.state} />
            <VoiceDebugRow label="agent pending" value={agent.isPending} />
            <VoiceDebugRow label="ready to speak" value={voiceReadyToSpeak} />
            <VoiceDebugRow label="can play audio" value={canPlayAudio} />
            <VoiceDebugRow
              label="output warmup"
              value={voiceOutputWarmupStatus}
            />
            <VoiceDebugRow label="agent speaking" value={voiceAgentSpeaking} />
            <VoiceDebugRow label="memory mode" value={effectiveVoiceMemoryMode} />
            <VoiceDebugRow label="room" value={voiceSessionInfo?.roomName} />
            <VoiceDebugRow
              label="participant"
              value={voiceSessionInfo?.identity}
            />
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

      {/* Mobile bottom tab bar */}
      <div className="oc-mobile-tabbar-wrap">
        <MobileTabBar />
      </div>
    </ConversationShell>
  );
}
