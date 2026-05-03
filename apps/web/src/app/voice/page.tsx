"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  useAgent,
  useAudioPlayback,
  useSessionContext,
} from "@livekit/components-react";
import { ConnectionState } from "livekit-client";
import {
  ASSISTANT_VOICE_OPTIONS,
  TRANSCRIPTION_LANGUAGE_OPTIONS,
  type AssistantVoiceOption,
  type TranscriptionLanguageOption,
} from "@/lib/api";
import { useCommandActions } from "@/lib/command-actions";
import { useSessionStore, type EndedSessionResult } from "@/lib/session";
import { CouchLogo } from "@/components/logo";
import { SessionPill } from "@/components/conversation-shell";

// Keep the mic closed until the agent and voice output path are ready. This
// avoids collecting speech during worker cold starts, when the user would not
// yet hear confirmation that the session is listening.
const VOICE_SESSION_START_OPTIONS = {
  tracks: {
    microphone: {
      enabled: false,
    },
  },
} as const;

const ENABLE_VOICE_DEBUG = process.env.NEXT_PUBLIC_ENABLE_VOICE_DEBUG === "true";
const VOICE_OUTPUT_WARMUP_TOPIC = "opencouch.voice_output_warmup";
const VOICE_OUTPUT_WARMUP_TIMEOUT_MS = 6_000;

type VoiceOutputWarmupStatus = "idle" | "requested" | "speaking" | "done";
type VoiceEndOptionsStatus = "idle" | "in_progress" | "completed" | "failed";

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

function assistantVoiceLabel(value: string): string {
  const match = ASSISTANT_VOICE_OPTIONS.find((option) => option.value === value);
  return match?.label ?? (value || "default");
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

function VoiceEndOptionsDialog({
  finalizationStatus,
  detail,
  endedSession,
  isPersistent,
  onContinueInChat,
  onReviewMemory,
  onStartNewSession,
  onStayOnVoice,
}: {
  finalizationStatus: VoiceEndOptionsStatus;
  detail: string | null;
  endedSession: EndedSessionResult | null;
  isPersistent: boolean;
  onContinueInChat: () => void;
  onReviewMemory: () => void;
  onStartNewSession: () => void;
  onStayOnVoice: () => void;
}) {
  const hasSummary = Boolean(endedSession?.summary);
  const reviewDisabled = !isPersistent || finalizationStatus === "in_progress";
  const statusCopy = (() => {
    if (!isPersistent) {
      return "This was an incognito voice session, so no memory will be saved.";
    }
    if (finalizationStatus === "in_progress") {
      return "Session memory is saving in the background. You can move around while it finishes.";
    }
    if (finalizationStatus === "completed") {
      return hasSummary
        ? "The session summary is ready."
        : detail || "The voice session has finished saving.";
    }
    if (finalizationStatus === "failed") {
      return (
        detail ||
        "The voice session ended, but memory saving did not finish cleanly."
      );
    }
    return "The voice session has ended.";
  })();

  return (
    <div
      className="fixed inset-0 z-[85] flex items-end justify-center bg-[rgba(21,32,29,0.36)] px-0 py-0 md:items-center md:px-6 md:py-8"
      role="presentation"
    >
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby="voice-ended-title"
        className="w-full max-w-[560px] rounded-t-[22px] border border-oc-line bg-white shadow-[0_28px_80px_-34px_rgba(21,32,29,0.58)] md:rounded-[22px]"
      >
        <div className="flex items-start justify-between gap-4 border-b border-oc-line-2 px-5 py-4 md:px-6 md:py-5">
          <div>
            <p className="font-mono text-[10.5px] uppercase tracking-[0.14em] text-oc-primary">
              Voice session
            </p>
            <h2
              id="voice-ended-title"
              className="mt-1 font-display text-[24px] font-semibold leading-tight text-oc-ink"
            >
              Session ended
            </h2>
            <p className="mt-1 text-[13.5px] leading-[1.55] text-oc-muted">
              {statusCopy}
            </p>
          </div>
          <button
            type="button"
            onClick={onStayOnVoice}
            aria-label="Close voice session options"
            className="mt-0.5 inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-oc-line bg-white text-oc-muted transition-[color,border-color,background] hover:border-oc-primary/30 hover:bg-oc-surface-tint hover:text-oc-primary"
          >
            x
          </button>
        </div>

        <div className="px-5 py-4 md:px-6 md:py-5">
          {hasSummary && (
            <div className="mb-4 rounded-xl border border-oc-teal-200 bg-oc-teal-50/70 px-4 py-3">
              <p className="font-mono text-[10.5px] uppercase tracking-[0.1em] text-oc-teal-700">
                Summary
              </p>
              <p className="mt-2 text-[13px] leading-relaxed text-oc-text">
                {endedSession?.summary}
              </p>
              {endedSession?.themes && endedSession.themes.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-2">
                  {endedSession.themes.map((theme) => (
                    <span
                      key={theme}
                      className="rounded-md border border-oc-teal-200/70 bg-white/70 px-2 py-0.5 text-[11px] font-mono text-oc-teal-700"
                    >
                      {theme}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}

          <div className="grid gap-2">
            <button
              type="button"
              onClick={onContinueInChat}
              className="rounded-xl bg-oc-primary px-4 py-3 text-left text-[14px] font-medium text-[#F6F1E5] transition-colors hover:bg-oc-primary-2"
            >
              Continue in chat
              <span className="mt-0.5 block text-[12px] font-normal opacity-80">
                Return to text without starting over.
              </span>
            </button>
            <button
              type="button"
              onClick={onStartNewSession}
              className="rounded-xl border border-oc-line bg-white px-4 py-3 text-left text-[14px] font-medium text-oc-ink transition-colors hover:bg-oc-surface-tint"
            >
              Start a new session
              <span className="mt-0.5 block text-[12px] font-normal text-oc-muted">
                Go home and choose mode, user, and thread again.
              </span>
            </button>
            <button
              type="button"
              onClick={onReviewMemory}
              disabled={reviewDisabled}
              className="rounded-xl border border-oc-line bg-white px-4 py-3 text-left text-[14px] font-medium text-oc-ink transition-colors hover:bg-oc-surface-tint disabled:cursor-not-allowed disabled:opacity-50"
            >
              Review memory
              <span className="mt-0.5 block text-[12px] font-normal text-oc-muted">
                {!isPersistent
                  ? "Not available for incognito sessions."
                  : finalizationStatus === "in_progress"
                    ? "Available after memory saving finishes."
                    : "Open saved facts, summaries, and style rules."}
              </span>
            </button>
            <button
              type="button"
              onClick={onStayOnVoice}
              className="rounded-xl border border-transparent px-4 py-2.5 text-center text-[13px] font-medium text-oc-muted transition-colors hover:bg-oc-surface-tint hover:text-oc-primary"
            >
              Stay on Voice
            </button>
          </div>
        </div>
      </section>
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
  waiting = false,
}: {
  size?: number;
  active?: boolean;
  waiting?: boolean;
}) {
  return (
    <div
      className={`oc-orb-wrap${active ? " oc-orb-active" : ""}${
        waiting ? " oc-orb-waiting" : ""
      }`}
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
  const userId = useSessionStore((s) => s.userId);
  const threadId = useSessionStore((s) => s.threadId);
  const sessionMode = useSessionStore((s) => s.sessionMode);
  const chatLoading = useSessionStore((s) => s.chatLoading);
  const voiceConnected = useSessionStore((s) => s.voiceConnected);
  const voiceAgentSpeaking = useSessionStore((s) => s.voiceAgentSpeaking);
  const voiceReadyToSpeak = useSessionStore((s) => s.voiceReadyToSpeak);
  const transcriptionLanguageSelected = useSessionStore(
    (s) => s.transcriptionLanguageSelected
  );
  const assistantVoiceSelected = useSessionStore(
    (s) => s.assistantVoiceSelected
  );
  const voiceTranscripts = useSessionStore((s) => s.voiceTranscripts);
  const voiceActivities = useSessionStore((s) => s.voiceActivities);
  const voiceFinalization = useSessionStore((s) => s.voiceFinalization);
  const voiceSessionInfo = useSessionStore((s) => s.voiceSessionInfo);
  const lastEndedSession = useSessionStore((s) => s.lastEndedSession);
  const clearLastEndedSession = useSessionStore((s) => s.clearLastEndedSession);
  const voiceError = useSessionStore((s) => s.voiceError);
  const setVoiceError = useSessionStore((s) => s.setVoiceError);
  const setChatNotice = useSessionStore((s) => s.setChatNotice);
  const setTranscriptionLanguageSelected = useSessionStore(
    (s) => s.setTranscriptionLanguageSelected
  );
  const setAssistantVoiceSelected = useSessionStore(
    (s) => s.setAssistantVoiceSelected
  );
  const clearVoiceTranscripts = useSessionStore((s) => s.clearVoiceTranscripts);
  const clearVoiceActivities = useSessionStore((s) => s.clearVoiceActivities);
  const voiceDisconnect = useSessionStore((s) => s.voiceDisconnect);
  const session = useSessionContext();
  const agent = useAgent();
  const { canPlayAudio, startAudio } = useAudioPlayback(session.room);
  const { startNewSession } = useCommandActions();
  const router = useRouter();
  const [voiceOutputWarmupStatus, setVoiceOutputWarmupStatus] =
    useState<VoiceOutputWarmupStatus>("idle");
  const [voiceEndOptionsOpen, setVoiceEndOptionsOpen] = useState(false);
  const [endedVoiceThreadId, setEndedVoiceThreadId] = useState<string | null>(null);
  const voiceOutputWarmupRequestedRef = useRef(false);
  const voiceOutputWarmupTimeoutRef = useRef<number | null>(null);
  const microphoneDesiredEnabledRef = useRef<boolean | null>(null);
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
    setVoiceEndOptionsOpen(false);
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
    const disconnectedThreadId = threadId;
    voiceDisconnect();
    resetVoiceOutputWarmup();
    setVoiceError(null);
    setEndedVoiceThreadId(disconnectedThreadId);
    setVoiceEndOptionsOpen(true);
    void session.end();
  }, [
    resetVoiceOutputWarmup,
    session,
    setVoiceError,
    threadId,
    voiceDisconnect,
  ]);

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
  const endOptionsFinalizationStatus: VoiceEndOptionsStatus =
    endedVoiceThreadId && voiceFinalization.threadId === endedVoiceThreadId
      ? voiceFinalization.status
      : "idle";
  const endOptionsFinalizationDetail =
    endedVoiceThreadId && voiceFinalization.threadId === endedVoiceThreadId
      ? voiceFinalization.detail
      : null;
  const endedVoiceSession =
    endedVoiceThreadId && lastEndedSession?.threadId === endedVoiceThreadId
      ? lastEndedSession
      : null;
  const voiceCanAcceptSpeech =
    voiceConnected &&
    canPlayAudio &&
    voiceReadyToSpeak &&
    !agent.isPending &&
    voiceOutputWarmupStatus === "done";
  const voiceIsWarming = voiceConnected && !voiceCanAcceptSpeech;

  const continueInChat = useCallback(() => {
    setVoiceEndOptionsOpen(false);
    if (endOptionsFinalizationStatus === "in_progress") {
      setChatNotice(
        "Voice session ended. Memory is saving in the background; you can keep using chat."
      );
    } else if (endOptionsFinalizationStatus === "failed") {
      setChatNotice(
        endOptionsFinalizationDetail ||
          "Voice session ended, but the latest memory may not be saved yet."
      );
    } else {
      setChatNotice(null);
    }
    router.push("/");
  }, [
    endOptionsFinalizationDetail,
    endOptionsFinalizationStatus,
    router,
    setChatNotice,
  ]);

  const reviewMemory = useCallback(() => {
    setVoiceEndOptionsOpen(false);
    clearLastEndedSession();
    router.push("/memory");
  }, [clearLastEndedSession, router]);

  const startFreshSession = useCallback(() => {
    setVoiceEndOptionsOpen(false);
    clearLastEndedSession();
    void startNewSession();
  }, [clearLastEndedSession, startNewSession]);

  useEffect(() => {
    if (!session.isConnected) {
      microphoneDesiredEnabledRef.current = null;
      return;
    }

    if (microphoneDesiredEnabledRef.current === voiceCanAcceptSpeech) {
      return;
    }

    microphoneDesiredEnabledRef.current = voiceCanAcceptSpeech;
    void session.room.localParticipant
      .setMicrophoneEnabled(voiceCanAcceptSpeech)
      .catch((error) => {
        microphoneDesiredEnabledRef.current = null;
        if (voiceCanAcceptSpeech) {
          const message =
            error instanceof Error
              ? error.message
              : "Unable to enable the microphone.";
          setVoiceError(message);
        }
      });
  }, [
    session.isConnected,
    session.room.localParticipant,
    setVoiceError,
    voiceCanAcceptSpeech,
  ]);

  const eyebrowState: "idle" | "warming" | "listening" | "speaking" = (() => {
    if (!voiceConnected) return "idle";
    if (voiceAgentSpeaking) return "speaking";
    if (voiceCanAcceptSpeech) return "listening";
    return "warming";
  })();

  const eyebrowLabel = (() => {
    if (eyebrowState === "idle") return "agent · idle";
    if (eyebrowState === "speaking") return "agent · speaking";
    if (eyebrowState === "listening") return "agent · listening";
    return "agent · wait";
  })();

  const statusText = (() => {
    if (session.connectionState === ConnectionState.Connecting) {
      return "connecting to the livekit session…";
    }
    if (!voiceConnected) {
      return "connect when you’re ready";
    }
    if (!canPlayAudio) {
      return "allow audio before speaking";
    }
    if (!voiceReadyToSpeak || agent.isPending) {
      return "please wait — the agent is warming up";
    }
    if (voiceOutputWarmupStatus !== "done") {
      return "please wait — checking voice output";
    }
    if (voiceAgentSpeaking) {
      return "agent speaking…";
    }
    if (agent.state === "thinking") {
      return "agent thinking…";
    }
    return "ready — speak when you want";
  })();

  const readinessNotice = (() => {
    if (!voiceConnected || voiceCanAcceptSpeech) {
      return null;
    }
    if (!canPlayAudio) {
      return {
        title: "Allow audio before speaking",
        detail: "The mic is muted until playback is enabled, so you can hear the agent first.",
      };
    }
    if (!voiceReadyToSpeak || agent.isPending) {
      return {
        title: "Hold on before speaking",
        detail: "The agent is still warming up. Your mic will open automatically when it can listen.",
      };
    }
    if (voiceOutputWarmupStatus !== "done") {
      return {
        title: "Checking voice output",
        detail: "The mic is still muted while the assistant verifies the audio path.",
      };
    }
    return null;
  })();

  const mobileVoiceTitle = (() => {
    if (!voiceConnected) return "Voice";
    if (voiceAgentSpeaking) return "speaking";
    if (voiceCanAcceptSpeech) return "listening";
    return "warming up";
  })();

  const memoryConsolidationMessage = (() => {
    if (isSavingDisconnectedSession) {
      return "Voice memory is saving in the background. You can use chat or other tabs while it finishes; reconnecting voice unlocks when the latest session memory is ready.";
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
    <>
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
            {mobileVoiceTitle}
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
          className={`oc-voice-eyebrow oc-voice-eyebrow--${eyebrowState} ${
            eyebrowState !== "idle" ? "is-ready" : ""
          }`}
        >
          <span className="dot" />
          {eyebrowLabel}
        </div>

        <VoiceOrb size={280} active={voiceAgentSpeaking} waiting={voiceIsWarming} />

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
                Saving session memory from the previous call. You can use chat
                while this finishes.
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
                  : "Connect voice"}
            </button>
            {isSavingDisconnectedSession && (
              <button
                type="button"
                className="oc-voice-cta oc-voice-cta--secondary mt-3"
                onClick={() => router.push("/")}
              >
                Use chat
              </button>
            )}

            {/* Voice controls — quiet inline controls, not cards */}
            <div
              className="mt-5 flex flex-wrap items-center justify-center gap-3"
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                color: "var(--color-oc-text-dim)",
                letterSpacing: "0.04em",
                textTransform: "uppercase",
              }}
            >
              <label className="flex items-center gap-2">
                <span>voice</span>
                <select
                  value={assistantVoiceSelected}
                  onChange={(event) =>
                    setAssistantVoiceSelected(
                      event.target.value as AssistantVoiceOption
                    )
                  }
                  disabled={voiceConnected}
                  className="rounded-lg border border-oc-border bg-white px-2.5 py-1.5 text-[12px] font-mono normal-case tracking-normal text-oc-text-secondary disabled:opacity-60"
                >
                  {ASSISTANT_VOICE_OPTIONS.map((option) => (
                    <option key={option.label} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex items-center gap-2">
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
            </div>

            {/* Single meta line — replaces the four debug cards. */}
            <div className="oc-voice-meta">
              <span>
                voice <b>{assistantVoiceLabel(assistantVoiceSelected)}</b>
              </span>
              <span className="sep">·</span>
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
            {readinessNotice && (
              <div
                className="oc-voice-readiness"
                role="status"
                aria-live="polite"
                aria-atomic="true"
              >
                <span className="oc-voice-readiness-icon">
                  <IconMic size={16} />
                </span>
                <span className="oc-voice-readiness-copy">
                  <b>{readinessNotice.title}</b>
                  <span>{readinessNotice.detail}</span>
                </span>
              </div>
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

            <div className="oc-voice-meta" style={{ marginTop: 18 }}>
              {latestActivity && (
                <>
                  <span>
                    {latestActivity.label} <b>{latestActivity.status}</b>
                  </span>
                  <span className="sep">·</span>
                </>
              )}
              <span>
                mic <b>{voiceCanAcceptSpeech ? "open" : "muted"}</b>
              </span>
              <span className="sep">·</span>
              <span>
                voice{" "}
                <b>
                  {assistantVoiceLabel(
                    voiceSessionInfo?.assistantVoice || assistantVoiceSelected
                  )}
                </b>
              </span>
            </div>
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

      {voiceEndOptionsOpen && (
        <VoiceEndOptionsDialog
          finalizationStatus={endOptionsFinalizationStatus}
          detail={endOptionsFinalizationDetail}
          endedSession={endedVoiceSession}
          isPersistent={isPersistent}
          onContinueInChat={continueInChat}
          onReviewMemory={reviewMemory}
          onStartNewSession={startFreshSession}
          onStayOnVoice={() => setVoiceEndOptionsOpen(false)}
        />
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
            <VoiceDebugRow label="mic enabled" value={voiceCanAcceptSpeech} />
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

    </>
  );
}
