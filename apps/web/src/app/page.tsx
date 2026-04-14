"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
  createChatStream,
  getHistory,
  getMemoryStatus,
  getThreads,
  type ChatResponse,
  type CrisisInfo,
  type StreamEvent,
  type Message,
  type MemoryStatus,
  type ThreadSummary,
} from "@/lib/api";
import { useSessionStore, type ChatMessage } from "@/lib/session";
import { CouchLogo } from "@/components/logo";

const CONVERSATION_STARTERS = [
  {
    label: "Check in",
    prompt: "Hi. What feels most worth putting down first today?",
    icon: "💬",
  },
  {
    label: "Breathing exercise",
    prompt: "I'm feeling wound up. Can we do a breathing exercise?",
    icon: "🌬️",
  },
  {
    label: "Examine a thought",
    prompt: "I have a thought that keeps pulling at me. Can we do a thought record?",
    icon: "💭",
  },
  {
    label: "I'm stuck",
    prompt: "I've been stuck all week and can't make myself do anything. Can we try something small?",
    icon: "🧱",
  },
  {
    label: "Self-compassion",
    prompt: "I'm being really hard on myself right now. Is there something we can do about that?",
    icon: "🤲",
  },
  {
    label: "Let go of a thought",
    prompt: "I want to try letting go of a thought instead of arguing with it. Can we do that?",
    icon: "🍃",
  },
  {
    label: "Reflect on patterns",
    prompt: "I've been noticing a pattern in how I react to stress. Can we explore it?",
    icon: "🔍",
  },
  {
    label: "What do you know?",
    prompt: "What do you remember about me from our past conversations?",
    icon: "🧠",
  },
];

export default function TextChatPage() {
  const { userId, threadId, sessionMode, setThreadId, messages, setMessages, addMessage, appendToLastMessage, updateLastMessage, clearMessages, chatLoading: isLoading, setChatLoading: setIsLoading } = useSessionStore();
  const [input, setInput] = useState("");
  const [stages, setStages] = useState<string[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Empty state data
  const [memoryStatus, setMemoryStatus] = useState<MemoryStatus | null>(null);
  const [recentThreads, setRecentThreads] = useState<ThreadSummary[]>([]);
  const lastLoadedThread = useRef<string | null>(null);

  // Load history when thread changes — NOT on every re-mount.
  // Messages live in Zustand so they survive tab switches. We only
  // reload from the backend when the threadId actually changes.
  useEffect(() => {
    if (lastLoadedThread.current === threadId) return;
    lastLoadedThread.current = threadId;
    clearMessages();
    if (sessionMode === "incognito") return;
    getHistory(threadId)
      .then((history) => {
        if (history.length > 0) {
          setMessages(
            history.map((m: Message) => ({
              role: m.role,
              content: m.content,
              mode: m.mode,
            }))
          );
        }
      })
      .catch(() => {});
  }, [threadId, sessionMode, clearMessages, setMessages]);

  // Load memory status and recent threads for empty state
  useEffect(() => {
    if (sessionMode === "incognito") return;
    getMemoryStatus(threadId, userId).then(setMemoryStatus).catch(() => {});
    getThreads(5).then(setRecentThreads).catch(() => {});
  }, [threadId, userId, sessionMode]);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, stages]);

  const sendMessage = useCallback((text?: string) => {
    const msg = (text || input).trim();
    if (!msg || isLoading) return;

    setInput("");
    setIsLoading(true);
    setStages([]);

    if (inputRef.current) {
      inputRef.current.style.height = "auto";
    }

    addMessage({ role: "user", content: msg });

    let done = false;
    let streamingStarted = false;

    const ws = createChatStream(msg, threadId, userId, (event: StreamEvent) => {
      if (event.type === "status") {
        setStages((prev) => [...prev, event.stage]);
      } else if (event.type === "chunk") {
        // v0.9 token streaming: chunks arrive in real time as the LLM
        // generates tokens. First chunk creates the message; subsequent
        // chunks append to it.
        if (!streamingStarted) {
          streamingStarted = true;
          addMessage({ role: "assistant", content: event.text });
          setStages([]);
          setIsLoading(false);
          inputRef.current?.focus();
        } else {
          appendToLastMessage(event.text);
        }
      } else if (event.type === "done") {
        done = true;
        const resp = event.response as ChatResponse;
        if (streamingStarted) {
          // Reconcile: patch the streamed message with the authoritative
          // response text and metadata from DoneEvent. Handles partial-
          // stream failures (where fallback text differs from chunks)
          // and attaches diagnostics/crisis/mode info.
          updateLastMessage({
            content: resp.response_text,
            mode: resp.mode,
            modeSource: resp.mode_source,
            modality: resp.modality,
            responseType: resp.response_type,
            crisis: resp.crisis,
            diagnostics: resp.diagnostics,
          });
        } else {
          // No chunks arrived — render full response from DoneEvent.
          addMessage({
            role: "assistant",
            content: resp.response_text,
            mode: resp.mode,
            modeSource: resp.mode_source,
            modality: resp.modality,
            responseType: resp.response_type,
            crisis: resp.crisis,
            diagnostics: resp.diagnostics,
          });
          setStages([]);
          setIsLoading(false);
          inputRef.current?.focus();
        }
        ws.close();
      }
    });

    ws.onerror = () => {
      if (done) return;
      setStages([]);
      setIsLoading(false);
      addMessage({
        role: "assistant",
        content:
          "Connection error — is the backend running?\nStart with: uv run uvicorn main:app --port 8000",
      });
    };

    ws.onclose = () => {
      if (!done && isLoading) {
        setStages([]);
        setIsLoading(false);
      }
    };
  }, [input, isLoading, threadId, userId, addMessage, setIsLoading]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    const el = e.target;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 120) + "px";
  };

  const turnCount = messages.filter((m) => m.role === "user").length;
  const totalFacts = memoryStatus
    ? Object.values(memoryStatus.counts).reduce((a, b) => a + b, 0)
    : 0;

  // Filter recent threads: exclude current, show only those with turns
  const otherThreads = recentThreads.filter(
    (t) => t.thread_id !== threadId && t.turn_count > 0
  );

  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <header className="px-6 py-3.5 border-b border-oc-border flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          <h1 className="font-display text-lg text-oc-teal-900">Chat</h1>
          {turnCount > 0 && (
            <span className="text-[12px] font-mono text-oc-text-dim">
              {turnCount} turn{turnCount !== 1 ? "s" : ""}
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          {isLoading && (
            <span className="text-[12px] font-mono text-oc-cta">thinking…</span>
          )}
          {messages.length > 0 && !isLoading && (
            <button
              onClick={() => clearMessages()}
              className="flex items-center gap-1.5 text-[12px] font-mono text-oc-text-muted hover:text-oc-red transition-colors"
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3.5 h-3.5">
                <polyline points="3 6 5 6 21 6" />
                <path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6" />
                <path d="M10 11v6M14 11v6" />
              </svg>
              clear chat
            </button>
          )}
        </div>
      </header>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-6">
        {/* ── Empty state ── */}
        {messages.length === 0 && !isLoading && (
          <div className="flex items-center justify-center h-full animate-fadeIn">
            <div className="w-full max-w-lg">
              {/* Greeting */}
              <div className="text-center mb-8">
                <div className="w-14 h-14 rounded-xl bg-oc-accent-glow flex items-center justify-center mx-auto mb-4">
                  <CouchLogo className="w-8 h-8" />
                </div>
                <p className="font-display text-xl text-oc-text">
                  {sessionMode === "incognito"
                    ? "Private session"
                    : userId
                      ? `Welcome back, ${userId}`
                      : "What\u2019s on your mind?"}
                </p>
                <p className="text-oc-text-muted text-sm mt-1.5 font-mono">
                  {sessionMode === "incognito"
                    ? "incognito · nothing is saved"
                    : totalFacts > 0
                      ? `${totalFacts} memories across ${Object.keys(memoryStatus?.counts || {}).filter(k => (memoryStatus?.counts[k] ?? 0) > 0).length} layers`
                      : "start a conversation to begin building memory"}
                </p>
              </div>

              {/* Memory snapshot — persistent mode only */}
              {sessionMode === "persistent" && memoryStatus && totalFacts > 0 && (
                <div className="flex justify-center gap-3 mb-8 animate-slideUp" style={{ animationDelay: "50ms" }}>
                  {Object.entries(memoryStatus.counts).map(([kind, count]) => (
                    <div
                      key={kind}
                      className="flex items-center gap-2 px-3 py-2 bg-oc-bg-card border border-oc-border rounded-lg"
                    >
                      <span className="text-lg font-display text-oc-teal-700 tabular-nums">{count}</span>
                      <span className="text-[12px] font-mono text-oc-text-muted">{kind}</span>
                    </div>
                  ))}
                </div>
              )}

              {/* Conversation starters */}
              <div className="space-y-2 mb-8 animate-slideUp" style={{ animationDelay: "100ms" }}>
                <p className="text-[11px] font-mono uppercase tracking-widest text-oc-text-dim text-center mb-3">
                  Try one of these
                </p>
                <div className="grid grid-cols-2 gap-2">
                  {CONVERSATION_STARTERS.map((starter) => (
                    <button
                      key={starter.label}
                      onClick={() => sendMessage(starter.prompt)}
                      className="text-left p-3.5 rounded-xl border border-oc-border bg-oc-bg hover:bg-oc-teal-50 hover:border-oc-teal-200 transition-all group"
                    >
                      <div className="flex items-start gap-2.5">
                        <span className="text-base mt-0.5">{starter.icon}</span>
                        <div>
                          <p className="text-[14px] font-medium text-oc-text group-hover:text-oc-teal-800 transition-colors">
                            {starter.label}
                          </p>
                          <p className="text-[12px] text-oc-text-muted mt-0.5 leading-relaxed line-clamp-2">
                            {starter.prompt}
                          </p>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Recent sessions — persistent mode only */}
              {sessionMode === "persistent" && otherThreads.length > 0 && (
                <div className="animate-slideUp" style={{ animationDelay: "150ms" }}>
                  <p className="text-[11px] font-mono uppercase tracking-widest text-oc-text-dim text-center mb-3">
                    Recent sessions
                  </p>
                  <div className="space-y-1.5">
                    {otherThreads.slice(0, 3).map((t) => (
                      <button
                        key={t.thread_id}
                        onClick={() => setThreadId(t.thread_id)}
                        className="w-full flex items-center justify-between px-4 py-2.5 rounded-lg border border-oc-border hover:bg-oc-warm-50 hover:border-oc-border-strong transition-all text-left"
                      >
                        <div className="flex items-center gap-2.5 min-w-0">
                          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-4 h-4 text-oc-text-dim shrink-0">
                            <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
                          </svg>
                          <span className="text-[13px] font-mono text-oc-text-secondary truncate">
                            {t.thread_id}
                          </span>
                        </div>
                        <span className="text-[12px] font-mono text-oc-text-dim shrink-0 ml-2">
                          {t.turn_count} turn{t.turn_count !== 1 ? "s" : ""}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── Message list ── */}
        <div className="space-y-5 max-w-2xl mx-auto">
          {messages.map((msg, i) => {
            const userTurnIndex = msg.role === "user"
              ? messages.slice(0, i + 1).filter((m) => m.role === "user").length
              : null;

            return (
              <div key={i} className="animate-slideUp" style={{ animationDelay: `${Math.min(i * 30, 200)}ms` }}>
                {/* User message */}
                {msg.role === "user" && (
                  <div className="flex items-start gap-3 justify-end">
                    <div className="bg-oc-teal-50 border border-oc-teal-100 rounded-2xl rounded-tr-md px-4 py-3 max-w-[80%]">
                      <p className="text-[15px] leading-relaxed text-oc-teal-900 whitespace-pre-wrap">
                        {msg.content}
                      </p>
                    </div>
                    <div className="w-7 h-7 rounded-full bg-oc-teal-100 flex items-center justify-center shrink-0 mt-0.5">
                      <span className="text-[11px] font-mono font-bold text-oc-teal-700">
                        {userTurnIndex}
                      </span>
                    </div>
                  </div>
                )}

                {/* Assistant message */}
                {msg.role === "assistant" && (
                  <div className="flex items-start gap-3">
                    <div className="w-7 h-7 rounded-full bg-oc-warm-200 flex items-center justify-center shrink-0 mt-0.5">
                      <CouchLogo variant="mono" className="w-4 h-4 text-oc-warm-700" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="bg-oc-bg-card border border-oc-border rounded-2xl rounded-tl-md px-4 py-3 max-w-[90%]">
                        <p className="text-[15px] leading-relaxed text-oc-text whitespace-pre-wrap">
                          {msg.content}
                        </p>
                      </div>
                      {msg.mode && <StateStrip msg={msg} />}
                    </div>
                  </div>
                )}
              </div>
            );
          })}

          {/* Loading indicator with live pipeline stages */}
          {isLoading && (
            <div className="flex items-start gap-3 animate-fadeIn">
              <div className="w-7 h-7 rounded-full bg-oc-warm-200 flex items-center justify-center shrink-0 mt-0.5">
                <span className="text-[11px] font-display font-bold text-oc-warm-700">O</span>
              </div>
              <div className="bg-oc-bg-card border border-oc-border rounded-2xl rounded-tl-md px-4 py-3 min-w-[180px]">
                {/* Completed stages */}
                {stages.length > 0 && (
                  <div className="space-y-1.5 mb-2">
                    {stages.map((s, i) => (
                      <div key={i} className="flex items-center gap-2 animate-fadeIn">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="w-3.5 h-3.5 text-oc-green shrink-0">
                          <polyline points="20 6 9 17 4 12" />
                        </svg>
                        <span className="text-[12px] font-mono text-oc-text-muted">{s}</span>
                      </div>
                    ))}
                  </div>
                )}
                {/* Active spinner for next stage */}
                <div className="flex items-center gap-2">
                  <span className="relative flex h-3.5 w-3.5 items-center justify-center shrink-0">
                    <span className="animate-ping absolute inline-flex h-2.5 w-2.5 rounded-full bg-oc-cta opacity-50" />
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-oc-cta" />
                  </span>
                  <span className="text-[12px] font-mono text-oc-cta">
                    {stages.length === 0 ? "starting…" : "processing…"}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Input */}
      <div className="px-6 py-3.5 border-t border-oc-border shrink-0 bg-oc-bg">
        <div className="max-w-2xl mx-auto flex gap-2.5 items-end">
          <textarea
            ref={inputRef}
            value={input}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder="Type a message..."
            disabled={isLoading}
            autoFocus
            rows={1}
            className="flex-1 bg-oc-bg-input border border-oc-border rounded-xl px-4 py-3 text-[15px] placeholder:text-oc-text-dim focus:outline-none focus:border-oc-teal-400 focus:ring-2 focus:ring-oc-accent-subtle transition-all disabled:opacity-50 resize-none overflow-hidden"
          />
          <button
            onClick={() => sendMessage()}
            disabled={isLoading || !input.trim()}
            className="bg-oc-teal-700 text-white px-4 py-3 rounded-xl text-[15px] font-medium hover:bg-oc-teal-600 transition-colors disabled:opacity-30 disabled:cursor-not-allowed shadow-sm"
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-5 h-5">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}


/* ── State Strip ── diagnostic pills under assistant messages ── */

function StateStrip({ msg }: { msg: ChatMessage }) {
  const [expanded, setExpanded] = useState(false);
  const crisis = msg.crisis;
  const diag = msg.diagnostics || {};

  const isCrisis = crisis?.needs_crisis_response;
  const safetyLabel = isCrisis
    ? "crisis"
    : crisis?.needs_clarification
      ? "check"
      : (crisis?.level ?? 0) >= 1
        ? "distress"
        : "safe";

  const totalMs = diag.turn_total_ms != null ? Number(diag.turn_total_ms) : null;

  return (
    <div className="mt-2 ml-0.5">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex flex-wrap items-center gap-2 group text-left"
      >
        <Pill variant="teal">{msg.mode}</Pill>
        {msg.modality && msg.modality !== "none" && (
          <Pill variant="teal">{msg.modality}</Pill>
        )}
        {msg.modeSource && <Pill variant="muted">{msg.modeSource}</Pill>}
        <Pill variant={safetyLabel === "safe" ? "green" : safetyLabel === "crisis" ? "red" : "amber"}>
          {safetyLabel}
        </Pill>
        {diag.retrieval_path != null ? (
          <Pill variant="muted">{String(diag.retrieval_path)}</Pill>
        ) : null}
        {totalMs != null && (
          <span className="text-[11px] font-mono text-oc-text-dim">
            {totalMs.toFixed(0)}ms
          </span>
        )}
        <span className="text-[11px] text-oc-text-dim group-hover:text-oc-text-muted transition-colors ml-0.5">
          {expanded ? "▾" : "▸"}
        </span>
      </button>

      {expanded && (
        <div className="mt-2.5 bg-oc-bg-state rounded-xl p-5 space-y-3.5 animate-slideUp shadow-lg border border-oc-warm-800/50">
          <div className="flex items-center gap-3">
            <span className="text-oc-warm-500 font-mono text-[11px] uppercase tracking-widest w-16 shrink-0">
              route
            </span>
            <div className="flex items-center gap-1.5 font-mono text-[13px]">
              <span className="text-oc-teal-300">crisis_gate</span>
              <span className="text-oc-warm-600">→</span>
              <span className={safetyLabel === "safe" ? "text-emerald-400" : "text-red-400"}>
                {safetyLabel}
              </span>
              <span className="text-oc-warm-600">→</span>
              <span className="text-oc-teal-300">
                {msg.responseType === "crisis" ? "crisis_response" : "therapeutic"}
              </span>
              <span className="text-oc-warm-600">→</span>
              <span className="text-amber-300">{msg.mode}</span>
              {msg.modality && msg.modality !== "none" && (
                <>
                  <span className="text-oc-warm-600">·</span>
                  <span className="text-purple-300">{msg.modality}</span>
                </>
              )}
            </div>
          </div>

          {crisis?.reason && (
            <div className="flex items-start gap-3">
              <span className="text-oc-warm-500 font-mono text-[11px] uppercase tracking-widest w-16 shrink-0 pt-px">
                reason
              </span>
              <span className="text-oc-warm-400 italic text-[13px]">{crisis.reason}</span>
            </div>
          )}

          <div className="flex items-start gap-3">
            <span className="text-oc-warm-500 font-mono text-[11px] uppercase tracking-widest w-16 shrink-0 pt-px">
              timing
            </span>
            <div className="grid grid-cols-2 gap-x-8 gap-y-1.5 flex-1 font-mono">
              <TimingRow label="load_memory" ms={diag.load_memory_ms} />
              <TimingRow label="crisis_gate" ms={diag.crisis_gate_ms} />
              <TimingRow label="extract_facts" ms={diag.extract_facts_ms} extra={diag.semantic_writes} />
              <TimingRow label="extract_rules" ms={diag.extract_procedural_ms} extra={diag.procedural_writes} />
              <TimingRow label="total" ms={diag.turn_total_ms} bold />
            </div>
          </div>

          <div className="flex items-start gap-3">
            <span className="text-oc-warm-500 font-mono text-[11px] uppercase tracking-widest w-16 shrink-0 pt-px">
              memory
            </span>
            <div className="flex flex-wrap gap-4 font-mono text-[12px] text-oc-warm-400">
              <span>sem: <span className="text-oc-teal-300">{String(diag.semantic_hits ?? 0)}</span>/{String(diag.semantic_store_size ?? 0)}</span>
              <span>epi: <span className="text-oc-teal-300">{String(diag.episodic_hits ?? 0)}</span>/{String(diag.episodic_store_size ?? 0)}</span>
              <span>proc: <span className="text-oc-teal-300">{String(diag.procedural_count ?? 0)}</span></span>
              <span>recall: <span className={diag.proactive_recall ? "text-emerald-400" : "text-oc-warm-600"}>{diag.proactive_recall ? "on" : "off"}</span></span>
            </div>
          </div>

          {(diag.extract_facts_reason != null || diag.extract_procedural_reason != null) && (
            <div className="flex items-start gap-3">
              <span className="text-oc-warm-500 font-mono text-[11px] uppercase tracking-widest w-16 shrink-0 pt-px">
                notes
              </span>
              <div className="space-y-1 text-[12px] font-mono text-oc-warm-500">
                {diag.extract_facts_reason != null && (
                  <div><span className="text-oc-teal-400">facts:</span> {String(diag.extract_facts_reason)}</div>
                )}
                {diag.extract_procedural_reason != null && (
                  <div><span className="text-oc-teal-400">rules:</span> {String(diag.extract_procedural_reason)}</div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}


function TimingRow({ label, ms, extra, bold }: { label: string; ms: unknown; extra?: unknown; bold?: boolean }) {
  const formatted = ms != null ? `${Number(ms).toFixed(0)}ms` : "—";
  const writes = extra != null ? ` (${String(extra)}w)` : "";
  return (
    <div className={`flex justify-between text-[12px] ${bold ? "text-oc-warm-200" : "text-oc-warm-500"}`}>
      <span>{label}</span>
      <span className="tabular-nums">
        {formatted}
        {writes && <span className="text-oc-warm-600">{writes}</span>}
      </span>
    </div>
  );
}


function Pill({ children, variant }: { children: React.ReactNode; variant: "teal" | "muted" | "green" | "red" | "amber" }) {
  const styles = {
    teal: "bg-oc-teal-50 text-oc-teal-700 border-oc-teal-200/60",
    muted: "bg-oc-warm-100 text-oc-warm-600 border-oc-warm-200",
    green: "bg-emerald-50 text-emerald-700 border-emerald-200/60",
    red: "bg-red-50 text-red-700 border-red-200/60",
    amber: "bg-amber-50 text-amber-700 border-amber-200/60",
  };

  return (
    <span className={`text-[11px] font-mono font-medium px-2 py-0.5 rounded-md border ${styles[variant]}`}>
      {children}
    </span>
  );
}
