"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
  createChatStream,
  type ChatResponse,
  type CrisisInfo,
  type StreamEvent,
  type Message,
  getHistory,
} from "@/lib/api";
import { useSessionStore } from "@/lib/session";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  mode?: string | null;
  modeSource?: string | null;
  responseType?: string | null;
  crisis?: CrisisInfo | null;
  diagnostics?: Record<string, unknown> | null;
}

export default function TextChatPage() {
  const { userId, threadId } = useSessionStore();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [stage, setStage] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setMessages([]);
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
  }, [threadId]);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, stage]);

  const sendMessage = useCallback(() => {
    const text = input.trim();
    if (!text || isLoading) return;

    setInput("");
    setIsLoading(true);
    setStage(null);

    setMessages((prev) => [...prev, { role: "user", content: text }]);

    let done = false;

    const ws = createChatStream(text, threadId, userId, (event: StreamEvent) => {
      if (event.type === "status") {
        setStage(event.stage);
      } else if (event.type === "done") {
        done = true;
        const resp = event.response as ChatResponse;
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: resp.response_text,
            mode: resp.mode,
            modeSource: resp.mode_source,
            responseType: resp.response_type,
            crisis: resp.crisis,
            diagnostics: resp.diagnostics,
          },
        ]);
        setStage(null);
        setIsLoading(false);
        ws.close();
        inputRef.current?.focus();
      }
    });

    ws.onerror = () => {
      if (done) return;
      setStage(null);
      setIsLoading(false);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "Connection error — is the backend running? Start with: uv run uvicorn main:app --port 8000",
        },
      ]);
    };

    ws.onclose = () => {
      if (!done && isLoading) {
        setStage(null);
        setIsLoading(false);
      }
    };
  }, [input, isLoading, threadId, userId]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <header className="px-6 py-3.5 border-b border-oc-border flex items-center justify-between shrink-0">
        <h1 className="text-sm font-semibold text-oc-teal-800">Text Chat</h1>
        {stage && (
          <div className="flex items-center gap-2 text-[11px] text-oc-cta font-medium">
            <div className="w-1.5 h-1.5 rounded-full bg-oc-cta animate-pulse" />
            {stage}
          </div>
        )}
      </header>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-5 space-y-4">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <p className="text-oc-text-secondary text-sm">What&apos;s on your mind?</p>
              <p className="text-oc-text-muted text-xs mt-1">Type a message to start.</p>
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i}>
            {/* Message bubble */}
            <div className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-[75%] rounded-xl px-4 py-2.5 text-[13px] leading-relaxed ${
                  msg.role === "user"
                    ? "bg-oc-teal-50 text-oc-teal-900 border border-oc-teal-100"
                    : "bg-oc-bg-card text-oc-text border border-oc-border"
                }`}
              >
                {msg.content}
              </div>
            </div>

            {/* State strip — assistant messages only */}
            {msg.role === "assistant" && msg.mode && (
              <StateStrip msg={msg} />
            )}
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-oc-bg-card border border-oc-border rounded-xl px-4 py-2.5">
              <div className="flex gap-1">
                <div className="w-1.5 h-1.5 rounded-full bg-oc-teal-400 animate-bounce [animation-delay:0ms]" />
                <div className="w-1.5 h-1.5 rounded-full bg-oc-teal-400 animate-bounce [animation-delay:150ms]" />
                <div className="w-1.5 h-1.5 rounded-full bg-oc-teal-400 animate-bounce [animation-delay:300ms]" />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="px-6 py-3.5 border-t border-oc-border shrink-0">
        <div className="flex gap-2.5 items-center">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type a message..."
            disabled={isLoading}
            autoFocus
            className="flex-1 bg-oc-bg border border-oc-border rounded-lg px-3.5 py-2.5 text-[13px] placeholder:text-oc-text-dim focus:outline-none focus:border-oc-border-strong focus:ring-2 focus:ring-oc-accent-subtle transition-all disabled:opacity-50"
          />
          <button
            onClick={sendMessage}
            disabled={isLoading || !input.trim()}
            className="bg-oc-teal-600 text-white px-4 py-2.5 rounded-lg text-[13px] font-medium hover:bg-oc-teal-500 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}


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
        : "normal";

  return (
    <div className="ml-0 mt-1.5 mb-1">
      {/* Collapsed: pills row */}
      <div
        className="flex flex-wrap items-center gap-1.5 cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <Pill color="teal">{msg.mode}</Pill>
        {msg.modeSource && <Pill color="gray">{msg.modeSource}</Pill>}
        <Pill color={safetyLabel === "normal" ? "green" : safetyLabel === "crisis" ? "red" : "orange"}>
          {safetyLabel}
        </Pill>
        {diag.retrieval_path != null ? (
          <Pill color="gray">{String(diag.retrieval_path)}</Pill>
        ) : null}
        <span className="text-[9px] text-oc-text-dim ml-1">
          {expanded ? "▾" : "▸"} {expanded ? "less" : "details"}
        </span>
      </div>

      {/* Expanded: full diagnostics */}
      {expanded && (
        <div className="mt-2 bg-oc-warm-50 border border-oc-border rounded-lg p-3 text-[11px] space-y-2 animate-in fade-in">
          {/* Route trace */}
          <div className="flex items-center gap-2">
            <span className="text-oc-text-muted font-medium uppercase tracking-wider text-[9px]">
              route
            </span>
            <code className="text-oc-teal-600 text-[11px]">
              crisis_gate({safetyLabel}) → {msg.responseType === "crisis" ? "crisis_response" : "therapeutic"} → {msg.mode}
            </code>
          </div>

          {/* Crisis reason */}
          {crisis?.reason && (
            <div className="text-oc-text-secondary italic text-[11px]">
              {crisis.reason}
            </div>
          )}

          {/* Timings table */}
          <div className="grid grid-cols-2 gap-x-6 gap-y-0.5 text-[10px]">
            <TimingRow label="load_memory" ms={diag.load_memory_ms} />
            <TimingRow label="crisis_gate" ms={diag.crisis_gate_ms} />
            <TimingRow label="extract_facts" ms={diag.extract_facts_ms} extra={`writes: ${String(diag.semantic_writes ?? "-")}`} />
            <TimingRow label="extract_procedural" ms={diag.extract_procedural_ms} extra={`writes: ${String(diag.procedural_writes ?? "-")}`} />
            <TimingRow label="turn_total" ms={diag.turn_total_ms} bold />
          </div>

          {/* Memory hits */}
          <div className="flex flex-wrap gap-3 text-[10px] text-oc-text-muted">
            <span>semantic: {String(diag.semantic_hits ?? 0)}/{String(diag.semantic_store_size ?? 0)}</span>
            <span>episodic: {String(diag.episodic_hits ?? 0)}/{String(diag.episodic_store_size ?? 0)}</span>
            <span>procedural: {String(diag.procedural_count ?? 0)}</span>
            <span>recall: {diag.proactive_recall ? "on" : "off"}</span>
          </div>

          {/* Extractor reasons */}
          {diag.extract_facts_reason != null ? (
            <div className="text-[10px] text-oc-text-muted">
              <span className="font-medium">facts:</span> {String(diag.extract_facts_reason)}
            </div>
          ) : null}
          {diag.extract_procedural_reason != null ? (
            <div className="text-[10px] text-oc-text-muted">
              <span className="font-medium">rules:</span> {String(diag.extract_procedural_reason)}
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
}


function TimingRow({ label, ms, extra, bold }: { label: string; ms: unknown; extra?: string; bold?: boolean }) {
  const formatted = ms != null ? `${Number(ms).toFixed(1)}ms` : "-";
  return (
    <div className={`flex justify-between ${bold ? "font-medium text-oc-text-secondary" : "text-oc-text-muted"}`}>
      <span>{label}</span>
      <span className="tabular-nums">
        {formatted}
        {extra && <span className="ml-2 text-oc-text-dim">({extra})</span>}
      </span>
    </div>
  );
}


function Pill({ children, color }: { children: React.ReactNode; color: "teal" | "gray" | "green" | "red" | "orange" }) {
  const styles = {
    teal: "bg-oc-teal-50 text-oc-teal-700 border-oc-teal-100",
    gray: "bg-oc-warm-100 text-oc-warm-700 border-oc-warm-200",
    green: "bg-emerald-50 text-emerald-700 border-emerald-100",
    red: "bg-red-50 text-red-700 border-red-100",
    orange: "bg-orange-50 text-orange-700 border-orange-100",
  };

  return (
    <span className={`text-[10px] font-semibold px-2 py-0.5 rounded border ${styles[color]}`}>
      {children}
    </span>
  );
}
