"use client";

import { useSessionStore } from "@/lib/session";
import type { MemoryFact } from "@/lib/api";

const CATEGORY_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  loss: { bg: "bg-purple-50", text: "text-purple-700", border: "border-purple-200/60" },
  preference: { bg: "bg-rose-50", text: "text-rose-700", border: "border-rose-200/60" },
  coping_strategy: { bg: "bg-teal-50", text: "text-teal-700", border: "border-teal-200/60" },
  relationship: { bg: "bg-blue-50", text: "text-blue-700", border: "border-blue-200/60" },
  trigger: { bg: "bg-amber-50", text: "text-amber-700", border: "border-amber-200/60" },
  goal: { bg: "bg-emerald-50", text: "text-emerald-700", border: "border-emerald-200/60" },
  context: { bg: "bg-indigo-50", text: "text-indigo-700", border: "border-indigo-200/60" },
};

const DEFAULT_COLORS = { bg: "bg-oc-warm-100", text: "text-oc-warm-600", border: "border-oc-warm-200" };

function formatTime(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch {
    return "";
  }
}

function formatCategory(cat: string): string {
  return cat.replace(/_/g, " ");
}

function FactCard({ fact, isNew }: { fact: MemoryFact; isNew: boolean }) {
  const colors = CATEGORY_COLORS[fact.category] || DEFAULT_COLORS;

  // Build a readable summary from the SPO triple
  const summary =
    fact.subject && fact.object
      ? `${fact.subject} ${fact.predicate.replace(/_/g, " ")} ${fact.object}`
      : fact.evidence_quote;

  return (
    <div
      className={`rounded-xl border border-oc-border bg-oc-bg-card p-3.5 space-y-2 ${isNew ? "animate-memoryIn" : ""}`}
    >
      <p className="text-[13px] leading-relaxed text-oc-text">
        {summary}
      </p>
      <div className="flex items-center gap-2">
        <span
          className={`text-[10px] font-mono font-medium px-1.5 py-0.5 rounded border ${colors.bg} ${colors.text} ${colors.border}`}
        >
          {formatCategory(fact.category)}
        </span>
        <span className="text-[11px] font-mono text-oc-text-dim">
          {formatTime(fact.created_at)}
        </span>
      </div>
    </div>
  );
}

export function MemoryPanel() {
  const memoryFacts = useSessionStore((s) => s.memoryFacts);
  const memoryPanelOpen = useSessionStore((s) => s.memoryPanelOpen);
  const setMemoryPanelOpen = useSessionStore((s) => s.setMemoryPanelOpen);

  if (!memoryPanelOpen) return null;

  return (
    <div className="w-[320px] shrink-0 border-l border-oc-border bg-oc-bg flex flex-col h-full animate-panelSlideIn">
      {/* Header */}
      <div className="px-4 py-3.5 border-b border-oc-border flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-4 h-4 text-oc-teal-600">
            <path d="M12 2a7 7 0 017 7c0 2.38-1.19 4.47-3 5.74V17a2 2 0 01-2 2h-4a2 2 0 01-2-2v-2.26C6.19 13.47 5 11.38 5 9a7 7 0 017-7z" />
            <path d="M10 21h4" />
          </svg>
          <h2 className="font-display text-sm text-oc-teal-900">Memories</h2>
          <span className="text-[11px] font-mono text-oc-text-dim">
            {memoryFacts.length}
          </span>
        </div>
        <button
          onClick={() => setMemoryPanelOpen(false)}
          aria-label="Close memories panel"
          className="text-oc-text-muted hover:text-oc-text transition-colors p-1"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-4 h-4">
            <path d="M18 6L6 18M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Facts list */}
      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-2.5">
        {memoryFacts.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center px-4">
            <div className="w-10 h-10 rounded-full bg-oc-accent-subtle flex items-center justify-center mb-3">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-5 h-5 text-oc-teal-500">
                <path d="M12 2a7 7 0 017 7c0 2.38-1.19 4.47-3 5.74V17a2 2 0 01-2 2h-4a2 2 0 01-2-2v-2.26C6.19 13.47 5 11.38 5 9a7 7 0 017-7z" />
                <path d="M10 21h4" />
              </svg>
            </div>
            <p className="text-[13px] text-oc-text-muted">
              No semantic facts yet
            </p>
            <p className="text-[12px] text-oc-text-dim mt-1">
              Session summaries and style rules appear on the Memory page
            </p>
          </div>
        ) : (
          // Show newest first
          [...memoryFacts].reverse().map((fact, i) => (
            <FactCard key={fact.key} fact={fact} isNew={i === 0} />
          ))
        )}
      </div>
    </div>
  );
}

export function MemoryToggleButton() {
  const memoryPanelOpen = useSessionStore((s) => s.memoryPanelOpen);
  const setMemoryPanelOpen = useSessionStore((s) => s.setMemoryPanelOpen);
  const memoryUnseenCount = useSessionStore((s) => s.memoryUnseenCount);
  const sessionMode = useSessionStore((s) => s.sessionMode);

  // Only show in persistent mode
  if (sessionMode === "incognito") return null;

  return (
    <button
      onClick={() => setMemoryPanelOpen(!memoryPanelOpen)}
      aria-label={memoryPanelOpen ? "Close memories panel" : "Open memories panel"}
      aria-expanded={memoryPanelOpen}
      className={`flex items-center gap-1.5 text-[12px] font-mono transition-colors ${
        memoryPanelOpen
          ? "text-oc-teal-700"
          : "text-oc-text-muted hover:text-oc-teal-700"
      }`}
    >
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3.5 h-3.5">
        <path d="M12 2a7 7 0 017 7c0 2.38-1.19 4.47-3 5.74V17a2 2 0 01-2 2h-4a2 2 0 01-2-2v-2.26C6.19 13.47 5 11.38 5 9a7 7 0 017-7z" />
        <path d="M10 21h4" />
      </svg>
      memories
      {memoryUnseenCount > 0 && (
        <span className="inline-flex items-center justify-center min-w-[18px] h-[18px] px-1 rounded-full bg-oc-teal-700 text-white text-[10px] font-bold animate-memoryBadge">
          +{memoryUnseenCount}
        </span>
      )}
    </button>
  );
}
