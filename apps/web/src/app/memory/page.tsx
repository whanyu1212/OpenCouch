"use client";

import { useState, useEffect, useCallback } from "react";
import {
  getMemoryStatus,
  getMemoryFacts,
  getMemorySessions,
  getMemoryRules,
  deleteMemoryFact,
  deleteMemorySession,
  deleteMemoryRule,
  updateMemoryRecall,
  type MemoryFact,
  type MemoryRule,
  type MemorySession,
  type MemoryStatus,
} from "@/lib/api";
import { useSessionStore } from "@/lib/session";
import { CouchLogo } from "@/components/logo";
import { SessionPill } from "@/components/conversation-shell";

type Tab = "overview" | "facts" | "sessions" | "rules";

export default function MemoryPage() {
  const userId = useSessionStore((s) => s.userId);
  const threadId = useSessionStore((s) => s.threadId);
  const sessionMode = useSessionStore((s) => s.sessionMode);
  const memoryRefreshVersion = useSessionStore((s) => s.memoryRefreshVersion);
  const bumpMemoryRefreshVersion = useSessionStore((s) => s.bumpMemoryRefreshVersion);
  const [tab, setTab] = useState<Tab>("overview");
  const [status, setStatus] = useState<MemoryStatus | null>(null);
  const [facts, setFacts] = useState<MemoryFact[]>([]);
  const [sessions, setSessions] = useState<MemorySession[]>([]);
  const [rules, setRules] = useState<MemoryRule[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [updatingRecall, setUpdatingRecall] = useState(false);

  const loadAll = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [s, f, sess, r] = await Promise.all([
        getMemoryStatus(threadId, userId, sessionMode),
        getMemoryFacts(threadId, userId, sessionMode),
        getMemorySessions(threadId, userId, sessionMode),
        getMemoryRules(threadId, userId, sessionMode),
      ]);
      setStatus(s);
      setFacts(f);
      setSessions(sess);
      setRules(r);
    } catch {
      setError("Could not load memory — is the backend running?");
    } finally {
      setLoading(false);
    }
  }, [sessionMode, threadId, userId]);

  useEffect(() => {
    loadAll();
  }, [loadAll, memoryRefreshVersion]);

  const handleDeleteFact = async (index: number) => {
    setError(null);
    try {
      await deleteMemoryFact(index, threadId, userId || undefined, sessionMode);
      bumpMemoryRefreshVersion();
    } catch {
      setError("Could not delete memory fact.");
    }
  };

  const handleDeleteSession = async (index: number) => {
    setError(null);
    try {
      await deleteMemorySession(index, threadId, userId || undefined, sessionMode);
      bumpMemoryRefreshVersion();
    } catch {
      setError("Could not delete memory session.");
    }
  };

  const handleDeleteRule = async (index: number) => {
    setError(null);
    try {
      await deleteMemoryRule(index, threadId, userId || undefined, sessionMode);
      bumpMemoryRefreshVersion();
    } catch {
      setError("Could not delete memory rule.");
    }
  };

  const handleRecallChange = async (enabled: boolean) => {
    setUpdatingRecall(true);
    setError(null);
    try {
      const result = await updateMemoryRecall(
        enabled,
        threadId,
        userId || undefined,
        sessionMode
      );
      setStatus((current) =>
        current
          ? {
              ...current,
              proactive_recall_enabled: result.proactive_recall_enabled,
            }
          : current
      );
      bumpMemoryRefreshVersion();
    } catch {
      setError("Could not update proactive recall.");
    } finally {
      setUpdatingRecall(false);
    }
  };

  const TABS: { key: Tab; label: string; count: number }[] = [
    { key: "overview", label: "Overview", count: 0 },
    { key: "facts", label: "Facts", count: facts.length },
    { key: "sessions", label: "Sessions", count: sessions.length },
    { key: "rules", label: "Rules", count: rules.length },
  ];

  return (
    <>
      {/* Desktop top bar — wrapper controls breakpoint visibility */}
      <div className="oc-app-top-wrap">
      <header className="oc-app-top">
        <div className="flex items-center gap-3">
          <span className="oc-mobile-mark">
            <CouchLogo className="w-4 h-4" />
          </span>
          <h2 className="oc-app-top-title">Memory</h2>
        </div>
        <div className="flex items-center gap-2.5">
          <button
            onClick={loadAll}
            disabled={loading}
            className="oc-tab-chip"
          >
            {loading ? "loading…" : "refresh"}
          </button>
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
          <h2 className="oc-mobile-top-title">Memory</h2>
        </div>
        <button
          onClick={loadAll}
          disabled={loading}
          className="oc-tab-chip"
          style={{ padding: "4px 8px", fontSize: 9.5 }}
        >
          {loading ? "loading…" : "refresh"}
        </button>
      </header>
      </div>

      {/* Tabs */}
      <div className="px-4 md:px-6 border-b border-oc-line-2 flex gap-0 shrink-0 overflow-x-auto">
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`px-4 py-3 text-[14px] font-medium border-b-2 transition-colors whitespace-nowrap ${
              tab === t.key
                ? "border-oc-teal-600 text-oc-teal-800"
                : "border-transparent text-oc-text-muted hover:text-oc-text-secondary"
            }`}
          >
            {t.label}
            {t.count > 0 && (
              <span className="ml-2 text-[11px] font-mono bg-oc-teal-50 text-oc-teal-600 px-2 py-0.5 rounded-md border border-oc-teal-200/60">
                {t.count}
              </span>
            )}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto px-4 md:px-6 py-5">
        {loading && (
          <div className="flex items-center gap-2 text-oc-text-muted text-sm font-mono">
            <div className="dot-pulse"><span /><span /><span /></div>
            loading
          </div>
        )}
        {error && (
          <div className="px-4 py-3 bg-oc-red-subtle border border-oc-red/20 rounded-xl text-oc-red text-[15px]">
            {error}
          </div>
        )}

        {!loading && !error && (
          <div className="animate-fadeIn">
            {tab === "overview" && status && (
              <OverviewTab
                status={status}
                updatingRecall={updatingRecall}
                onRecallChange={handleRecallChange}
              />
            )}
            {tab === "facts" && <FactsTab facts={facts} onDelete={handleDeleteFact} />}
            {tab === "sessions" && <SessionsTab sessions={sessions} onDelete={handleDeleteSession} />}
            {tab === "rules" && <RulesTab rules={rules} onDelete={handleDeleteRule} />}
          </div>
        )}
      </div>

    </>
  );
}

function OverviewTab({
  status,
  updatingRecall,
  onRecallChange,
}: {
  status: MemoryStatus;
  updatingRecall: boolean;
  onRecallChange: (enabled: boolean) => Promise<void>;
}) {
  const recallEnabled = status.proactive_recall_enabled;

  return (
    <div className="space-y-5 max-w-lg">
      {/* Count cards */}
      <div className="grid grid-cols-3 gap-3">
        {Object.entries(status.counts).map(([kind, count]) => (
          <div key={kind} className="bg-oc-bg-card border border-oc-border rounded-xl p-5 text-center">
            <p className="text-3xl font-display text-oc-teal-700 tabular-nums">{count}</p>
            <p className="text-[12px] text-oc-text-muted mt-1.5 font-mono uppercase tracking-wider">{kind}</p>
          </div>
        ))}
      </div>

      {/* Config card */}
      <div className="bg-oc-bg-card border border-oc-border rounded-xl divide-y divide-oc-border">
        <MetaRow label="Mode" value={status.memory_mode} />
        <MetaRow label="Owner" value={status.owner_id} mono />
        <MetaRow label="Crisis log" value={String(status.crisis_log_count)} />
        <MetaRow
          label="Session feedback"
          value={String(status.session_feedback_count)}
        />
        <MetaRow
          label="Proactive recall"
          value={recallEnabled ? "On" : "Off"}
          accent={recallEnabled}
        />
      </div>

      <div className="bg-oc-bg-card border border-oc-border rounded-xl p-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-[15px] font-medium text-oc-text">
              Proactive recall
            </p>
            <p className="mt-1.5 text-[13px] leading-relaxed text-oc-text-muted">
              When on, OpenCouch may mention relevant past sessions without
              being asked. When off, it should only reference saved memories
              when you ask or when safety and style preferences require it.
            </p>
          </div>
          <button
            type="button"
            disabled={updatingRecall}
            aria-pressed={recallEnabled}
            onClick={() => void onRecallChange(!recallEnabled)}
            className={`shrink-0 rounded-full border px-3.5 py-2 text-[12px] font-mono transition-all disabled:opacity-50 ${
              recallEnabled
                ? "border-oc-teal-300 bg-oc-teal-50 text-oc-teal-800"
                : "border-oc-border bg-oc-bg text-oc-text-secondary hover:bg-oc-warm-50"
            }`}
          >
            {updatingRecall ? "saving..." : recallEnabled ? "on" : "off"}
          </button>
        </div>
      </div>
    </div>
  );
}

function MetaRow({ label, value, mono, accent }: { label: string; value: string; mono?: boolean; accent?: boolean }) {
  return (
    <div className="flex justify-between px-5 py-3 text-[14px]">
      <span className="text-oc-text-muted">{label}</span>
      <span className={`${mono ? "font-mono text-[13px]" : ""} ${accent ? "text-oc-green font-medium" : "text-oc-text-secondary"}`}>
        {value}
      </span>
    </div>
  );
}

function FactsTab({ facts, onDelete }: { facts: MemoryFact[]; onDelete: (index: number) => Promise<void> }) {
  const [filter, setFilter] = useState<string | null>(null);

  if (facts.length === 0) return <Empty label="No semantic facts stored yet." />;

  const categories = Array.from(new Set(facts.map(f => f.category))).filter(Boolean);
  const filteredFacts = filter ? facts.filter(f => f.category === filter) : facts;

  return (
    <div className="space-y-4 max-w-2xl">
      {categories.length > 1 && (
        <div className="flex flex-wrap gap-2 mb-2">
          <button
            onClick={() => setFilter(null)}
            className={`px-3 py-1.5 rounded-full text-[12px] font-medium transition-colors ${!filter ? "bg-oc-ink-2 text-white" : "bg-oc-warm-100 text-oc-text-secondary hover:bg-oc-warm-200"}`}
          >
            All
          </button>
          {categories.map(cat => (
            <button
              key={cat}
              onClick={() => setFilter(cat)}
              className={`px-3 py-1.5 rounded-full text-[12px] font-medium transition-colors ${filter === cat ? "bg-oc-ink-2 text-white" : "bg-oc-warm-100 text-oc-text-secondary hover:bg-oc-warm-200"}`}
            >
              {cat.replace(/_/g, " ")}
            </button>
          ))}
        </div>
      )}
      <div className="space-y-3">
        {filteredFacts.map((f, i) => (
          <div key={i} className="group bg-oc-bg-card border border-oc-border rounded-xl p-5 hover:border-oc-border-strong transition-colors shadow-sm">
            <div className="flex items-start justify-between gap-4">
              <div className="flex-1 min-w-0">
                <p className="text-[15px] text-oc-text leading-relaxed italic">
                  &ldquo;{String(f.evidence_quote)}&rdquo;
                </p>
                <div className="flex flex-wrap gap-2 mt-3">
                  <Tag>{f.category}</Tag>
                  <Tag>{f.subject} → {f.predicate} → {f.object}</Tag>
                  <Tag muted>{f.confidence}</Tag>
                </div>
              </div>
              <div className="flex flex-col items-end gap-2 shrink-0">
                <DeleteButton onDelete={() => onDelete(f.index)} label="fact" />
                <span className="text-[11px] font-mono text-oc-text-dim mt-auto pt-2">#{f.index}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function SessionsTab({ sessions, onDelete }: { sessions: MemorySession[]; onDelete: (index: number) => Promise<void> }) {
  if (sessions.length === 0) return <Empty label="No episodic session arcs yet." />;
  return (
    <div className="space-y-4 max-w-2xl relative">
      {/* Add a subtle timeline vertical line on the left */}
      <div className="absolute left-6 top-4 bottom-4 w-px bg-oc-line-2 z-0 hidden sm:block" />
      {sessions.map((s, i) => (
        <div key={i} className="group relative z-10 flex gap-4">
          <div className="hidden sm:flex flex-col items-center mt-2 shrink-0 w-12">
            <div className="w-3 h-3 rounded-full bg-oc-warm-200 border-2 border-white shadow-sm" />
          </div>
          <div className="flex-1 bg-white border border-oc-border rounded-xl p-5 hover:border-oc-border-strong transition-colors shadow-sm">
            <div className="flex items-start justify-between gap-4">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-[12px] font-mono font-medium text-oc-text-secondary">
                    {s.ended_at.slice(0, 10)}
                  </span>
                  <span className="text-[12px] text-oc-text-dim px-2 py-0.5 rounded-full bg-oc-warm-50 border border-oc-warm-200">
                    {s.turn_count} turns
                  </span>
                </div>
                <p className="text-[15px] text-oc-ink-2 leading-relaxed">{s.summary}</p>
                <div className="flex flex-wrap gap-2 mt-4">
                  <div className="flex items-center gap-1.5 px-2 py-1 bg-gradient-to-r from-oc-warm-50 to-oc-teal-50 border border-oc-line-2 rounded-md text-[11px] font-mono text-oc-text-secondary">
                    <span>{s.mood_opened}</span>
                    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="opacity-50"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
                    <span>{s.mood_closed}</span>
                  </div>
                  {s.themes.map((t) => (
                    <Tag key={t} muted>{t}</Tag>
                  ))}
                </div>
              </div>
              <DeleteButton onDelete={() => onDelete(s.index)} label="session" />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function RulesTab({ rules, onDelete }: { rules: MemoryRule[]; onDelete: (index: number) => Promise<void> }) {
  if (rules.length === 0) return <Empty label="No procedural style rules yet." />;
  return (
    <div className="space-y-4 max-w-2xl">
      {rules.map((r, i) => (
        <div key={i} className="group bg-oc-teal-50 border border-oc-teal-200/60 rounded-xl p-5 hover:border-oc-teal-300/80 transition-colors shadow-sm relative overflow-hidden">
          {/* Subtle accent bar */}
          <div className="absolute left-0 top-0 bottom-0 w-1 bg-oc-teal-400 opacity-50" />
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1.5">
                <span className="text-[10px] font-mono uppercase tracking-widest text-oc-teal-700/70">
                  Style Rule
                </span>
                <span className="text-oc-text-dim">·</span>
                <span className="text-[11px] font-mono text-oc-teal-700/60">
                  {r.confidence} confidence
                </span>
              </div>
              <p className="text-[15px] font-medium text-oc-teal-900 leading-relaxed mb-3">
                {r.rule}
              </p>
              {r.evidence.length > 0 ? (
                <div className="mt-3 pt-3 border-t border-oc-teal-200/40">
                  <p className="text-[12px] text-oc-teal-800/70 italic font-mono leading-relaxed">
                    Based on: &quot;{r.evidence.join('&quot;, &quot;')}&quot;
                  </p>
                </div>
              ) : null}
            </div>
            <div className="flex flex-col items-end gap-2 shrink-0">
              <DeleteButton onDelete={() => onDelete(r.index)} label="rule" />
              {r.added_at ? <span className="text-[10px] font-mono text-oc-teal-700/50 mt-auto pt-2">{r.added_at.slice(0, 10)}</span> : null}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function DeleteButton({ onDelete, label }: { onDelete: () => Promise<void>; label: string }) {
  const [confirming, setConfirming] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const handleClick = async () => {
    if (!confirming) {
      setConfirming(true);
      return;
    }
    setDeleting(true);
    try {
      await onDelete();
    } finally {
      setDeleting(false);
      setConfirming(false);
    }
  };

  return (
    <button
      onClick={handleClick}
      onBlur={() => setConfirming(false)}
      disabled={deleting}
      className={`shrink-0 text-[11px] font-mono px-2 py-1 rounded-md border transition-all ${
        confirming
          ? "bg-red-50 text-red-600 border-red-200 opacity-100"
          : "text-oc-text-dim border-transparent opacity-0 group-hover:opacity-100 hover:text-red-500 hover:border-red-200"
      } disabled:opacity-50`}
      title={`Delete this ${label}`}
    >
      {deleting ? "…" : confirming ? "confirm?" : "forget"}
    </button>
  );
}

function Tag({ children, muted }: { children: React.ReactNode; muted?: boolean }) {
  return (
    <span className={`text-[11px] font-mono font-medium px-2 py-0.5 rounded-md border ${
      muted
        ? "bg-oc-warm-100 text-oc-warm-600 border-oc-warm-200"
        : "bg-oc-teal-50 text-oc-teal-700 border-oc-teal-200/60"
    }`}>
      {children}
    </span>
  );
}

function Empty({ label }: { label: string }) {
  return (
    <div className="text-center py-16 text-oc-text-muted text-base font-mono">{label}</div>
  );
}
