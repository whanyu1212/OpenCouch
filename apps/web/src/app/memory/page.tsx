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
  type MemoryFact,
  type MemoryRule,
  type MemorySession,
  type MemoryStatus,
} from "@/lib/api";
import { useSessionStore } from "@/lib/session";

type Tab = "overview" | "facts" | "sessions" | "rules";

export default function MemoryPage() {
  const { userId, threadId, memoryRefreshVersion, bumpMemoryRefreshVersion } = useSessionStore();
  const [tab, setTab] = useState<Tab>("overview");
  const [status, setStatus] = useState<MemoryStatus | null>(null);
  const [facts, setFacts] = useState<MemoryFact[]>([]);
  const [sessions, setSessions] = useState<MemorySession[]>([]);
  const [rules, setRules] = useState<MemoryRule[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const loadAll = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [s, f, sess, r] = await Promise.all([
        getMemoryStatus(threadId, userId),
        getMemoryFacts(threadId, userId),
        getMemorySessions(threadId, userId),
        getMemoryRules(threadId, userId),
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
  }, [threadId, userId]);

  useEffect(() => {
    loadAll();
  }, [loadAll, memoryRefreshVersion]);

  const handleDeleteFact = async (index: number) => {
    await deleteMemoryFact(index, threadId, userId || undefined);
    bumpMemoryRefreshVersion();
  };

  const handleDeleteSession = async (index: number) => {
    await deleteMemorySession(index, threadId, userId || undefined);
    bumpMemoryRefreshVersion();
  };

  const handleDeleteRule = async (index: number) => {
    await deleteMemoryRule(index, threadId, userId || undefined);
    bumpMemoryRefreshVersion();
  };

  const TABS: { key: Tab; label: string; count: number }[] = [
    { key: "overview", label: "Overview", count: 0 },
    { key: "facts", label: "Facts", count: facts.length },
    { key: "sessions", label: "Sessions", count: sessions.length },
    { key: "rules", label: "Rules", count: rules.length },
  ];

  return (
    <div className="flex flex-col h-screen">
      <header className="px-6 py-3.5 border-b border-oc-border shrink-0 flex items-center justify-between">
        <h1 className="font-display text-lg text-oc-teal-900">Memory</h1>
        <button
          onClick={loadAll}
          disabled={loading}
          className="text-[13px] font-mono text-oc-teal-600 hover:text-oc-teal-500 transition-colors disabled:opacity-50"
        >
          {loading ? "loading…" : "refresh"}
        </button>
      </header>

      {/* Tabs */}
      <div className="px-6 border-b border-oc-border flex gap-0 shrink-0">
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`px-4 py-3 text-[14px] font-medium border-b-2 transition-colors ${
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

      <div className="flex-1 overflow-y-auto px-6 py-5">
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
            {tab === "overview" && status && <OverviewTab status={status} />}
            {tab === "facts" && <FactsTab facts={facts} onDelete={handleDeleteFact} />}
            {tab === "sessions" && <SessionsTab sessions={sessions} onDelete={handleDeleteSession} />}
            {tab === "rules" && <RulesTab rules={rules} onDelete={handleDeleteRule} />}
          </div>
        )}
      </div>
    </div>
  );
}

function OverviewTab({ status }: { status: MemoryStatus }) {
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
          value={status.proactive_recall_enabled ? "On" : "Off"}
          accent={status.proactive_recall_enabled}
        />
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
  if (facts.length === 0) return <Empty label="No semantic facts stored yet." />;
  return (
    <div className="space-y-3 max-w-2xl">
      {facts.map((f, i) => (
        <div key={i} className="group bg-oc-bg-card border border-oc-border rounded-xl p-5 hover:border-oc-border-strong transition-colors">
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
            <div className="flex items-center gap-2 shrink-0">
              <span className="text-[11px] font-mono text-oc-text-dim">#{f.index}</span>
              <DeleteButton onDelete={() => onDelete(f.index)} label="fact" />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function SessionsTab({ sessions, onDelete }: { sessions: MemorySession[]; onDelete: (index: number) => Promise<void> }) {
  if (sessions.length === 0) return <Empty label="No episodic session arcs yet." />;
  return (
    <div className="space-y-3 max-w-2xl">
      {sessions.map((s, i) => (
        <div key={i} className="group bg-oc-bg-card border border-oc-border rounded-xl p-5 hover:border-oc-border-strong transition-colors">
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1 min-w-0">
              <p className="text-[15px] text-oc-text leading-relaxed">{s.summary}</p>
              <div className="flex flex-wrap gap-2 mt-3">
                {s.themes.map((t) => (
                  <Tag key={t}>{t}</Tag>
                ))}
                <Tag muted>{s.mood_opened} → {s.mood_closed}</Tag>
                <Tag muted>{s.turn_count} turns</Tag>
                <Tag muted>{s.ended_at.slice(0, 10)}</Tag>
              </div>
            </div>
            <DeleteButton onDelete={() => onDelete(s.index)} label="session" />
          </div>
        </div>
      ))}
    </div>
  );
}

function RulesTab({ rules, onDelete }: { rules: MemoryRule[]; onDelete: (index: number) => Promise<void> }) {
  if (rules.length === 0) return <Empty label="No procedural style rules yet." />;
  return (
    <div className="space-y-3 max-w-2xl">
      {rules.map((r, i) => (
        <div key={i} className="group bg-oc-bg-card border border-oc-border rounded-xl p-5 hover:border-oc-border-strong transition-colors">
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1 min-w-0">
              <p className="text-[15px] text-oc-text">{r.rule}</p>
              <div className="flex flex-wrap gap-2 mt-3">
                <Tag muted>{r.confidence}</Tag>
                {r.added_at ? <Tag muted>{r.added_at.slice(0, 10)}</Tag> : null}
              </div>
              {r.evidence.length > 0 ? (
                <p className="text-[13px] text-oc-text-muted mt-2.5 italic font-mono leading-relaxed">
                  evidence: {r.evidence.join("; ")}
                </p>
              ) : null}
            </div>
            <DeleteButton onDelete={() => onDelete(r.index)} label="rule" />
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
