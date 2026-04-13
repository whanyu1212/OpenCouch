"use client";

import { useState, useEffect, useCallback } from "react";
import {
  getMemoryStatus,
  getMemoryFacts,
  getMemorySessions,
  getMemoryRules,
  type MemoryStatus,
} from "@/lib/api";
import { useSessionStore } from "@/lib/session";

type Tab = "overview" | "facts" | "sessions" | "rules";

export default function MemoryPage() {
  const { userId, threadId } = useSessionStore();
  const [tab, setTab] = useState<Tab>("overview");
  const [status, setStatus] = useState<MemoryStatus | null>(null);
  const [facts, setFacts] = useState<Record<string, unknown>[]>([]);
  const [sessions, setSessions] = useState<Record<string, unknown>[]>([]);
  const [rules, setRules] = useState<Record<string, unknown>[]>([]);
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
  }, [loadAll]);

  const TABS: { key: Tab; label: string; count: number }[] = [
    { key: "overview", label: "Overview", count: 0 },
    { key: "facts", label: "Facts", count: facts.length },
    { key: "sessions", label: "Sessions", count: sessions.length },
    { key: "rules", label: "Rules", count: rules.length },
  ];

  return (
    <div className="flex flex-col h-screen">
      <header className="px-6 py-3.5 border-b border-oc-border shrink-0 flex items-center justify-between">
        <h1 className="text-sm font-semibold text-oc-teal-800">Memory</h1>
        <button
          onClick={loadAll}
          className="text-[11px] text-oc-teal-500 hover:text-oc-teal-400 transition-colors"
        >
          Refresh
        </button>
      </header>

      {/* Tabs */}
      <div className="px-6 border-b border-oc-border flex gap-0 shrink-0">
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`px-3 py-2.5 text-[12px] font-medium border-b-2 transition-colors ${
              tab === t.key
                ? "border-oc-teal-500 text-oc-teal-700"
                : "border-transparent text-oc-text-muted hover:text-oc-text-secondary"
            }`}
          >
            {t.label}
            {t.count > 0 && (
              <span className="ml-1.5 text-[10px] bg-oc-teal-50 text-oc-teal-600 px-1.5 py-0.5 rounded-full">
                {t.count}
              </span>
            )}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-5">
        {loading && <p className="text-oc-text-muted text-xs">Loading...</p>}
        {error && (
          <div className="px-3.5 py-2.5 bg-red-50 border border-red-200 rounded-lg text-oc-red text-[13px]">
            {error}
          </div>
        )}

        {!loading && !error && tab === "overview" && status && (
          <OverviewTab status={status} />
        )}
        {!loading && !error && tab === "facts" && <FactsTab facts={facts} />}
        {!loading && !error && tab === "sessions" && <SessionsTab sessions={sessions} />}
        {!loading && !error && tab === "rules" && <RulesTab rules={rules} />}
      </div>
    </div>
  );
}

function OverviewTab({ status }: { status: MemoryStatus }) {
  return (
    <div className="space-y-4 max-w-md">
      <div className="grid grid-cols-3 gap-3">
        {Object.entries(status.counts).map(([kind, count]) => (
          <div key={kind} className="bg-oc-bg-card border border-oc-border rounded-xl p-3.5 text-center">
            <p className="text-2xl font-bold text-oc-teal-600 tabular-nums">{count}</p>
            <p className="text-[11px] text-oc-text-secondary mt-0.5 capitalize">{kind}</p>
          </div>
        ))}
      </div>
      <div className="bg-oc-bg-card border border-oc-border rounded-xl p-3.5 text-[12px] space-y-1.5">
        <Row label="Mode" value={status.memory_mode} />
        <Row label="Owner" value={status.owner_id} mono />
        <Row label="Crisis log" value={String(status.crisis_log_count)} />
        <Row label="Proactive recall" value={status.proactive_recall_enabled ? "On" : "Off"} highlight={status.proactive_recall_enabled} />
      </div>
    </div>
  );
}

function Row({ label, value, mono, highlight }: { label: string; value: string; mono?: boolean; highlight?: boolean }) {
  return (
    <div className="flex justify-between">
      <span className="text-oc-text-muted">{label}</span>
      <span className={`${mono ? "font-mono text-[11px]" : ""} ${highlight ? "text-oc-green font-medium" : "text-oc-text-secondary"}`}>
        {value}
      </span>
    </div>
  );
}

function FactsTab({ facts }: { facts: Record<string, unknown>[] }) {
  if (facts.length === 0) return <Empty label="No semantic facts stored yet." />;
  return (
    <div className="space-y-2">
      {facts.map((f, i) => (
        <div key={i} className="bg-oc-bg-card border border-oc-border rounded-lg p-3.5">
          <div className="flex items-start justify-between gap-3">
            <div className="flex-1 min-w-0">
              <p className="text-[13px] text-oc-text leading-relaxed">
                &ldquo;{String(f.evidence_quote)}&rdquo;
              </p>
              <div className="flex flex-wrap gap-2 mt-2">
                <Tag>{String(f.category)}</Tag>
                <Tag>{String(f.subject)} → {String(f.predicate)} → {String(f.object)}</Tag>
                <Tag muted>{String(f.confidence)}</Tag>
              </div>
            </div>
            <span className="text-[10px] text-oc-text-dim shrink-0">#{String(f.index)}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

function SessionsTab({ sessions }: { sessions: Record<string, unknown>[] }) {
  if (sessions.length === 0) return <Empty label="No episodic session arcs yet." />;
  return (
    <div className="space-y-2">
      {sessions.map((s, i) => (
        <div key={i} className="bg-oc-bg-card border border-oc-border rounded-lg p-3.5">
          <p className="text-[13px] text-oc-text leading-relaxed">{String(s.summary)}</p>
          <div className="flex flex-wrap gap-2 mt-2">
            {(s.themes as string[] || []).map((t: string) => (
              <Tag key={t}>{t}</Tag>
            ))}
            <Tag muted>{String(s.mood_opened)} → {String(s.mood_closed)}</Tag>
            <Tag muted>{String(s.turn_count)} turns</Tag>
            <Tag muted>{String(s.ended_at).slice(0, 10)}</Tag>
          </div>
        </div>
      ))}
    </div>
  );
}

function RulesTab({ rules }: { rules: Record<string, unknown>[] }) {
  if (rules.length === 0) return <Empty label="No procedural style rules yet." />;
  return (
    <div className="space-y-2">
      {rules.map((r, i) => (
        <div key={i} className="bg-oc-bg-card border border-oc-border rounded-lg p-3.5">
          <p className="text-[13px] text-oc-text">{String(r.rule)}</p>
          <div className="flex flex-wrap gap-2 mt-2">
            <Tag muted>{String(r.confidence)}</Tag>
            {r.added_at ? <Tag muted>{String(r.added_at).slice(0, 10)}</Tag> : null}
          </div>
          {Array.isArray(r.evidence) && r.evidence.length > 0 ? (
            <p className="text-[11px] text-oc-text-muted mt-1.5 italic">
              Evidence: {(r.evidence as string[]).join("; ")}
            </p>
          ) : null}
        </div>
      ))}
    </div>
  );
}

function Tag({ children, muted }: { children: React.ReactNode; muted?: boolean }) {
  return (
    <span className={`text-[10px] font-medium px-2 py-0.5 rounded-full ${
      muted
        ? "bg-oc-warm-100 text-oc-text-muted"
        : "bg-oc-teal-50 text-oc-teal-700"
    }`}>
      {children}
    </span>
  );
}

function Empty({ label }: { label: string }) {
  return (
    <div className="text-center py-12 text-oc-text-muted text-sm">{label}</div>
  );
}
