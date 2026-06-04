"use client";

import { useState, useEffect, useCallback } from "react";
import { getThreadState } from "@/lib/api";
import { useSessionStore } from "@/lib/session";
import { CouchLogo } from "@/components/logo";
import { SessionPill } from "@/components/conversation-shell";

/**
 * State Inspector — displays the full agent state dict for the current
 * thread. This is the developer dashboard equivalent of `GET /api/threads/{id}/state`.
 *
 * Shows collapsible sections for the current split LangGraph state:
 * response fields, session memory, procedural profile, session progress,
 * exercise state, memory control, crisis, diagnostics, and transcript.
 */

const STATE_SECTIONS: { key: string; label: string; desc: string; icon: string }[] = [
  { key: "route", label: "Route", desc: "Top-level graph branch", icon: "⇢" },
  { key: "response_style", label: "Response Style", desc: "Current reply style", icon: "◆" },
  { key: "therapeutic_approach", label: "Approach", desc: "Therapeutic approach overlay", icon: "◇" },
  { key: "response_text", label: "Response Text", desc: "Generated reply text", icon: "¶" },
  { key: "crisis", label: "Crisis", desc: "Safety assessment: level, confidence, flags", icon: "⚑" },
  { key: "session_memory", label: "Session Memory", desc: "Summary, concerns, loops, goal", icon: "◉" },
  { key: "procedural_profile", label: "Procedural Profile", desc: "Style rules and recall toggle", icon: "☷" },
  { key: "session_progress", label: "Session Progress", desc: "Turn count", icon: "▸" },
  { key: "exercise_state", label: "Exercise State", desc: "Guided exercise continuity", icon: "◎" },
  { key: "memory_control", label: "Memory Control", desc: "Pending memory action", icon: "⌁" },
  { key: "grounded_lookup", label: "Grounded Lookup", desc: "Factual lookup query and status", icon: "⌕" },
  { key: "resource_lookup_status", label: "Crisis Resources", desc: "Crisis resource lookup status", icon: "✚" },
  { key: "inferred_location", label: "Inferred Location", desc: "User-stated crisis location", icon: "⌖" },
  { key: "found_resources", label: "Found Resources", desc: "Verified crisis resources", icon: "☑" },
  { key: "diagnostics", label: "Diagnostics", desc: "Per-turn timings and write counts", icon: "⏱" },
  { key: "transcript", label: "Transcript", desc: "Full conversation history", icon: "¶" },
  { key: "history", label: "History", desc: "Raw history array", icon: "↻" },
  { key: "working_memory", label: "Working Memory", desc: "Cross-turn working memory entries", icon: "⊞" },
];

export default function StateInspectorPage() {
  const threadId = useSessionStore((s) => s.threadId);
  const sessionMode = useSessionStore((s) => s.sessionMode);
  const [state, setState] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(false);

  const fetchState = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const s = await getThreadState(threadId, sessionMode);
      setState(s);
      if (!s) setError("No state found for this thread.");
    } catch {
      setError("Failed to load state — is the backend running?");
    } finally {
      setLoading(false);
    }
  }, [sessionMode, threadId]);

  useEffect(() => {
    fetchState();
  }, [fetchState]);

  // Auto-refresh every 3 seconds when enabled
  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(fetchState, 3000);
    return () => clearInterval(interval);
  }, [autoRefresh, fetchState]);

  // Extract top-level info
  const turnCount = state
    ? (state.session_progress as Record<string, unknown> | undefined)?.turn_count
    : null;
  const responseStyle = state
    ? state.response_style
    : null;

  return (
    <>
      {/* Desktop top bar — wrapper controls breakpoint visibility */}
      <div className="oc-app-top-wrap">
      <header className="oc-app-top">
        <div className="flex items-center gap-3">
          <span className="oc-mobile-mark">
            <CouchLogo className="w-4 h-4" />
          </span>
          <h2 className="oc-app-top-title">State</h2>
          {turnCount != null && (
            <span className="text-[12px] font-mono text-oc-text-dim">
              turn {String(turnCount)}
            </span>
          )}
          {responseStyle != null && String(responseStyle) !== "pending" && (
            <span className="text-[11px] font-mono font-medium px-2 py-0.5 rounded-md border bg-oc-teal-50 text-oc-teal-700 border-oc-teal-200/60">
              {String(responseStyle)}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2.5">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="w-3.5 h-3.5 accent-oc-teal-600"
            />
            <span className="text-[12px] font-mono text-oc-text-muted">auto</span>
          </label>
          <button
            onClick={fetchState}
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
          <h2 className="oc-mobile-top-title">State</h2>
        </div>
        <button
          onClick={fetchState}
          disabled={loading}
          className="oc-tab-chip"
          style={{ padding: "4px 8px", fontSize: 9.5 }}
        >
          {loading ? "loading…" : "refresh"}
        </button>
      </header>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        <div className="px-6 pt-4">
          <div className="px-4 py-3 rounded-xl border border-oc-border bg-oc-warm-50 text-[13px] text-oc-text-secondary">
            <span className="font-semibold text-oc-text-primary">Developer/debug view.</span>{" "}
            This page shows raw runtime state from <code>/api/threads/&lbrace;thread_id&rbrace;/state</code>,
            including transcript, safety, memory, routing, and diagnostics fields. Use the typed chat,
            history, memory, and voice APIs for product workflows.
          </div>
        </div>

        {error && !state && (
          <div className="px-6 py-5">
            <div className="px-4 py-3 bg-oc-warm-100 border border-oc-border rounded-xl text-[15px] text-oc-text-secondary">
              {error}
            </div>
          </div>
        )}

        {state && (
          <div className="animate-fadeIn">
            {/* Thread meta bar */}
            <div className="px-6 py-3 bg-oc-bg-card border-b border-oc-border flex items-center gap-4 text-[12px] font-mono text-oc-text-muted">
              <span>thread: <span className="text-oc-text-secondary">{threadId}</span></span>
              <span>·</span>
              <span>channel: <span className="text-oc-text-secondary">{String(state.channel ?? "—")}</span></span>
              <span>·</span>
              <span>user: <span className="text-oc-text-secondary">{String(state.user_id ?? "—")}</span></span>
              <span>·</span>
              <span>keys: <span className="text-oc-text-secondary">{Object.keys(state).length}</span></span>
            </div>

            {/* State sections */}
            <div className="px-6 py-4 space-y-2">
              {STATE_SECTIONS.map(({ key, label, desc, icon }) => {
                const value = state[key];
                if (value === undefined) return null;
                return (
                  <StateSection
                    key={key}
                    sectionKey={key}
                    label={label}
                    desc={desc}
                    icon={icon}
                    value={value}
                  />
                );
              })}

              {/* Remaining keys not in the predefined sections */}
              {Object.keys(state)
                .filter(
                  (k) =>
                    !STATE_SECTIONS.some((s) => s.key === k) &&
                    ![
                      "message",
                      "channel",
                      "user_id",
                      "session_id",
                      "installed_skills",
                      "should_persist_memory",
                      "crisis_audit",
                    ].includes(k)
                )
                .map((key) => (
                  <StateSection
                    key={key}
                    sectionKey={key}
                    label={key}
                    desc=""
                    icon="·"
                    value={state[key]}
                  />
                ))}
            </div>

            {/* Raw JSON toggle */}
            <RawJsonSection state={state} />
          </div>
        )}
      </div>

    </>
  );
}


/* ── Collapsible State Section ── */

function StateSection({
  sectionKey,
  label,
  desc,
  icon,
  value,
}: {
  sectionKey: string;
  label: string;
  desc: string;
  icon: string;
  value: unknown;
}) {
  const [expanded, setExpanded] = useState(false);

  // Summary line for collapsed state
  const summary = getSummary(sectionKey, value);

  return (
    <div className="border border-oc-border rounded-xl overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-4 py-3 flex items-center gap-3 hover:bg-oc-warm-50 transition-colors text-left group"
      >
        <span className="text-oc-teal-500 font-mono text-base w-6 text-center">{icon}</span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-display text-[15px] text-oc-text">{label}</span>
            {desc && (
              <span className="text-[12px] text-oc-text-dim hidden sm:inline">{desc}</span>
            )}
          </div>
          {summary && !expanded && (
            <p className="text-[13px] font-mono text-oc-text-muted mt-0.5 truncate">
              {summary}
            </p>
          )}
        </div>
        <span className="text-[12px] text-oc-text-dim group-hover:text-oc-text-muted transition-colors">
          {expanded ? "▾" : "▸"}
        </span>
      </button>

      {expanded && (
        <div className="border-t border-oc-border bg-oc-bg-state px-4 py-3 overflow-x-auto animate-slideUp">
          <div className="state-tree">
            <JsonValue value={value} depth={0} />
          </div>
        </div>
      )}
    </div>
  );
}


/* ── JSON Renderer with syntax highlighting ── */

function JsonValue({ value, depth }: { value: unknown; depth: number }) {
  if (value === null || value === undefined) {
    return <span className="null">{value === null ? "null" : "undefined"}</span>;
  }

  if (typeof value === "boolean") {
    return <span className="boolean">{value ? "true" : "false"}</span>;
  }

  if (typeof value === "number") {
    return <span className="number">{value}</span>;
  }

  if (typeof value === "string") {
    // Truncate very long strings
    const display = value.length > 200 ? value.slice(0, 200) + "…" : value;
    return <span className="string">&quot;{display}&quot;</span>;
  }

  if (Array.isArray(value)) {
    return <JsonArray items={value} depth={depth} />;
  }

  if (typeof value === "object") {
    return <JsonObject obj={value as Record<string, unknown>} depth={depth} />;
  }

  return <span>{String(value)}</span>;
}

function JsonArray({ items, depth }: { items: unknown[]; depth: number }) {
  const [collapsed, setCollapsed] = useState(depth > 1 && items.length > 3);

  if (items.length === 0) {
    return <span className="bracket">[]</span>;
  }

  if (collapsed) {
    return (
      <span>
        <span className="toggle" onClick={() => setCollapsed(false)}>▸</span>
        <span className="bracket">[</span>
        <span className="null"> {items.length} items </span>
        <span className="bracket">]</span>
      </span>
    );
  }

  return (
    <span>
      <span className="toggle" onClick={() => setCollapsed(true)}>▾</span>
      <span className="bracket">[</span>
      <div style={{ paddingLeft: 16 }}>
        {items.map((item, i) => (
          <div key={i}>
            <JsonValue value={item} depth={depth + 1} />
            {i < items.length - 1 && <span className="bracket">,</span>}
          </div>
        ))}
      </div>
      <span className="bracket">]</span>
    </span>
  );
}

function JsonObject({ obj, depth }: { obj: Record<string, unknown>; depth: number }) {
  const keys = Object.keys(obj);
  const [collapsed, setCollapsed] = useState(depth > 1 && keys.length > 5);

  if (keys.length === 0) {
    return <span className="bracket">{"{}"}</span>;
  }

  if (collapsed) {
    return (
      <span>
        <span className="toggle" onClick={() => setCollapsed(false)}>▸</span>
        <span className="bracket">{"{"}</span>
        <span className="null"> {keys.length} keys </span>
        <span className="bracket">{"}"}</span>
      </span>
    );
  }

  return (
    <span>
      {depth > 0 && (
        <span className="toggle" onClick={() => setCollapsed(true)}>▾</span>
      )}
      <span className="bracket">{"{"}</span>
      <div style={{ paddingLeft: 16 }}>
        {keys.map((key, i) => (
          <div key={key}>
            <span className="key">&quot;{key}&quot;</span>
            <span className="bracket">: </span>
            <JsonValue value={obj[key]} depth={depth + 1} />
            {i < keys.length - 1 && <span className="bracket">,</span>}
          </div>
        ))}
      </div>
      <span className="bracket">{"}"}</span>
    </span>
  );
}


/* ── Summary helpers ── */

function getSummary(key: string, value: unknown): string {
  if (value == null) return "";
  const obj = value as Record<string, unknown>;

  switch (key) {
    case "route":
    case "response_style":
    case "therapeutic_approach":
    case "resource_lookup_status":
    case "inferred_location":
      return String(value || "—");
    case "grounded_lookup":
      return `status=${String(obj.status ?? "—")}`;
    case "response_text":
      return String(value || "").slice(0, 80);
    case "crisis": {
      const level = obj.level ?? obj.level;
      return `level=${String(level ?? 0)} conf=${String(obj.confidence ?? "—")} crisis=${String(obj.needs_crisis_response ?? false)}`;
    }
    case "session_memory":
      return `goal=${String(obj.current_goal ?? "none")} concerns=${Array.isArray(obj.active_concerns) ? obj.active_concerns.length : 0} loops=${Array.isArray(obj.open_loops) ? obj.open_loops.length : 0}`;
    case "procedural_profile":
      return `rules=${Array.isArray(obj.procedural_rules) ? obj.procedural_rules.length : 0} recall=${String(obj.proactive_recall_enabled ?? false)}`;
    case "session_progress":
      return `turn=${String(obj.turn_count ?? "—")}`;
    case "exercise_state":
      return `type=${String(obj.exercise_type ?? "none")} step=${String(obj.exercise_step ?? "—")} approach=${String(obj.exercise_therapeutic_approach ?? "—")}`;
    case "memory_control": {
      const pending = obj.pending_action;
      return pending ? "pending confirmation" : "no pending action";
    }
    case "diagnostics": {
      const total = obj.turn_total_ms;
      return total != null ? `total=${Number(total).toFixed(0)}ms` : "empty";
    }
    case "transcript":
    case "history":
      return Array.isArray(value) ? `${value.length} messages` : "";
    case "working_memory":
      return Array.isArray(value) ? `${value.length} entries` : "";
    default:
      if (typeof value === "object" && !Array.isArray(value)) {
        return `${Object.keys(value as object).length} keys`;
      }
      if (Array.isArray(value)) return `${value.length} items`;
      return String(value).slice(0, 80);
  }
}


/* ── Raw JSON toggle ── */

function RawJsonSection({ state }: { state: Record<string, unknown> }) {
  const [show, setShow] = useState(false);

  return (
    <div className="px-6 py-4 border-t border-oc-border">
      <button
        onClick={() => setShow(!show)}
        className="text-[13px] font-mono text-oc-text-muted hover:text-oc-text-secondary transition-colors"
      >
        {show ? "▾ hide raw JSON" : "▸ show raw JSON"}
      </button>
      {show && (
        <pre className="mt-3 bg-oc-bg-state text-oc-warm-400 rounded-xl p-5 text-[12px] font-mono overflow-auto max-h-96 leading-relaxed border border-oc-warm-800/50">
          {JSON.stringify(state, null, 2)}
        </pre>
      )}
    </div>
  );
}
