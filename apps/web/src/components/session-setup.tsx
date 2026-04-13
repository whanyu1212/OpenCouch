"use client";

import { useState } from "react";
import { useSessionStore, type SessionMode } from "@/lib/session";
import { CouchLogo } from "@/components/logo";

/**
 * Session setup screen shown before the user starts interacting.
 *
 * Two mode cards:
 * - Persistent — loads memory, continues existing threads
 * - Incognito — fresh thread, no memory loaded, nothing saved
 */
export function SessionSetup() {
  const { startSession } = useSessionStore();
  const [selectedMode, setSelectedMode] = useState<SessionMode>("persistent");
  const [editUserId, setEditUserId] = useState("");
  const [editThreadId, setEditThreadId] = useState("");

  const handleStart = () => {
    startSession(selectedMode, editUserId, editThreadId);
  };

  return (
    <div className="flex-1 flex items-center justify-center p-8 animate-fadeIn">
      <div className="w-full max-w-lg">
        {/* Header */}
        <div className="text-center mb-10">
          <div className="w-16 h-16 rounded-2xl bg-oc-warm-100 border border-oc-border flex items-center justify-center mx-auto mb-5 shadow-sm">
            <CouchLogo className="w-10 h-10" />
          </div>
          <h1 className="font-display text-3xl text-oc-teal-900 mb-2">
            OpenCouch
          </h1>
          <p className="text-[15px] text-oc-text-muted">
            Choose how you&apos;d like this session to work.
          </p>
        </div>

        {/* Mode cards */}
        <div className="space-y-3 mb-8">
          <ModeCard
            mode="persistent"
            selected={selectedMode === "persistent"}
            onSelect={() => setSelectedMode("persistent")}
            title="Persistent"
            description="Loads memory and chat history. The agent remembers your previous sessions and builds on what it knows about you."
            icon={
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-6 h-6">
                <path d="M4 7h16M4 12h16M4 17h10" />
              </svg>
            }
          />
          <ModeCard
            mode="incognito"
            selected={selectedMode === "incognito"}
            onSelect={() => setSelectedMode("incognito")}
            title="Incognito"
            description="Fresh start with no memory. Nothing from this session is saved or carried over. A new thread ID is generated automatically."
            icon={
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-6 h-6">
                <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                <circle cx="12" cy="12" r="3" />
                <line x1="1" y1="1" x2="23" y2="23" />
              </svg>
            }
          />
        </div>

        {/* Session config — only shown in persistent mode */}
        {selectedMode === "persistent" && (
          <div className="border border-oc-border rounded-xl p-5 space-y-4 mb-8 bg-oc-bg-card animate-fadeIn">
            <div>
              <label className="text-[11px] font-mono font-medium uppercase tracking-widest text-oc-text-muted block mb-1.5">
                user id
              </label>
              <input
                type="text"
                value={editUserId}
                onChange={(e) => setEditUserId(e.target.value)}
                className="w-full px-3.5 py-2.5 text-[14px] font-mono bg-oc-bg-input border border-oc-border rounded-lg focus:outline-none focus:border-oc-teal-400 focus:ring-1 focus:ring-oc-accent-subtle transition-all placeholder:text-oc-text-dim/60"
                placeholder="e.g. alice, dev-user-01, hanyu"
              />
              <p className="text-[12px] text-oc-text-dim mt-1.5 leading-relaxed">
                Memory is scoped to this ID. Same ID across threads shares memory.
              </p>
            </div>
            <div>
              <label className="text-[11px] font-mono font-medium uppercase tracking-widest text-oc-text-muted block mb-1.5">
                thread id
              </label>
              <input
                type="text"
                value={editThreadId}
                onChange={(e) => setEditThreadId(e.target.value)}
                className="w-full px-3.5 py-2.5 text-[14px] font-mono bg-oc-bg-input border border-oc-border rounded-lg focus:outline-none focus:border-oc-teal-400 focus:ring-1 focus:ring-oc-accent-subtle transition-all placeholder:text-oc-text-dim/60"
                placeholder="e.g. session-1, friday-checkin, grief-work"
              />
              <p className="text-[12px] text-oc-text-dim mt-1.5 leading-relaxed">
                Reuse an ID to continue a conversation, or pick a new one to start fresh.
              </p>
            </div>
          </div>
        )}

        {/* Incognito note */}
        {selectedMode === "incognito" && (
          <div className="border border-oc-border-subtle rounded-xl p-5 mb-8 bg-oc-warm-50 animate-fadeIn">
            <p className="text-[14px] text-oc-text-secondary leading-relaxed">
              A random thread ID will be generated. No user ID is sent, so the memory store has nothing to retrieve. The crisis safety log still operates (it&apos;s always-on and anonymous).
            </p>
          </div>
        )}

        {/* Start button */}
        <button
          onClick={handleStart}
          className="w-full bg-oc-teal-700 text-white py-3.5 rounded-xl text-base font-medium hover:bg-oc-teal-600 transition-all shadow-sm"
        >
          Start Session
        </button>

        <p className="text-center text-[11px] text-oc-text-dim mt-5 font-mono">
          development prototype · not a therapist
        </p>
      </div>
    </div>
  );
}


function ModeCard({
  mode,
  selected,
  onSelect,
  title,
  description,
  icon,
}: {
  mode: SessionMode;
  selected: boolean;
  onSelect: () => void;
  title: string;
  description: string;
  icon: React.ReactNode;
}) {
  return (
    <button
      onClick={onSelect}
      className={`w-full text-left p-5 rounded-xl border-2 transition-all ${
        selected
          ? "border-oc-teal-500 bg-oc-teal-50/50 shadow-sm"
          : "border-oc-border hover:border-oc-border-strong bg-oc-bg"
      }`}
    >
      <div className="flex items-start gap-4">
        <div className={`w-11 h-11 rounded-lg flex items-center justify-center shrink-0 ${
          selected ? "bg-oc-teal-100 text-oc-teal-700" : "bg-oc-warm-100 text-oc-warm-600"
        }`}>
          {icon}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2.5">
            <span className={`font-display text-lg ${selected ? "text-oc-teal-800" : "text-oc-text"}`}>
              {title}
            </span>
            {mode === "incognito" && (
              <span className="text-[10px] font-mono uppercase tracking-widest text-oc-text-dim border border-oc-border px-2 py-0.5 rounded">
                no memory
              </span>
            )}
          </div>
          <p className="text-[13px] text-oc-text-muted leading-relaxed mt-1">
            {description}
          </p>
        </div>
        {/* Radio indicator */}
        <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center shrink-0 mt-1 ${
          selected ? "border-oc-teal-500" : "border-oc-warm-300"
        }`}>
          {selected && <div className="w-2.5 h-2.5 rounded-full bg-oc-teal-500" />}
        </div>
      </div>
    </button>
  );
}
