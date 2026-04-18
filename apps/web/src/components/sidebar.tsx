"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useRef, useState, useCallback } from "react";
import { useSessionStore } from "@/lib/session";
import { CouchLogo } from "@/components/logo";

const NAV_ITEMS = [
  {
    label: "Chat",
    href: "/",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-[18px] h-[18px]">
        <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
      </svg>
    ),
  },
  {
    label: "Voice",
    href: "/voice",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-[18px] h-[18px]">
        <path d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z" />
        <path d="M19 10v2a7 7 0 01-14 0v-2" />
        <line x1="12" y1="19" x2="12" y2="23" />
      </svg>
    ),
  },
  {
    label: "Memory",
    href: "/memory",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-[18px] h-[18px]">
        <path d="M4 7h16M4 12h16M4 17h10" />
      </svg>
    ),
  },
  {
    label: "State",
    href: "/state",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-[18px] h-[18px]">
        <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
        <polyline points="14 2 14 8 20 8" />
        <line x1="9" y1="13" x2="15" y2="13" />
        <line x1="9" y1="17" x2="13" y2="17" />
      </svg>
    ),
  },
];

const MIN_WIDTH = 220;
const MAX_WIDTH = 420;
const DEFAULT_WIDTH = 260;

export function Sidebar() {
  const pathname = usePathname();
  const { userId, threadId, sessionMode, setUserId, setThreadId, newSession, chatLoading, voiceConnected } = useSessionStore();
  const isIncognito = sessionMode === "incognito";
  const [width, setWidth] = useState(DEFAULT_WIDTH);
  const isResizing = useRef(false);

  const handleMouseDown = useCallback(() => {
    isResizing.current = true;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";

    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing.current) return;
      const newWidth = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, e.clientX));
      setWidth(newWidth);
    };

    const handleMouseUp = () => {
      isResizing.current = false;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
  }, []);

  return (
    <aside
      className="relative border-r border-oc-border bg-oc-bg-sidebar flex flex-col shrink-0 oc-surface-noise"
      style={{ width }}
    >
      {/* Logo */}
      <div className="relative z-10 px-4 pt-5 pb-4">
        <Link href="/" className="flex items-center gap-3 group">
          <div className="w-9 h-9 rounded-lg bg-oc-warm-100 border border-oc-border flex items-center justify-center shadow-sm group-hover:bg-oc-warm-50 transition-colors">
            <CouchLogo className="w-6 h-6" />
          </div>
          <div>
            <span className="font-display text-[17px] text-oc-teal-900 block leading-none">
              OpenCouch
            </span>
            <span className="text-[11px] text-oc-text-muted font-mono tracking-wide uppercase">
              v0.8 · lab
            </span>
          </div>
        </Link>
      </div>

      {/* Session config */}
      <div className="relative z-10 mx-3 px-3 py-3 border border-oc-border rounded-lg bg-oc-bg/60 backdrop-blur-sm space-y-2.5">
        {/* Mode indicator — clickable to go back to mode picker */}
        <button
          onClick={newSession}
          className={`w-full flex items-center justify-between px-2.5 py-2 rounded-md border transition-all cursor-pointer ${
            isIncognito
              ? "bg-oc-warm-100 border-oc-warm-200 hover:bg-oc-warm-200/70"
              : "bg-oc-teal-50 border-oc-teal-200/60 hover:bg-oc-teal-100/60"
          }`}
        >
          <div className="flex items-center gap-2">
            {isIncognito ? (
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3.5 h-3.5 text-oc-warm-600">
                <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                <circle cx="12" cy="12" r="3" />
                <line x1="1" y1="1" x2="23" y2="23" />
              </svg>
            ) : (
              <div className="w-2 h-2 rounded-full bg-oc-green" />
            )}
            <span className={`text-[11px] font-mono font-medium uppercase tracking-widest ${
              isIncognito ? "text-oc-warm-600" : "text-oc-teal-700"
            }`}>
              {isIncognito ? "incognito" : "persistent"}
            </span>
          </div>
          <span className={`text-[11px] font-mono flex items-center gap-1 ${
            isIncognito ? "text-oc-warm-400" : "text-oc-teal-400"
          }`}>
            switch
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-3 h-3">
              <polyline points="9 18 15 12 9 6" />
            </svg>
          </span>
        </button>

        <div>
          <label className="text-[11px] font-mono font-medium uppercase tracking-widest text-oc-text-muted block mb-1">
            user
          </label>
          <input
            type="text"
            value={isIncognito ? "(anonymous)" : userId}
            onChange={(e) => setUserId(e.target.value)}
            disabled={isIncognito}
            placeholder="e.g. alice"
            className="w-full px-2.5 py-2 text-[13px] font-mono bg-oc-bg-input border border-oc-border rounded-md focus:outline-none focus:border-oc-teal-400 focus:ring-1 focus:ring-oc-accent-subtle transition-all disabled:opacity-50 disabled:bg-oc-warm-50 placeholder:text-oc-text-dim/60"
          />
        </div>
        <div>
          <label className="text-[11px] font-mono font-medium uppercase tracking-widest text-oc-text-muted block mb-1">
            thread
          </label>
          <input
            type="text"
            value={threadId}
            onChange={(e) => setThreadId(e.target.value)}
            disabled={isIncognito}
            placeholder="e.g. session-1"
            className="w-full px-2.5 py-2 text-[13px] font-mono bg-oc-bg-input border border-oc-border rounded-md focus:outline-none focus:border-oc-teal-400 focus:ring-1 focus:ring-oc-accent-subtle transition-all disabled:opacity-50 disabled:bg-oc-warm-50 placeholder:text-oc-text-dim/60"
          />
        </div>
        <button
          onClick={newSession}
          className="w-full text-[12px] font-medium text-oc-teal-700 hover:text-oc-teal-600 py-2 border border-oc-border-subtle rounded-md hover:bg-oc-teal-50 hover:border-oc-teal-200 transition-all"
        >
          + New Session
        </button>
      </div>

      {/* Navigation */}
      <nav className="relative z-10 flex-1 px-3 pt-5 space-y-1">
        {chatLoading && (
          <div className="flex items-center gap-2 px-2.5 py-2 mb-2 rounded-lg bg-oc-cta-subtle border border-oc-cta/15 animate-fadeIn">
            <span className="relative flex h-2 w-2 shrink-0">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-oc-cta opacity-50" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-oc-cta" />
            </span>
            <span className="text-[11px] font-mono text-oc-cta leading-tight">
              Response in progress — stay on Chat until it completes
            </span>
          </div>
        )}
        <p className="text-[11px] font-mono font-medium uppercase tracking-widest text-oc-text-dim px-2.5 mb-2">
          Navigate
        </p>
        {NAV_ITEMS.map((item) => {
          const isActive = pathname === item.href;
          const showVoiceLive = item.href === "/voice" && voiceConnected && !isActive;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-[14px] transition-all ${
                isActive
                  ? "bg-oc-teal-50 text-oc-teal-800 font-medium border border-oc-teal-100 shadow-sm"
                  : "text-oc-text-secondary hover:text-oc-text hover:bg-oc-warm-100/60 border border-transparent"
              }`}
            >
              <span className={isActive ? "text-oc-teal-600" : "text-oc-text-muted"}>
                {item.icon}
              </span>
              {item.label}
              {showVoiceLive && (
                <span className="ml-auto flex items-center gap-1.5">
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-oc-green opacity-50" />
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-oc-green" />
                  </span>
                  <span className="text-[11px] font-mono text-oc-green">live</span>
                </span>
              )}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="relative z-10 px-3 py-3 border-t border-oc-border-subtle">
        <p className="text-[11px] text-oc-text-dim font-mono leading-relaxed">
          prototype · not a therapist
        </p>
      </div>

      {/* Drag handle */}
      <div
        onMouseDown={handleMouseDown}
        className="absolute top-0 right-0 w-1 h-full cursor-col-resize hover:bg-oc-teal-300/30 active:bg-oc-teal-400/40 transition-colors z-20"
      />
    </aside>
  );
}
