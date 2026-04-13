"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useRef, useState, useCallback } from "react";
import { useSessionStore } from "@/lib/session";

const NAV_ITEMS = [
  {
    label: "Text Chat",
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
        <circle cx="12" cy="12" r="10" />
        <path d="M12 16v-4M12 8h.01" />
      </svg>
    ),
  },
];

const MIN_WIDTH = 200;
const MAX_WIDTH = 400;
const DEFAULT_WIDTH = 240;

export function Sidebar() {
  const pathname = usePathname();
  const { userId, threadId, setUserId, setThreadId, newSession } = useSessionStore();
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
      className="relative border-r border-oc-border bg-oc-bg-card flex flex-col shrink-0"
      style={{ width }}
    >
      {/* Logo */}
      <div className="px-4 py-3.5 border-b border-oc-border">
        <Link href="/" className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-md bg-oc-teal-600 flex items-center justify-center text-white font-bold text-[10px]">
            OC
          </div>
          <span className="font-semibold text-[13px] text-oc-teal-800">OpenCouch</span>
        </Link>
      </div>

      {/* Session config */}
      <div className="px-3 py-2.5 border-b border-oc-border space-y-1.5">
        <div>
          <label className="text-[9px] font-medium uppercase tracking-wider text-oc-text-muted block mb-0.5">
            User ID
          </label>
          <input
            type="text"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            className="w-full px-2 py-1 text-[11px] bg-oc-bg border border-oc-border rounded focus:outline-none focus:border-oc-border-strong transition-colors"
          />
        </div>
        <div>
          <label className="text-[9px] font-medium uppercase tracking-wider text-oc-text-muted block mb-0.5">
            Thread ID
          </label>
          <input
            type="text"
            value={threadId}
            onChange={(e) => setThreadId(e.target.value)}
            className="w-full px-2 py-1 text-[11px] bg-oc-bg border border-oc-border rounded focus:outline-none focus:border-oc-border-strong transition-colors"
          />
        </div>
        <button
          onClick={newSession}
          className="w-full text-[10px] font-medium text-oc-teal-600 hover:text-oc-teal-500 py-1 border border-oc-border rounded hover:bg-oc-teal-50 transition-colors"
        >
          + New Session
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-2 space-y-0.5">
        {NAV_ITEMS.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-2.5 px-2.5 py-2 rounded-md text-[12px] transition-colors ${
                isActive
                  ? "bg-oc-teal-50 text-oc-teal-700 font-medium"
                  : "text-oc-text-secondary hover:text-oc-text hover:bg-oc-warm-50"
              }`}
            >
              {item.icon}
              {item.label}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="px-3 py-2.5 border-t border-oc-border">
        <p className="text-[9px] text-oc-text-muted leading-relaxed">
          Development prototype · Not a therapist
        </p>
      </div>

      {/* Drag handle */}
      <div
        onMouseDown={handleMouseDown}
        className="absolute top-0 right-0 w-1 h-full cursor-col-resize hover:bg-oc-teal-300/30 active:bg-oc-teal-400/40 transition-colors"
      />
    </aside>
  );
}
