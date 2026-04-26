"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  useCommandActions,
  type CommandAction,
} from "@/lib/command-actions";

const GROUP_ORDER: CommandAction["group"][] = [
  "Session",
  "Memory",
  "Navigation",
  "Preferences",
];

export function CommandPalette() {
  const {
    actions,
    commandPaletteOpen,
    closeCommandPalette,
    openCommandPalette,
    runAction,
  } = useCommandActions();
  const [query, setQuery] = useState("");
  const handleClose = useCallback(() => {
    setQuery("");
    closeCommandPalette();
  }, [closeCommandPalette]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const key = event.key.toLowerCase();
      if ((event.metaKey || event.ctrlKey) && key === "k") {
        event.preventDefault();
        if (commandPaletteOpen) {
          handleClose();
        } else {
          openCommandPalette();
        }
      }

      if (event.key === "Escape" && commandPaletteOpen) {
        handleClose();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [commandPaletteOpen, handleClose, openCommandPalette]);

  const visibleActions = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return actions
      .filter((action) => action.id !== "show_help")
      .filter((action) => {
        if (!needle) return true;
        return [action.label, action.description, action.group]
          .join(" ")
          .toLowerCase()
          .includes(needle);
      });
  }, [actions, query]);

  const groupedActions = GROUP_ORDER.map((group) => ({
    group,
    actions: visibleActions.filter((action) => action.group === group),
  })).filter((entry) => entry.actions.length > 0);

  if (!commandPaletteOpen) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="OpenCouch actions"
      className="fixed inset-0 z-50 bg-oc-teal-950/25 backdrop-blur-sm flex items-start justify-center px-4 pt-[12vh] animate-fadeIn"
      onMouseDown={handleClose}
    >
      <div
        className="w-full max-w-xl rounded-2xl border border-oc-border-strong bg-oc-bg-card shadow-2xl overflow-hidden"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="px-4 py-3 border-b border-oc-border bg-oc-bg-input flex items-center gap-3">
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.7"
            className="w-4 h-4 text-oc-text-muted"
          >
            <circle cx="11" cy="11" r="7" />
            <path d="M20 20l-3.5-3.5" />
          </svg>
          <input
            autoFocus
            aria-label="Search actions"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search actions..."
            className="flex-1 bg-transparent text-[15px] text-oc-text outline-none placeholder:text-oc-text-dim"
          />
          <span className="text-[11px] font-mono text-oc-text-dim border border-oc-border rounded-md px-2 py-1">
            Esc
          </span>
        </div>

        <div className="max-h-[60vh] overflow-y-auto p-3">
          {groupedActions.length === 0 ? (
            <p className="px-3 py-6 text-center text-[13px] text-oc-text-muted">
              No actions match that search.
            </p>
          ) : (
            groupedActions.map((entry) => (
              <section key={entry.group} className="mb-3 last:mb-0">
                <p className="px-2 pb-1.5 text-[11px] font-mono uppercase tracking-widest text-oc-text-dim">
                  {entry.group}
                </p>
                <div className="space-y-1">
                  {entry.actions.map((action) => (
                    <button
                      key={action.id}
                      type="button"
                      disabled={action.disabled}
                      onClick={() => {
                        handleClose();
                        void runAction(action.id);
                      }}
                      className="w-full text-left rounded-xl px-3 py-2.5 border border-transparent hover:border-oc-teal-100 hover:bg-oc-teal-50 transition-all disabled:opacity-45 disabled:cursor-not-allowed disabled:hover:border-transparent disabled:hover:bg-transparent"
                    >
                      <span className="block text-[14px] font-medium text-oc-text">
                        {action.label}
                      </span>
                      <span className="block mt-0.5 text-[12px] leading-relaxed text-oc-text-muted">
                        {action.description}
                      </span>
                    </button>
                  ))}
                </div>
              </section>
            ))
          )}
        </div>

        <div className="px-4 py-2.5 border-t border-oc-border bg-oc-bg/70 flex justify-between text-[11px] font-mono text-oc-text-dim">
          <span>Use /help in chat to open this menu.</span>
          <span>Cmd/Ctrl K</span>
        </div>
      </div>
    </div>
  );
}
