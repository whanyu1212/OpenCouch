"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { useSessionStore } from "@/lib/session";
import { useCommandActions } from "@/lib/command-actions";
import { CouchLogo } from "@/components/logo";

/**
 * ConversationShell — shared chrome for /chat and /voice.
 *
 * Replaces the resizable Sidebar on these two routes with a 64px icon
 * NavRail (desktop) and a 4-tab bottom bar (mobile). Session controls
 * collapse into a SessionPill popover so the conversation can take
 * the full canvas.
 *
 * Memory and State pages keep the existing wide Sidebar — switching
 * to the rail is governed by the route in app-shell.tsx.
 */

const NAV_ITEMS: Array<{ id: string; label: string; href: string }> = [
  { id: "chat", label: "Chat", href: "/" },
  { id: "voice", label: "Voice", href: "/voice" },
  { id: "memory", label: "Memory", href: "/memory" },
  { id: "state", label: "State", href: "/state" },
];

function activeIdForPath(pathname: string): string {
  if (pathname === "/" || pathname.startsWith("/chat")) return "chat";
  if (pathname.startsWith("/voice")) return "voice";
  if (pathname.startsWith("/memory")) return "memory";
  if (pathname.startsWith("/state")) return "state";
  return "chat";
}

const IconChat = ({ size = 16 }: { size?: number }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.6"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <path d="M21 12a8 8 0 0 1-11.5 7.2L4 21l1.8-5.5A8 8 0 1 1 21 12z" />
  </svg>
);

const IconMic = ({ size = 16 }: { size?: number }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.6"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <rect x="9" y="3" width="6" height="11" rx="3" />
    <path d="M5 11a7 7 0 0 0 14 0" />
    <path d="M12 18v3" />
  </svg>
);

const IconBrain = ({ size = 16 }: { size?: number }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.6"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <path d="M9 4a3 3 0 0 0-3 3 3 3 0 0 0-2 5 3 3 0 0 0 2 5 3 3 0 0 0 3 3" />
    <path d="M15 4a3 3 0 0 1 3 3 3 3 0 0 1 2 5 3 3 0 0 1-2 5 3 3 0 0 1-3 3" />
    <path d="M9 4v16M15 4v16" />
  </svg>
);

const IconState = ({ size = 16 }: { size?: number }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.6"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <path d="M4 6h16M4 12h16M4 18h10" />
  </svg>
);

const IconPlus = ({ size = 14 }: { size?: number }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.8"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <path d="M12 5v14M5 12h14" />
  </svg>
);

const IconHistory = ({ size = 14 }: { size?: number }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.6"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <circle cx="12" cy="12" r="9" />
    <path d="M12 7v5l3 2" />
    <path d="M3 12a9 9 0 0 1 16-5" />
    <path d="M19 3v4h-4" />
  </svg>
);

const IconStop = ({ size = 14 }: { size?: number }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.6"
    aria-hidden="true"
  >
    <rect x="6" y="6" width="12" height="12" rx="2" />
  </svg>
);

const NAV_ICONS: Record<
  string,
  ({ size }: { size?: number }) => React.ReactElement
> = {
  chat: IconChat,
  voice: IconMic,
  memory: IconBrain,
  state: IconState,
};

export function NavRail() {
  const pathname = usePathname();
  const active = activeIdForPath(pathname);
  const { startNewSession, openThreadDrawer, endCurrentSession, isBusy, canEndSession } =
    useCommandActions();
  const voiceConnected = useSessionStore((s) => s.voiceConnected);
  const userId = useSessionStore((s) => s.userId);
  const sessionMode = useSessionStore((s) => s.sessionMode);

  const avatarLetter = (userId || "·").trim().charAt(0).toUpperCase() || "·";

  return (
    <aside className="oc-rail">
      <div className="oc-rail-brand" title="OpenCouch">
        <Link
          href="/"
          aria-label="OpenCouch home"
          className="inline-flex h-9 w-9 items-center justify-center rounded-[10px] bg-white border border-[var(--color-oc-line)] text-[var(--color-oc-primary)] shadow-sm"
        >
          <CouchLogo className="w-5 h-5" />
        </Link>
      </div>
      <nav className="oc-rail-nav" aria-label="Primary navigation">
        {NAV_ITEMS.map((item) => {
          const Icon = NAV_ICONS[item.id];
          const isActive = active === item.id;
          const showVoiceLive =
            item.id === "voice" && voiceConnected && !isActive;
          return (
            <Link
              key={item.id}
              href={item.href}
              className={`oc-rail-item ${isActive ? "is-active" : ""}`}
              aria-current={isActive ? "page" : undefined}
              title={item.label}
            >
              <span style={{ position: "relative" }}>
                <Icon size={18} />
                {showVoiceLive && (
                  <span
                    aria-hidden="true"
                    style={{
                      position: "absolute",
                      top: -2,
                      right: -4,
                      width: 6,
                      height: 6,
                      borderRadius: "50%",
                      background: "var(--color-oc-pulse)",
                      boxShadow: "0 0 0 2px var(--color-oc-bg-sidebar)",
                    }}
                  />
                )}
              </span>
              <span className="oc-rail-label">{item.label}</span>
            </Link>
          );
        })}
        <div
          aria-hidden="true"
          style={{
            width: 32,
            height: 1,
            background: "var(--color-oc-line-2)",
            margin: "10px auto",
          }}
        />
        <button
          type="button"
          className="oc-rail-item"
          title="New session"
          onClick={() => void startNewSession()}
          disabled={isBusy}
        >
          <IconPlus size={18} />
          <span className="oc-rail-label">New</span>
        </button>
        {sessionMode === "persistent" && (
          <button
            type="button"
            className="oc-rail-item"
            title="Previous sessions"
            onClick={openThreadDrawer}
          >
            <IconHistory size={18} />
            <span className="oc-rail-label">Past</span>
          </button>
        )}
        {sessionMode === "persistent" && (
          <button
            type="button"
            className="oc-rail-item"
            title="End session"
            onClick={() => void endCurrentSession()}
            disabled={!canEndSession}
          >
            <IconStop size={18} />
            <span className="oc-rail-label">End</span>
          </button>
        )}
      </nav>
      <div className="oc-rail-foot">
        <div className="oc-rail-avatar" title={userId || "no user"}>
          {avatarLetter}
        </div>
      </div>
    </aside>
  );
}

export function MobileTabBar() {
  const pathname = usePathname();
  const active = activeIdForPath(pathname);
  return (
    <div className="oc-tabbar" aria-label="Primary navigation">
      {NAV_ITEMS.map((item) => {
        const Icon = NAV_ICONS[item.id];
        const isActive = active === item.id;
        return (
          <Link
            key={item.id}
            href={item.href}
            className={`oc-tab ${isActive ? "is-active" : ""}`}
            aria-current={isActive ? "page" : undefined}
          >
            <span className="oc-tab-icon">
              <Icon size={20} />
            </span>
            <span className="oc-tab-label">{item.label}</span>
          </Link>
        );
      })}
    </div>
  );
}

export function SessionPill() {
  const userId = useSessionStore((s) => s.userId);
  const threadId = useSessionStore((s) => s.threadId);
  const sessionMode = useSessionStore((s) => s.sessionMode);
  const responseModelTier = useSessionStore((s) => s.responseModelTier);
  const {
    canEndSession,
    endCurrentSession,
    endingSession,
    openThreadDrawer,
    setResponseTier,
    showTextResponseTier,
    startNewSession,
    isBusy,
  } = useCommandActions();

  const [open, setOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handleClick = (event: MouseEvent) => {
      if (!wrapperRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", handleClick);
    document.addEventListener("keydown", handleKey);
    return () => {
      document.removeEventListener("mousedown", handleClick);
      document.removeEventListener("keydown", handleKey);
    };
  }, [open]);

  const isIncognito = sessionMode === "incognito";

  const close = useCallback(() => setOpen(false), []);

  return (
    <div
      ref={wrapperRef}
      className={`oc-session-pill-wrap${open ? " is-open" : ""}`}
    >
      <button
        type="button"
        className="oc-session-pill"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-haspopup="dialog"
      >
        <span
          className={`oc-session-dot${isIncognito ? " oc-session-dot--incognito" : ""}`}
        />
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            letterSpacing: "0.04em",
            color: "var(--color-oc-muted)",
          }}
        >
          {isIncognito ? "incognito" : "persistent"}
        </span>
        <span
          style={{
            width: 1,
            height: 12,
            background: "var(--color-oc-line)",
          }}
        />
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 11.5,
            color: "var(--color-oc-ink-2)",
          }}
        >
          {isIncognito ? "anon" : userId || "—"}
        </span>
        <span style={{ color: "var(--color-oc-text-dim)" }}>/</span>
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 11.5,
            color: "var(--color-oc-ink-2)",
            maxWidth: 140,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {prettyThreadName(threadId)}
        </span>
      </button>
      {open && (
        <div className="oc-popover" role="dialog" aria-label="Session controls">
          <div className="oc-popover-section">
            <div className="oc-popover-eyebrow">Session</div>
            <div className="oc-popover-row">
              <span className="label">mode</span>
              <span>{isIncognito ? "incognito" : "persistent"}</span>
            </div>
            <div className="oc-popover-row">
              <span className="label">user</span>
              <span>{isIncognito ? "anonymous" : userId || "—"}</span>
            </div>
            <div className="oc-popover-row">
              <span className="label">thread</span>
              <span
                style={{
                  maxWidth: 180,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {threadId}
              </span>
            </div>
          </div>

          {showTextResponseTier && (
            <div className="oc-popover-section">
              <div className="oc-popover-eyebrow">Response speed</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                <button
                  type="button"
                  className="oc-popover-action"
                  style={{
                    border: "1px solid",
                    borderColor:
                      responseModelTier === "fast"
                        ? "var(--color-oc-primary-soft)"
                        : "var(--color-oc-line)",
                    background:
                      responseModelTier === "fast"
                        ? "var(--color-oc-primary-tint)"
                        : "transparent",
                    justifyContent: "center",
                    fontSize: 12,
                  }}
                  onClick={() => {
                    setResponseTier("fast");
                  }}
                >
                  Fast
                </button>
                <button
                  type="button"
                  className="oc-popover-action"
                  style={{
                    border: "1px solid",
                    borderColor:
                      responseModelTier === "quality"
                        ? "var(--color-oc-primary-soft)"
                        : "var(--color-oc-line)",
                    background:
                      responseModelTier === "quality"
                        ? "var(--color-oc-primary-tint)"
                        : "transparent",
                    justifyContent: "center",
                    fontSize: 12,
                  }}
                  onClick={() => {
                    setResponseTier("quality");
                  }}
                >
                  Quality
                </button>
              </div>
            </div>
          )}

          <div className="oc-popover-section">
            <button
              type="button"
              className="oc-popover-action"
              onClick={() => {
                close();
                void startNewSession();
              }}
              disabled={isBusy}
            >
              <IconPlus size={14} />
              New session
            </button>
            {!isIncognito && (
              <button
                type="button"
                className="oc-popover-action"
                onClick={() => {
                  close();
                  openThreadDrawer();
                }}
              >
                <IconHistory size={14} />
                Previous sessions
              </button>
            )}
            {!isIncognito && (
              <button
                type="button"
                className="oc-popover-action oc-popover-action--danger"
                onClick={() => {
                  close();
                  void endCurrentSession();
                }}
                disabled={!canEndSession}
              >
                <IconStop size={14} />
                {endingSession ? "ending…" : "End session"}
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Wraps a route's content with the new chat/voice shell:
 * desktop nav rail on the left, mobile tab bar pinned to the bottom.
 *
 * The shell itself is a flex row on md+ and a flex column on mobile.
 * Pages render their own top bar + content + composer inside `children`.
 */
export function ConversationShell({
  children,
  withWash = false,
}: {
  children: ReactNode;
  withWash?: boolean;
}) {
  return (
    <div
      style={{
        display: "flex",
        height: "100vh",
        width: "100%",
        position: "relative",
      }}
    >
      {withWash && <div className="oc-wash" />}
      {/* Desktop rail — hidden on small screens via .oc-rail-wrap */}
      <div className="oc-rail-wrap" style={{ position: "relative", zIndex: 2 }}>
        <NavRail />
      </div>
      {/* Inner column: top bar + content + composer (pages provide); on
          mobile we add the bottom tab bar via children layout. */}
      <div
        style={{
          flex: 1,
          minWidth: 0,
          display: "flex",
          flexDirection: "column",
          position: "relative",
          zIndex: 1,
        }}
      >
        {children}
      </div>
    </div>
  );
}

function prettyThreadName(threadId: string): string {
  if (!threadId) return "—";
  if (threadId.length <= 24) return threadId;
  // Telegram-style raw IDs: telegram:dn:5376052137:session:abc — keep last 14 chars.
  if (threadId.includes(":")) {
    const parts = threadId.split(":");
    return parts[parts.length - 1].slice(-14);
  }
  return threadId.slice(0, 10) + "…" + threadId.slice(-8);
}
