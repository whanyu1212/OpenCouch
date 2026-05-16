"use client";

import { useEffect, useState, type ReactNode } from "react";
import { useSessionStore, type SessionMode } from "@/lib/session";
import { CouchLogo } from "@/components/logo";

/**
 * Session setup / landing screen.
 *
 * Responsive: mobile = single-column (Direction A); md+ = two-column
 * masthead + panel (Direction B). Same component tree, layout switches
 * on the `md:` breakpoint.
 */
export function SessionSetup() {
  const startSession = useSessionStore((s) => s.startSession);
  // Prefill from the last session's persisted values (mode, user, thread).
  // The user still has to click Start — we just save them re-typing.
  const persistedMode = useSessionStore((s) => s.sessionMode);
  const persistedUserId = useSessionStore((s) => s.userId);
  const persistedThreadId = useSessionStore((s) => s.threadId);
  const [mode, setMode] = useState<SessionMode>(persistedMode);
  const [userId, setUserId] = useState(persistedUserId);
  const [threadId, setThreadId] = useState(persistedThreadId);

  // Keep the form's thread input in sync with the store. When the user
  // returns home mid-session, newSession() generates a fresh
  // threadId in the store — without this effect, the form would still
  // hold the previous prefilled value because useState only captures the
  // initial value at mount.
  useEffect(() => {
    setThreadId(persistedThreadId);
  }, [persistedThreadId]);

  const startDisabled = mode === "persistent" && !userId.trim();
  const handleStart = () => {
    if (startDisabled) return;
    startSession(mode, userId, threadId);
  };

  return (
    <div className="flex min-h-screen w-full flex-1 flex-col bg-oc-bg text-oc-ink">
      <TopBar />

      <main className="flex flex-1 items-start justify-center px-5 pt-2 pb-10 md:items-center md:px-20 md:pt-0">
        <div className="grid w-full max-w-[1280px] grid-cols-1 gap-10 md:grid-cols-[1.05fr_1fr] md:gap-20">
          <Masthead mode={mode} />

          <Panel
            mode={mode}
            setMode={setMode}
            userId={userId}
            setUserId={setUserId}
            threadId={threadId}
            setThreadId={setThreadId}
            onStart={handleStart}
            startDisabled={startDisabled}
          />
        </div>
      </main>
    </div>
  );
}

/* ── Top chrome ─────────────────────────────────────────────────────── */

function TopBar() {
  return (
    <div className="flex items-center px-5 py-5 md:px-7 md:py-5">
      <div className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.04em] text-oc-ink-2">
        <span className="h-1.5 w-1.5 rounded-full bg-oc-primary" />
        opencouch
      </div>
    </div>
  );
}

/* ── Left masthead (B) / hero (A) ───────────────────────────────────── */

function Masthead({ mode }: { mode: SessionMode }) {
  return (
    <section className="flex flex-col items-center text-center md:items-start md:pb-10 md:text-left">
      {/* Mobile: stacked logomark above wordmark; Desktop: horizontal lockup */}
      <div className="mb-3.5 flex flex-col items-center gap-3.5 md:mb-7 md:flex-row md:items-center md:gap-3.5">
        <Logomark />
        <h1
          className="font-display font-semibold leading-none text-oc-ink"
          style={{ fontSize: "clamp(36px, 6vw, 56px)", letterSpacing: "-0.018em" }}
        >
          OpenCouch
        </h1>
      </div>

      <p
        className="mx-auto max-w-[360px] font-display text-[17px] leading-[1.55] text-oc-muted md:mx-0 md:max-w-[480px] md:text-[22px] md:leading-[1.45] md:text-oc-ink-2"
      >
        A quiet space to think out loud — voice or text, with optional memory
        that keeps what mattered last time.
      </p>

      {/* Body paragraph — desktop only, mobile reads it from the helper line below */}
      <p className="mt-4 hidden max-w-[440px] text-[14.5px] leading-[1.6] text-oc-muted md:block">
        Choose <strong className="font-medium text-oc-ink-2">Persistent</strong> to let the
        agent remember across sessions, or{" "}
        <strong className="font-medium text-oc-ink-2">Incognito</strong> for a fresh,
        ephemeral conversation.
      </p>

      {/* Capability strip — desktop only */}
      <div className="mt-7 hidden items-center gap-5 font-mono text-[11px] uppercase tracking-[0.08em] text-oc-text-muted md:flex">
        <span className="inline-flex items-center gap-2">
          <IconMic /> voice
        </span>
        <span className="opacity-40">·</span>
        <span>text</span>
        <span className="opacity-40">·</span>
        <span className="inline-flex items-center gap-2">
          <IconLock size={11} /> private
        </span>
      </div>

      <MemoryModelDiagram mode={mode} />

      {/* Aria-live region keeps the desktop body in sync with mode without visual change */}
      <span className="sr-only" aria-live="polite">
        {mode === "persistent"
          ? "Persistent mode selected"
          : "Incognito mode selected"}
      </span>
    </section>
  );
}

function MemoryModelDiagram({ mode }: { mode: SessionMode }) {
  const persistent = mode === "persistent";

  return (
    <div className="mt-10 hidden w-full max-w-[500px] text-left lg:block">
      <div className="flex items-center justify-between gap-4">
        <p className="font-mono text-[10.5px] uppercase tracking-[0.14em] text-oc-muted">
          Memory model
        </p>
        <span
          className={`rounded-full border px-2.5 py-1 font-mono text-[10px] uppercase tracking-[0.08em] ${
            persistent
              ? "border-[rgba(31,79,70,0.18)] bg-oc-primary-tint text-oc-primary"
              : "border-oc-line bg-oc-surface-tint text-oc-muted"
          }`}
        >
          {persistent ? "persistent" : "incognito"}
        </span>
      </div>

      <div className="mt-5 space-y-5 border-l border-oc-line pl-4">
        {persistent ? (
          <>
            <MemoryFlowRow
              first="User ID"
              firstDetail="who memory belongs to"
              second="Local memory"
              secondDetail="summaries + key facts"
              third="Future sessions"
              thirdDetail="context returns"
              active
            />
            <MemoryFlowRow
              first="Thread ID"
              firstDetail="conversation lane"
              second="Current session"
              secondDetail="voice or text"
              third="Session summary"
              thirdDetail="saved when useful"
            />
          </>
        ) : (
          <>
            <MemoryFlowRow
              first="Incognito"
              firstDetail="anonymous start"
              second="Current session"
              secondDetail="voice or text"
              third="Discarded"
              thirdDetail="nothing saved"
            />
            <div className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.08em] text-oc-muted">
              <span className="h-px flex-1 bg-oc-line" aria-hidden />
              Memory lookup and writes are disabled
              <span className="h-px flex-1 bg-oc-line" aria-hidden />
            </div>
          </>
        )}
      </div>

      <p className="mt-4 text-[12.5px] leading-[1.55] text-oc-muted">
        {persistent
          ? "Same user ID shares memory across threads. Thread ID picks which conversation to continue."
          : "Incognito creates a fresh temporary thread and skips saved memory."}
      </p>
    </div>
  );
}

function MemoryFlowRow({
  first,
  firstDetail,
  second,
  secondDetail,
  third,
  thirdDetail,
  active = false,
}: {
  first: string;
  firstDetail: string;
  second: string;
  secondDetail: string;
  third: string;
  thirdDetail: string;
  active?: boolean;
}) {
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_34px_minmax(0,1fr)_34px_minmax(0,1fr)] items-start gap-2">
      <MemoryNode title={first} detail={firstDetail} active={active} />
      <span className="mt-[13px] flex items-center text-oc-text-dim" aria-hidden>
        <span className="h-px flex-1 bg-oc-line" />
        <IconArrow size={12} />
      </span>
      <MemoryNode title={second} detail={secondDetail} active={active} />
      <span className="mt-[13px] flex items-center text-oc-text-dim" aria-hidden>
        <span className="h-px flex-1 bg-oc-line" />
        <IconArrow size={12} />
      </span>
      <MemoryNode title={third} detail={thirdDetail} active={active} />
    </div>
  );
}

function MemoryNode({
  title,
  detail,
  active,
}: {
  title: string;
  detail: string;
  active: boolean;
}) {
  return (
    <div className="min-w-0">
      <div className="flex items-center gap-2">
        <span
          className={`h-2 w-2 shrink-0 rounded-full ${
            active ? "bg-oc-primary" : "bg-oc-warm-300"
          }`}
        />
        <div className="font-display text-[14.5px] font-semibold leading-tight text-oc-ink">
          {title}
        </div>
      </div>
      <div className="mt-1.5 pl-4 text-[11.5px] leading-snug text-oc-muted">
        {detail}
      </div>
    </div>
  );
}

function Logomark() {
  return (
    <div
      className="inline-flex items-center justify-center rounded-[14px] border border-oc-line bg-white text-oc-primary"
      style={{
        width: 56,
        height: 56,
        boxShadow:
          "0 1px 0 rgba(255,255,255,0.6) inset, 0 14px 30px -22px rgba(31,79,70,0.45)",
      }}
    >
      <CouchLogo className="h-7 w-7" />
    </div>
  );
}

/* ── Right panel ────────────────────────────────────────────────────── */

interface PanelProps {
  mode: SessionMode;
  setMode: (m: SessionMode) => void;
  userId: string;
  setUserId: (s: string) => void;
  threadId: string;
  setThreadId: (s: string) => void;
  onStart: () => void;
  startDisabled: boolean;
}

function Panel({
  mode,
  setMode,
  userId,
  setUserId,
  threadId,
  setThreadId,
  onStart,
  startDisabled,
}: PanelProps) {
  return (
    <section className="mx-auto w-full max-w-[480px] rounded-none border-0 bg-transparent p-0 md:rounded-[22px] md:border md:border-oc-line md:bg-white md:p-[28px_30px] md:[box-shadow:0_1px_0_rgba(255,255,255,0.6)_inset,0_32px_64px_-36px_rgba(31,79,70,0.30)]">
      <p className="mb-3.5 hidden font-mono text-[10.5px] uppercase tracking-[0.14em] text-oc-muted md:block">
        How this session works
      </p>

      <div className="space-y-3">
        <ModeCard
          value="persistent"
          selected={mode === "persistent"}
          onSelect={setMode}
          icon={<IconLines />}
          title="Persistent"
          desc="Loads memory and chat history. The agent remembers your previous sessions and builds on what it knows about you."
        />
        <ModeCard
          value="incognito"
          selected={mode === "incognito"}
          onSelect={setMode}
          icon={<IconEyeOff />}
          title="Incognito"
          tag="no memory"
          desc="Fresh start with no memory. Nothing from this session is saved. A new thread ID is generated for you."
        />
      </div>

      {mode === "persistent" && (
        <div className="mt-5">
          <PrivacyNote />
        </div>
      )}

      <IdFields
        mode={mode}
        userId={userId}
        setUserId={setUserId}
        threadId={threadId}
        setThreadId={setThreadId}
      />

      <GettingStartedGuide />

      <p className="mt-6 text-[12.5px] leading-[1.55] text-oc-text-muted md:hidden">
        Choose <strong className="font-medium text-oc-ink-2">Persistent</strong> to let the
        agent remember across sessions, or{" "}
        <strong className="font-medium text-oc-ink-2">Incognito</strong> for a fresh,
        ephemeral conversation. Voice and text both supported.
      </p>

      <button
        type="button"
        onClick={onStart}
        disabled={startDisabled}
        className="mt-5 inline-flex w-full items-center justify-center gap-2.5 rounded-xl bg-oc-primary px-5 py-4 font-display text-[17px] font-medium text-[#F6F1E5] transition-[background,transform,box-shadow] hover:bg-oc-primary-2 active:translate-y-px disabled:cursor-not-allowed disabled:opacity-50"
        style={{
          boxShadow:
            "0 1px 0 rgba(23,62,55,0.4) inset, 0 8px 24px -12px rgba(23,62,55,0.5)",
          letterSpacing: "0.005em",
        }}
      >
        <span>Start session</span>
        <IconArrow />
      </button>
    </section>
  );
}

function GettingStartedGuide() {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!open) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [open]);

  return (
    <div className="mt-5 border-t border-oc-line-2 pt-5">
      <div className="flex justify-center">
        <button
          type="button"
          onClick={() => setOpen(true)}
          className="inline-flex items-center gap-2 rounded-lg border border-oc-line bg-oc-surface-tint px-3.5 py-2 font-mono text-[10.5px] uppercase tracking-[0.08em] text-oc-primary transition-[background,border-color] hover:border-oc-primary/40 hover:bg-white"
        >
          <IconHelp />
          Getting started
        </button>
      </div>

      {open && (
        <div
          className="fixed inset-0 z-[80] flex items-end justify-center bg-[rgba(21,32,29,0.34)] px-0 py-0 md:items-center md:px-6 md:py-8"
          onMouseDown={() => setOpen(false)}
        >
          <section
            role="dialog"
            aria-modal="true"
            aria-labelledby="getting-started-title"
            className="max-h-[88vh] w-full overflow-hidden rounded-t-[22px] border border-oc-line bg-white shadow-[0_28px_80px_-34px_rgba(21,32,29,0.58)] md:max-w-[700px] md:rounded-[22px]"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <div className="flex items-start justify-between gap-4 border-b border-oc-line-2 px-5 py-4 md:px-6 md:py-5">
              <div>
                <p className="font-mono text-[10.5px] uppercase tracking-[0.14em] text-oc-primary">
                  OpenCouch
                </p>
                <h2
                  id="getting-started-title"
                  className="mt-1 font-display text-[24px] font-semibold leading-tight text-oc-ink"
                >
                  Getting started
                </h2>
                <p className="mt-1 max-w-[560px] text-[13.5px] leading-[1.55] text-oc-muted">
                  Use this as an operating guide, not a checklist. Start with
                  the mode and conversation style that fit the moment.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setOpen(false)}
                aria-label="Close getting started guide"
                className="mt-0.5 inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-oc-line bg-white text-oc-muted transition-[color,border-color,background] hover:border-oc-primary/30 hover:bg-oc-surface-tint hover:text-oc-primary"
              >
                <IconClose />
              </button>
            </div>

            <div className="max-h-[calc(88vh-116px)] overflow-y-auto px-5 py-3 md:px-6 md:py-4">
              {GETTING_STARTED_SECTIONS.map((section) => (
                <details
                  key={section.title}
                  className="group border-b border-oc-line-2 py-3 last:border-b-0"
                  open={section.defaultOpen}
                >
                  <summary className="flex cursor-pointer list-none items-center justify-between gap-4 [&::-webkit-details-marker]:hidden">
                    <span>
                      <span className="block font-display text-[17px] font-semibold text-oc-ink">
                        {section.title}
                      </span>
                      <span className="mt-0.5 block text-[12.5px] leading-[1.45] text-oc-muted">
                        {section.summary}
                      </span>
                    </span>
                    <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-oc-surface-tint text-oc-primary transition-transform group-open:rotate-180">
                      <IconChevronDown />
                    </span>
                  </summary>
                  <div className="mt-3 space-y-2.5 pb-1">
                    {section.points.map((point) => (
                      <p
                        key={point}
                        className="text-[13px] leading-[1.55] text-oc-ink-2"
                      >
                        {point}
                      </p>
                    ))}
                    {section.examples && (
                      <div className="mt-3 rounded-xl border border-oc-line bg-oc-surface-tint px-3.5 py-3">
                        <p className="font-mono text-[10.5px] uppercase tracking-[0.1em] text-oc-muted">
                          Useful openings
                        </p>
                        <div className="mt-2 grid gap-2">
                          {section.examples.map((example) => (
                            <p
                              key={example}
                              className="rounded-lg bg-white px-3 py-2 text-[12.5px] leading-[1.45] text-oc-ink-2"
                            >
                              {example}
                            </p>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </details>
              ))}
            </div>
          </section>
        </div>
      )}
    </div>
  );
}

const GETTING_STARTED_SECTIONS = [
  {
    title: "Start with the right mode",
    summary: "Pick the memory behavior before starting.",
    defaultOpen: true,
    points: [
      "Persistent uses your User ID to load prior memory and save useful session summaries after you end a session.",
      "Incognito starts fresh, hides the identity fields, and does not save session memory.",
      "Use the same User ID when you want continuity. Use a new Thread ID when you want a separate conversation under the same memory profile.",
    ],
  },
  {
    title: "Choose how you want to talk",
    summary: "Chat and Voice fit different moments.",
    defaultOpen: true,
    points: [
      "Chat is best for careful writing, pasted context, comparing options, or reviewing what was said.",
      "Voice is best when typing feels hard, you want real-time support, or you want a grounding/check-in style conversation.",
      "On Voice, choose the spoken voice and transcript language before connecting. Wait until the mic says open before speaking.",
      "The voice selector changes the spoken voice. For response tone, ask directly for gentler, shorter, slower, or more practical replies.",
    ],
  },
  {
    title: "Ask for the kind of help you want",
    summary: "You do not need a perfect opening question.",
    points: [
      "Start with what is happening, what you want from the session, and how direct or gentle the reply should be.",
      "Ask for structure when you need it: name the feeling, sort the problem, slow down, examine a thought, or find one next step.",
      "If advice feels too fast, say so. The assistant can shift into listening, reflection, or a more practical mode.",
    ],
    examples: [
      "I want help sorting out what I am feeling, without jumping to advice yet.",
      "Can you help me slow down and think clearly about what happened?",
      "I want one practical next step, but please keep it gentle.",
      "I have a thought that keeps pulling at me. Can we examine it?",
    ],
  },
  {
    title: "Use memory intentionally",
    summary: "Memory is for durable context, not every detail.",
    points: [
      "Persistent mode can save recurring themes, useful preferences, session summaries, and insights that may help future sessions.",
      "Open Memory to review facts, summaries, and style rules. Delete anything that is wrong or no longer useful.",
      "Say clearly when something should or should not be remembered. Use Incognito for conversations that should not affect future sessions.",
    ],
  },
  {
    title: "Move around and finish cleanly",
    summary: "Navigation should not trap you in one surface.",
    points: [
      "Home returns to this setup flow for a new session. Chat, Voice, Memory, and State stay in the primary navigation.",
      "Memory shows what was saved. State shows current session signals and diagnostics when you want to inspect behavior.",
      "End session creates a clean stopping point. After the summary finishes, continue in the thread, start a new session, or review memory.",
    ],
  },
] satisfies Array<{
  title: string;
  summary: string;
  defaultOpen?: boolean;
  points: string[];
  examples?: string[];
}>;

/* ── Mode card ──────────────────────────────────────────────────────── */

interface ModeCardProps {
  value: SessionMode;
  selected: boolean;
  onSelect: (v: SessionMode) => void;
  icon: ReactNode;
  title: string;
  tag?: string;
  desc: string;
}

function ModeCard({ value, selected, onSelect, icon, title, tag, desc }: ModeCardProps) {
  return (
    <button
      type="button"
      onClick={() => onSelect(value)}
      aria-pressed={selected}
      className={`relative flex w-full gap-4 rounded-2xl border bg-white p-5 text-left transition-[border-color,background,box-shadow,transform] duration-200 hover:[border-color:#D6CCB8] ${
        selected ? "border-oc-primary bg-oc-primary-tint" : "border-oc-line"
      }`}
      style={
        selected
          ? {
              boxShadow:
                "0 1px 0 rgba(31,79,70,0.08), 0 18px 36px -28px rgba(31,79,70,0.45)",
            }
          : undefined
      }
    >
      <span
        className={`flex h-[38px] w-[38px] shrink-0 items-center justify-center rounded-[10px] text-oc-primary ${
          selected ? "bg-[#CFE0D5]" : "bg-oc-primary-soft"
        }`}
      >
        {icon}
      </span>

      <span className="min-w-0 flex-1">
        <span className="flex items-center gap-2.5">
          <span
            className="font-display text-[18px] font-semibold text-oc-ink"
            style={{ letterSpacing: "-0.005em" }}
          >
            {title}
          </span>
          {tag ? (
            <span
              className={`inline-flex rounded font-mono text-[10px] uppercase tracking-[0.05em] ${
                selected
                  ? "border border-[rgba(31,79,70,0.18)] bg-[rgba(31,79,70,0.05)] px-2 py-px text-oc-primary"
                  : "border border-oc-line bg-oc-bg px-2 py-px text-oc-muted"
              }`}
            >
              {tag}
            </span>
          ) : null}
        </span>
        <span className="mt-1.5 block text-[13.5px] leading-[1.55] text-oc-muted">
          {desc}
        </span>
      </span>

      <span
        aria-hidden
        className={`absolute right-5 top-5 flex h-[18px] w-[18px] items-center justify-center rounded-full border-[1.5px] bg-white transition-colors ${
          selected ? "border-oc-primary" : "border-oc-line"
        }`}
      >
        <span
          className="h-2 w-2 rounded-full bg-oc-primary transition-transform duration-200"
          style={{ transform: selected ? "scale(1)" : "scale(0)" }}
        />
      </span>
    </button>
  );
}

/* ── Privacy note ───────────────────────────────────────────────────── */

function PrivacyNote() {
  return (
    <div
      role="note"
      className="flex items-start gap-2.5 rounded-lg border border-dashed border-oc-line bg-oc-surface-tint px-3.5 py-3 text-[12.5px] leading-[1.55] text-oc-muted"
    >
      <span className="mt-0.5 shrink-0 text-oc-primary">
        <IconLock />
      </span>
      <span>
        Persistent mode stores conversation summaries and memory locally to your user ID.
        You can clear or export memory any time from Settings.
      </span>
    </div>
  );
}

/* ── ID fields ──────────────────────────────────────────────────────── */

interface IdFieldsProps {
  mode: SessionMode;
  userId: string;
  setUserId: (s: string) => void;
  threadId: string;
  setThreadId: (s: string) => void;
}

function IdFields({ mode, userId, setUserId, threadId, setThreadId }: IdFieldsProps) {
  const incognito = mode === "incognito";
  return (
    <div className="mt-5 border-t border-oc-line-2 pt-5">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 md:gap-3.5">
        <Field
          id="oc-uid"
          label="user id"
          placeholder="e.g. alice, dev-user-01"
          value={userId}
          onChange={setUserId}
          disabled={incognito}
        />
        <Field
          id="oc-tid"
          label="thread id"
          placeholder={incognito ? "auto-generated for incognito" : "e.g. friday-checkin"}
          value={incognito ? "" : threadId}
          onChange={setThreadId}
          disabled={incognito}
        />
      </div>
      <p className="mt-2.5 text-[13px] leading-[1.45] text-oc-text-muted md:text-[12.5px]">
        {incognito
          ? "A new thread ID is generated automatically — nothing is saved."
          : "Reuse an ID to continue a conversation. Same User ID across threads shares memory."}
      </p>
    </div>
  );
}

interface FieldProps {
  id: string;
  label: string;
  placeholder: string;
  value: string;
  onChange: (v: string) => void;
  disabled: boolean;
}

function Field({ id, label, placeholder, value, onChange, disabled }: FieldProps) {
  return (
    <div>
      <label
        htmlFor={id}
        className="mb-1.5 block font-mono text-[10.5px] uppercase tracking-[0.08em] text-oc-muted"
      >
        {label}
      </label>
      <input
        id={id}
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        disabled={disabled}
        className="w-full rounded-lg border border-oc-line bg-white px-3.5 py-3 font-mono text-[13px] text-oc-ink outline-none transition-[border-color,box-shadow,background] duration-150 placeholder:text-oc-text-muted/70 focus:border-oc-primary focus:[box-shadow:0_0_0_3px_rgba(31,79,70,0.10)] disabled:cursor-not-allowed disabled:bg-oc-surface-tint disabled:text-oc-text-muted"
      />
    </div>
  );
}

/* ── Inline icons ───────────────────────────────────────────────────── */

function IconLines({ size = 18 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      aria-hidden
    >
      <path d="M5 7h14M5 12h10M5 17h14" />
    </svg>
  );
}

function IconEyeOff({ size = 18 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M3 3l18 18" />
      <path d="M10.6 6.1A10.5 10.5 0 0 1 12 6c5 0 9 4 10 6-0.4 0.8-1.2 2-2.4 3.2" />
      <path d="M6.4 7.6C4.4 9 3.2 10.7 2 12c1 2 5 6 10 6 1.7 0 3.2-0.5 4.5-1.2" />
      <path d="M9.5 9.8a3 3 0 0 0 4.2 4.2" />
    </svg>
  );
}

function IconLock({ size = 14 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <rect x="4" y="11" width="16" height="9" rx="2" />
      <path d="M8 11V8a4 4 0 0 1 8 0v3" />
    </svg>
  );
}

function IconArrow({ size = 16 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M5 12h14" />
      <path d="M13 6l6 6-6 6" />
    </svg>
  );
}

function IconClose({ size = 16 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M6 6l12 12" />
      <path d="M18 6L6 18" />
    </svg>
  );
}

function IconChevronDown({ size = 15 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M6 9l6 6 6-6" />
    </svg>
  );
}

function IconHelp({ size = 14 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <circle cx="12" cy="12" r="9" />
      <path d="M9.6 9a2.6 2.6 0 1 1 3.4 2.5c-0.8 0.3-1 0.8-1 1.5" />
      <path d="M12 17h0.01" />
    </svg>
  );
}

function IconMic({ size = 13 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <rect x="9" y="3" width="6" height="11" rx="3" />
      <path d="M5 11a7 7 0 0 0 14 0" />
      <path d="M12 18v3" />
    </svg>
  );
}
