"use client";

import dynamic from "next/dynamic";
import { usePathname } from "next/navigation";
import { useSessionStore, useSessionStoreHydrated } from "@/lib/session";
import { SessionSetup } from "@/components/session-setup";
import { CommandPalette } from "@/components/command-palette";
import { ThreadDrawer } from "@/components/thread-drawer";
import { CommandActionsProvider } from "@/lib/command-actions";

const DynamicVoiceSessionProvider = dynamic(
  () =>
    import("@/components/voice-session-provider").then(
      (mod) => mod.VoiceSessionProvider
    ),
  { ssr: false }
);

/**
 * AppShell — chooses between the landing setup screen and the main
 * conversation layout.
 *
 * Pages own their own chrome via `<ConversationShell>` — a slim icon
 * NavRail on desktop and a bottom tab bar on mobile. AppShell just wires
 * up the providers (command palette, thread drawer, optional LiveKit
 * provider) and the hydration loader.
 */
export function AppShell({ children }: { children: React.ReactNode }) {
  const hydrated = useSessionStoreHydrated();
  const isSetup = useSessionStore((s) => s.isSetup);
  const voiceConnected = useSessionStore((s) => s.voiceConnected);
  const voiceFinalizationStatus = useSessionStore(
    (s) => s.voiceFinalization.status
  );
  const pathname = usePathname();

  if (!hydrated) {
    return (
      <div className="flex min-h-screen flex-1 items-center justify-center bg-oc-bg px-6 text-oc-text">
        <div className="flex items-center gap-3 rounded-xl border border-oc-border bg-oc-bg-card px-4 py-3">
          <span className="relative flex h-3.5 w-3.5 items-center justify-center">
            <span className="absolute inline-flex h-2.5 w-2.5 animate-ping rounded-full bg-oc-cta opacity-50" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-oc-cta" />
          </span>
          <span className="font-mono text-[13px] text-oc-text-muted">
            loading
          </span>
        </div>
      </div>
    );
  }

  if (!isSetup) {
    return <SessionSetup />;
  }

  const content = (
    <CommandActionsProvider>
      <main className="flex-1 flex min-w-0">{children}</main>
      <CommandPalette />
      <ThreadDrawer />
    </CommandActionsProvider>
  );

  if (
    pathname === "/voice" ||
    voiceConnected ||
    voiceFinalizationStatus === "in_progress"
  ) {
    return <DynamicVoiceSessionProvider>{content}</DynamicVoiceSessionProvider>;
  }

  return content;
}
