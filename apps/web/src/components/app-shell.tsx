"use client";

import dynamic from "next/dynamic";
import { usePathname } from "next/navigation";
import { useSessionStore, useSessionStoreHydrated } from "@/lib/session";
import { SessionSetup } from "@/components/session-setup";
import { Sidebar } from "@/components/sidebar";
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
 * AppShell — conditionally renders the setup screen or the
 * sidebar + page content based on session state.
 *
 * When isSetup is false, the full viewport shows the session
 * setup screen. Once the user picks a mode and starts, the
 * normal sidebar + content layout appears.
 *
 * The LiveKit voice session provider is loaded only when voice is needed
 * so chat, memory, and state routes do not pay for the voice bundle.
 */
export function AppShell({ children }: { children: React.ReactNode }) {
  const hydrated = useSessionStoreHydrated();
  const isSetup = useSessionStore((s) => s.isSetup);
  const voiceConnected = useSessionStore((s) => s.voiceConnected);
  const voiceFinalizationStatus = useSessionStore((s) => s.voiceFinalization.status);
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
      <Sidebar />
      <main className="flex-1 flex flex-col min-w-0">{children}</main>
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
