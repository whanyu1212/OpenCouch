"use client";

import dynamic from "next/dynamic";
import { useEffect } from "react";
import { usePathname } from "next/navigation";
import { useSessionStore, useSessionStoreHydrated } from "@/lib/session";
import { SessionSetup } from "@/components/session-setup";
import { CommandPalette } from "@/components/command-palette";
import { ThreadDrawer } from "@/components/thread-drawer";
import { CommandActionsProvider } from "@/lib/command-actions";
import { ConversationShell } from "@/components/conversation-shell";
import { shouldUseRealtimeVoiceProvider } from "@/lib/voice-provider-routing";
import { VoiceSafetyOverlay } from "@/components/voice-safety-overlay";

const DynamicRealtimeVoiceSessionProvider = dynamic(
  () =>
    import("@/components/realtime-voice-session-provider").then(
      (mod) => mod.RealtimeVoiceSessionProvider
    ),
  { ssr: false }
);

/**
 * AppShell — chooses between the landing setup screen and the main
 * conversation layout.
 *
 * AppShell owns the persistent conversation chrome so tab navigation only
 * swaps page content instead of remounting the rail/tab bar. It also wires
 * up the providers (command palette, thread drawer, optional Realtime voice
 * provider) and the hydration loader.
 */
export function AppShell({ children }: { children: React.ReactNode }) {
  const hydrated = useSessionStoreHydrated();
  const isSetup = useSessionStore((s) => s.isSetup);
  const voiceConnected = useSessionStore((s) => s.voiceConnected);
  const voiceConnectionPending = useSessionStore((s) => s.voiceConnectionPending);
  const voiceFinalizationStatus = useSessionStore(
    (s) => s.voiceFinalization.status
  );
  const sessionMode = useSessionStore((s) => s.sessionMode);
  const voiceSafetyOverlayActive = useSessionStore(
    (s) => s.voiceSafetyOverlay?.open ?? false
  );
  const voiceSafetyResourceWorkActive = useSessionStore(
    (s) => s.voiceSafetyResourceWorkActive
  );
  const pathname = usePathname();
  const withWash = pathname === "/voice" || (pathname === "/" && sessionMode === "persistent");

  // Preload the voice provider chunk after setup so navigating to
  // /voice doesn't cause a loading gap that briefly unmounts the tree
  // and kills an active chat WebSocket.
  useEffect(() => {
    if (isSetup) {
      import("@/components/realtime-voice-session-provider");
    }
  }, [isSetup]);

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
      <ConversationShell withWash={withWash}>
        <main className="flex-1 flex min-w-0 flex-col">{children}</main>
      </ConversationShell>
      <CommandPalette />
      <ThreadDrawer />
      <VoiceSafetyOverlay />
    </CommandActionsProvider>
  );

  if (
    shouldUseRealtimeVoiceProvider({
      pathname,
      voiceConnected,
      voiceConnectionPending,
      voiceFinalizationStatus,
      voiceSafetyOverlayActive,
      voiceSafetyResourceWorkActive,
    })
  ) {
    return (
      <DynamicRealtimeVoiceSessionProvider>
        {content}
      </DynamicRealtimeVoiceSessionProvider>
    );
  }

  return content;
}
