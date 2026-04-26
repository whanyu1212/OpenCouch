"use client";

import { useSessionStore } from "@/lib/session";
import { SessionSetup } from "@/components/session-setup";
import { Sidebar } from "@/components/sidebar";
import { VoiceSessionProvider } from "@/components/voice-session-provider";
import { CommandPalette } from "@/components/command-palette";
import { ThreadDrawer } from "@/components/thread-drawer";
import { CommandActionsProvider } from "@/lib/command-actions";

/**
 * AppShell — conditionally renders the setup screen or the
 * sidebar + page content based on session state.
 *
 * When isSetup is false, the full viewport shows the session
 * setup screen. Once the user picks a mode and starts, the
 * normal sidebar + content layout appears.
 *
 * The LiveKit voice session provider wraps the app layout so a voice
 * room can survive in-app tab switches.
 */
export function AppShell({ children }: { children: React.ReactNode }) {
  const isSetup = useSessionStore((s) => s.isSetup);

  if (!isSetup) {
    return <SessionSetup />;
  }

  return (
    <VoiceSessionProvider>
      <CommandActionsProvider>
        <Sidebar />
        <main className="flex-1 flex flex-col min-w-0">{children}</main>
        <CommandPalette />
        <ThreadDrawer />
      </CommandActionsProvider>
    </VoiceSessionProvider>
  );
}
