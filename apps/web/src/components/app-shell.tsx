"use client";

import { useEffect } from "react";
import { useSessionStore } from "@/lib/session";
import { getVoiceRefs } from "@/lib/session";
import { SessionSetup } from "@/components/session-setup";
import { Sidebar } from "@/components/sidebar";

/**
 * AppShell — conditionally renders the setup screen or the
 * sidebar + page content based on session state.
 *
 * When isSetup is false, the full viewport shows the session
 * setup screen. Once the user picks a mode and starts, the
 * normal sidebar + content layout appears.
 *
 * Also owns the global visibilitychange listener so the voice
 * AudioContext is resumed when the browser tab returns to
 * foreground — this must live here (not in voice/page.tsx)
 * because the voice session survives in-app tab switches.
 */
export function AppShell({ children }: { children: React.ReactNode }) {
  const isSetup = useSessionStore((s) => s.isSetup);

  // Resume suspended AudioContext when the browser tab becomes visible.
  // Lives here so it survives navigation between in-app pages.
  useEffect(() => {
    function handleVisibility() {
      if (document.hidden) return;
      const { audioCtx } = getVoiceRefs();
      if (audioCtx && audioCtx.state === "suspended") {
        audioCtx.resume();
      }
    }
    document.addEventListener("visibilitychange", handleVisibility);
    return () =>
      document.removeEventListener("visibilitychange", handleVisibility);
  }, []);

  if (!isSetup) {
    return <SessionSetup />;
  }

  return (
    <>
      <Sidebar />
      <main className="flex-1 flex flex-col min-w-0">{children}</main>
    </>
  );
}
