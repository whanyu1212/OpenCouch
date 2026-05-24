type VoiceFinalizationStatus = "idle" | "in_progress" | "completed" | "failed";

export function shouldUseRealtimeVoiceProvider({
  pathname,
  voiceConnected,
  voiceFinalizationStatus,
}: {
  pathname: string;
  voiceConnected: boolean;
  voiceFinalizationStatus: VoiceFinalizationStatus;
}): boolean {
  if (pathname.startsWith("/voice/realtime-dev")) {
    return false;
  }

  return (
    pathname === "/voice" ||
    voiceConnected ||
    voiceFinalizationStatus === "in_progress"
  );
}
