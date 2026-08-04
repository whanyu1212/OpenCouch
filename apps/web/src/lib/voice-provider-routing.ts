type VoiceFinalizationStatus = "idle" | "in_progress" | "completed" | "failed";

export function shouldUseRealtimeVoiceProvider({
  pathname,
  voiceConnected,
  voiceConnectionPending = false,
  voiceFinalizationStatus,
  voiceSafetyOverlayActive = false,
  voiceSafetyResourceWorkActive = false,
}: {
  pathname: string;
  voiceConnected: boolean;
  voiceConnectionPending?: boolean;
  voiceFinalizationStatus: VoiceFinalizationStatus;
  voiceSafetyOverlayActive?: boolean;
  voiceSafetyResourceWorkActive?: boolean;
}): boolean {
  if (pathname.startsWith("/voice/realtime-dev")) {
    return false;
  }
  if (voiceSafetyOverlayActive || voiceSafetyResourceWorkActive) {
    return true;
  }

  return (
    pathname === "/voice" ||
    voiceConnected ||
    voiceConnectionPending ||
    voiceFinalizationStatus === "in_progress" ||
    voiceFinalizationStatus === "failed"
  );
}
