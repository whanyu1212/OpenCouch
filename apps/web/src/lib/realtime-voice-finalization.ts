export async function finalizeAfterPendingRealtimeVoiceTurn<T>(
  pendingRecording: Promise<void> | null,
  finalize: () => Promise<T>
): Promise<T> {
  await pendingRecording;
  return finalize();
}
