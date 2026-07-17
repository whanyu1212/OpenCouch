export async function finalizeAfterPendingRealtimeVoiceTurn<T>(
  pendingRecording: Promise<void> | null,
  finalize: () => Promise<T>
): Promise<T> {
  await pendingRecording;
  return finalize();
}

export function onRealtimeVoiceTurnRecordingSettled(
  pendingRecording: Promise<void>,
  onSettled: (recording: Promise<void>) => void
): void {
  void pendingRecording.then(
    () => onSettled(pendingRecording),
    () => onSettled(pendingRecording)
  );
}
