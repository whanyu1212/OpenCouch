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

export async function clearHandleAfterSuccessfulDisconnect(
  disconnect: () => Promise<void>,
  clearHandle: () => void
): Promise<void> {
  await disconnect();
  clearHandle();
}

export class RealtimeVoiceDisconnectCoordinator {
  private disconnected = false;
  private pendingAttempt: Promise<void> | null = null;

  disconnect(attempt: () => Promise<void>): Promise<void> {
    if (this.disconnected) return Promise.resolve();
    if (this.pendingAttempt) return this.pendingAttempt;

    const pendingAttempt = attempt().then(() => {
      this.disconnected = true;
    });
    this.pendingAttempt = pendingAttempt;
    void pendingAttempt.then(
      () => this.clearAttempt(pendingAttempt),
      () => this.clearAttempt(pendingAttempt)
    );
    return pendingAttempt;
  }

  private clearAttempt(attempt: Promise<void>): void {
    if (this.pendingAttempt === attempt) {
      this.pendingAttempt = null;
    }
  }
}
