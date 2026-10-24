// The conversation this browser is in, so a reload picks it back up (HY-350).
const STORAGE_KEY = "opencouch.threadId";

/** The saved conversation id, or a new one if there isn't one (or storage is blocked). */
export function savedThreadId(): string {
  try {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (saved) return saved;
  } catch {
    // Private windows and blocked storage throw; the conversation then just
    // doesn't survive a reload.
  }
  return startThread();
}

/** Start a new conversation, remembered from now on. */
export function startThread(): string {
  const id = crypto.randomUUID();
  try {
    window.localStorage.setItem(STORAGE_KEY, id);
  } catch {
    // See `savedThreadId`.
  }
  return id;
}
