// The conversation this browser is in, so a reload picks it back up (HY-350).
const STORAGE_KEY = "opencouch.threadId";
// Who that conversation belongs to.
const USER_KEY = "opencouch.userId";

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

/**
 * Note who just signed in. Someone other than the last person starts their
 * own conversation, so a shared browser never reopens someone else's.
 */
export function rememberSignedInUser(userId: string): void {
  try {
    if (window.localStorage.getItem(USER_KEY) !== userId) forgetThread();
    window.localStorage.setItem(USER_KEY, userId);
  } catch {
    // See `savedThreadId`.
  }
}

/** Forget the saved conversation, so the next person on this browser doesn't open it. */
export function forgetThread(): void {
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // See `savedThreadId`.
  }
}

/**
 * Call `onChange` when another tab changes the saved conversation or who is
 * signed in, as signing out or in does.
 *
 * @returns A function that stops listening.
 */
export function onThreadChangedElsewhere(onChange: () => void): () => void {
  const listener = (event: StorageEvent) => {
    if (event.key === STORAGE_KEY || event.key === USER_KEY) onChange();
  };
  window.addEventListener("storage", listener);
  return () => window.removeEventListener("storage", listener);
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
