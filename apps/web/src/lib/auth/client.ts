import { createAuthClient } from "@neondatabase/auth/next";

/** Neon Auth in the browser, through our own `/api/auth` routes. */
export const authClient = createAuthClient();

/** The session ended or was never there, so the person has to sign in again. */
export class SignedOutError extends Error {
  constructor() {
    super("Not signed in");
    this.name = "SignedOutError";
  }
}

/** A Neon Auth call's outcome; `status` is the HTTP status of a failure, if it had one. */
export type AuthOutcome<T> = { ok: true; data: T } | { ok: false; status: number | undefined };

function statusOf(failure: unknown): number | undefined {
  if (typeof failure === "object" && failure !== null && "status" in failure) {
    return typeof failure.status === "number" ? failure.status : undefined;
  }
  return undefined;
}

/**
 * Run a Neon Auth client call. The client returns some failures and throws
 * others (a 502 from our proxy, say), so both come back the same way.
 */
export async function callNeonAuth<T>(
  call: () => Promise<{ data: T; error: unknown }>,
): Promise<AuthOutcome<T>> {
  try {
    const { data, error } = await call();
    return error ? { ok: false, status: statusOf(error) } : { ok: true, data };
  } catch (failure) {
    return { ok: false, status: statusOf(failure) };
  }
}

/**
 * Headers proving who is sending a request to the agent: a Neon Auth JWT,
 * which lasts 15 minutes, so a fresh one is fetched for every request.
 *
 * @throws {SignedOutError} If there's no session.
 * @throws If the session can't be checked right now.
 */
export async function agentAuthHeaders(): Promise<Record<string, string>> {
  // Straight from our token route: Neon's client can answer `token()` from
  // its session cache, and that session holds Better Auth's opaque session
  // token, not the JWT the agent checks.
  const response = await fetch("/api/auth/token", { cache: "no-store" });
  if (response.status === 401) throw new SignedOutError();
  if (!response.ok) throw new Error(`Sign-in couldn't be checked (${response.status})`);
  const body: unknown = await response.json();
  const token =
    typeof body === "object" && body !== null && "token" in body ? body.token : null;
  if (typeof token !== "string" || !token) throw new SignedOutError();
  return { Authorization: `Bearer ${token}` };
}
