import { agentUrl } from "./agent-url";

/** The address can't be one, so the agent refused it. */
export class InvalidEmailError extends Error {
  constructor() {
    super("Not an email address");
    this.name = "InvalidEmailError";
  }
}

/**
 * Ask for early access to the private beta (HY-353). The reply is the same
 * whether the address is new, has asked before or is already invited.
 *
 * @throws {InvalidEmailError} If the agent says the address can't be one.
 * @throws If the request doesn't go through.
 */
export async function requestAccess(email: string): Promise<void> {
  // Next to the agent endpoint (`…/api/agent` → `…/api/access-requests`).
  const response = await fetch(new URL("access-requests", agentUrl()), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email }),
    signal: AbortSignal.timeout(10_000),
  });
  if (response.status === 422) throw new InvalidEmailError();
  if (!response.ok) throw new Error(`Access request failed with ${response.status}`);
}
