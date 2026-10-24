const CONFIGURED_URL = process.env.NEXT_PUBLIC_AGENT_URL ?? "http://localhost:8080/api/agent";

/**
 * The agent's AG-UI endpoint, as an absolute URL; its other routes are
 * resolved next to it. It may be configured as a path, such as `/api/agent`
 * when the agent is served from the same site. Only call it in the browser.
 */
export function agentUrl(): string {
  return new URL(CONFIGURED_URL, window.location.href).href;
}
