"use client";

import { useEffect, useState } from "react";

import { agentUrl } from "./agent-url";
import { isRecord } from "./is-record";

// Past this, the card falls back to the emergency number alone.
const CATALOG_TIMEOUT_MS = 10_000;

/** One way to reach a vetted crisis service. */
export type CrisisChannel =
  | { kind: "whatsapp"; label: string; display: string; url: string }
  | { kind: "call" | "sms"; label: string; display: string };

/** A vetted crisis service from the agent's catalog (HY-209). */
export type CrisisResource = {
  key: string;
  name: string;
  hours: string | null;
  channels: CrisisChannel[];
};

/** The catalog, while it loads and once it has loaded or failed. */
export type CrisisCatalog =
  | { status: "loading" }
  | { status: "ready"; resources: ReadonlyMap<string, CrisisResource> }
  | { status: "unavailable" };

// The response is untrusted input, so it is checked field by field.
function parseChannel(value: unknown): CrisisChannel | null {
  if (!isRecord(value)) return null;
  const { kind, label, display, url } = value;
  if (typeof label !== "string" || typeof display !== "string") return null;
  if (kind === "call" || kind === "sms") return { kind, label, display };
  // Only a WhatsApp link is ever opened from the card.
  if (kind === "whatsapp" && typeof url === "string" && url.startsWith("https://wa.me/")) {
    return { kind, label, display, url };
  }
  return null;
}

function parseResource(value: unknown): CrisisResource | null {
  if (!isRecord(value)) return null;
  const { key, name, hours, channels } = value;
  if (typeof key !== "string" || typeof name !== "string" || !Array.isArray(channels)) return null;
  const parsed = channels.map(parseChannel);
  if (parsed.some((channel) => channel === null)) return null;
  return {
    key,
    name,
    hours: typeof hours === "string" ? hours : null,
    channels: parsed.filter((channel) => channel !== null),
  };
}

/**
 * Check the agent's catalog response and key its resources by catalog key.
 *
 * @throws If the response, or any entry in it, isn't in the expected shape.
 */
export function parseCrisisCatalog(body: unknown): Map<string, CrisisResource> {
  const raw =
    typeof body === "object" && body !== null && "resources" in body ? body.resources : null;
  if (!Array.isArray(raw)) throw new Error("Catalog response was not in the expected shape");
  const resources = raw.map(parseResource);
  if (resources.some((resource) => resource === null)) {
    throw new Error("Catalog response had an entry not in the expected shape");
  }
  return new Map(
    resources.filter((resource) => resource !== null).map((resource) => [resource.key, resource]),
  );
}

/**
 * Load the vetted crisis resources from the agent.
 *
 * @throws If the agent can't be reached in time, or answers with something unexpected.
 */
async function loadCrisisCatalog(signal: AbortSignal): Promise<Map<string, CrisisResource>> {
  // Resolved next to the agent endpoint, like the history request.
  const url = new URL("crisis-resources", agentUrl());
  const response = await fetch(url, {
    signal: AbortSignal.any([signal, AbortSignal.timeout(CATALOG_TIMEOUT_MS)]),
  });
  if (!response.ok) throw new Error(`Catalog request failed with ${response.status}`);
  return parseCrisisCatalog(await response.json());
}

/**
 * The vetted crisis catalog, loaded when the chat opens so the resource card
 * can show the moment a crisis turn starts.
 *
 * @param needed Whether a card is showing. A load that failed is tried again
 *   when one is, so a brief outage at open doesn't leave every later card short.
 */
export function useCrisisCatalog(needed: boolean): CrisisCatalog {
  const [catalog, setCatalog] = useState<CrisisCatalog>({ status: "loading" });
  const retry = needed && catalog.status === "unavailable";

  // A retry that works loads the catalog once more as `retry` turns back off:
  // one small request, only after a failure, whose own failure is ignored.
  useEffect(() => {
    const controller = new AbortController();
    loadCrisisCatalog(controller.signal)
      .then((resources) => {
        if (!controller.signal.aborted) setCatalog({ status: "ready", resources });
      })
      .catch(() => {
        if (controller.signal.aborted) return;
        // A catalog already loaded is never thrown away by a later failure.
        setCatalog((current) => (current.status === "ready" ? current : { status: "unavailable" }));
      });
    return () => controller.abort();
  }, [retry]);

  return catalog;
}

/** The resource card's catalog keys in server-sent state, or none. */
export function cardKeysIn(state: unknown): string[] {
  if (typeof state !== "object" || state === null || !("crisis_resources" in state)) return [];
  const keys = state.crisis_resources;
  return Array.isArray(keys) && keys.every((key) => typeof key === "string") ? keys : [];
}
