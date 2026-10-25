import { isInvited } from "@/lib/auth/invites";
import { auth } from "@/lib/auth/server";

// Neon Auth's endpoints, proxied on our own site so its session cookie is first-party.
const neonAuth = auth.handler();
export const { GET, PUT, DELETE, PATCH } = neonAuth;

const SEND_CODE_PATH = "email-otp/send-verification-otp";
// About how long Neon takes to send a code, so an address that isn't invited
// doesn't stand out by answering faster.
const SEND_DELAY_MS = { min: 600, spread: 800 };

type RouteContext = { params: Promise<{ path: string[] }> };

/**
 * Sends a code only to invited addresses (HY-353). Anyone else gets the same
 * reply as a sent code, so whether an address is invited stays private. This
 * spares uninvited people an account and an email; the agent's invite check
 * is still the gate, since Neon's endpoint can be called directly.
 */
export async function POST(request: Request, context: RouteContext): Promise<Response> {
  const { path } = await context.params;
  if (path.join("/") !== SEND_CODE_PATH) return neonAuth.POST(request, context);
  const email = await requestedEmail(request);
  if (email !== null && isInvited(email)) {
    const sent = await neonAuth.POST(request, context);
    // Rebuilt from its status and body alone: Neon's own headers (a request
    // ID, a charset) would set it apart from the reply below.
    return new Response(await sent.text(), {
      status: sent.status,
      headers: { "content-type": "application/json" },
    });
  }
  await new Promise((resolve) =>
    setTimeout(resolve, SEND_DELAY_MS.min + Math.random() * SEND_DELAY_MS.spread),
  );
  return Response.json({ success: true });
}

// The address a send-code request names, read from a copy so Neon still gets the body.
async function requestedEmail(request: Request): Promise<string | null> {
  const body: unknown = await request
    .clone()
    .json()
    .catch(() => null);
  if (typeof body !== "object" || body === null || !("email" in body)) return null;
  return typeof body.email === "string" ? body.email : null;
}
