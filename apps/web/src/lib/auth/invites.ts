/**
 * Whether an address is invited to the private beta (HY-353), from the same
 * `OPENCOUCH_BETA_INVITES` the agent reads. With none set, nobody is.
 * `OPENCOUCH_BETA_OPEN=true` invites everyone, but only outside a production
 * build, so it can't open a deploy even if it's set there.
 */
export function isInvited(email: string): boolean {
  const local = process.env.NODE_ENV !== "production";
  if (local && process.env.OPENCOUCH_BETA_OPEN === "true") return true;
  const invites = (process.env.OPENCOUCH_BETA_INVITES ?? "")
    .split(",")
    .map(normalizedEmail)
    .filter(Boolean);
  return invites.includes(normalizedEmail(email));
}

// Matches the agent's `normalized_email`, which uses Python's `lower()`.
function normalizedEmail(email: string): string {
  return email.trim().toLowerCase();
}
