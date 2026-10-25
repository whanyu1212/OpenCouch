/**
 * Whether a 403's body is the agent saying the person isn't invited to the
 * private beta (HY-353), rather than a 403 from something in front of it.
 */
export function isNotInvited(body: unknown): boolean {
  if (typeof body !== "object" || body === null || !("detail" in body)) return false;
  const { detail } = body;
  return (
    typeof detail === "object" &&
    detail !== null &&
    "code" in detail &&
    detail.code === "not_invited"
  );
}
