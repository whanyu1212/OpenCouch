import { createNeonAuth } from "@neondatabase/auth/next/server";

function required(name: string): string {
  const value = process.env[name];
  if (!value) throw new Error(`${name} is required: see apps/web/.env.example`);
  return value;
}

/** Neon Auth on the server: the auth routes, route protection and sessions (HY-207). */
export const auth = createNeonAuth({
  baseUrl: required("NEON_AUTH_BASE_URL"),
  cookies: { secret: required("NEON_AUTH_COOKIE_SECRET") },
});
