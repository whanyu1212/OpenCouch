import { auth } from "@/lib/auth/server";

// Neon Auth's endpoints, proxied on our own site so its session cookie is first-party.
export const { GET, POST, PUT, DELETE, PATCH } = auth.handler();
