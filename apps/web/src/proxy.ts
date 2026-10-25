import { auth } from "@/lib/auth/server";

// Sends anyone not signed in to /sign-in. Only the chat is matched: the
// sign-in page, the auth routes and static files must load without a session.
export const proxy = auth.middleware({ loginUrl: "/sign-in" });

export const config = {
  matcher: ["/"],
};
