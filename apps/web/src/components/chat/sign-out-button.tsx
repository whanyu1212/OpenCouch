"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { authClient, callNeonAuth } from "@/lib/auth/client";
import { forgetThread } from "@/lib/thread-id";

export function SignOutButton({ disabled }: { disabled: boolean }) {
  const router = useRouter();
  const [pending, setPending] = useState(false);

  async function signOut() {
    setPending(true);
    if (!(await callNeonAuth(() => authClient.signOut())).ok) {
      // Still signed in, so the button comes back to try again.
      setPending(false);
      return;
    }
    // Whoever signs in next on this browser starts their own conversation.
    forgetThread();
    router.replace("/sign-in");
  }

  return (
    <button
      type="button"
      onClick={() => void signOut()}
      disabled={disabled || pending}
      className="rounded-full px-3 py-1.5 text-sm font-medium underline-offset-2 hover:underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:opacity-40"
    >
      Sign out
    </button>
  );
}
