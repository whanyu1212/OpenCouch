import { CrisisServices } from "@/components/sign-in/crisis-services";
import { PRIMARY } from "@/components/sign-in/styles";

import { SignOutButton } from "./sign-out-button";

/** For someone signed in but not invited to the private beta (HY-353). */
export function Waitlist() {
  return (
    <div className="flex min-h-dvh flex-col">
      <header className="border-b border-line px-5 pt-[max(1rem,env(safe-area-inset-top))] pb-3">
        <div className="mx-auto flex max-w-md items-center justify-between gap-4">
          <p className="text-xl font-semibold tracking-tight">opencouch</p>
          <SignOutButton disabled={false} />
        </div>
      </header>
      <main className="px-5 pt-16 pb-[max(2rem,env(safe-area-inset-bottom))]">
        <div className="mx-auto max-w-md">
          <h1 className="text-3xl font-semibold tracking-tight text-balance">Not quite yet</h1>
          <p className="mt-3 text-ink-soft">
            OpenCouch is in a small private beta, and this address isn&apos;t on the invite
            list. You can ask for a place, or sign out and sign in with the address you were
            invited with.
          </p>
          <a href="/sign-in?path=request" className={`block text-center ${PRIMARY}`}>
            Request early access
          </a>
          <CrisisServices />
        </div>
      </main>
    </div>
  );
}
