import type { Metadata } from "next";

import { CrisisServices } from "@/components/sign-in/crisis-services";
import { SignInForm } from "@/components/sign-in/sign-in-form";

export const metadata: Metadata = { title: "Sign in · OpenCouch" };

export default function SignIn() {
  return (
    <div className="flex min-h-dvh flex-col">
      <header className="border-b border-line px-5 pt-[max(1rem,env(safe-area-inset-top))] pb-3">
        <p className="mx-auto max-w-md text-xl font-semibold tracking-tight">opencouch</p>
      </header>
      <main className="px-5 pt-16 pb-[max(2rem,env(safe-area-inset-bottom))]">
        <div className="mx-auto max-w-md">
          <h1 className="text-3xl font-semibold tracking-tight text-balance">Sign in</h1>
          <p className="mt-3 text-ink-soft">
            We&apos;ll email you a code. No password, and no name needed.
          </p>
          <SignInForm />
          <CrisisServices />
        </div>
      </main>
    </div>
  );
}
