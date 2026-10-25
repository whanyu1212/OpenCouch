import type { Metadata } from "next";

import { CrisisServices } from "@/components/sign-in/crisis-services";
import { Entry, type EntryPath } from "@/components/sign-in/entry";

export const metadata: Metadata = { title: "Sign in · OpenCouch" };

export default async function SignIn({ searchParams }: PageProps<"/sign-in">) {
  // The waitlist links here with `?path=request` to open the request form.
  const { path } = await searchParams;
  const initialPath: EntryPath = path === "request" ? "request" : "choose";

  return (
    <div className="flex min-h-dvh flex-col">
      <header className="border-b border-line px-5 pt-[max(1rem,env(safe-area-inset-top))] pb-3">
        <p className="mx-auto max-w-md text-xl font-semibold tracking-tight">opencouch</p>
      </header>
      <main className="px-5 pt-16 pb-[max(2rem,env(safe-area-inset-bottom))]">
        <div className="mx-auto max-w-md">
          <h1 className="text-3xl font-semibold tracking-tight text-balance">
            A coach for when worry takes over
          </h1>
          <p className="mt-3 text-ink-soft">
            A guided self-help program for anxiety, with a coach that helps you practise a few
            simple skills. We&apos;re in a small private beta for now.
          </p>
          <Entry initialPath={initialPath} />
          <CrisisServices />
        </div>
      </main>
    </div>
  );
}
