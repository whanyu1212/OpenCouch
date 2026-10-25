"use client";

import { useState } from "react";

import { RequestAccessForm } from "./request-access-form";
import { SignInForm } from "./sign-in-form";
import { PRIMARY, SECONDARY, TEXT_BUTTON } from "./styles";

/** Which way in someone is taking. */
export type EntryPath = "choose" | "invited" | "request";

/** The two ways in: sign in with an invite, or ask for one (HY-353). */
export function Entry({ initialPath }: { initialPath: EntryPath }) {
  const [path, setPath] = useState<EntryPath>(initialPath);

  if (path === "choose") {
    return (
      <div className="mt-8">
        <button type="button" onClick={() => setPath("invited")} className={PRIMARY}>
          I&apos;ve been invited
        </button>
        <button type="button" onClick={() => setPath("request")} className={SECONDARY}>
          Request early access
        </button>
      </div>
    );
  }

  return (
    <div>
      {path === "invited" ? <SignInForm /> : <RequestAccessForm />}
      <button type="button" onClick={() => setPath("choose")} className={`mt-4 ${TEXT_BUTTON}`}>
        Back
      </button>
    </div>
  );
}
