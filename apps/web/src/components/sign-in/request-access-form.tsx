"use client";

import { type SubmitEvent, useState } from "react";

import { InvalidEmailError, requestAccess } from "@/lib/access-requests";

import { FIELD, PRIMARY } from "./styles";

// Who to write to for a request to be deleted.
const REMOVAL_CONTACT = "whanyu47@gmail.com";

/** Leave an email to be invited when there's a place in the beta (HY-353). */
export function RequestAccessForm() {
  const [email, setEmail] = useState("");
  const [pending, setPending] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const [requested, setRequested] = useState<string | null>(null);

  async function submit(address: string) {
    setPending(true);
    setProblem(null);
    try {
      await requestAccess(address);
      setRequested(address);
    } catch (error) {
      setProblem(
        error instanceof InvalidEmailError
          ? "That doesn't look like an email address."
          : "That didn't go through. Check your connection and try again.",
      );
    } finally {
      setPending(false);
    }
  }

  function handleSubmit(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault();
    if (pending) return;
    void submit(email.trim());
  }

  if (requested !== null) {
    return (
      <p role="status" className="mt-8 rounded-2xl bg-accent-wash px-4 py-3">
        Thanks. We&apos;ll write to <span className="font-medium">{requested}</span> when
        there&apos;s a place for you.
      </p>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="mt-8">
      <p className="text-ink-soft">
        Leave your email and we&apos;ll let you know when there&apos;s a place. We&apos;ll only
        use it for that. To have it deleted, email{" "}
        <a
          href={`mailto:${REMOVAL_CONTACT}`}
          className="font-medium text-ink underline underline-offset-2"
        >
          {REMOVAL_CONTACT}
        </a>
        .
      </p>
      <label className="mt-4 block text-sm font-medium">
        Email
        <input
          type="email"
          required
          autoComplete="email"
          value={email}
          onChange={(event) => setEmail(event.target.value)}
          placeholder="you@example.com"
          className={FIELD}
        />
      </label>

      {problem && (
        <p role="alert" className="mt-3 rounded-2xl bg-notice-wash px-4 py-2.5 text-sm text-notice">
          {problem}
        </p>
      )}

      <button type="submit" disabled={pending} className={PRIMARY}>
        Request early access
      </button>
    </form>
  );
}
