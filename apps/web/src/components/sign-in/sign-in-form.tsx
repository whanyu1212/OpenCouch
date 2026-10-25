"use client";

import { useRouter } from "next/navigation";
import { type SubmitEvent, useState } from "react";

import { authClient, callNeonAuth } from "@/lib/auth/client";
import { rememberSignedInUser } from "@/lib/thread-id";

import { FIELD, PRIMARY } from "./styles";

type Step = { name: "email" } | { name: "code"; email: string };

/** Sign in with a one-time code sent to an email address; a first sign-in creates the account. */
export function SignInForm() {
  const router = useRouter();
  const [step, setStep] = useState<Step>({ name: "email" });
  const [email, setEmail] = useState("");
  const [code, setCode] = useState("");
  const [pending, setPending] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  async function sendCode(address: string) {
    setPending(true);
    setProblem(null);
    const sent = await callNeonAuth(() =>
      authClient.emailOtp.sendVerificationOtp({ email: address, type: "sign-in" }),
    );
    setPending(false);
    if (!sent.ok) {
      setProblem("The code couldn't be sent. Check the address and try again.");
      return;
    }
    setCode("");
    setStep({ name: "code", email: address });
  }

  async function checkCode(address: string) {
    setPending(true);
    setProblem(null);
    const signedIn = await callNeonAuth(() =>
      authClient.signIn.emailOtp({ email: address, otp: code }),
    );
    if (!signedIn.ok || !signedIn.data) {
      setPending(false);
      setProblem("That code didn't work. Check it, or send a new one.");
      return;
    }
    rememberSignedInUser(signedIn.data.user.id);
    router.replace("/");
  }

  function chooseAnotherEmail() {
    setProblem(null);
    setStep({ name: "email" });
  }

  function handleSubmit(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault();
    if (pending) return;
    void (step.name === "email" ? sendCode(email.trim()) : checkCode(step.email));
  }

  return (
    <form onSubmit={handleSubmit} className="mt-8">
      {step.name === "email" ? (
        <label className="block text-sm font-medium">
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
      ) : (
        <>
          {/* The same words whether or not the address is invited, so the
              page never says who is. */}
          <p className="text-ink-soft">
            If <span className="font-medium text-ink">{step.email}</span> is invited, we&apos;ve
            sent it a 6-digit code.
          </p>
          <label className="mt-4 block text-sm font-medium">
            Code
            <input
              required
              inputMode="numeric"
              autoComplete="one-time-code"
              pattern="[0-9]{6}"
              maxLength={6}
              value={code}
              onChange={(event) => setCode(event.target.value.replace(/\D/g, ""))}
              className={`${FIELD} tracking-[0.3em] tabular-nums`}
            />
          </label>
        </>
      )}

      {problem && (
        <p role="alert" className="mt-3 rounded-2xl bg-notice-wash px-4 py-2.5 text-sm text-notice">
          {problem}
        </p>
      )}

      <button type="submit" disabled={pending} className={PRIMARY}>
        {step.name === "email" ? "Email me a code" : "Sign in"}
      </button>

      {step.name === "code" && (
        <div className="mt-3 flex justify-between text-sm">
          <button
            type="button"
            disabled={pending}
            onClick={chooseAnotherEmail}
            className="font-medium underline-offset-2 hover:underline disabled:opacity-40"
          >
            Use another email
          </button>
          <button
            type="button"
            disabled={pending}
            onClick={() => void sendCode(step.email)}
            className="font-medium underline-offset-2 hover:underline disabled:opacity-40"
          >
            Send a new code
          </button>
        </div>
      )}
    </form>
  );
}
