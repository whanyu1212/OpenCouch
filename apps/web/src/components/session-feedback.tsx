"use client";

import { useState } from "react";

import {
  submitSessionFeedback,
  type ApiMemoryMode,
  type SessionFeedbackLabel,
  type SessionFeedbackModality,
} from "@/lib/api";

type FeedbackState = "idle" | "submitting" | "submitted" | "error";

const OPTIONS: Array<{ label: SessionFeedbackLabel; title: string }> = [
  { label: "positive", title: "Positive" },
  { label: "negative", title: "Negative" },
  { label: "skip", title: "Skip" },
];

export function SessionFeedback({
  threadId,
  memoryMode,
  modality,
  onSubmitted,
  className = "",
}: {
  threadId: string;
  memoryMode: ApiMemoryMode;
  modality: SessionFeedbackModality;
  onSubmitted?: (label: SessionFeedbackLabel) => void;
  className?: string;
}) {
  const [state, setState] = useState<FeedbackState>("idle");
  const [selected, setSelected] = useState<SessionFeedbackLabel | null>(null);

  const submit = async (label: SessionFeedbackLabel) => {
    setSelected(label);
    setState("submitting");
    try {
      const response = await submitSessionFeedback(
        threadId,
        label,
        memoryMode,
        modality
      );
      if (!response.recorded) {
        setState("error");
        return;
      }
      setState("submitted");
      onSubmitted?.(label);
    } catch {
      setState("error");
    }
  };

  const disabled = state === "submitting" || state === "submitted";

  return (
    <div className={className}>
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-[10.5px] uppercase tracking-[0.1em] text-oc-text-dim">
          Session feedback
        </span>
        {OPTIONS.map((option) => {
          const isSelected = selected === option.label;
          return (
            <button
              key={option.label}
              type="button"
              onClick={() => void submit(option.label)}
              disabled={disabled}
              aria-pressed={isSelected && state === "submitted"}
              className={[
                "rounded-lg border px-2.5 py-1.5 text-[12px] font-medium transition-colors disabled:cursor-not-allowed",
                isSelected && state === "submitted"
                  ? "border-oc-teal-300 bg-oc-teal-100 text-oc-teal-800"
                  : "border-oc-line bg-white/80 text-oc-text-secondary hover:border-oc-primary/30 hover:bg-white",
                disabled && !(isSelected && state === "submitted")
                  ? "opacity-55"
                  : "",
              ].join(" ")}
            >
              {state === "submitting" && isSelected ? "Saving..." : option.title}
            </button>
          );
        })}
      </div>
      <p
        className="mt-1 min-h-[18px] text-[12px] text-oc-text-dim"
        role="status"
        aria-live="polite"
      >
        {state === "submitted"
          ? "Feedback saved."
          : state === "error"
            ? "Could not save feedback. Try again."
            : ""}
      </p>
    </div>
  );
}
