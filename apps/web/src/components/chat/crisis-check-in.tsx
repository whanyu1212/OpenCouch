import { CRISIS_CHECK_IN_CHOICES, type CrisisCheckInAnswer } from "@/lib/crisis-check-in";

/** Structured answers for the current server-issued safety check-in. */
export function CrisisCheckIn({
  disabled,
  onAnswer,
}: {
  disabled: boolean;
  onAnswer: (answer: CrisisCheckInAnswer) => void;
}) {
  return (
    <section aria-labelledby="crisis-check-in-title" className="mt-5">
      <h2 id="crisis-check-in-title" className="sr-only">
        Safety check-in
      </h2>
      <div className="flex flex-wrap gap-2">
        {CRISIS_CHECK_IN_CHOICES.map(({ answer, text }) => (
          <button
            key={answer}
            type="button"
            onClick={() => onAnswer(answer)}
            disabled={disabled}
            className="rounded-full border border-line bg-canvas px-4 py-2 text-sm font-medium focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:opacity-40"
          >
            {text}
          </button>
        ))}
      </div>
    </section>
  );
}
