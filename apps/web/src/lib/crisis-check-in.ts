export type CrisisCheckInAnswer = "safe" | "not_sure" | "need_help";

const CRISIS_CHECK_IN_TEXT: Readonly<Record<CrisisCheckInAnswer, string>> = {
  safe: "I'm safe right now",
  not_sure: "I'm not sure",
  need_help: "I need help now",
};

export const CRISIS_CHECK_IN_CHOICES: readonly {
  answer: CrisisCheckInAnswer;
  text: string;
}[] = [
  { answer: "safe", text: CRISIS_CHECK_IN_TEXT.safe },
  { answer: "not_sure", text: CRISIS_CHECK_IN_TEXT.not_sure },
  { answer: "need_help", text: CRISIS_CHECK_IN_TEXT.need_help },
];

/** The one-run payload that distinguishes a server-offered tap from typed text. */
export function crisisCheckInRunParameters(offerId: string, answer: CrisisCheckInAnswer) {
  return { forwardedProps: { crisis_check_in: { offer_id: offerId, answer } } };
}

export function crisisCheckInText(answer: CrisisCheckInAnswer): string {
  return CRISIS_CHECK_IN_TEXT[answer];
}
