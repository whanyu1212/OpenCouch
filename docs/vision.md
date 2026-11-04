# OpenCouch: the big picture

Why OpenCouch exists, what it is betting on, where it stands, and what would
show it's wrong. How it's built is in the architecture plan,
[#356](https://github.com/whanyu1212/OpenCouch/issues/356); the work is
tracked in the [Linear project](https://linear.app/queryverse/project/opencouch-0d8e0d95078d).

Last reviewed: 2026-11-04.

## The problem

**Guided self-help for anxiety works, and is hard to get.** Internet-delivered
CBT for anxiety works about as well as face-to-face therapy when someone guides
it. Unguided programs lose most of their users. The guide is the expensive
part.

**General AI assistants have made worry worse in one specific way.** For
someone with generalized anxiety, an assistant that is always there and built to
answer is a reassurance machine. "Is this symptom serious?", "Did my boss sound
annoyed?", "Are you sure it'll be fine?": each answer brings a minute of relief
and teaches the mind that the worry needed an answer, so it comes back
stronger. That reassurance-seeking keeps anxiety going is well established in
health anxiety and OCD (Salkovskis & Warwick 1986; Kobori & Salkovskis 2013).
The evidence specific to GAD still needs to be gathered. People with OCD now
describe turning to chatbots for reassurance, with mixed results: some
chatbots decline and others feed the compulsion (Barkhuff 2026). Clinicians
have started to describe how general-purpose chatbots can keep OCD and anxiety
going (Golden & Aboujaoude 2026).

**Feeling supported isn't getting better.** Companion apps are tuned for
engagement. A recent trial reported an AI CBT app that people used far more
than a digital workbook, with no extra drop in symptoms. More conversation is
not the goal; getting better often means needing the app less.

## What OpenCouch is

A structured, time-limited program of skills for worry, with an AI coach as
the guide. The coach helps people practise the skills instead of reassuring
them.

- **Program:** short sessions with the coach that review progress, teach a
  skill and set practice.
- **Daily practice:** quick logging between sessions. Current thinking is a
  worry log with a daily worry time.
- **Coach chat:** always available and aware of where the person is in the
  program. When it detects reassurance-seeking or a worry loop, it redirects
  to a skill instead of answering.
- **Progress:** a standard questionnaire (GAD-7) over time, and what the
  person's own practice shows. The program ends, with a plan for afterwards.
- **Safety:** screening on every turn, in code, invisible until it's needed.

It is **not** an open-ended companion, therapy, a diagnosis tool, or a library
of articles and meditations. It's presented as a skills program, not as
treatment for a disorder.

## What we're betting on

Each bet comes with what would show it's wrong.

1. **An AI coach can be the guide in guided self-help.** The evidence for
   guided programs comes from human guides, and nobody has shown an AI coach
   does the same job.
   *Wrong if:* people drop out the way they do from unguided programs, or
   chat with the coach without doing the practice.
2. **Redirecting instead of reassuring helps, and people stay.** Withholding
   an answer is uncomfortable, and ChatGPT is one tab away.
   *Wrong if:* people leave or switch to another assistant after redirects, or
   the redirect fires on questions that deserved an answer.
3. **The coach gets better as it learns the person.** The worry log and the
   formulation (a personal map of what they worry about and what keeps it
   going) should make redirects specific to them, which a general assistant
   can't copy.
   *Wrong if:* personalised redirects do no better than generic ones.
4. **Code-owned structure makes it safe enough to put in front of anxious
   people.** Safety screening, crisis routing and state changes run in code,
   and the model only chooses words and a few safe tools.
   *Wrong if:* the crisis evals miss cases, or the safety classifier drifts
   without anyone noticing.
5. **Practice moves outcomes.** The bet underneath all the others.
   *Wrong if:* GAD-7 doesn't move for people who practise. This can't be
   tested properly without users and a comparison group, and it's the
   biggest risk.

## Where we stand

**Built (on `develop`):**

- **Safety:**
  - the gate on every turn;
  - conversation-level risk, with a careful step-down from CRISIS and a
    per-user floor across threads;
  - a tool-less crisis responder with a vetted resource card and check-in
    buttons;
  - an audit log;
  - a weekly drift check on the safety classifier (Jev).

  Crisis detection on the BHH test split: AUC 0.993, with 97% recall at a 5%
  false-alarm rate (2026-10-28 recording).
- **Product:**
  - sign-in;
  - threads owned by the user who started them;
  - an invite-only private beta;
  - a per-user records store (program position, worries, check-ins,
    formulation);
  - an interim coach prompt with three provisional skills and
    reassurance-redirect guidance.

**Not built:** the program itself (state machine and content), the
reassurance classifier (HY-337), coach instructions built from program state
(HY-215), the multi-turn eval runner (HY-239), the reply-quality judge
(HY-240), questionnaires and reminders.

**The imbalance:** safety is well ahead of the differentiator. Safety had to
come first, because nothing should talk to users without it. The reassurance
redirect, the thing that makes OpenCouch different, is now the priority, and
it has no evidence behind it yet.

## Where it sits

These are categories, not reviews of particular products:

- **General assistants:** free, always available, and built to answer
  everything. For worry, that makes them the reassurance machine described
  above. This is the real competitor.
- **Companion apps:** tuned for engagement and open-ended conversation.
- **Clinical tools:** sold to clinics and services. They assign exercises
  and support people between therapy sessions, with a human clinician in the
  loop.
- **Research chatbots:** therapy-style bots tested in trials, usually
  against a waitlist rather than an active alternative.

OpenCouch goes straight to the person, with no clinic in between. It is
narrow, about worry only, and refuses to reassure by design. The risk of
having no clinic is that the AI coach has to do the guiding a clinician
would otherwise do (bet 1).

## Open questions

- **Clinical content.** Which approach, which skills, in what order, over how
  long. The candidates are CBT-family treatments for worry. This needs
  clinical input, and how to get that input hasn't been decided. It is the
  main thing blocking the program (M4, M5).
- **The evidence base for GAD.** The clinical sources above are about health
  anxiety and OCD. Find the GAD-specific evidence on reassurance-seeking and
  intolerance of uncertainty before writing up HY-425.
- **Who it's for.** People with diagnosed GAD, or anyone whose worry gets in
  the way? This affects the claims, the safety population and the
  questionnaires.
- **Where.** Crisis resources are Singapore-only for now. Other regions come
  with M6.
- **Rules.** Medical-device rules depend on what the product claims to do.
  Check the companion-chatbot laws (e.g. California SB 243) before any
  launch outside a private beta.
- **Phone app.** Web first as an installable PWA. A native app is decided
  when reminders are built (HY-342).

## Research

OpenCouch can produce real findings without a clinical trial, as long as they
are about behaviour and engineering, not anxiety outcomes.

1. **The reassurance-redirect study** (HY-425), the main candidate. It
   compares four designs:
   - a plain assistant;
   - a prompt-only coach;
   - a gate in code;
   - a gate plus a redirect personalised from the user's records.

   It measures redirect recall, false redirects on real questions (including
   medical red flags), and whether the coach gives in when the user keeps
   asking. A prior-work check on 2026-11-04 found the problem argued in
   position papers and qualitative studies. It found no study that compares
   redirect designs or measures these costs. Re-check novelty with a full
   literature sweep before writing it up.
2. **Safety classifier drift.** The same questions gave about ten points more
   false alarms at WATCH between 2026-10-19 and 2026-10-23, with no change on
   our side ([evals README](../apps/agent/evals/README.md); monitoring in
   [DRIFT.md](../apps/agent/evals/crisis/DRIFT.md)). This is a short note or
   a section of a paper unless the observation window gets longer.
3. **Conversation-level risk and step-down.** It over-escalates less than
   screening each message alone, and misses nothing extra. The step-down cases
   were written by the person who tuned the rule, so they need independent
   cases before they can support a claim.

The bar for any of these:

- write a study plan before running anything;
- treat each independent scenario, not each paraphrase or rerun, as one data
  point;
- tune on one set of scenario families and report on held-out ones;
- have someone independent review the labels;
- include ordinary cases, so a system that refuses everything can't look
  good;
- release the code, cases and recordings.

## What success looks like

- **End of M5:** one person can go through the weeks 1–2 loop end to end, and
  HY-425 has results on held-out scenarios.
- **Private beta:** a small group uses the loop for several weeks. Before it
  starts, decide on targets for:
  - practice completion;
  - whether people keep practising after being redirected;
  - drop-off;
  - GAD-7 change, which without a comparison group is description, not
    evidence.
- **Later:** a clinical partner, a comparison against a simpler alternative
  (a workbook, or the coach without the redirect), and a preprint.

## Set aside

Parked after the pivot from a companion on 2026-10-23, kept rather than
cancelled:

- long-term inferred memory;
- grounded web search, because searching symptoms feeds reassurance-seeking;
- an open-ended companion mode;
- voice.

The reasons are in #356.

## Sources

- Salkovskis, P. M., & Warwick, H. M. C. (1986). Morbid preoccupations,
  health anxiety and reassurance. *Behaviour Research and Therapy*, 24(5),
  597–602.
- Kobori, O., & Salkovskis, P. M. (2013). Patterns of reassurance seeking and
  reassurance-related behaviours in OCD and anxiety disorders. *Behavioural
  and Cognitive Psychotherapy*.
- Barkhuff (2026). Reassurance Robots: OCD in the Age of Generative AI.
  arXiv:2602.19401.
- Golden & Aboujaoude (2026). A transdiagnostic model for how general purpose
  AI chatbots can perpetuate OCD and anxiety disorders. *npj Digital
  Medicine*. doi:10.1038/s41746-026-02531-7. Not read in full.
- Weilnhammer et al. (2026). A clinically validated framework for auditing AI
  chatbot behavior in mental health interactions (SIM-VAIL). arXiv:2602.01347.
- The trial of an AI CBT app against digital workbooks (540 adults, 2026) and
  the claims about guided and unguided internet CBT are cited from secondary
  summaries. Find and read the primary papers before quoting them outside
  this repo.
