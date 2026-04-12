import React, { useState } from 'react';
import styles from './ScenarioReplay.module.css';

/* ── Types ──────────────────────────────────────────────────────────────────── */

type ModeType = 'therapeutic' | 'crisis';

interface Turn {
  user: string;
  agent: string;
  route: string;
  mode: string;
  modeType: ModeType;
  why: string;
}

interface Scenario {
  id: string;
  title: string;
  subtitle: string;
  turns: Turn[];
}

/* ── Scenario data ──────────────────────────────────────────────────────────── */
/*
 * Updated 2026-04-12 to match the actual shipped graph topology.
 *
 * Mode names match the 6 therapeutic subgraph nodes: supportive,
 * reflective, clarifying, psychoeducation, guided_exercise, closing.
 * Plus the crisis-gate outcomes: crisis_response, safety_check.
 *
 * "orientation", "out_of_scope", "realignment", "pattern_reflection",
 * and "supportive_conversation" were v0.1 design-doc names that don't
 * exist as graph nodes. Removed or mapped to their actual equivalents.
 *
 * Session stages (opening, deepening, stabilizing, closing) are
 * currently static — the state field exists but dynamic inference is
 * not wired. Removed from the turn metadata to avoid showing
 * behavior the system doesn't actually produce.
 *
 * Named intents (guided_cbt_work, just_need_to_vent, etc.) are not
 * a dispatcher concept in the shipped code. The dispatcher uses
 * inline regex + LLM classification, not sticky intent labels.
 * Removed from turn metadata.
 */

const SCENARIOS: Scenario[] = [
  {
    id: 'new_user',
    title: 'New user journey',
    subtitle: 'clarifying → support → exercise',
    turns: [
      {
        user: "Hi, I'm not really sure how this works. Is this like therapy?",
        agent: "Welcome — I'm glad you're here. OpenCouch isn't therapy, and I'm not a therapist. Think of me as a support tool: I can help you talk through what's on your mind, try grounding exercises, or reflect on patterns. What's bringing you here today?",
        route: 'crisis_gate(safe) → therapeutic_subgraph → clarifying',
        mode: 'clarifying', modeType: 'therapeutic',
        why: 'First turn with no history. "How does this work" is ambiguous — the dispatcher asks for context before picking a therapeutic direction.',
      },
      {
        user: "I've been really stressed about work lately. My manager keeps piling things on and I can't say no.",
        agent: "That sounds exhausting — carrying more than your share with no room to push back. It makes sense that the stress is building. What feels most heavy about it right now?",
        route: 'crisis_gate(safe) → therapeutic_subgraph → supportive',
        mode: 'supportive', modeType: 'therapeutic',
        why: 'General distress, no exercise or reflection pattern. Dispatcher picks supportive as the default for emotional sharing.',
      },
      {
        user: "Can we try a grounding exercise? I feel like I need to just calm down.",
        agent: "Let's do that. Try the 5-4-3-2-1 technique with me: name 5 things you can see right now. Take your time — there's no rush.",
        route: 'crisis_gate(safe) → therapeutic_subgraph → guided_exercise',
        mode: 'guided_exercise', modeType: 'therapeutic',
        why: 'Explicit exercise request: "grounding exercise" + "calm down" matched the guided_exercise regex fast path.',
      },
    ],
  },
  {
    id: 'crisis',
    title: 'Crisis escalation',
    subtitle: 'support → safety check → crisis response',
    turns: [
      {
        user: "I've been having a really rough week. Everything feels like it's falling apart.",
        agent: "I hear you — when everything hits at once, it can feel genuinely overwhelming. You don't have to sort through all of it right now. What feels most pressing?",
        route: 'crisis_gate(safe) → therapeutic_subgraph → supportive',
        mode: 'supportive', modeType: 'therapeutic',
        why: 'General distress, no crisis signals. Default supportive mode.',
      },
      {
        user: "Sometimes I wonder if anyone would even notice if I wasn't here anymore.",
        agent: "I want to pause and check on something important. When you say you wonder if anyone would notice if you weren't here — are you having thoughts of harming yourself or ending your life?",
        route: 'crisis_gate(level 1, needs_clarification) → crisis_response_node',
        mode: 'safety_check', modeType: 'crisis',
        why: 'Ambiguous concerning language. Crisis gate set needs_clarification=true. Routed to crisis_response_node which generates a safety check before proceeding.',
      },
      {
        user: "I've been thinking about ending things. I have pills saved up. I was going to do it this weekend.",
        agent: "Thank you for telling me. I take this seriously and I want to make sure you're safe right now. Please contact the 988 Suicide & Crisis Lifeline by calling or texting 988, or go to your nearest emergency room. Is there someone you trust who can be with you right now?",
        route: 'crisis_gate(level 3, imminent_risk override) → crisis_response_node',
        mode: 'crisis_response', modeType: 'crisis',
        why: 'Plan + means + timeline detected. Level 3 imminent-risk override fires in the deterministic regex tier. All normal conversation stops.',
      },
    ],
  },
  {
    id: 'guided_work',
    title: 'Guided exercise session',
    subtitle: 'structured work with wrap-up',
    turns: [
      {
        user: "Can we do a thought record? I keep catastrophizing about a presentation I have next week.",
        agent: "Absolutely. Let's slow it down. What's the specific situation? Tell me about the presentation — when is it, and what's the thought that keeps showing up.",
        route: 'crisis_gate(safe) → therapeutic_subgraph → guided_exercise',
        mode: 'guided_exercise', modeType: 'therapeutic',
        why: 'Explicit "thought record" pattern matched the guided_exercise regex fast path.',
      },
      {
        user: "It's a team update on Friday. The thought is 'everyone will think I'm incompetent and I'll get fired.'",
        agent: "Okay — so the hot thought is \"everyone will think I'm incompetent and I'll get fired.\" Before we look at the evidence, what emotion comes with that thought? And how strong does it feel, roughly 0 to 10?",
        route: 'crisis_gate(safe) → therapeutic_subgraph → guided_exercise',
        mode: 'guided_exercise', modeType: 'therapeutic',
        why: 'Continuing structured exercise. The dispatcher detects the active exercise via routing.exercise_type and stays in guided_exercise.',
      },
      {
        user: "That actually helps to see it written out. What would a more balanced thought be?",
        agent: "Based on the evidence: \"The presentation might not go perfectly, but my team knows my work and one update won't define my whole reputation.\" How does that land — does it feel realistic?",
        route: 'crisis_gate(safe) → therapeutic_subgraph → guided_exercise',
        mode: 'guided_exercise', modeType: 'therapeutic',
        why: 'Still in the exercise. Step tracking keeps the agent in the same mode until the user signals completion.',
      },
      {
        user: "I think I'm good for today. Thanks for walking me through that.",
        agent: "Good work. The key things: you caught the catastrophizing thought, examined the evidence, and landed on a more balanced take. Noticing the thought is the hardest part — and you did that. Take care.",
        route: 'crisis_gate(safe) → therapeutic_subgraph → closing',
        mode: 'closing', modeType: 'therapeutic',
        why: '"I\'m good for today" + "thanks" matched the closing regex fast path. Warm wrap-up with summary, no new exploration opened.',
      },
    ],
  },
  {
    id: 'psychoed',
    title: 'Psychoeducation → reflection',
    subtitle: 'understanding → noticing → exploring',
    turns: [
      {
        user: "My chest gets tight when I'm anxious and I don't understand what's happening to me. Why does this happen?",
        agent: "That tightness is your nervous system doing its job — just at the wrong time. When your brain detects a threat (even a social one), it triggers a fight-or-flight response: adrenaline, faster heartbeat, tightened muscles. Your chest muscles tense, your breathing shallows. It's not dangerous, but it feels awful.",
        route: 'crisis_gate(safe) → therapeutic_subgraph → psychoeducation',
        mode: 'psychoeducation', modeType: 'therapeutic',
        why: 'User describes a specific reaction (chest tightness) AND asks for understanding ("why does this happen"). Both pieces present — matches the psychoeducation dispatcher rule.',
      },
      {
        user: "That makes sense. I notice it happens every Sunday night before work starts on Monday.",
        agent: "That's a really useful observation — the pattern is tied to anticipation, not the work itself. Your body is responding to the *idea* of Monday before it even arrives. Do you notice it peaks at a particular moment on Sunday, or is it more of a slow build?",
        route: 'crisis_gate(safe) → therapeutic_subgraph → reflective',
        mode: 'reflective', modeType: 'therapeutic',
        why: '"I notice" + "happens every" signals the user is naming a recurring pattern. Dispatcher picks reflective mode to explore it.',
      },
      {
        user: "Why do I keep dreading Mondays even though nothing bad actually happens at work?",
        agent: "That's a really common pattern — anticipatory anxiety often has nothing to do with the actual outcome. Your brain learned \"Sunday night = dread\" and now replays it automatically, even when the evidence says Monday is usually fine. The dread feels real, but it's running on autopilot, not on data.",
        route: 'crisis_gate(safe) → therapeutic_subgraph → reflective',
        mode: 'reflective', modeType: 'therapeutic',
        why: '"Why do I keep" — the user has already named the pattern and is inviting deeper exploration. Reflective continues.',
      },
    ],
  },
  {
    id: 'grief_long',
    title: 'Grief across a full session',
    subtitle: '7 turns of support and reflection',
    turns: [
      {
        user: "My dad died three weeks ago. I don't really know why I'm here.",
        agent: "I'm sorry about your dad. You don't need a reason to be here — just showing up is enough. There's no right way to do this. What feels most present for you right now?",
        route: 'crisis_gate(safe) → therapeutic_subgraph → supportive',
        mode: 'supportive', modeType: 'therapeutic',
        why: 'Grief language detected. No exercise or reflection request. Supportive mode holds space.',
      },
      {
        user: "I keep swinging between feeling completely numb and then crying in the car on the way to work. Is that normal?",
        agent: "That's very normal — grief doesn't move in a straight line. The numbness and the waves of sadness can alternate for a while, sometimes within the same hour. Your mind is processing something enormous, and it does it in bursts, not all at once.",
        route: 'crisis_gate(safe) → therapeutic_subgraph → psychoeducation',
        mode: 'psychoeducation', modeType: 'therapeutic',
        why: '"Is that normal" + grief context — user describes a reaction and seeks understanding. Psychoeducation provides normalizing framing.',
      },
      {
        user: "Everyone keeps telling me to be strong. But I'm not strong. I'm just going through the motions.",
        agent: "\"Be strong\" is one of those things people say when they don't know what else to offer. It's not a standard you need to meet. Going through the motions is what getting through looks like right now — and that's enough.",
        route: 'crisis_gate(safe) → therapeutic_subgraph → supportive',
        mode: 'supportive', modeType: 'therapeutic',
        why: 'Emotional sharing, no question or pattern being named. Supportive mode validates without redirecting.',
      },
      {
        user: "I feel guilty too. We had an argument the week before he died and I never got to apologize.",
        agent: "That weight — the unfinished conversation — is one of the hardest parts of losing someone. The guilt doesn't mean you did something wrong. It means the relationship mattered. You don't have to resolve that tonight.",
        route: 'crisis_gate(safe) → therapeutic_subgraph → supportive',
        mode: 'supportive', modeType: 'therapeutic',
        why: 'Continuing support. Guilt + grief. No reframing, no silver linings — holding space.',
      },
      {
        user: "I notice I keep replaying the argument. Like my brain won't let it go. Why does it do that?",
        agent: "Rumination after loss is your brain trying to rewrite the ending. It replays the scene looking for a version where you said the right thing, even though that version doesn't exist. It's not a character flaw — it's grief trying to problem-solve something that can't be solved.",
        route: 'crisis_gate(safe) → therapeutic_subgraph → reflective',
        mode: 'reflective', modeType: 'therapeutic',
        why: '"I notice I keep" — the user is naming a recurring pattern (rumination). "Why does it do that" invites exploration of the pattern.',
      },
      {
        user: "That actually makes sense. I think I've been beating myself up for not being able to stop thinking about it.",
        agent: "And now you have a different frame for it — the replaying isn't something broken in you, it's grief doing its (frustrating) work. That shift from 'why can't I stop' to 'oh, this is what grief does' is real progress, even if it doesn't feel like it yet.",
        route: 'crisis_gate(safe) → therapeutic_subgraph → supportive',
        mode: 'supportive', modeType: 'therapeutic',
        why: '"That actually makes sense" — consolidation. User is integrating the insight. Supportive reflects it back.',
      },
      {
        user: "I think I need to stop for today. This was a lot.",
        agent: "That was a lot, and you stayed with it. The main thing: the numbness and the waves are normal, the guilt about the argument doesn't mean you failed him, and the replaying is grief trying to fix something unfixable. Take care of yourself tonight.",
        route: 'crisis_gate(safe) → therapeutic_subgraph → closing',
        mode: 'closing', modeType: 'therapeutic',
        why: '"Need to stop for today" matched the closing regex fast path. Summary + gentle landing. No new exploration opened.',
      },
    ],
  },
  {
    id: 'venting',
    title: 'Just need to vent',
    subtitle: 'hold space — no advice, no exercises',
    turns: [
      {
        user: "I don't want advice or exercises. I just need to get this out.",
        agent: "Go ahead — I'm here. You don't need to organize it or make it make sense. Just say what needs to be said.",
        route: 'crisis_gate(safe) → therapeutic_subgraph → supportive',
        mode: 'supportive', modeType: 'therapeutic',
        why: 'Emotional sharing with an explicit "don\'t want advice" signal. Supportive mode holds space without pivoting to structure.',
      },
      {
        user: "My sister called me selfish for not visiting mum more. She has no idea what my life looks like right now. I work 60 hours a week and I'm barely keeping it together.",
        agent: "That's a lot to carry — the work, the pressure from your sister, and the guilt underneath it. You don't sound selfish. You sound stretched thin.",
        route: 'crisis_gate(safe) → therapeutic_subgraph → supportive',
        mode: 'supportive', modeType: 'therapeutic',
        why: 'Continuing support. Venting — reflect without redirecting.',
      },
      {
        user: "And then mum says 'your sister's just worried about me' like I'm the problem. I can't win.",
        agent: "Caught between both of them, and no version of you is enough for either. That 'I can't win' feeling — it sounds exhausting.",
        route: 'crisis_gate(safe) → therapeutic_subgraph → supportive',
        mode: 'supportive', modeType: 'therapeutic',
        why: 'Continued venting. No exercise request, no pattern being explored. Supportive stays in hold-space mode.',
      },
      {
        user: "Yeah. Sorry, I know this is just me ranting.",
        agent: "You don't need to apologize for that. This is exactly what this is for.",
        route: 'crisis_gate(safe) → therapeutic_subgraph → supportive',
        mode: 'supportive', modeType: 'therapeutic',
        why: 'User apologizing for venting. Short validation — no pivoting to structure.',
      },
      {
        user: "I think I just needed someone to hear it without telling me what to do. Thanks.",
        agent: "Heard. And for what it's worth — you're managing a lot more than most people see.",
        route: 'crisis_gate(safe) → therapeutic_subgraph → closing',
        mode: 'closing', modeType: 'therapeutic',
        why: '"Thanks" + natural ending signal matched the closing regex fast path. Brief close without homework or next steps.',
      },
    ],
  },
  {
    id: 'multi_mode',
    title: 'Multi-mode session',
    subtitle: 'support → psychoeducation → exercise → reflection → closing',
    turns: [
      {
        user: "I've been having panic attacks and I don't know what to do about them.",
        agent: "That sounds really frightening. Panic attacks can feel overwhelming, especially when you don't know what's triggering them. When did the last one happen, and what was it like?",
        route: 'crisis_gate(safe) → therapeutic_subgraph → supportive',
        mode: 'supportive', modeType: 'therapeutic',
        why: 'General distress about panic. No specific request for explanation or exercise. Default supportive.',
      },
      {
        user: "Yesterday at the grocery store. My heart was racing, I couldn't breathe, I thought I was dying. What's actually happening when that happens?",
        agent: "During a panic attack, your amygdala fires a false alarm — it sends the same signals as if you were in real danger. Your heart rate spikes, breathing shallows, muscles tense, and adrenaline floods your system. The 'I'm dying' feeling is your brain misinterpreting those intense body signals. It peaks in about 10 minutes and always passes, even though it doesn't feel like it will.",
        route: 'crisis_gate(safe) → therapeutic_subgraph → psychoeducation',
        mode: 'psychoeducation', modeType: 'therapeutic',
        why: 'User describes a specific reaction (heart racing, couldn\'t breathe) AND asks "what\'s actually happening." Both pieces present for psychoeducation.',
      },
      {
        user: "Okay that helps to know. Is there something I can do when it's actually happening? Like in the moment?",
        agent: "Yes — try this next time it starts. It's called physiological sighing: take a double inhale through your nose (two short sniffs), then a long slow exhale through your mouth. The double inhale reinflates your lung sacs, and the long exhale activates your vagus nerve to slow your heart rate. Do it three times. It's the fastest way to manually downshift your nervous system.",
        route: 'crisis_gate(safe) → therapeutic_subgraph → guided_exercise',
        mode: 'guided_exercise', modeType: 'therapeutic',
        why: '"Something I can do in the moment" — explicit exercise request matched the guided_exercise path.',
      },
      {
        user: "I notice it tends to happen in crowded places. Grocery stores, trains, concerts. Is there a pattern there?",
        agent: "There is — the common thread is environments where you feel you can't easily escape if something goes wrong. Crowded, enclosed, unpredictable. Your brain isn't afraid of the store itself — it's afraid of being trapped with the panic. That 'what if I have an attack here' thought becomes the trigger, which is why it generalizes across similar settings.",
        route: 'crisis_gate(safe) → therapeutic_subgraph → reflective',
        mode: 'reflective', modeType: 'therapeutic',
        why: '"I notice" + "is there a pattern" — user is naming a pattern and inviting exploration. Reflective mode.',
      },
      {
        user: "That makes a lot of sense actually. So it's not the places, it's the fear of the fear.",
        agent: "Exactly — 'fear of the fear' is the core of how panic disorder works. The good news is that once you see the loop, it starts to lose some of its power. The breathing technique gives you a real tool for the body side, and understanding the pattern weakens the anticipation side.",
        route: 'crisis_gate(safe) → therapeutic_subgraph → supportive',
        mode: 'supportive', modeType: 'therapeutic',
        why: '"Makes a lot of sense" — consolidation. User is integrating the insight. Supportive reflects and affirms.',
      },
      {
        user: "This was really helpful. I feel like I understand what's happening now. Can we wrap up?",
        agent: "Here's what we covered: panic attacks are false alarms from your amygdala, not actual danger. The double-inhale sighing technique can help in the moment. And the pattern you noticed — it's the fear of the fear, not the places themselves, that drives the avoidance. One thing to try this week: next time you notice the anticipation building, name it ('that's the fear of the fear') before it escalates. Take care.",
        route: 'crisis_gate(safe) → therapeutic_subgraph → closing',
        mode: 'closing', modeType: 'therapeutic',
        why: '"Can we wrap up" matched the closing regex fast path. Summary of key takeaways + one actionable next step.',
      },
    ],
  },
  {
    id: 'ambivalence',
    title: 'Ambivalence about change',
    subtitle: 'MI-flavored support — evoking, not directing',
    turns: [
      {
        user: "I know I should leave my job. Everyone tells me to. But I can't seem to actually do it.",
        agent: "There's a gap between knowing and doing — and it sounds like you're stuck right in it. Part of you sees the reasons to leave, and part of you has reasons to stay. What's keeping you there?",
        route: 'crisis_gate(safe) → therapeutic_subgraph → supportive',
        mode: 'supportive', modeType: 'therapeutic',
        why: 'Ambivalence about change. MI overlay shapes the supportive response toward reflective listening and evoking, not directing.',
      },
      {
        user: "Stability, I guess. I know what to expect. And I'm scared that if I leave, the next thing will be worse.",
        agent: "So the current job is painful but predictable, and leaving means trading known pain for unknown risk. That's not irrational — it's a real calculation. What would 'worse' actually look like for you?",
        route: 'crisis_gate(safe) → therapeutic_subgraph → supportive',
        mode: 'supportive', modeType: 'therapeutic',
        why: 'Continuing MI-flavored support. Evoking the user\'s own framing. Not pushing toward action.',
      },
      {
        user: "I guess worse would be starting over and feeling incompetent again. I worked so hard to get where I am.",
        agent: "There's a lot of identity wrapped up in this — not just a paycheck, but proof that you're competent, that the hard work mattered. Leaving feels like it might erase that, even if staying is costing you something else.",
        route: 'crisis_gate(safe) → therapeutic_subgraph → supportive',
        mode: 'supportive', modeType: 'therapeutic',
        why: 'MI reflection: naming the identity stake without pushing toward change.',
      },
      {
        user: "Yeah, exactly. But I also know I'm burnt out. I dread Mondays. I'm short with my kids. Something has to change.",
        agent: "So on one side: stability, identity, proof of competence. On the other: burnout, dreading the week, and it spilling into your family. You're holding both of those at the same time. Which side feels more urgent right now?",
        route: 'crisis_gate(safe) → therapeutic_subgraph → supportive',
        mode: 'supportive', modeType: 'therapeutic',
        why: 'MI summary of both sides. Not resolving the ambivalence — letting the user weigh it.',
      },
      {
        user: "The family side. I don't want my kids to remember me as the angry tired parent.",
        agent: "That tells you something important — when you think about what actually matters, it's not the job title. It's the version of yourself you want your kids to see. You don't have to decide about the job tonight, but that clarity about what matters is worth paying attention to.",
        route: 'crisis_gate(safe) → therapeutic_subgraph → supportive',
        mode: 'supportive', modeType: 'therapeutic',
        why: 'User articulated their own value. MI: reflecting the change talk back without prescribing action.',
      },
    ],
  },
];

/* ── Mode type colors ───────────────────────────────────────────────────────── */

const MODE_TYPE_CLS: Record<ModeType, string> = {
  therapeutic: 'mtTherapeutic',
  crisis: 'mtCrisis',
};

/* ── Helpers ─────────────────────────────────────────────────────────────────── */

function uniqueModeTypes(turns: Turn[]): ModeType[] {
  const seen = new Set<ModeType>();
  for (const t of turns) seen.add(t.modeType);
  return (['therapeutic', 'crisis'] as ModeType[]).filter(mt => seen.has(mt));
}

function uniqueModes(turns: Turn[]): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const t of turns) {
    if (!seen.has(t.mode)) { seen.add(t.mode); result.push(t.mode); }
  }
  return result;
}

/* ── Component ──────────────────────────────────────────────────────────────── */

export default function ScenarioReplay() {
  const [activeId, setActiveId] = useState<string | null>(null);
  const scenario = activeId ? SCENARIOS.find(s => s.id === activeId) ?? null : null;

  return (
    <div className={styles.root}>
      {/* Legend */}
      <div className={styles.legend}>
        <span className={styles.legendItem}><span className={[styles.legendDot, styles.mtTherapeutic].join(' ')} />therapeutic</span>
        <span className={styles.legendItem}><span className={[styles.legendDot, styles.mtCrisis].join(' ')} />crisis</span>
      </div>

      {/* Scenario card grid */}
      <div className={styles.cardGrid}>
        {SCENARIOS.map(s => {
          const mts = uniqueModeTypes(s.turns);
          const modes = uniqueModes(s.turns);
          return (
            <button
              key={s.id}
              className={[styles.card, activeId === s.id ? styles.cardActive : ''].join(' ')}
              onClick={() => setActiveId(p => p === s.id ? null : s.id)}
            >
              <div className={styles.cardTop}>
                <span className={styles.cardTitle}>{s.title}</span>
                <span className={styles.cardTurns}>{s.turns.length} turns</span>
              </div>
              <span className={styles.cardSub}>{s.subtitle}</span>
              <div className={styles.cardMeta}>
                <div className={styles.cardDots}>
                  {mts.map(mt => (
                    <span key={mt} className={[styles.cardDot, styles[MODE_TYPE_CLS[mt]]].join(' ')} title={mt} />
                  ))}
                </div>
                <div className={styles.cardModes}>
                  {modes.map(m => (
                    <span key={m} className={styles.cardMode}>{m}</span>
                  ))}
                </div>
              </div>
            </button>
          );
        })}
      </div>

      {/* Conversation turns */}
      {scenario && (
      <div className={styles.conversation} key={scenario.id}>
        {scenario.turns.map((turn, i) => (
          <div key={i} className={styles.turn}>
            {/* Turn number */}
            <div className={styles.turnMarker}>
              <span className={styles.turnNum}>{i + 1}</span>
              {i < scenario.turns.length - 1 && <div className={styles.turnLine} />}
            </div>

            {/* Chat + state */}
            <div className={styles.turnBody}>
              {/* Chat bubbles */}
              <div className={styles.chat}>
                <div className={styles.bubbleUser}>
                  <span className={styles.bubbleRole}>user</span>
                  <p className={styles.bubbleText}>{turn.user}</p>
                </div>
                <div className={styles.bubbleAgent}>
                  <span className={styles.bubbleRole}>opencouch</span>
                  <p className={styles.bubbleText}>{turn.agent}</p>
                </div>
              </div>

              {/* State strip */}
              <div className={styles.stateStrip}>
                <div className={styles.routeTrace}>
                  <span className={styles.routeLabel}>route</span>
                  <code className={styles.routePath}>{turn.route}</code>
                </div>
                <div className={styles.statePills}>
                  <span className={[styles.pill, styles[MODE_TYPE_CLS[turn.modeType]]].join(' ')}>
                    {turn.mode}
                  </span>
                  <span className={styles.pillType}>{turn.modeType}</span>
                </div>
                <p className={styles.why}>{turn.why}</p>
              </div>
            </div>
          </div>
        ))}
      </div>
      )}
    </div>
  );
}
