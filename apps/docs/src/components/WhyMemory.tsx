import React from 'react';
import s from './WhyMemory.module.css';

/* ================================================================
   WhyMemory — Visual "Why Memory Matters" component

   Four sections:
   1. Side-by-side contrast: with memory vs without
   2. Three notebook cards showing what each layer stores
   3. Safety principles strip
   4. Compact flow diagram
   ================================================================ */

function ContrastPanel() {
  return (
    <div className={s.contrast}>
      {/* Without memory */}
      <div className={s.contrastPane}>
        <div className={s.contrastLabel}>
          <span className={s.labelDotRed} />
          Without memory
        </div>
        <div className={`${s.chatBubble} ${s.userBubble}`}>
          I had a rough day at work again.
        </div>
        <div className={`${s.chatBubble} ${s.agentBubble}`}>
          I'm sorry to hear that. Can you tell me more about your work situation?
        </div>
        <div className={`${s.chatBubble} ${s.userBubble}`}>
          I already told you about my boss last week...
        </div>
        <div className={`${s.chatBubble} ${s.agentBubble}`}>
          I understand. What's been happening with your boss?
        </div>
      </div>

      {/* With memory */}
      <div className={s.contrastPane}>
        <div className={s.contrastLabel}>
          <span className={s.labelDotGreen} />
          With memory
        </div>
        <div className={`${s.chatBubble} ${s.userBubble}`}>
          I had a rough day at work again.
        </div>
        <div className={`${s.chatBubble} ${s.agentBubbleGood}`}>
          Last time you mentioned <span className={s.bubbleHighlight}>your boss being
          difficult about deadlines</span>. Was it the same issue today, or
          something new?
        </div>
        <div className={`${s.chatBubble} ${s.userBubble}`}>
          Same thing, but worse now.
        </div>
        <div className={`${s.chatBubble} ${s.agentBubbleGood}`}>
          That sounds exhausting. The <span className={s.bubbleHighlight}>breathing
          exercise helped last session</span> — would you like to try it
          again, or would you rather just talk through it?
        </div>
      </div>
    </div>
  );
}

function NotebookCards() {
  return (
    <div className={s.notebooks}>
      {/* Semantic */}
      <div className={s.notebook}>
        <div className={s.notebookHeader}>
          <span className={s.notebookIcon}>{'\u25C6'}</span>
          <div>
            <div className={s.notebookTitle}>Fact Notebook</div>
            <div className={s.notebookSub}>semantic memory</div>
          </div>
        </div>
        <div className={s.notebookBody}>
          <div className={s.notebookEntry}>
            <span className={`${s.entryLabel} ${s.labelSemantic}`}>relationship</span>
            User worries about work — "my boss is terrible"
          </div>
          <div className={s.notebookEntry}>
            <span className={`${s.entryLabel} ${s.labelSemantic}`}>coping</span>
            User uses fluoxetine — "I take it daily"
          </div>
          <div className={s.notebookEntry}>
            <span className={`${s.entryLabel} ${s.labelSemantic}`}>relationship</span>
            User knows Sarah — "my sister visited last week"
          </div>
        </div>
      </div>

      {/* Episodic */}
      <div className={s.notebook}>
        <div className={s.notebookHeader}>
          <span className={s.notebookIcon}>{'\u25CB'}</span>
          <div>
            <div className={s.notebookTitle}>Session Diary</div>
            <div className={s.notebookSub}>episodic memory</div>
          </div>
        </div>
        <div className={s.notebookBody}>
          <div className={s.notebookEntry}>
            <span className={`${s.entryLabel} ${s.labelEpisodic}`}>apr 18</span>
            Panic attack at work. Tried box breathing. Mood: anxious {'\u2192'} calmer.
          </div>
          <div className={s.notebookEntry}>
            <span className={`${s.entryLabel} ${s.labelEpisodic}`}>apr 15</span>
            Sleep trouble and the move. Open loop: apartment search.
          </div>
          <div className={s.notebookEntry}>
            <span className={`${s.entryLabel} ${s.labelEpisodic}`}>apr 12</span>
            Work stress, argument with partner. Resolved: apologized.
          </div>
        </div>
      </div>

      {/* Procedural */}
      <div className={s.notebook}>
        <div className={s.notebookHeader}>
          <span className={s.notebookIcon}>{'\u25A0'}</span>
          <div>
            <div className={s.notebookTitle}>Style Guide</div>
            <div className={s.notebookSub}>procedural memory</div>
          </div>
        </div>
        <div className={s.notebookBody}>
          <div className={s.notebookEntry}>
            <span className={`${s.entryLabel} ${s.labelProcedural}`}>rule</span>
            Don't suggest meditation — makes user more anxious.
          </div>
          <div className={s.notebookEntry}>
            <span className={`${s.entryLabel} ${s.labelProcedural}`}>rule</span>
            Keep responses short and direct.
          </div>
          <div className={s.notebookEntry}>
            <span className={`${s.entryLabel} ${s.labelProcedural}`}>rule</span>
            Avoid phrases like "I understand" — feels dismissive.
          </div>
        </div>
      </div>
    </div>
  );
}

function SafetyStrip() {
  return (
    <div className={s.safetyStrip}>
      <div className={s.safetyCard}>
        <span className={s.safetyIcon}>{'\uD83D\uDD12'}</span>
        <div className={s.safetyTitle}>You own your data</div>
        <div className={s.safetyDesc}>
          Inspect, delete, or wipe anything. No hidden stores.
        </div>
      </div>
      <div className={s.safetyCard}>
        <span className={s.safetyIcon}>{'\uD83D\uDEE1\uFE0F'}</span>
        <div className={s.safetyTitle}>Private by default</div>
        <div className={s.safetyDesc}>
          Guest mode stores nothing. Persistent mode keeps data local on your machine.
        </div>
      </div>
      <div className={s.safetyCard}>
        <span className={s.safetyIcon}>{'\u26A1'}</span>
        <div className={s.safetyTitle}>Safety overrides memory</div>
        <div className={s.safetyDesc}>
          Crisis responses ignore all style rules. Safety is hardcoded.
        </div>
      </div>
    </div>
  );
}

function FlowDiagram() {
  const steps = [
    { icon: '\uD83D\uDCAC', label: 'You speak' },
    { icon: '\uD83D\uDEE1\uFE0F', label: 'Safety check' },
    { icon: '\uD83E\uDDE0', label: 'Load memory' },
    { icon: '\u2728', label: 'Generate reply' },
    { icon: '\uD83D\uDCDD', label: 'Learn from turn' },
    { icon: '\uD83D\uDCDA', label: 'Session end' },
  ];

  return (
    <div className={s.flow}>
      {steps.map((step, i) => (
        <React.Fragment key={step.label}>
          {i > 0 && <span className={s.flowArrow}>{'\u2192'}</span>}
          <div className={s.flowStep}>
            <span className={s.flowStepIcon}>{step.icon}</span>
            <span className={s.flowStepLabel}>{step.label}</span>
          </div>
        </React.Fragment>
      ))}
    </div>
  );
}

export default function WhyMemory(): React.JSX.Element {
  return (
    <div className={s.root}>
      <ContrastPanel />
      <NotebookCards />
      <SafetyStrip />
      <FlowDiagram />
    </div>
  );
}
