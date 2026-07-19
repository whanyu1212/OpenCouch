import React, { useState } from 'react';
import s from './ToolRegistry.module.css';

/* ================================================================
   ToolRegistry — Tool catalog + trigger flow for agent tools

   Renders:
   - A registry header strip (status, provider backends)
   - A tool card per registered tool (expandable)
   - A trigger-flow diagram inside each expanded tool showing
     the invocation pipeline (stages, system prompts, fallbacks)

   Designed to scale from 1 tool to many without redesign.
   ================================================================ */

type Provider = 'openai' | 'app' | 'none';

interface Stage {
  id: string;
  label: string;
  systemPrompt: string;
  temperature: number;
  useSearch: boolean;
  onFailure: string;
  produces: string;
}

interface Tool {
  id: string;
  name: string;
  status: 'active' | 'planned';
  triggerPath: string; // where in the runtime this fires from
  triggerCondition: string;
  providers: Provider[]; // which providers support this tool
  description: string;
  pipeline: Stage[];
  outputFields: string[];
  gracefulDegradation: string;
  file: string;
  fn: string;
  tests?: string;
}

const TOOLS: Tool[] = [
  {
    id: 'therapeutic_response_skill',
    name: 'load_therapeutic_response_skill',
    status: 'active',
    triggerPath: 'TherapeuticAgent SDK tool · Realtime voice tool',
    triggerCondition: 'ordinary non-crisis therapeutic reply needs response-style guidance',
    providers: ['app'],
    description:
      'Loads prompt-ready guidance for one response style and optional therapeutic approach. The tool is side-effect-free and gives the model the exact local skill context it should use before drafting the reply.',
    pipeline: [
      {
        id: 'select_style',
        label: 'Select style',
        systemPrompt:
          'Text uses specialist/runtime state; voice uses Realtime instructions and tool descriptions.',
        temperature: 0,
        useSearch: false,
        onFailure: 'falls back to ordinary supportive guidance only when locally allowed',
        produces: 'response_style + therapeutic_approach',
      },
      {
        id: 'render_skill',
        label: 'Render skill context',
        systemPrompt:
          'Render the response-style prompt fragment and approach overlay from reviewed local sources.',
        temperature: 0,
        useSearch: false,
        onFailure: 'schema/tool error surfaces to the caller',
        produces: 'TherapeuticResponseSkillToolResult',
      },
    ],
    outputFields: [
      'context.tool_results.therapeutic_response_skill',
      'response_style',
      'therapeutic_approach',
    ],
    gracefulDegradation:
      'The tool has no side effects. Invalid styles or schema failures surface instead of silently loading unrelated guidance.',
    file: 'agent/tools/therapeutic.py',
    fn: 'execute_therapeutic_response_skill_tool',
  },
  {
    id: 'memory_control',
    name: 'memory control tools',
    status: 'active',
    triggerPath: 'memory-control text branch · persistent Realtime voice tools',
    triggerCondition: 'user explicitly asks to inspect or change saved memory state',
    providers: ['app'],
    description:
      'Reads or mutates saved memory through app-owned memory-control services. Destructive actions stage a pending deletion first and require a later confirm/cancel action.',
    pipeline: [
      {
        id: 'normalize_request',
        label: 'Normalize request',
        systemPrompt:
          'Turn explicit memory commands into a structured action such as list, status, set recall, save preference, forget by index, or forget by query.',
        temperature: 0,
        useSearch: false,
        onFailure: 'validation error surfaces; no memory mutation occurs',
        produces: 'MemoryControlRequest',
      },
      {
        id: 'execute_action',
        label: 'Execute service',
        systemPrompt: '(local service execution)',
        temperature: 0,
        useSearch: false,
        onFailure: 'write failures surface; read-only failures do not mutate state',
        produces: 'MemoryToolResult / MemoryReadToolResult',
      },
    ],
    outputFields: [
      'state.memory_control',
      'state.procedural_profile',
      'response_text',
    ],
    gracefulDegradation:
      'Incognito voice rejects persistent-only memory tools. Deletes are two-step and non-idempotent confirm actions are not marked retry-safe.',
    file: 'agent/tools/memory.py',
    fn: 'execute_memory_tool_action',
  },
  {
    id: 'crisis_resource_search',
    name: 'lookup_crisis_resources',
    status: 'active',
    triggerPath: 'CrisisAgent SDK tool · Realtime voice tool',
    triggerCondition: 'crisis specialist is selected for a level 2/3 crisis response and llm_client is available',
    providers: ['openai'],
    description:
      'Surfaces verified crisis hotlines local to the user through the crisis specialist tool surface. Uses OpenAI web search grounding. The lookup chains two structured LLM calls: first classify location availability from the conversation, then search for resources with grounding enabled.',
    pipeline: [
      {
        id: 'extract_location',
        label: 'Extract location',
        systemPrompt:
          'Extract location information from mental health support conversations. Return only the location mentioned, or empty string if none.',
        temperature: 0,
        useSearch: false,
        onFailure: 'provider error retries/surfaces; no_location and location_refused are explicit statuses',
        produces: 'provided/refused/not_provided + optional location',
      },
      {
        id: 'lookup_resources',
        label: 'Search with grounding',
        systemPrompt:
          'You are a factual assistant helping to find official crisis support resources. Use your web search capability to find verified hotlines. Format: - Name | Phone | Website',
        temperature: 0,
        useSearch: true,
        onFailure: 'provider error retries/surfaces; no verified results returns an empty list status',
        produces: 'structured list with name/phone/url/region (max 5)',
      },
    ],
    outputFields: [
      'state.inferred_location',
      'state.found_resources',
      'state.resource_lookup_status',
    ],
    gracefulDegradation:
      'No-location, location-refused, and no-verified-result cases return explicit statuses. Missing LLM configuration or provider failures retry or surface through the runtime instead of silently continuing.',
    file: 'agent/tools/grounded_search.py',
    fn: 'execute_crisis_resource_lookup_tool',
    tests: 'tests/unit/runtime/test_grounded_search_crisis_resources.py',
  },
  {
    id: 'grounded_lookup',
    name: 'answer_grounded_lookup',
    status: 'active',
    triggerPath: 'grounded lookup text branch · Realtime voice tool',
    triggerCondition: 'turn triage routes an explicit factual lookup request AND llm_client is available',
    providers: ['openai'],
    description:
      'Answers explicit, non-therapeutic factual lookup requests ("look up the eligibility for…", "search for the latest guidelines on…", "verify whether X is true"). Uses provider-native search grounding via use_search=True. Returns ("answer", status) where status reports whether the answer is verified or not verified. The TherapeuticAgent does not generate an ordinary therapeutic reply on these turns — the user gets a single grounded reply with sources.',
    pipeline: [
      {
        id: 'detect_intent',
        label: 'Turn dispatch',
        systemPrompt: 'LLM structured routing decision',
        temperature: 0,
        useSearch: false,
        onFailure: 'runtime retry/error; no silent regex fallback',
        produces: 'state.grounded_lookup.query (str) when matched',
      },
      {
        id: 'grounded_answer',
        label: 'Search-grounded answer',
        systemPrompt:
          'Answer factual lookup requests using web search/grounding. Prefer official, primary, or otherwise reputable sources. Never invent facts, contact details, eligibility rules, prices, dates, or source names. If you cannot verify the answer, say so clearly. Include a short Sources: list when available.',
        temperature: 0,
        useSearch: true,
        onFailure: 'provider error retries/surfaces; weak or missing sources return no_verified_answer',
        produces: 'str — concise answer with Sources: section',
      },
      {
        id: 'verify',
        label: 'Verification check',
        systemPrompt: '(local normalization — no user-facing claims added)',
        temperature: 0,
        useSearch: false,
        onFailure: 'flags answer as no_verified_answer',
        produces: 'FactualLookupStatus literal',
      },
    ],
    outputFields: [
      'state.response_text',
      'state.grounded_lookup.status',
    ],
    gracefulDegradation:
      'Weak or missing sources produce an explicit "I couldn\'t verify that" reply rather than an invented answer. Missing LLM configuration or provider failures retry or surface. The status field (answered / no_verified_answer) drives observability.',
    file: 'agent/tools/grounded.py',
    fn: 'execute_grounded_lookup_tool',
  },
  {
    id: 'guided_exercise',
    name: 'guided exercise tools',
    status: 'active',
    triggerPath: 'GuidedExerciseAgent SDK tools · Realtime voice tools',
    triggerCondition: 'user explicitly asks for a structured practice or an active exercise needs lifecycle handling',
    providers: ['app'],
    description:
      'Discovers channel-appropriate exercises and loads runtime-selected exercise skill context. Text progress is handled by the app-owned guided-exercise lifecycle; voice records progress through a server-validated Realtime tool because it does not run the text lifecycle before speaking.',
    pipeline: [
      {
        id: 'discover',
        label: 'Discover candidates',
        systemPrompt:
          'Filter the shared exercise catalog by channel, installed skills, and therapeutic approach.',
        temperature: 0,
        useSearch: false,
        onFailure: 'returns no available exercises rather than inventing one',
        produces: 'GuidedExerciseSkillDiscoveryToolResult',
      },
      {
        id: 'load_skill',
        label: 'Load skill',
        systemPrompt:
          'Render the runtime-selected exercise step and action into prompt-ready guidance.',
        temperature: 0,
        useSearch: false,
        onFailure: 'invalid exercise/action surfaces to the caller',
        produces: 'GuidedExerciseSkillToolResult',
      },
      {
        id: 'record_voice_progress',
        label: 'Record voice progress',
        systemPrompt:
          'Validate expected skill/step and compute the next runtime-owned exercise_state delta for Realtime voice.',
        temperature: 0,
        useSearch: false,
        onFailure: 'state mismatch surfaces; active exercise is not advanced',
        produces: 'GuidedExerciseProgressToolResult',
      },
    ],
    outputFields: [
      'state.exercise_state',
      'state.response_style',
      'diagnostics.guided_exercise',
    ],
    gracefulDegradation:
      'The catalog is validated at import time. Voice can only receive exercises marked as voice-suitable or available for the voice channel.',
    file: 'agent/tools/guided_exercise.py',
    fn: 'execute_guided_exercise_skill_tool / execute_guided_exercise_progress_tool',
  },
];

const PROVIDER_META: Record<Provider, { label: string; color: string }> = {
  openai: { label: 'OpenAI', color: 'providerOpenai' },
  app: { label: 'App-owned', color: 'providerAnthropic' },
  none: { label: '—', color: 'providerNone' },
};

function ToolCard({ tool, expanded, onToggle }: {
  tool: Tool;
  expanded: boolean;
  onToggle: () => void;
}) {
  const isActive = tool.status === 'active';

  return (
    <article className={`${s.toolCard} ${expanded ? s.toolCardExpanded : ''}`}>
      <button
        className={s.toolHeader}
        onClick={onToggle}
        aria-expanded={expanded}
      >
        {/* Status dot */}
        <span className={`${s.statusDot} ${isActive ? s.statusActive : s.statusPlanned}`} />

        {/* Name */}
        <div className={s.toolHeaderLeft}>
          <h3 className={s.toolName}>{tool.name}</h3>
          <span className={s.toolPath}>{'\u25B8'} fires from {tool.triggerPath}</span>
        </div>

        {/* Providers */}
        <div className={s.toolProviders}>
          {tool.providers.map((p) => (
            <span key={p} className={`${s.providerChip} ${s[PROVIDER_META[p].color]}`}>
              {PROVIDER_META[p].label}
            </span>
          ))}
        </div>

        {/* Chevron */}
        <span className={`${s.chevron} ${expanded ? s.chevronOpen : ''}`}>
          {'\u25BE'}
        </span>
      </button>

      {expanded && (
        <div className={s.toolBody}>
          {/* Trigger */}
          <div className={s.metaRow}>
            <span className={s.metaKey}>trigger</span>
            <span className={s.metaVal}>{tool.triggerCondition}</span>
          </div>

          {/* Description */}
          <p className={s.description}>{tool.description}</p>

          {/* Pipeline stages */}
          <div className={s.pipelineLabel}>Pipeline</div>
          <div className={s.pipeline}>
            {tool.pipeline.map((stage, idx) => (
              <React.Fragment key={stage.id}>
                <StageBox stage={stage} idx={idx + 1} />
                {idx < tool.pipeline.length - 1 && (
                  <div className={s.pipeArrow}>{'\u2192'}</div>
                )}
              </React.Fragment>
            ))}
          </div>

          {/* Outputs */}
          <div className={s.outputBlock}>
            <span className={s.outputLabel}>Writes to</span>
            <div className={s.outputFields}>
              {tool.outputFields.map((f) => (
                <code key={f} className={s.outputField}>{f}</code>
              ))}
            </div>
          </div>

          {/* Graceful degradation */}
          <div className={s.degradation}>
            <span className={s.degradationLabel}>{'\u26A0'} On failure</span>
            <span className={s.degradationText}>{tool.gracefulDegradation}</span>
          </div>

          {/* Fingerprint */}
          <footer className={s.fingerprint}>
            <span className={s.fpFile}>{tool.file}</span>
            <span className={s.fpDivider}>{'::'}</span>
            <span className={s.fpFn}>{tool.fn}</span>
            {tool.tests && (
              <>
                <span className={s.fpSep}>{'\u00B7'}</span>
                <span className={s.fpTests}>{tool.tests}</span>
              </>
            )}
          </footer>
        </div>
      )}
    </article>
  );
}

function StageBox({ stage, idx }: { stage: Stage; idx: number }) {
  return (
    <div className={s.stage}>
      <div className={s.stageHeader}>
        <span className={s.stageIdx}>{String(idx).padStart(2, '0')}</span>
        <span className={s.stageLabel}>{stage.label}</span>
      </div>
      <div className={s.stageProps}>
        <div className={s.stageProp}>
          <span className={s.stagePropKey}>system</span>
          <span className={s.stagePropVal}>{stage.systemPrompt}</span>
        </div>
        <div className={s.stagePropInline}>
          <span className={`${s.stagePill} ${stage.useSearch ? s.stagePillGrounding : s.stagePillDeterministic}`}>
            {stage.useSearch ? 'use_search=true' : stage.systemPrompt.startsWith('(') ? 'deterministic' : 'use_search=false'}
          </span>
          <span className={s.stagePillNeutral}>temp={stage.temperature}</span>
        </div>
        <div className={s.stageProp}>
          <span className={s.stagePropKey}>produces</span>
          <span className={s.stagePropVal}><code className={s.stageCode}>{stage.produces}</code></span>
        </div>
        <div className={s.stageProp}>
          <span className={s.stagePropKey}>on failure</span>
          <span className={s.stagePropValMuted}>{stage.onFailure}</span>
        </div>
      </div>
    </div>
  );
}

export default function ToolRegistry(): React.JSX.Element {
  const [expanded, setExpanded] = useState<string | null>(TOOLS[0]?.id ?? null);

  const activeCount = TOOLS.filter((t) => t.status === 'active').length;
  const plannedCount = TOOLS.filter((t) => t.status === 'planned').length;

  return (
    <div className={s.root}>
      {/* Registry header */}
      <header className={s.registryHeader}>
        <div className={s.registryTitle}>
          <span className={s.titleBrand}>TOOLS</span>
          <span className={s.titleSub}>agent-invokable capabilities</span>
        </div>
        <div className={s.registryStats}>
          <div className={s.regStat}>
            <span className={s.regStatDot + ' ' + s.statusActive} />
            <span className={s.regStatValue}>{activeCount}</span>
            <span className={s.regStatLabel}>active</span>
          </div>
          <div className={s.regStat}>
            <span className={s.regStatDot + ' ' + s.statusPlanned} />
            <span className={s.regStatValue}>{plannedCount}</span>
            <span className={s.regStatLabel}>planned</span>
          </div>
        </div>
      </header>

      {/* Invocation pattern note */}
      <div className={s.patternNote}>
        <span className={s.patternNoteLabel}>pattern</span>
        <span className={s.patternNoteText}>
          Tools in OpenCouch are <strong>app-owned</strong>. Text binds them as
          SDK function tools; voice binds a subset as Realtime function schemas.
          Both transports call the same backend services for state mutation,
          memory reads, grounded lookup, crisis resources, and exercise progress.
        </span>
      </div>

      {/* Tool cards */}
      <div className={s.toolList}>
        {TOOLS.map((tool) => (
          <ToolCard
            key={tool.id}
            tool={tool}
            expanded={expanded === tool.id}
            onToggle={() => setExpanded(expanded === tool.id ? null : tool.id)}
          />
        ))}

        {/* Placeholder slot for future tools */}
        <article className={s.toolCardGhost}>
          <div className={s.ghostHeader}>
            <span className={s.ghostDot} />
            <span className={s.ghostLabel}>next tool</span>
            <span className={s.ghostHint}>
              future candidates: structured assessment lookup, clinician-reviewed
              resource directory, session pacing and availability stubs
            </span>
          </div>
        </article>
      </div>
    </div>
  );
}
