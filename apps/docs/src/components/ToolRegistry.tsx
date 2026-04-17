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

type Provider = 'gemini' | 'openai' | 'anthropic' | 'none';

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
  triggerPath: string; // where in the graph this fires from
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
    id: 'crisis_resource_search',
    name: 'find_local_crisis_resources',
    status: 'active',
    triggerPath: 'crisis_response_node',
    triggerCondition: 'crisis gate returns needs_crisis_response AND llm_client is available',
    providers: ['gemini', 'openai'],
    description:
      'Surfaces verified crisis hotlines local to the user. Uses provider-native web search grounding (Google Search for Gemini, web_search tool for OpenAI) — not a custom tool attachment. The call graph chains two deterministic LLM calls: first extract the user\'s location from the conversation, then search for resources with grounding enabled.',
    pipeline: [
      {
        id: 'extract_location',
        label: 'Extract location',
        systemPrompt:
          'Extract location information from mental health support conversations. Return only the location mentioned, or empty string if none.',
        temperature: 0,
        useSearch: false,
        onFailure: 'returns empty location, pipeline aborts, resources=[]',
        produces: 'str — e.g. "Singapore", "UK", ""',
      },
      {
        id: 'lookup_resources',
        label: 'Search with grounding',
        systemPrompt:
          'You are a factual assistant helping to find official crisis support resources. Use your web search capability to find verified hotlines. Format: - Name | Phone | Website',
        temperature: 0,
        useSearch: true,
        onFailure: 'logs warning, returns empty list',
        produces: 'str — pipe-separated or markdown-bold lines',
      },
      {
        id: 'parse_resources',
        label: 'Parse + normalize',
        systemPrompt: '(deterministic — no LLM)',
        temperature: 0,
        useSearch: false,
        onFailure: 'drops unparseable rows, keeps valid ones',
        produces: 'list[dict] with name/phone/url/region (max 5)',
      },
    ],
    outputFields: [
      'state.response.inferred_location',
      'state.response.found_resources',
    ],
    gracefulDegradation:
      'Any stage failure returns empty results. The crisis response proceeds without resources rather than blocking on a third-party outage.',
    file: 'agent/tools/web_search.py',
    fn: 'find_local_crisis_resources',
    tests: 'tests/test_web_search_parser.py (13 parser tests)',
  },
];

const PROVIDER_META: Record<Provider, { label: string; color: string }> = {
  gemini: { label: 'Gemini', color: 'providerGemini' },
  openai: { label: 'OpenAI', color: 'providerOpenai' },
  anthropic: { label: 'Anthropic', color: 'providerAnthropic' },
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
          Tools in OpenCouch are <strong>node-invoked</strong>, not LangGraph-registered.
          A node calls the tool function directly when its condition is met.
          Provider-native grounding (Google Search, OpenAI web_search) is enabled
          via the <code>use_search=True</code> kwarg on <code>generate_text()</code>.
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
              future candidates: session-arc summarizer, structured assessment lookup,
              skill-library retrieval
            </span>
          </div>
        </article>
      </div>
    </div>
  );
}
