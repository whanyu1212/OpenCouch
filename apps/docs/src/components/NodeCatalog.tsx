import React, { useState, useMemo } from 'react';
import s from './NodeCatalog.module.css';

/* ================================================================
   NodeCatalog — Technical spec sheet for every graph node

   Each node is rendered as a component datasheet card:
   - Classification strip (SAFETY, MEMORY, ROUTING, etc.)
   - Function signature (inputs -> outputs)
   - Policy chips (retry, reducer, parallel, LLM)
   - Skip conditions for gated nodes
   - File fingerprint + line range

   Filterable by category and capability.
   ================================================================ */

type Category = 'SAFETY' | 'MEMORY' | 'ROUTING' | 'EXTRACTION' | 'TERMINAL';

interface NodeSpec {
  id: string;
  name: string;
  category: Category;
  order: number; // execution position for sorting

  inputs: string[];
  outputs: string[];

  retry: boolean;
  llm: boolean;
  reducer?: string; // if node writes to a reducer-backed field
  parallel: boolean;
  subgraph: boolean;

  description: string;
  skipConditions?: string[];

  file: string;
  fn: string;
}

const NODES: NodeSpec[] = [
  {
    id: 'crisis_gate',
    name: 'crisis_gate_node',
    category: 'SAFETY',
    order: 1,
    inputs: ['state.message', 'state.history[-6:]'],
    outputs: ['state.crisis', 'state.route', 'state.crisis_audit', 'state.diagnostics'],
    retry: true,
    llm: true,
    parallel: false,
    subgraph: false,
    description:
      'Hard safety boundary. Runs BEFORE memory retrieval. LLM-primary classification: (1) deterministic override (imminent risk, idiomatic-safe, safety denial), (2) LLM structured classifier as the primary path, (3) deterministic regex ladder as fallback, (4) truth-table normalization. Returns Command(goto=...) that routes between the crisis and operational/therapeutic branches.',
    file: 'agent/nodes/crisis_gate.py',
    fn: 'run_crisis_gate_node',
  },
  {
    id: 'crisis_resource_lookup',
    name: 'crisis_resource_lookup_node',
    category: 'SAFETY',
    order: 2,
    inputs: ['state.message', 'state.history'],
    outputs: ['state.inferred_location', 'state.found_resources', 'state.resource_lookup_status'],
    retry: true,
    llm: true,
    parallel: false,
    subgraph: false,
    description:
      'Crisis branch only. Extracts the user\'s region from recent context with a cheap LLM call, then performs a search-grounded lookup for verified hotlines. Failures degrade to empty results so the crisis response is never blocked by a third-party outage.',
    skipConditions: ['no llm_client (returns empty resources)'],
    file: 'agent/nodes/crisis_resource_lookup.py',
    fn: 'run_crisis_resource_lookup_node',
  },
  {
    id: 'crisis_response',
    name: 'crisis_response_node',
    category: 'SAFETY',
    order: 3,
    inputs: ['state.crisis', 'state.found_resources', 'state.history'],
    outputs: ['state.response_text', 'state.response_style'],
    retry: true,
    llm: true,
    parallel: false,
    subgraph: false,
    description:
      'Generates the crisis reply with safety-overlay framing and any resources surfaced by crisis_resource_lookup_node. Streams chunks via get_stream_writer(). Falls back to a deterministic, escalation-aware reply on LLM failure.',
    file: 'agent/nodes/crisis_response.py',
    fn: 'run_crisis_response_node',
  },
  {
    id: 'crisis_log',
    name: 'crisis_log_node',
    category: 'SAFETY',
    order: 4,
    inputs: ['state.crisis', 'state.crisis_audit', 'state.session_id'],
    outputs: ['crisis_log backend (side effect)'],
    retry: true,
    llm: false,
    parallel: false,
    subgraph: false,
    description:
      'Always-on audit log. Appends a CrisisLogRecord regardless of memory mode — the privacy asymmetry is deliberate. Reads classifier path, override outcome, and LLM-failure flag from state.crisis_audit. Logs loudly on backend failure but never breaks the turn.',
    file: 'agent/nodes/crisis_log.py',
    fn: 'run_crisis_log_node',
  },
  {
    id: 'turn_dispatch',
    name: 'turn_dispatch_node',
    category: 'ROUTING',
    order: 5,
    inputs: ['state.message', 'state.memory_control.pending_action'],
    outputs: ['state.memory_control.action', 'state.grounded_lookup.query', 'Command(goto=...)'],
    retry: true,
    llm: true,
    parallel: false,
    subgraph: false,
    description:
      'LLM-primary safe-turn router. Routes explicit saved-memory commands to memory_control_node, explicit external factual lookup requests to grounded_answer_node, and ordinary support to load_memory_node. Also handles confirm/cancel for a pending destructive memory action.',
    file: 'agent/nodes/turn_dispatch.py',
    fn: 'run_turn_dispatch_node',
  },
  {
    id: 'memory_control',
    name: 'memory_control_node',
    category: 'ROUTING',
    order: 6,
    inputs: ['state.memory_control.action', 'memory store'],
    outputs: ['state.response_text', 'state.memory_control.pending_action'],
    retry: true,
    llm: false,
    parallel: false,
    subgraph: false,
    description:
      'Executes the detected memory command: list/status, recall toggle, save preference, forget by index/query, or confirm/cancel a pending action. Single destructive deletes write a pending_action to state for the next turn to confirm; namespace-wide wipes are gated by typed-word confirmation in the CLI layer. Returns a deterministic user-facing reply — no LLM call.',
    skipConditions: ['incognito mode short-circuits durable writes'],
    file: 'agent/nodes/memory_control.py',
    fn: 'run_memory_control_node',
  },
  {
    id: 'grounded_answer',
    name: 'grounded_answer_node',
    category: 'ROUTING',
    order: 8,
    inputs: ['state.grounded_lookup.query', 'state.message'],
    outputs: ['state.response_text', 'state.grounded_lookup.status'],
    retry: true,
    llm: true,
    parallel: false,
    subgraph: false,
    description:
      'Answers explicit factual lookup requests with provider-native search grounding (use_search=True). Status field reports answered, no_verified_answer, search_failed, or search_unavailable. Never invents facts — surfaces "I cannot verify that" when the result looks unverified.',
    skipConditions: ['no llm_client (returns search_unavailable)'],
    file: 'agent/nodes/grounded_answer.py',
    fn: 'run_grounded_answer_node',
  },
  {
    id: 'load_memory',
    name: 'load_memory_node',
    category: 'MEMORY',
    order: 9,
    inputs: ['state.message', 'state.session_id', 'state.user_id'],
    outputs: ['state.working_memory', 'state.session_memory', 'state.procedural_profile', 'state.diagnostics'],
    retry: true,
    llm: false,
    parallel: false,
    subgraph: false,
    description:
      'Therapeutic branch only. Retrieves semantic + episodic + procedural context via hybrid RRF retrieval. First-turn catch-up injects the most recent episodic arc. Returns structured WorkingMemoryEntry dicts — formatting happens on demand at prompt-build time.',
    skipConditions: ['crisis branch skips retrieval', 'memory_control / grounded_lookup branches skip retrieval', 'incognito mode returns empty'],
    file: 'agent/nodes/load_memory.py',
    fn: 'run_load_memory_node',
  },
  {
    id: 'therapeutic',
    name: 'therapeutic_subgraph',
    category: 'ROUTING',
    order: 10,
    inputs: ['state.message', 'state.working_memory', 'state.exercise_state'],
    outputs: ['state.therapeutic_approach', 'state.response_*', 'state.exercise_state'],
    retry: false,
    llm: true,
    parallel: false,
    subgraph: true,
    description:
      'Compiled subgraph with dispatcher + 7 response style nodes (supportive, reflective, clarifying, psychoeducation, technique, guided_exercise, closing). Uses a narrow TherapeuticSubgraphOutput to restrict what flows back to the parent, preventing reducer double-counting on transcript/history. Each child node has its own RetryPolicy.',
    file: 'agent/therapeutic/graph.py',
    fn: 'build_therapeutic_subgraph',
  },
  {
    id: 'finalize_turn',
    name: 'finalize_turn_node',
    category: 'TERMINAL',
    order: 11,
    inputs: ['state.response_text', 'state.response_style'],
    outputs: ['state.transcript [+1]', 'state.history [+1]'],
    retry: false,
    llm: false,
    reducer: 'operator.add',
    parallel: false,
    subgraph: false,
    description:
      'Appends the assistant reply as a single-element delta. The operator.add reducer on transcript/history handles accumulation. Stamps response_style onto the assistant turn for round-trip persistence. Returns empty delta for blank/whitespace responses to keep the transcript clean. No I/O, so no retry.',
    file: 'agent/nodes/finalize_turn.py',
    fn: 'run_finalize_turn_node',
  },
  {
    id: 'extract_facts',
    name: 'extract_semantic_facts_node',
    category: 'EXTRACTION',
    order: 12,
    inputs: ['state.message', 'state.response_text'],
    outputs: ['memory_store / session buffer (side effect)', 'state.diagnostics'],
    retry: true,
    llm: true,
    reducer: '_merge_dicts',
    parallel: true,
    subgraph: false,
    description:
      'LLM structured-output call that extracts semantic candidates, then runs deterministic write policy. Low-risk stable facts (relationship/preference/coping_strategy/goal) commit immediately; sensitive or interpretive candidates (loss/trigger, negative self-belief) are held for session end or repetition-gated. Runs in parallel with extract_procedural_rules_node.',
    skipConditions: [
      'route in {crisis, memory_control, grounded_lookup}',
      'no llm_client',
      'incognito mode',
      'small-talk gate triggered',
      'early-turn negative self-belief',
    ],
    file: 'agent/nodes/extract_facts.py',
    fn: 'run_extract_semantic_facts_node',
  },
  {
    id: 'extract_procedural',
    name: 'extract_procedural_rules_node',
    category: 'EXTRACTION',
    order: 13,
    inputs: ['state.message', 'state.response_text'],
    outputs: ['procedural profile / session buffer (side effect)', 'state.diagnostics'],
    retry: true,
    llm: true,
    reducer: '_merge_dicts',
    parallel: true,
    subgraph: false,
    description:
      'LLM structured-output call that extracts procedural candidates, then runs deterministic write policy. Explicit durable instructions ("please keep replies short") commit immediately; implicit or turn-scoped preferences are held or dropped. Safety-conflict requests ("skip the safety check") are always dropped. Same parallel lane as extract_semantic_facts_node.',
    skipConditions: [
      'route in {crisis, memory_control, grounded_lookup}',
      'no llm_client',
      'incognito mode',
      'small-talk gate triggered',
    ],
    file: 'agent/nodes/extract_procedural_rules.py',
    fn: 'run_extract_procedural_rules_node',
  },
];

const CATEGORY_META: Record<
  Category,
  { label: string; blurb: string; hue: string }
> = {
  SAFETY: {
    label: 'Safety',
    blurb: 'Cannot be bypassed by response style or therapeutic approach.',
    hue: 'safety',
  },
  MEMORY: {
    label: 'Memory',
    blurb: 'Retrieval and structured working context.',
    hue: 'memory',
  },
  ROUTING: {
    label: 'Routing',
    blurb: 'Dispatches to the right response pathway.',
    hue: 'routing',
  },
  EXTRACTION: {
    label: 'Extraction',
    blurb: 'Post-response LLM side effects. Parallel fan-out.',
    hue: 'extraction',
  },
  TERMINAL: {
    label: 'Terminal',
    blurb: 'Transcript finalization. Pure state, no I/O.',
    hue: 'terminal',
  },
};

type FilterKey = 'ALL' | Category | 'llm' | 'retry' | 'reducer' | 'parallel';

const FILTERS: { key: FilterKey; label: string; kind: 'category' | 'capability' }[] = [
  { key: 'ALL', label: 'All nodes', kind: 'category' },
  { key: 'SAFETY', label: 'Safety', kind: 'category' },
  { key: 'MEMORY', label: 'Memory', kind: 'category' },
  { key: 'ROUTING', label: 'Routing', kind: 'category' },
  { key: 'EXTRACTION', label: 'Extraction', kind: 'category' },
  { key: 'TERMINAL', label: 'Terminal', kind: 'category' },
  { key: 'llm', label: 'LLM calls', kind: 'capability' },
  { key: 'retry', label: 'RetryPolicy', kind: 'capability' },
  { key: 'reducer', label: 'Reducer-backed', kind: 'capability' },
  { key: 'parallel', label: 'Parallel', kind: 'capability' },
];

function nodeMatchesFilter(node: NodeSpec, filter: FilterKey): boolean {
  if (filter === 'ALL') return true;
  if (filter === 'llm') return node.llm;
  if (filter === 'retry') return node.retry;
  if (filter === 'reducer') return Boolean(node.reducer);
  if (filter === 'parallel') return node.parallel;
  return node.category === filter;
}

function NodeCard({ node }: { node: NodeSpec }) {
  const meta = CATEGORY_META[node.category];
  const orderStr = String(node.order).padStart(2, '0');

  return (
    <article className={`${s.card} ${s[`hue_${meta.hue}`]}`}>
      {/* Classification strip */}
      <header className={s.classStrip}>
        <span className={s.classOrder}>{orderStr}</span>
        <span className={s.classLabel}>{meta.label}</span>
        <span className={s.classBlurb}>{meta.blurb}</span>
      </header>

      {/* Name + policy chips */}
      <div className={s.nameRow}>
        <h3 className={s.nodeName}>{node.name}</h3>
        <div className={s.chips}>
          {node.llm && <span className={`${s.chip} ${s.chipLlm}`}>LLM</span>}
          {node.retry && <span className={`${s.chip} ${s.chipRetry}`}>retry=2</span>}
          {node.reducer && (
            <span className={`${s.chip} ${s.chipReducer}`}>{node.reducer}</span>
          )}
          {node.parallel && (
            <span className={`${s.chip} ${s.chipParallel}`}>parallel</span>
          )}
          {node.subgraph && (
            <span className={`${s.chip} ${s.chipSubgraph}`}>subgraph</span>
          )}
        </div>
      </div>

      {/* Description */}
      <p className={s.description}>{node.description}</p>

      {/* Signature (inputs -> outputs) */}
      <div className={s.signature}>
        <div className={s.signatureSide}>
          <span className={s.sigHeader}>{'\u25B8'} inputs</span>
          {node.inputs.map((i) => (
            <code key={i} className={s.sigField}>
              {i}
            </code>
          ))}
        </div>
        <div className={s.sigArrow} aria-hidden>
          {'\u2192'}
        </div>
        <div className={s.signatureSide}>
          <span className={s.sigHeader}>{'\u25B8'} outputs</span>
          {node.outputs.map((o) => (
            <code key={o} className={s.sigField}>
              {o}
            </code>
          ))}
        </div>
      </div>

      {/* Skip conditions (only for gated nodes) */}
      {node.skipConditions && node.skipConditions.length > 0 && (
        <div className={s.skipBlock}>
          <span className={s.skipLabel}>skip when</span>
          <div className={s.skipList}>
            {node.skipConditions.map((c) => (
              <span key={c} className={s.skipItem}>
                {c}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* File fingerprint */}
      <footer className={s.fingerprint}>
        <span className={s.fpFile}>{node.file}</span>
        <span className={s.fpDivider}>{'::'}</span>
        <span className={s.fpFn}>{node.fn}</span>
      </footer>
    </article>
  );
}

export default function NodeCatalog(): React.JSX.Element {
  const [filter, setFilter] = useState<FilterKey>('ALL');

  const filtered = useMemo(
    () =>
      NODES.filter((n) => nodeMatchesFilter(n, filter)).sort(
        (a, b) => a.order - b.order,
      ),
    [filter],
  );

  const stats = useMemo(() => {
    const total = NODES.length;
    const llm = NODES.filter((n) => n.llm).length;
    const retry = NODES.filter((n) => n.retry).length;
    const reducer = NODES.filter((n) => n.reducer).length;
    const parallel = NODES.filter((n) => n.parallel).length;
    return { total, llm, retry, reducer, parallel };
  }, []);

  return (
    <div className={s.root}>
      {/* Stats strip */}
      <div className={s.statsStrip}>
        <div className={s.stat}>
          <span className={s.statValue}>{stats.total}</span>
          <span className={s.statLabel}>nodes</span>
        </div>
        <div className={s.statDivider} />
        <div className={s.stat}>
          <span className={s.statValue}>{stats.llm}</span>
          <span className={s.statLabel}>LLM</span>
        </div>
        <div className={s.statDivider} />
        <div className={s.stat}>
          <span className={s.statValue}>{stats.retry}</span>
          <span className={s.statLabel}>retry=2</span>
        </div>
        <div className={s.statDivider} />
        <div className={s.stat}>
          <span className={s.statValue}>{stats.reducer}</span>
          <span className={s.statLabel}>reducers</span>
        </div>
        <div className={s.statDivider} />
        <div className={s.stat}>
          <span className={s.statValue}>{stats.parallel}</span>
          <span className={s.statLabel}>parallel</span>
        </div>
      </div>

      {/* Filter bar */}
      <nav className={s.filterBar} aria-label="Filter nodes">
        {FILTERS.map((f) => (
          <button
            key={f.key}
            className={[
              s.filterBtn,
              filter === f.key ? s.filterBtnActive : '',
              f.kind === 'capability' ? s.filterBtnCap : '',
            ].join(' ')}
            onClick={() => setFilter(f.key)}
            aria-pressed={filter === f.key}
          >
            {f.label}
            {filter !== f.key && (
              <span className={s.filterCount}>
                {f.key === 'ALL' ? NODES.length : NODES.filter((n) => nodeMatchesFilter(n, f.key)).length}
              </span>
            )}
          </button>
        ))}
      </nav>

      {/* Result count */}
      <div className={s.resultMeta}>
        showing {filtered.length} of {NODES.length}
        {filter !== 'ALL' && (
          <>
            {' '}
            <span className={s.filterHint}>
              (filter: <code>{FILTERS.find((f) => f.key === filter)?.label}</code>)
            </span>
          </>
        )}
      </div>

      {/* Cards */}
      <div className={s.cards}>
        {filtered.map((node) => (
          <NodeCard key={node.id} node={node} />
        ))}
      </div>

      {filtered.length === 0 && (
        <div className={s.empty}>No nodes match this filter.</div>
      )}
    </div>
  );
}
