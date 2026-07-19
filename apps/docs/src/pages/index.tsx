import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Layout from '@theme/Layout';
import {useEffect} from 'react';

const architectureCards = [
  {
    title: 'Safety Runtime',
    label: 'Start here',
    description: 'Crisis gate, risk levels, routing policy, and safety boundaries.',
    href: '/docs/philosophy/crisis-gate',
    source: 'apps/backend/agent',
  },
  {
    title: 'Agent Runtime',
    label: 'Core runtime',
    description: 'Router topology, app-owned state, tools, flows, and dispatch.',
    href: '/docs/agent/graph',
    source: 'apps/backend/agent',
  },
  {
    title: 'Memory System',
    label: 'Context layer',
    description: 'Semantic, episodic, and procedural memory with retrieval and privacy controls.',
    href: '/docs/memory/overview',
    source: 'apps/backend/agent/memory',
  },
  {
    title: 'Voice Runtime',
    label: 'Realtime',
    description: 'OpenAI Realtime lifecycle, session policy, tools, and persistence.',
    href: '/docs/voice',
    source: 'apps/backend/agent/voice',
  },
  {
    title: 'Web/API Surface',
    label: 'Product surface',
    description: 'Backend API contracts and the Next.js web interface.',
    href: '/docs/system/api-reference',
    source: 'apps/web · apps/backend',
  },
];

const runtimeSteps = [
  {
    label: 'User message',
    detail: 'The current turn enters the backend runtime.',
  },
  {
    label: 'crisis_gate',
    detail: 'Safety check runs before memory retrieval.',
    badge: 'required',
  },
  {
    label: 'turn_dispatch',
    detail: 'Safe turns route to memory, tools, or support.',
  },
  {
    label: 'load_memory',
    detail: 'Context loads only after safety clears.',
  },
  {
    label: 'TherapeuticAgent',
    detail: 'Generates bounded support or guided exercise turns.',
  },
  {
    label: 'finalize',
    detail: 'Appends response, diagnostics, and transcript state.',
  },
];

const pathCards = [
  {
    title: 'For contributors',
    description: 'Run the stack locally, understand the backend runtime, and find the main source areas.',
    href: '/docs/quickstart',
    links: [
      {label: 'Quickstart', href: '/docs/quickstart'},
      {label: 'Backend runtime', href: '/docs/backend/runtime'},
    ],
  },
  {
    title: 'For safety reviewers',
    description: 'Review the crisis gate, safety boundaries, and response policy.',
    href: '/docs/philosophy/crisis-gate',
    links: [
      {label: 'Crisis gate', href: '/docs/philosophy/crisis-gate'},
      {label: 'Approach', href: '/docs/philosophy/approach'},
    ],
  },
  {
    title: 'For agent developers',
    description: 'Understand runtime routing, prompt assembly, state, tools, and therapeutic behavior.',
    href: '/docs/agent/graph',
    links: [
      {label: 'Agent runtime', href: '/docs/agent/graph'},
      {label: 'Prompt assembly', href: '/docs/agent/prompt-assembly'},
    ],
  },
  {
    title: 'For memory/privacy reviewers',
    description: 'Inspect semantic, episodic, and procedural memory plus privacy controls.',
    href: '/docs/memory/overview',
    links: [
      {label: 'Memory overview', href: '/docs/memory/overview'},
      {label: 'Privacy controls', href: '/docs/memory/privacy'},
    ],
  },
  {
    title: 'For voice developers',
    description: 'Trace the OpenAI Realtime lifecycle, tool policy, persistence, and dogfood flow.',
    href: '/docs/voice',
    links: [
      {label: 'Voice overview', href: '/docs/voice'},
      {label: 'Realtime lifecycle', href: '/docs/voice/realtime-lifecycle'},
    ],
  },
];

const trustCards = [
  {
    title: 'Safety gate before memory',
    description: 'Crisis assessment runs before memory retrieval, so unsafe turns do not load contextual memory first.',
    artifact: 'crisis_gate → turn_dispatch → load_memory',
    href: '/docs/philosophy/crisis-gate',
  },
  {
    title: 'User-controlled memory',
    description: 'Memory is documented as inspectable and controllable through explicit user commands and privacy controls.',
    artifact: 'semantic · episodic · procedural',
    href: '/docs/memory/privacy',
  },
  {
    title: 'Honest boundaries',
    description: 'OpenCouch provides support without pretending to be therapy.',
    artifact: 'onboarding · prompts · crisis responses',
    href: '/docs/philosophy/approach',
  },
  {
    title: 'Self-hostable infrastructure',
    description: 'Backend, web, docs, and persistence modes can be run locally or on your own infrastructure.',
    artifact: 'backend · web · docs',
    href: '/docs/quickstart',
  },
  {
    title: 'AGPL-3.0 licensed',
    description: 'Deployment changes must remain visible, reducing the chance that safety mechanisms are stripped in closed forks.',
    artifact: 'open source safety preservation',
    href: '/docs/intro',
  },
];

const localCommands = [
  {
    label: 'Backend API',
    command: 'cd apps/backend && .venv/bin/python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000',
  },
  {
    label: 'Web app',
    command: 'cd apps/web && pnpm dev',
  },
  {
    label: 'Docs site',
    command: 'cd apps/docs && pnpm start',
  },
];

const sourceMap = [
  {
    label: 'Backend API',
    path: 'apps/backend',
    href: '/docs/backend/overview',
  },
  {
    label: 'Agent runtime',
    path: 'apps/backend/agent',
    href: '/docs/agent/graph',
  },
  {
    label: 'Voice runtime',
    path: 'apps/backend/agent/voice',
    href: '/docs/voice',
  },
  {
    label: 'Web app',
    path: 'apps/web',
    href: '/docs/system/web-ui',
  },
  {
    label: 'Docs site',
    path: 'apps/docs',
    href: '/docs/intro',
  },
  {
    label: 'Evaluation',
    path: 'eval',
    href: '/docs/observability/overview',
  },
];

const flowSteps = [
  {
    label: 'Private by default',
    n: '01',
    body: 'No data sold. No third-party training. Self-hostable so data never leaves your infrastructure if you choose.'
  },
  {
    label: 'Safe by design',
    n: '02',
    body: 'Crisis detection runs as a hard gate before every response — deterministic patterns plus an LLM classifier, always.'
  },
  {
    label: 'Honest boundaries',
    n: '03',
    body: 'OpenCouch is not a therapist. It says so clearly, in onboarding, in prompts, and in every crisis response.'
  },
  {
    label: 'Where you already are',
    n: '04',
    body: 'Web chat and OpenAI Realtime voice are the dogfood surfaces today. Additional messaging channels are on the roadmap.'
  },
  {
    label: 'AGPL-3.0 licensed',
    n: '05',
    body: 'Not MIT — intentionally. AGPL ensures that anyone who deploys a modified version must publish their changes. Safety features like the crisis gate cannot be silently stripped in a closed fork.'
  }
];

const stats = [
  { value: 'Open source', label: 'AGPL-3.0 licensed' },
  { value: 'Self-hostable', label: 'your infrastructure' },
  { value: 'Crisis-first', label: 'safety architecture' },
  { value: 'Multi-surface', label: 'text · voice · messaging' },
];

function RuntimePipelineCard(): JSX.Element {
  return (
    <div className="runtime-card" aria-label="Safety-first runtime pipeline">
      <div className="runtime-card__header">
        <span className="runtime-card__eyebrow">Every turn</span>
        <strong>Safety-first runtime pipeline</strong>
      </div>
      <ol className="runtime-pipeline">
        {runtimeSteps.map((step, index) => (
          <li key={step.label} className="runtime-pipeline__step">
            <span className="runtime-pipeline__index">{String(index + 1).padStart(2, '0')}</span>
            <div className="runtime-pipeline__copy">
              <div className="runtime-pipeline__topline">
                <code className="runtime-pipeline__label">{step.label}</code>
                {step.badge ? <span className="runtime-pipeline__badge">{step.badge}</span> : null}
              </div>
              <p className="runtime-pipeline__detail">{step.detail}</p>
            </div>
          </li>
        ))}
      </ol>
      <p className="runtime-card__footer">OpenCouch loads memory only after the safety gate clears the turn.</p>
    </div>
  );
}

export default function Home(): JSX.Element {
  const glyphSrc = useBaseUrl('/img/opencouch-glyph-1024.png');

  useEffect(() => {
    const nodes = Array.from(
      document.querySelectorAll<HTMLElement>('[data-reveal]')
    );
    if (!nodes.length) {
      return;
    }

    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (reducedMotion.matches || typeof IntersectionObserver === 'undefined') {
      nodes.forEach((node) => node.classList.add('is-visible'));
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) {
            return;
          }
          const node = entry.target as HTMLElement;
          const delay = node.dataset.revealDelay;
          if (delay) {
            node.style.transitionDelay = `${delay}ms`;
          }
          node.classList.add('is-visible');
          observer.unobserve(node);
        });
      },
      {
        threshold: 0.16,
        rootMargin: '0px 0px -8% 0px',
      }
    );

    nodes.forEach((node) => observer.observe(node));
    return () => observer.disconnect();
  }, []);

  return (
    <Layout
      title="OpenCouch Docs"
      description="Architecture, safety, and contributor documentation for OpenCouch">
      <header className={clsx('hero hero--opencouch')}>
        <div className="container">
          <div className="hero-shell">
            <div className="hero-copy">
              <p className="hero__eyebrow">Open Source</p>
              <h1 className="hero__title">
                <img className="hero__logo" src={glyphSrc} alt="" aria-hidden="true" />
                OpenCouch
              </h1>
              <p className="hero__subtitle">
                A calm, session-aware mental health support companion that helps you
                feel heard, make sense of what you're going through, and find one
                grounded next step — without pretending to be therapy.
              </p>
              <div className="hero-cta-row">
                <Link className="button button--primary button--lg" to="/docs/quickstart">
                  Start with Quickstart
                </Link>
                <Link className="button button--secondary button--lg" to="/docs/philosophy/crisis-gate">
                  Explore Safety Model
                </Link>
                <Link className="hero-inline-link" to="/docs/agent/graph">
                  View Agent Runtime →
                </Link>
              </div>
            </div>

            <div className="hero-visual">
              <RuntimePipelineCard />
            </div>
          </div>

          <div className="doc-card-grid">
            {architectureCards.map((card) => (
              <Link key={card.title} className="doc-card" to={card.href}>
                <div className="doc-card__header">
                  <div>
                    <span className="doc-card__label">{card.label}</span>
                    <strong>{card.title}</strong>
                  </div>
                </div>
                <p>{card.description}</p>
                <span className="doc-card__source">{card.source}</span>
              </Link>
            ))}
          </div>
        </div>
      </header>

      <section className="stats-strip reveal-on-scroll" data-reveal data-reveal-delay="40">
        <div className="container">
          <div className="stats-shell">
            <div className="stats-grid">
              {stats.map((s) => (
                <div
                  key={s.label}
                  className="stat-item">
                  <div className="stat-item__value">{s.value}</div>
                  <div className="stat-item__label">{s.label}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <main className="landing-shell">
        <section className="landing-section">
          <div className="container">
            <div className="section-heading reveal-on-scroll" data-reveal data-reveal-delay="70">
              <p className="eyebrow">Choose your path</p>
              <h2>Start with the docs closest to your work</h2>
              <p>
                OpenCouch spans safety policy, backend runtime, memory, voice, and web surfaces.
                Start with the path closest to what you want to inspect or change.
              </p>
            </div>
            <div className="path-card-grid">
              {pathCards.map((card, index) => (
                <article
                  key={card.title}
                  className="path-card reveal-on-scroll"
                  data-reveal
                  data-reveal-delay={110 + index * 45}>
                  <Link className="path-card__title" to={card.href}>
                    {card.title}
                  </Link>
                  <p>{card.description}</p>
                  <div className="path-card__links">
                    {card.links.map((link) => (
                      <Link key={link.href} className="path-card__link" to={link.href}>
                        {link.label}
                      </Link>
                    ))}
                  </div>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section className="landing-section landing-section--tinted">
          <div className="container">
            <div className="section-heading reveal-on-scroll" data-reveal data-reveal-delay="70">
              <p className="eyebrow">Trust model</p>
              <h2>Concrete safety and privacy mechanisms</h2>
              <p>
                Each principle is backed by a visible implementation artifact instead of a vague promise.
              </p>
            </div>
            <div className="trust-card-grid">
              {trustCards.map((card, index) => (
                <Link
                  key={card.title}
                  className="trust-card reveal-on-scroll"
                  to={card.href}
                  data-reveal
                  data-reveal-delay={120 + index * 45}>
                  <strong>{card.title}</strong>
                  <p>{card.description}</p>
                  <code className="trust-card__artifact">{card.artifact}</code>
                </Link>
              ))}
            </div>
          </div>
        </section>

        <section className="landing-section">
          <div className="container">
            <div className="section-heading reveal-on-scroll" data-reveal data-reveal-delay="70">
              <p className="eyebrow">Run locally</p>
              <h2>Start the stack from your checkout</h2>
              <p>
                Use the project virtualenv for backend commands and run each surface from its app directory.
              </p>
            </div>
            <div className="local-command-panel reveal-on-scroll" data-reveal data-reveal-delay="120">
              {localCommands.map((item) => (
                <div key={item.label} className="local-command">
                  <span className="local-command__label">{item.label}</span>
                  <code className="local-command__code">{item.command}</code>
                </div>
              ))}
            </div>
            <Link className="section-link" to="/docs/quickstart">
              Full quickstart →
            </Link>
          </div>
        </section>

        <section className="landing-section">
          <div className="container">
            <div className="section-heading reveal-on-scroll" data-reveal data-reveal-delay="70">
              <p className="eyebrow">Source map</p>
              <h2>Jump from docs concept to code</h2>
              <p>
                The docs map directly to the backend, agent, voice, web, and evaluation areas of the repository.
              </p>
            </div>
            <div className="source-map reveal-on-scroll" data-reveal data-reveal-delay="120">
              {sourceMap.map((item) => (
                <Link key={item.label} className="source-map__row" to={item.href}>
                  <span className="source-map__label">{item.label}</span>
                  <code className="source-map__path">{item.path}</code>
                </Link>
              ))}
            </div>
          </div>
        </section>

        <section className="landing-section">
          <div className="container">
            <div className="section-heading reveal-on-scroll" data-reveal data-reveal-delay="70">
              <p className="eyebrow">Core principles</p>
              <h2>Built to be trustworthy, not just capable</h2>
              <p>
                Most AI products are built for engagement. OpenCouch is built
                for trust — with clear safety boundaries, user-controlled memory,
                and an architecture that can run entirely on your own infrastructure.
              </p>
            </div>
            <div className="flow-grid">
              {flowSteps.map((step, index) => (
                <div
                  key={step.label}
                  className="flow-card reveal-on-scroll"
                  data-reveal
                  data-reveal-delay={140 + index * 55}>
                  <span className="flow-badge">{step.label}</span>
                  <p>{step.body}</p>
                  <span className="flow-card__number">{step.n}</span>
                </div>
              ))}
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
