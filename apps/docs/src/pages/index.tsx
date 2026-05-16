import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Layout from '@theme/Layout';
import {useEffect} from 'react';

type IconProps = {
  className?: string;
};

function ArchitectureIcon({className}: IconProps): JSX.Element {
  return (
    <svg viewBox="0 0 24 24" fill="none" className={className} aria-hidden="true">
      <path d="M12 3 4 7v10l8 4 8-4V7l-8-4Z" stroke="currentColor" strokeWidth="1.5" />
      <path d="M4 7l8 4m8-4-8 4m0 10V11" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}

function BackendIcon({className}: IconProps): JSX.Element {
  return (
    <svg viewBox="0 0 24 24" fill="none" className={className} aria-hidden="true">
      <rect x="3.5" y="4" width="17" height="6" rx="2" stroke="currentColor" strokeWidth="1.5" />
      <rect x="3.5" y="14" width="17" height="6" rx="2" stroke="currentColor" strokeWidth="1.5" />
      <circle cx="8" cy="7" r="1" fill="currentColor" />
      <circle cx="8" cy="17" r="1" fill="currentColor" />
    </svg>
  );
}

function SafetyIcon({className}: IconProps): JSX.Element {
  return (
    <svg viewBox="0 0 24 24" fill="none" className={className} aria-hidden="true">
      <path
        d="M12 3s-5 2-8 3v6c0 5.5 4 8 8 9 4-1 8-3.5 8-9V6c-3-1-8-3-8-3Z"
        stroke="currentColor"
        strokeWidth="1.5"
      />
      <path d="m9 12 2 2 4-4" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}

function AgentIcon({className}: IconProps): JSX.Element {
  return (
    <svg viewBox="0 0 24 24" fill="none" className={className} aria-hidden="true">
      <rect x="5" y="6" width="14" height="12" rx="3" stroke="currentColor" strokeWidth="1.5" />
      <circle cx="10" cy="12" r="1.2" fill="currentColor" />
      <circle cx="14" cy="12" r="1.2" fill="currentColor" />
      <path d="M9 15.2c.9.8 1.9 1.2 3 1.2s2.1-.4 3-1.2" stroke="currentColor" strokeWidth="1.5" />
      <path d="M12 3v2" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}

function PromptIcon({className}: IconProps): JSX.Element {
  return (
    <svg viewBox="0 0 24 24" fill="none" className={className} aria-hidden="true">
      <rect x="4" y="4" width="16" height="16" rx="2.5" stroke="currentColor" strokeWidth="1.5" />
      <path d="M8 9h8M8 12h8M8 15h5" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}

const cards = [
  {
    title: 'Architecture & Backend',
    href: '/docs/backend/overview',
    description: 'Turn pipeline, key decisions, persistence, and feature summary.',
    icon: ArchitectureIcon,
  },
  {
    title: 'Safety',
    href: '/docs/philosophy/crisis-gate',
    description: 'Crisis gate design, risk levels, and routing policy.',
    icon: SafetyIcon,
  },
  {
    title: 'Agent',
    href: '/docs/agent/graph',
    description: 'Response styles, therapeutic approaches, and graph routing.',
    icon: AgentIcon,
  },
  {
    title: 'Prompt Assembly',
    href: '/docs/agent/prompt-assembly',
    description: 'How identity, policy, response, approach, memory, and task layers compose.',
    icon: PromptIcon,
  }
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
    body: 'Web chat, LiveKit voice, and Telegram DMs are already dogfood surfaces. WhatsApp and Discord are planned next.'
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

export default function Home(): JSX.Element {
  const glyphSrc = useBaseUrl('/img/opencouch-glyph-1024.png');
  const landingSrc = useBaseUrl('/img/landing.png');

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
                <Link className="button button--primary button--lg" to="/docs/intro">
                  Read the docs
                </Link>
                <Link className="button button--secondary button--lg" to="/docs/philosophy/crisis-gate">
                  Safety model
                </Link>
              </div>
            </div>

            <div className="hero-visual">
              <img className="hero-visual__image" src={landingSrc} alt="OpenCouch landing page" />
            </div>
          </div>

          <div className="doc-card-grid">
            {cards.map((card) => (
              <Link key={card.title} className="doc-card" to={card.href}>
                <div className="doc-card__header">
                  <card.icon className="doc-card__icon" />
                  <strong>{card.title}</strong>
                </div>
                <p>{card.description}</p>
              </Link>
            ))}
          </div>
        </div>
      </header>

      {/* Stats strip */}
      <section className="stats-strip reveal-on-scroll" data-reveal data-reveal-delay="40">
        <div className="container">
          <div className="stats-shell">
            <div className="stats-grid">
              {stats.map((s, i) => (
                <div
                  key={s.label}
                  className="stat-item"
                  style={{
                    borderRight: i < stats.length - 1
                      ? '1px solid var(--oc-border)'
                      : 'none',
                  }}>
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
