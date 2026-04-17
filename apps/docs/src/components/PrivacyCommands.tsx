import React, { useState } from 'react';
import s from './PrivacyCommands.module.css';

/* ================================================================
   PrivacyCommands — Three-tier command visualization

   Shows Inspect / Delete / Wipe as three escalating tiers with
   commands inside each. Click a tier to expand and see its commands.
   ================================================================ */

interface Command {
  cmd: string;
  desc: string;
}

interface Tier {
  id: string;
  label: string;
  icon: string;
  risk: 'low' | 'medium' | 'high';
  confirmation: string;
  commands: Command[];
}

const TIERS: Tier[] = [
  {
    id: 'inspect',
    label: 'Inspect',
    icon: '\u2315',
    risk: 'low',
    confirmation: 'None — read-only',
    commands: [
      { cmd: '/memory status', desc: 'Per-namespace counts, mode, recall toggle, owner_id' },
      { cmd: '/memory list', desc: 'All semantic facts + episodic arcs' },
      { cmd: '/memory list facts', desc: 'Semantic facts only' },
      { cmd: '/memory list sessions', desc: 'Episodic arcs only' },
      { cmd: '/memory list rules', desc: 'Procedural style rules' },
    ],
  },
  {
    id: 'delete',
    label: 'Delete',
    icon: '\u2717',
    risk: 'medium',
    confirmation: 'Preview panel + y/N (default N)',
    commands: [
      { cmd: '/memory forget fact <n>', desc: 'Delete one semantic fact by index' },
      { cmd: '/memory forget session <n>', desc: 'Delete one episodic arc by index' },
      { cmd: '/memory forget rule <n>', desc: 'Delete one procedural rule by index' },
    ],
  },
  {
    id: 'wipe',
    label: 'Wipe',
    icon: '\u26A0',
    risk: 'high',
    confirmation: 'Must type the literal word to proceed',
    commands: [
      { cmd: '/memory clear facts', desc: 'Wipe all semantic facts' },
      { cmd: '/memory clear sessions', desc: 'Wipe all episodic arcs' },
      { cmd: '/memory clear rules', desc: 'Wipe all procedural rules (preserves recall toggle)' },
      { cmd: '/memory clear all', desc: 'Wipe everything' },
      { cmd: '/memory purge-crisis [days]', desc: 'Retention-purge crisis log (default: 90 days)' },
    ],
  },
];

export default function PrivacyCommands(): React.JSX.Element {
  const [expanded, setExpanded] = useState<string>('inspect');

  return (
    <div className={s.root}>
      {/* Tier headers */}
      <div className={s.tierBar}>
        {TIERS.map((tier) => (
          <button
            key={tier.id}
            className={[
              s.tierTab,
              s[`risk_${tier.risk}`],
              expanded === tier.id ? s.tierTabActive : '',
            ].join(' ')}
            onClick={() => setExpanded(tier.id)}
          >
            <span className={s.tierIcon}>{tier.icon}</span>
            <span className={s.tierLabel}>{tier.label}</span>
          </button>
        ))}
      </div>

      {/* Expanded tier */}
      {TIERS.filter((t) => t.id === expanded).map((tier) => (
        <div key={tier.id} className={`${s.tierBody} ${s[`risk_${tier.risk}`]}`}>
          <div className={s.tierMeta}>
            <span className={s.metaKey}>Confirmation</span>
            <span className={s.metaVal}>{tier.confirmation}</span>
          </div>

          <div className={s.cmdList}>
            {tier.commands.map((cmd) => (
              <div key={cmd.cmd} className={s.cmdRow}>
                <code className={s.cmdCode}>{cmd.cmd}</code>
                <span className={s.cmdDesc}>{cmd.desc}</span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
