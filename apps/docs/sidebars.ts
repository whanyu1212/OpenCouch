import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    'intro',
    'quickstart',
    {
      type: 'category',
      label: 'Philosophy',
      items: ['philosophy/approach', 'philosophy/crisis-gate']
    },
    {
      type: 'category',
      label: 'System',
      items: ['backend/overview', 'backend/runtime']
    },
    {
      type: 'category',
      label: 'Agent',
      items: [
        'agent/graph',
        'agent/state',
        'agent/prompt-assembly',
        'agent/context-management',
        {
          type: 'category',
          label: 'Therapeutic Modes',
          collapsed: false,
          items: ['agent/scenarios', 'agent/guided-exercises'],
        },
      ]
    },
    {
      type: 'category',
      label: 'Memory',
      items: ['memory/overview', 'memory/retrieval', 'memory/privacy']
    },
    {
      type: 'category',
      label: 'Voice',
      items: ['voice/overview']
    },
    {
      type: 'category',
      label: 'Observability',
      items: ['observability/overview', 'observability/session-feedback']
    },
    'roadmap',
  ]
};

export default sidebars;
