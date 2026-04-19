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
        {
          type: 'category',
          label: 'Agent Graph',
          link: { type: 'doc', id: 'agent/graph' },
          collapsed: false,
          items: [
            'agent/state',
            'agent/nodes',
            'agent/tools',
          ],
        },
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
    'voice/overview',
    {
      type: 'category',
      label: 'Observability',
      items: ['observability/overview', 'observability/session-feedback']
    },
    'roadmap',
  ]
};

export default sidebars;
