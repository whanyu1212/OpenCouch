import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    'intro',
    'quickstart',
    {
      type: 'category',
      label: 'Philosophy',
      items: ['philosophy/graph-vs-react', 'philosophy/approach', 'philosophy/crisis-gate']
    },
    {
      type: 'category',
      label: 'System',
      items: [
        'backend/overview',
        'backend/runtime',
        'system/api-reference',
        'system/web-ui',
      ]
    },
    {
      type: 'category',
      label: 'Agent',
      items: [
        {
          type: 'category',
          label: 'Agent Runtime',
          collapsed: false,
          items: [
            'agent/graph',
            'agent/state',
            'agent/tools',
            'agent/flows',
          ],
        },
        'agent/routing-classifiers',
        'agent/prompt-assembly',
        'agent/context-management',
        {
          type: 'category',
          label: 'Therapeutic Responses',
          collapsed: false,
          items: ['agent/scenarios', 'agent/guided-exercises'],
        },
      ]
    },
    {
      type: 'category',
      label: 'Memory',
      items: ['memory/why-memory', 'memory/overview', 'memory/retrieval', 'memory/privacy']
    },
    {
      type: 'category',
      label: 'Voice',
      items: [
        'voice/overview',
        'voice/realtime-lifecycle',
        'voice/tools-and-policy',
        'voice/persistence',
        'voice/dogfood',
      ],
    },
    {
      type: 'category',
      label: 'Observability & Evaluation',
      items: ['observability/overview', 'observability/session-feedback']
    },
    'roadmap',
  ]
};

export default sidebars;
