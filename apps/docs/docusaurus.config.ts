import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'OpenCouch Docs',
  tagline: 'Architecture, safety, and contributor documentation for OpenCouch',
  url: 'https://whanyu1212.github.io',
  baseUrl: '/OpenCouch/',
  onBrokenLinks: 'throw',
  organizationName: 'whanyu1212',
  projectName: 'OpenCouch',
  trailingSlash: false,
  favicon: 'img/favicon.svg',

  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  themes: ['@docusaurus/theme-mermaid'],

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          routeBasePath: 'docs',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    colorMode: {
      defaultMode: 'light',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },

    mermaid: {
      theme: {light: 'base', dark: 'dark'},
      options: {
        themeVariables: {
          primaryColor: '#215f5a',
          primaryTextColor: '#ffffff',
          primaryBorderColor: '#1b4542',
          lineColor: '#3d9990',
          secondaryColor: '#dfe8de',
          tertiaryColor: '#f5f3ea',
          mainBkg: '#ffffff',
          secondBkg: '#eef7f6',
          tertiaryBkg: '#f5f6f3',
          clusterBkg: '#eef7f6',
          clusterBorder: '#9dd2cb',
          nodeBorder: '#1b4542',
          defaultLinkColor: '#3d9990',
          titleColor: '#143432',
          textColor: '#143432',
          edgeLabelBackground: '#f5f6f3',
          actorBkg: '#215f5a',
          actorBorder: '#1b4542',
          actorTextColor: '#ffffff',
          signalColor: '#143432',
        },
      },
    },

    prism: {
      theme: {
        plain: {color: '#c6e8e4', backgroundColor: '#0d2422'},
        styles: [],
      },
      darkTheme: {
        plain: {color: '#c6e8e4', backgroundColor: '#080f0e'},
        styles: [],
      },
      additionalLanguages: ['python', 'bash', 'json', 'yaml', 'toml'],
    },

    docs: {
      sidebar: {
        hideable: true,
        autoCollapseCategories: false,
      },
    },

    navbar: {
      title: 'OpenCouch',
      hideOnScroll: false,
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          href: 'https://github.com/opencouch',
          label: 'GitHub',
          position: 'right',
          className: 'navbar__link--github',
        },
      ],
    },

    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {label: 'Overview', to: '/docs/intro'},
            {label: 'Backend', to: '/docs/backend/overview'},
            {label: 'Safety', to: '/docs/philosophy/crisis-gate'},
          ],
        },
        {
          title: 'Agent',
          items: [
            {label: 'Graph', to: '/docs/agent/graph'},
            {label: 'Prompt Assembly', to: '/docs/agent/prompt-assembly'},
            {label: 'Backend', to: '/docs/backend/overview'},
          ],
        },
        {
          title: 'Memory',
          items: [
            {label: 'Memory Layers', to: '/docs/memory/overview'},
            {label: 'Hybrid Retrieval', to: '/docs/memory/retrieval'},
            {label: 'Privacy Controls', to: '/docs/memory/privacy'},
          ],
        },
      ],
      copyright: `Copyright ${new Date().getFullYear()} OpenCouch`,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
