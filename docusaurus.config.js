// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Zhiyuan Pan',
  tagline: 'LLM & AI Infra 知识库',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://your-docusaurus-site.example.com',
  baseUrl: '/',

  organizationName: 'CyperPan',
  projectName: 'personal_blog_site',

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  markdown: {
    format: 'md',
  },

  i18n: {
    defaultLocale: 'zh-Hans',
    locales: ['zh-Hans'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
        },
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          onInlineTags: 'warn',
          onInlineAuthors: 'warn',
          onUntruncatedBlogPosts: 'warn',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: 'img/docusaurus-social-card.jpg',
      colorMode: {
        respectPrefersColorScheme: true,
      },
      navbar: {
        title: 'Zhiyuan Pan',
        logo: {
          alt: 'Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: '知识库',
          },
          {
            to: '/docs/category/llm推理',
            label: '推理',
            position: 'left',
          },
          {
            to: '/docs/category/llm训练',
            label: '训练',
            position: 'left',
          },
          {
            to: '/docs/category/数学推导',
            label: '数学推导',
            position: 'left',
          },
          {
            to: '/docs/category/编程题',
            label: '编程题',
            position: 'left',
          },
          {
            to: '/docs/category/面经',
            label: '面经',
            position: 'left',
          },
          {to: '/blog', label: '博客', position: 'left'},
          {
            href: 'https://github.com/CyperPan',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: '知识库',
            items: [
              {label: 'LLM推理', to: '/docs/category/llm推理'},
              {label: 'LLM训练', to: '/docs/category/llm训练'},
              {label: '数学推导', to: '/docs/category/数学推导'},
              {label: '编程题', to: '/docs/category/编程题'},
            ],
          },
          {
            title: '更多',
            items: [
              {label: '面经', to: '/docs/category/面经'},
              {label: '博客', to: '/blog'},
              {label: 'GitHub', href: 'https://github.com/CyperPan'},
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Zhiyuan Pan. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['python', 'bash', 'cpp', 'java'],
      },
    }),
};

export default config;
