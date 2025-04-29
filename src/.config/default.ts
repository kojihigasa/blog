import type { ThemeConfig } from '~/types'

// This is the default configuration for the template, please do not modify it directly.
// You can override this configuration in the `.config/user.ts` file.

export const defaultConfig: ThemeConfig = {
  site: {
    title: 'Koji Higasa Blog',
    subtitle: 'What you want is what you need',
    author: 'Koji Higasa',
    description: 'Koji Higasa Blog',
    website: 'https://kojihigasa.github.io',
    pageSize: 5,
    socialLinks: [
      {
        name: 'github',
        href: 'https://github.com/kojihigasa',
      },
      {
        name: 'rss',
        href: '/atom.xml',
      },
      {
        name: 'twitter',
        href: 'https://twitter.com/kojihigasa',
      },
      {
        name: 'linkedin',
        href: 'https://www.linkedin.com/in/koji-higasa-21a83a171/',
      },
    ],
    navLinks: [
      {
        name: 'Posts',
        href: '/blog',
      },
      {
        name: 'Archive',
        href: '/blog/archive',
      },
      {
        name: 'Categories',
        href: '/blog/categories',
      },
      {
        name: 'About',
        href: '/blog/about',
      },
    ],
    categoryMap: [{ name: '胡适', path: 'hu-shi' }],
    footer: [
      '© %year <a target="_blank" href="%website">%author</a>',
    ],
  },
  appearance: {
    theme: 'system',
    locale: 'en-us',
    colorsLight: {
      primary: '#2e405b',
      background: '#ffffff',
    },
    colorsDark: {
      primary: '#FFFFFF',
      background: '#232222',
    },
    fonts: {
      header:
        '"HiraMinProN-W6","Source Han Serif CN","Source Han Serif SC","Source Han Serif TC",serif',
      ui: '"Source Sans Pro","Roboto","Helvetica","Helvetica Neue","Source Han Sans SC","Source Han Sans TC","PingFang SC","PingFang HK","PingFang TC",sans-serif',
    },
  },
  seo: {
    twitter: '@kojihigasa',
    meta: [],
    link: [],
  },
  rss: {
    fullText: true,
  },
  comment: {
    disqus: { shortname: "koji-higasa-blog" },
  },
  analytics: {
    googleAnalyticsId: '',
    umamiAnalyticsId: '',
  },
  latex: {
    katex: false,
  },
}
