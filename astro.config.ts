import mdx from '@astrojs/mdx'
import sitemap from '@astrojs/sitemap'
import swup from '@swup/astro'
import robotsTxt from 'astro-robots-txt'
import { defineConfig } from 'astro/config'
import rehypeKatex from 'rehype-katex'
import remarkMath from 'remark-math'
import UnoCSS from 'unocss/astro'
import { themeConfig } from './src/.config'

// https://astro.build/config
export default defineConfig({
  site: themeConfig.site.website,
  base: 'blog',
  prefetch: true,
  markdown: {
    remarkPlugins: [
      remarkMath,
    ],
    rehypePlugins: [
      [rehypeKatex, { output: 'mathml' }] // thanks to https://weekend-thinker.com/blog/computer/7/
    ],
    shikiConfig: {
      theme: 'dracula',
      wrap: true,
    },
  },
  integrations: [
    UnoCSS({ injectReset: true }),
    mdx({}),
    robotsTxt(),
    sitemap(),
    swup({
      theme: false,
      animationClass: 'transition-swup-',
      cache: true,
      preload: true,
      accessibility: true,
      smoothScrolling: true,
      updateHead: true,
      updateBodyClass: true,
    }),
  ],
})
