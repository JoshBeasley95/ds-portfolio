// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';

// Static site (Vercel auto-detects Astro and serves dist/). Set `site` to the live
// URL once deployed so canonical links / sitemaps resolve correctly.
export default defineConfig({
  site: process.env.SITE_URL || 'https://joshuabeasley.dev',
  integrations: [mdx()],
});
