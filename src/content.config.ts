import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

// Each project's site writeup is one MDX file in src/content/projects/<slug>.mdx,
// emitted by the Job Hunter project generator (chart paths rewritten to /projects/<slug>/charts/).
// Writeups are emitted as .md (not .mdx): they're plain Markdown + frontmatter, and .md
// renders raw `<` / `{` literally — robust against arbitrary LLM-generated prose.
const projects = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/projects' }),
  schema: z.object({
    title: z.string(),
    company: z.string().optional(),
    role: z.string().optional(),
    date: z.string(),
    tags: z.array(z.string()).default([]),
    summary: z.string(),
    featured: z.boolean().optional(),
  }),
});

export const collections = { projects };
