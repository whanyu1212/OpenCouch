<!-- BEGIN:nextjs-agent-rules -->

# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` (resolved from this file's directory; in monorepos the `next` package may not be visible from the repo root) before writing any code. Heed deprecation notices.

This block is written and re-added by `next dev` — verify at `node_modules/next/dist/server/lib/generate-agent-files.js`. Removing it from a diff only re-creates the uncommitted change; committing it with your work keeps the tree clean.

<!-- END:nextjs-agent-rules -->

# Web app

The root `AGENTS.md` covers the layout, commands and architecture rules. These
are the standards for the TypeScript code here.

## TypeScript standards

- TypeScript strict; no `any`. ESLint must pass.
- Prettier formats the code (`pnpm format:web`); CI fails on unformatted files.
- Components render from typed AG-UI messages and state; keep protocol handling
  in `src/lib/`, not in components.
- Tests use Vitest and sit next to the code as `*.test.ts`. Keep the logic
  worth testing in plain functions in `src/lib/`, so it can be tested without
  a DOM or a network.
