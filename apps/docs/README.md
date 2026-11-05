# Rewrite documentation

Astro Starlight documentation for the implementation on `develop`.
From the repository root:

```sh
pnpm install --frozen-lockfile
pnpm dev:docs
pnpm check:docs
pnpm build:docs
pnpm preview:docs
```

Open `http://localhost:4321/OpenCouch/`. The preview serves the production build,
including search. Node 22.12+ is required by the pinned Astro 6 release.

Relevant PRs and pushes to `develop` build-check the site. Only relevant pushes
to `main` deploy to GitHub Pages. Merging into `develop` does not replace the
existing public site. There is no manual deployment trigger.

Keep pages grounded in the current code. Mark planned capabilities explicitly;
date results and link their cases, methodology, and detailed audit notes.
Raw local eval recordings are not needed to build this site.
