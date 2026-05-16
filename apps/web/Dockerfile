FROM node:22-bookworm-slim AS base

WORKDIR /workspace

ENV CI=true

RUN corepack enable && corepack prepare pnpm@10.33.0 --activate

FROM base AS dev

EXPOSE 3000

CMD ["sh", "apps/web/docker-entrypoint.dev.sh"]

FROM base AS builder

COPY pnpm-lock.yaml pnpm-workspace.yaml ./
COPY apps/web/package.json apps/web/package.json

RUN --mount=type=cache,target=/root/.local/share/pnpm/store \
    pnpm --dir apps/web install --frozen-lockfile

COPY apps/web apps/web

ARG NEXT_PUBLIC_API_URL=http://localhost:8080/api
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}

RUN pnpm --dir apps/web build

FROM base AS prod

ENV NODE_ENV=production

COPY --from=builder /workspace/node_modules node_modules
COPY --from=builder /workspace/apps/web/node_modules apps/web/node_modules
COPY --from=builder /workspace/apps/web/.next apps/web/.next
COPY --from=builder /workspace/apps/web/public apps/web/public
COPY --from=builder /workspace/apps/web/package.json apps/web/package.json
COPY pnpm-workspace.yaml ./

EXPOSE 3000

CMD ["pnpm", "--dir", "apps/web", "start", "--hostname", "0.0.0.0"]
