FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/opencouch-backend-venv

WORKDIR /workspace/apps/backend

COPY apps/backend/pyproject.toml apps/backend/uv.lock ./

FROM base AS api-dev

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

WORKDIR /workspace
COPY apps/backend ./apps/backend
WORKDIR /workspace/apps/backend

ENV PATH="/opt/opencouch-backend-venv/bin:$PATH" \
    HOST=0.0.0.0 \
    PORT=8080

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]

FROM base AS prod

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

WORKDIR /workspace
COPY apps/backend ./apps/backend

WORKDIR /workspace/apps/backend

ENV PATH="/opt/opencouch-backend-venv/bin:$PATH" \
    PORT=8080

EXPOSE 8080

CMD ["python", "run_server.py"]
