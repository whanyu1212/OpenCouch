FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /workspace/apps/backend

COPY apps/backend/pyproject.toml apps/backend/uv.lock ./
RUN uv sync --locked --no-dev

WORKDIR /workspace

COPY apps/backend ./apps/backend

WORKDIR /workspace/apps/backend

ENV PATH="/workspace/apps/backend/.venv/bin:$PATH" \
    PORT=8080

EXPOSE 8080

CMD ["python", "run_server.py"]
