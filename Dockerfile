# Reproducible runtime for the `triage` CLI.
#   docker build -t pytest-triage .
#   docker run --rm --env-file .env -v "$PWD/data:/app/data" -v "$PWD/eval:/app/eval" pytest-triage stats
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.28 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependencies first: this layer is cached until uv.lock changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
RUN uv sync --locked --no-dev

# Run as an unprivileged user.
RUN useradd --create-home triage && chown -R triage /app
USER triage

ENTRYPOINT ["uv", "run", "--no-sync", "triage"]
CMD ["--help"]
